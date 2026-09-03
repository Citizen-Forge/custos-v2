"""
Runs in sandbox-runner only -- the one piece of code in this entire
project with write access to the real repo tree (see docker-compose.
yml's comment on that mount) and the Docker socket needed to rebuild
the live services afterward. Never exposed as an agent tool anywhere;
only reachable via `docker compose --profile sandbox run`, issued
directly on the host.

For each `approved` self-modification proposal: applies its diff to
the REAL `/repo-src` tree, re-runs the REAL test suite there as a
final, non-LLM-dependent gate (the sandboxed run in run_self_mod_
sandbox.py used a separate, ephemeral scratch copy -- re-testing the
actual tree catches drift between then and now, not just trusts a
stale result), and only if that's clean: commits (git history is this
pipeline's actual safety net, not a pre-deployment human step -- see
self_mod.py's module docstring for why) and rebuilds/restarts the
affected services. A proposal whose sandboxed run didn't have a clean
all-passing result never reaches here regardless of its review verdict
-- see _has_clean_sandbox_result below; the reviewer's own judgment is
what approves a proposal, but it can't talk its way around real test
failures being real.

    docker compose --profile sandbox run --rm sandbox-runner \\
        python scripts/run_self_mod_deploy.py

A failed final test run rolls back the applied diff (`git checkout --
.`) and leaves the proposal `approved` (not `deployed`) for a human to
investigate -- it does not retry silently or mark anything reviewed
again.

Real Docker-out-of-Docker issue hit live, not assumed (2026-08-30): a
first version used `docker compose ... run`/`... up --build` directly
for both the final test and the rebuild, both cwd'd at REPO_HOST_PATH
-- but that path only means anything to the HOST daemon, not to this
process itself (a Linux container can't chdir into a Windows/host
path string at all, immediate FileNotFoundError). The final test now
reuses run_self_mod_sandbox.py's own raw docker build/run/network
approach instead of compose (see _build_and_test). The real rebuild
still needs actual Compose (it has to recreate the REAL project's
containers, not a throwaway one) -- split in two: `docker build` with
an explicit container-local context path (no cwd ambiguity, a build
context is always read locally regardless of daemon location), then
`docker compose --project-directory REPO_HOST_PATH up -d` with NO
`--build` flag, so Compose never needs to locally read a build context
it can't reach -- `--project-directory` only needs to work as a string
Compose hands to the daemon for resolving volume paths, not something
this process itself has to access.
"""

import os
import re
import subprocess
import time
import uuid

import psycopg

from harness import self_mod

REPO_SRC = "/repo-src"
REPO_HOST_PATH = os.environ["REPO_HOST_PATH"]
TEST_TIMEOUT_SECONDS = 900
POSTGRES_READY_TIMEOUT_SECONDS = 60

# Derived from REPO_HOST_PATH's basename, matching Compose's own default
# project-naming rule (no explicit `name:` in this project's yaml) --
# what the already-running containers/images/network for the real
# deployment are actually named, on either box (local dev or Unraid).
PROJECT_NAME = os.path.basename(REPO_HOST_PATH.rstrip("/\\"))


def _compose_file_args() -> list[str]:
    args = ["-f", os.path.join(REPO_SRC, "docker-compose.yml")]
    prod_overlay = os.path.join(REPO_SRC, "docker-compose.prod.yml")
    if os.path.exists(prod_overlay):
        args += ["-f", prod_overlay]
    return args


def _parse_pytest_summary(stdout: str) -> tuple[int | None, int | None]:
    """Same parsing as run_self_mod_sandbox.py's own copy -- kept
    separate rather than shared across the two scripts/ entrypoints,
    which aren't set up as an importable package (see test_scheduler.py's
    own sys.path workaround for testing this kind of script directly)."""
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    error_match = re.search(r"(\d+) error", stdout)
    if passed_match is None and failed_match is None and error_match is None:
        return None, None
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = (int(failed_match.group(1)) if failed_match else 0) + (int(error_match.group(1)) if error_match else 0)
    return passed, failed


def _build_and_test(patched_src: str) -> dict:
    """Final gate before committing -- same raw docker build/run/network
    approach as run_self_mod_sandbox.py's own isolated test (see that
    script's module docstring for why: `docker compose` can't be made to
    work across the Docker-out-of-Docker boundary here, since the build
    context needs a container-local read while volume mounts need a
    host-visible path, and one cwd can't satisfy both). This re-tests
    the REAL tree (already `git apply`'d in place by the caller) rather
    than a copy, catching drift between the sandboxed proposal-time
    result and right now."""
    run_id = uuid.uuid4().hex[:12]
    image_tag = f"self-mod-final-{run_id}"
    network_name = f"self-mod-final-net-{run_id}"
    pg_name = f"self-mod-final-pg-{run_id}"

    try:
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, patched_src], capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS
        )
        if build_result.returncode != 0:
            return {"ok": False, "output": f"image build failed:\n{build_result.stderr[-3000:]}"}

        subprocess.run(["docker", "network", "create", network_name], check=True, capture_output=True)
        try:
            subprocess.run(
                [
                    "docker", "run", "-d", "--name", pg_name, "--network", network_name,
                    "-e", "POSTGRES_USER=custos", "-e", "POSTGRES_PASSWORD=custos", "-e", "POSTGRES_DB=custos_harness",
                    "postgres:16-alpine",
                ],
                check=True,
                capture_output=True,
            )
            deadline = time.monotonic() + POSTGRES_READY_TIMEOUT_SECONDS
            pg_ready = False
            while time.monotonic() < deadline:
                if subprocess.run(["docker", "exec", pg_name, "pg_isready", "-U", "custos"], capture_output=True).returncode == 0:
                    pg_ready = True
                    break
                time.sleep(1)
            if not pg_ready:
                return {"ok": False, "output": f"throwaway postgres never became ready within {POSTGRES_READY_TIMEOUT_SECONDS}s"}

            test_result = subprocess.run(
                [
                    "docker", "run", "--rm", "--network", network_name,
                    "-e", f"DATABASE_URL=postgresql://custos:custos@{pg_name}:5432/custos_harness",
                    "-e", "HARNESS_WORKSPACE=/workspace",
                    image_tag, "pytest", "-q",
                ],
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
            )
            passed, failed = _parse_pytest_summary(test_result.stdout)
            ok = test_result.returncode == 0 and passed is not None and failed == 0
            return {"ok": ok, "output": test_result.stdout[-2000:], "passed": passed, "failed": failed}
        finally:
            subprocess.run(["docker", "rm", "-f", pg_name], capture_output=True)
            subprocess.run(["docker", "network", "rm", network_name], capture_output=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"timed out after {TEST_TIMEOUT_SECONDS}s"}
    finally:
        subprocess.run(["docker", "image", "rm", "-f", image_tag], capture_output=True)


def _has_clean_sandbox_result(proposal: dict) -> bool:
    """The hard, mechanical precondition run_self_mod_deploy.py itself
    enforces, independent of what the reviewer model said -- a proposal
    with any real test failure, or an incomplete/never-ran sandboxed
    test (tests_passed is None -- see self_mod.record_sandbox_result's
    own docstring for that distinction), never reaches deployment no
    matter its review_verdict."""
    return (
        proposal["sandbox_tests_passed"] is not None
        and proposal["sandbox_tests_failed"] == 0
        and proposal["sandbox_exit_code"] == 0
    )


def _deploy_one(conn, proposal: dict) -> None:
    proposal_id = proposal["id"]
    print(f"deploying #{proposal_id}: {proposal['description']}")

    if not _has_clean_sandbox_result(proposal):
        print(
            f"#{proposal_id}: refusing to deploy -- sandboxed run wasn't clean "
            f"(passed={proposal['sandbox_tests_passed']}, failed={proposal['sandbox_tests_failed']}, "
            f"exit={proposal['sandbox_exit_code']}), regardless of review verdict"
        )
        return

    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_SRC, capture_output=True, text=True)
    if status.stdout.strip():
        print(f"#{proposal_id}: real tree has uncommitted changes -- refusing to deploy onto a dirty tree")
        return

    patch_path = os.path.join(REPO_SRC, "_self_mod_deploy.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(proposal["diff"])
    apply_result = subprocess.run(["git", "apply", "_self_mod_deploy.patch"], cwd=REPO_SRC, capture_output=True, text=True)
    os.remove(patch_path)
    if apply_result.returncode != 0:
        print(f"#{proposal_id}: diff no longer applies to the real tree (drifted since proposed):\n{apply_result.stderr}")
        return

    print(f"#{proposal_id}: diff applied, running the real test suite against the real tree...")
    final_result = _build_and_test(REPO_SRC)
    print(final_result["output"])
    if not final_result["ok"]:
        print(f"#{proposal_id}: final test run failed against the real tree -- rolling back, NOT deploying")
        subprocess.run(["git", "checkout", "--", "."], cwd=REPO_SRC)
        return

    commit_message = (
        f"Self-modification #{proposal_id}: {proposal['description']}\n\n"
        f"Proposed by: {proposal['proposed_by']}\n"
        f"Review: {proposal['review_verdict']} -- {proposal['review_reasoning']}\n"
        f"Sandboxed test result: {proposal['sandbox_tests_passed']} passed, "
        f"{proposal['sandbox_tests_failed']} failed\n\n"
        "Deployed automatically by scripts/run_self_mod_deploy.py -- no human "
        "review gate; git history is the rollback mechanism for this pipeline."
    )
    subprocess.run(["git", "add", "-A"], cwd=REPO_SRC, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=REPO_SRC, check=True)
    print(f"#{proposal_id}: committed. Rebuilding and restarting affected services...")

    # Build once (container-local context read -- see this module's own
    # docstring for why `docker compose ... --build` can't be trusted
    # across this Docker-out-of-Docker boundary), then tag under every
    # service name that shares this same Dockerfile/context, matching
    # Compose's own per-service image-naming convention.
    image_base = f"{PROJECT_NAME}-harness"
    build_result = subprocess.run(
        ["docker", "build", "-t", f"{image_base}:latest", REPO_SRC], capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS
    )
    if build_result.returncode != 0:
        print(f"#{proposal_id}: real image build failed after a clean test run (unexpected) -- rolling back:\n{build_result.stderr[-2000:]}")
        subprocess.run(["git", "checkout", "--", "."], cwd=REPO_SRC)
        return
    subprocess.run(["docker", "tag", f"{image_base}:latest", f"{PROJECT_NAME}-api:latest"], check=True)
    subprocess.run(["docker", "tag", f"{image_base}:latest", f"{PROJECT_NAME}-scheduler:latest"], check=True)

    # `--project-directory` (a plain string, not something this process
    # needs to itself read) is what lets Compose resolve THIS project's
    # relative volume paths (./src, etc.) against the real HOST location
    # rather than sandbox-runner's own /repo-src view -- `-f` above is
    # already an absolute, container-visible path so reading the yaml
    # itself needs no such trick. Deliberately no `--build` here: the
    # image was already built explicitly above, for the reason in this
    # module's own docstring.
    up_result = subprocess.run(
        ["docker", "compose", "--project-directory", REPO_HOST_PATH, *_compose_file_args(), "up", "-d", "harness", "api", "scheduler"],
        capture_output=True,
        text=True,
    )
    if up_result.returncode != 0:
        print(f"#{proposal_id}: WARNING -- committed but recreating the live services failed:\n{up_result.stderr[-2000:]}")
        print(f"#{proposal_id}: the commit stands (git history has it); a human needs to manually rebuild/restart the affected services.")
        self_mod.mark_deployed(conn, proposal_id)
        return

    self_mod.mark_deployed(conn, proposal_id)
    print(f"#{proposal_id}: deployed.")


def deploy_proposal(conn, proposal: dict) -> None:
    """Public name for _deploy_one, so the unattended loop
    (run_self_mod_loop.py) drives exactly the same deployment path this
    script does -- including its hard gates: a proposal without a clean
    sandbox result never deploys, deployment refuses a dirty tree, and a
    failed final test run rolls the applied diff back."""
    _deploy_one(conn, proposal)


def main() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        self_mod.init_table(conn)
        approved = self_mod.list_by_status(conn, "approved")

        if not approved:
            print("no approved self-modification proposals awaiting deployment")
            return

        for proposal in approved:
            _deploy_one(conn, proposal)


if __name__ == "__main__":
    main()
