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
"""

import os
import subprocess

import psycopg

from harness import self_mod

REPO_SRC = "/repo-src"
REPO_HOST_PATH = os.environ["REPO_HOST_PATH"]
TEST_TIMEOUT_SECONDS = 900


def _compose_files() -> list[str]:
    files = ["docker-compose.yml"]
    if os.path.exists(os.path.join(REPO_SRC, "docker-compose.prod.yml")):
        files.append("docker-compose.prod.yml")
    return files


def _compose_cmd(*args: str) -> list[str]:
    cmd = ["docker", "compose"]
    for f in _compose_files():
        cmd += ["-f", f]
    return cmd + list(args)


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
    try:
        test_run = subprocess.run(
            _compose_cmd("run", "--rm", "harness", "pytest", "-q"),
            cwd=REPO_HOST_PATH,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(f"#{proposal_id}: final test run timed out -- rolling back, NOT deploying")
        subprocess.run(["git", "checkout", "--", "."], cwd=REPO_SRC)
        return

    print(test_run.stdout[-2000:])
    if test_run.returncode != 0:
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

    subprocess.run(_compose_cmd("up", "-d", "--build", "harness", "api", "scheduler"), cwd=REPO_HOST_PATH, check=True)

    self_mod.mark_deployed(conn, proposal_id)
    print(f"#{proposal_id}: deployed.")


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
