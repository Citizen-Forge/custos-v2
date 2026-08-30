"""
Runs in sandbox-runner only (needs the real Docker socket to build and
run a fully isolated test stack against a real diff -- same privilege
boundary as run_sandbox_for_proposals.py, never harness/api). For each
`pending` self-modification proposal: copies the real repo to an
ephemeral scratch directory, applies the diff there with `patch` (works
on a plain directory, no git repo needed in the copy), builds a real
image from the patched copy, and runs the REAL test suite against it in
a throwaway, uniquely-named isolated network with its own ephemeral
Postgres -- torn down after, never touches the live database or the
real running services. Records real pass/fail counts, not just an exit
code, so the reviewer agent has actual evidence to judge, not a guess.

    docker compose --profile sandbox run --rm sandbox-runner \\
        python scripts/run_self_mod_sandbox.py

Deliberately raw `docker build`/`docker run`/`docker network` calls, NOT
`docker compose` -- found live, not assumed: Docker-out-of-Docker makes
compose's own relative-path resolution unsatisfiable across this
boundary. Build context is read LOCALLY by whichever process invokes
docker (sandbox-runner's own filesystem view, i.e. SCRATCH_CONTAINER_
PATH), but any bind-mount volume compose resolves from a relative path
gets handed to the DAEMON as-is, which needs a HOST-visible path
instead (same class of gotcha sandbox.py's own module docstring already
documents for its simpler single `docker run`) -- one `cwd` can't
satisfy both at once for a multi-service compose invocation. The fix
here is to need no runtime volumes at all: the built image already has
everything baked in via the Dockerfile's own COPY commands (same as the
real deployment), and postgres needs no persistent volume for a
throwaway test database, so nothing here needs a host-resolved path
except the build context itself, which is a plain positional argument
to `docker build`, not something compose has to infer from cwd.

Needs a mount of the WHOLE repo (not just src/tests/scripts, unlike
this project's other sandbox-runner scripts) -- building a real,
testable image needs Dockerfile/requirements.txt too, which harness/
other sandbox scripts never needed since they only ever ran one
already-isolated script, not a full application build. That mount is
read-write (see docker-compose.yml's comment on it), but this script
itself only ever COPIES from it to an ephemeral scratch dir before
applying anything -- it never writes back. Only run_self_mod_deploy.py
actually writes to the real tree.
"""

import os
import re
import shutil
import subprocess
import time
import uuid

import psycopg

from harness import self_mod

REPO_SRC = "/repo-src"
SCRATCH_CONTAINER_PATH = os.environ["SANDBOX_SCRATCH_CONTAINER_PATH"]
# A full image build + real pytest run takes minutes, not seconds --
# measured live elsewhere in this project at 3-6 min for the real suite
# alone, plus build time on top.
TEST_TIMEOUT_SECONDS = 900
POSTGRES_READY_TIMEOUT_SECONDS = 60


def _parse_pytest_summary(stdout: str) -> tuple[int | None, int | None]:
    """Parses pytest's own final summary line (e.g. "12 passed, 2 failed
    in 4.56s"). Returns (None, None) if no such line was ever reached --
    a crash/import error before any test ran, not "zero tests", and the
    reviewer prompt treats that distinction as real evidence, not noise."""
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    error_match = re.search(r"(\d+) error", stdout)
    if passed_match is None and failed_match is None and error_match is None:
        return None, None
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = (int(failed_match.group(1)) if failed_match else 0) + (int(error_match.group(1)) if error_match else 0)
    return passed, failed


def _run_isolated_test(diff: str) -> dict:
    run_id = uuid.uuid4().hex[:12]
    container_dir = os.path.join(SCRATCH_CONTAINER_PATH, run_id)
    image_tag = f"self-mod-test-{run_id}"
    network_name = f"self-mod-net-{run_id}"
    pg_name = f"self-mod-pg-{run_id}"

    try:
        shutil.copytree(
            REPO_SRC,
            container_dir,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "workspace", "sandbox-scratch", ".pytest_cache"
            ),
        )

        patch_path = os.path.join(container_dir, "_self_mod.patch")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff)
        apply_result = subprocess.run(
            ["patch", "-p1", "--input", "_self_mod.patch"], cwd=container_dir, capture_output=True, text=True
        )
        os.remove(patch_path)
        if apply_result.returncode != 0:
            return {
                "exit_code": apply_result.returncode,
                "stdout": apply_result.stdout,
                "stderr": f"diff did not apply cleanly:\n{apply_result.stderr}",
                "tests_passed": None,
                "tests_failed": None,
            }

        # Build context is a plain positional arg -- read locally by
        # whatever process runs `docker build`, i.e. sandbox-runner's own
        # filesystem view. No cwd/relative-path ambiguity here.
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, container_dir],
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
        )
        if build_result.returncode != 0:
            return {
                "exit_code": build_result.returncode,
                "stdout": build_result.stdout,
                "stderr": f"image build failed:\n{build_result.stderr[-3000:]}",
                "tests_passed": None,
                "tests_failed": None,
            }

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
                ready = subprocess.run(
                    ["docker", "exec", pg_name, "pg_isready", "-U", "custos"], capture_output=True
                )
                if ready.returncode == 0:
                    pg_ready = True
                    break
                time.sleep(1)
            if not pg_ready:
                return {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"throwaway postgres never became ready within {POSTGRES_READY_TIMEOUT_SECONDS}s",
                    "tests_passed": None,
                    "tests_failed": None,
                }

            try:
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
                return {
                    "exit_code": test_result.returncode,
                    "stdout": test_result.stdout,
                    "stderr": test_result.stderr,
                    "tests_passed": passed,
                    "tests_failed": failed,
                }
            except subprocess.TimeoutExpired as e:
                return {
                    "exit_code": -1,
                    "stdout": (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                    "stderr": f"timed out after {TEST_TIMEOUT_SECONDS}s",
                    "tests_passed": None,
                    "tests_failed": None,
                }
        finally:
            subprocess.run(["docker", "rm", "-f", pg_name], capture_output=True)
            subprocess.run(["docker", "network", "rm", network_name], capture_output=True)
            subprocess.run(["docker", "image", "rm", "-f", image_tag], capture_output=True)
    finally:
        shutil.rmtree(container_dir, ignore_errors=True)


def main() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        self_mod.init_table(conn)
        pending = self_mod.list_by_status(conn, "pending")

        if not pending:
            print("no self-modification proposals awaiting sandboxing")
            return

        for proposal in pending:
            print(f"sandboxing #{proposal['id']}: {proposal['description'][:80]}")
            result = _run_isolated_test(proposal["diff"])
            self_mod.record_sandbox_result(
                conn,
                proposal["id"],
                result["exit_code"],
                result["stdout"],
                result["stderr"],
                result["tests_passed"],
                result["tests_failed"],
            )
            status = "timed out" if result["exit_code"] == -1 else f"exit {result['exit_code']}"
            print(f"#{proposal['id']}: {status}, passed={result['tests_passed']} failed={result['tests_failed']}")


if __name__ == "__main__":
    main()
