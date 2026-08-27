"""
Phase 7: the sandbox itself -- spawns a maximally-restricted, ephemeral,
single-use sibling container to run candidate tool code, via the host's
Docker daemon (Docker-out-of-Docker through the mounted socket, not a
nested daemon). Trusted control-plane code, never agent-editable -- see
PLAN.md Phase 7's "who holds the Docker socket" section for why that
boundary matters more than any flag below it: this module only runs
inside the dedicated `sandbox-runner` service, which is never reachable
from a ticket's `shell_exec`/`write_file` tool calls.

Every flag here was verified live against real Docker before this module
existed, not assumed:
- `--network none`: no exfiltration, no downloading a second payload.
- `--read-only` + a noexec tmpfs for `/tmp`: candidate code can't persist
  anything, including a second-stage payload to execute later.
- `--cap-drop=ALL`, `--security-opt=no-new-privileges`, non-root `--user`:
  no privilege-escalation path.
- `--pids-limit`: verified live this actually caps a fork-bomb attempt
  (stopped at the configured limit, not silently ignored).
- `--memory` / `--cpus`: resource-exhaustion caps.
- No `-e` flags at all: verified live a candidate script sees no
  environment variables by default -- secrets never leak unless this
  code is changed to explicitly forward them, which it never does.

Docker-out-of-Docker path gotcha, worth stating plainly: `docker run -v
HOST_PATH:...` issued from inside a container is resolved by the HOST
daemon against the HOST filesystem, not the calling container's own
filesystem. A path that exists inside sandbox-runner's own container but
isn't ALSO bind-mounted from the host at the identical host path will
not resolve. `SANDBOX_SCRATCH_HOST_PATH` is the one absolute host path
this module is allowed to know about, used only for wiring the mount --
never treated as a container-local path.
"""

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass

SANDBOX_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def run_sandboxed(source_code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> SandboxResult:
    scratch_root = os.environ["SANDBOX_SCRATCH_CONTAINER_PATH"]
    host_root = os.environ["SANDBOX_SCRATCH_HOST_PATH"]

    run_id = uuid.uuid4().hex[:12]
    container_dir = os.path.join(scratch_root, run_id)
    host_dir = f"{host_root}/{run_id}"
    container_name = f"custos-sandbox-{run_id}"
    os.makedirs(container_dir, exist_ok=True)

    with open(os.path.join(container_dir, "candidate.py"), "w", encoding="utf-8") as f:
        f.write(source_code)

    docker_cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "64",
        "--memory", "128m",
        "--cpus", "1",
        "--user", "1000:1000",
        "--tmpfs", "/tmp:size=8m,noexec",
        "-v", f"{host_dir}:/sandbox:ro",
        "--workdir", "/sandbox",
        SANDBOX_IMAGE,
        "python", "/sandbox/candidate.py",
    ]

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        return SandboxResult(stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode, timed_out=False)
    except subprocess.TimeoutExpired as e:
        # Killing the `docker run` client process does not stop the
        # container on the daemon side -- it's just a client attached to
        # the container's output stream. Explicitly kill it by name so a
        # timed-out sandbox doesn't keep running unbounded.
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return SandboxResult(stdout=e.stdout or "", stderr=e.stderr or "", exit_code=-1, timed_out=True)
    finally:
        shutil.rmtree(container_dir, ignore_errors=True)
