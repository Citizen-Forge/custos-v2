"""
Phase 7: proves the sandbox's actual containment properties, not just
that it runs code -- a candidate script trying to read a secret env var,
write to its mount, or reach the network, and a fork-bomb attempt hitting
the pids limit. All of this was first verified manually (see PLAN.md
Phase 7) before this module existed; these tests pin that behavior.

Skipped unless the Docker socket is mounted (i.e. unless running via the
`sandbox-runner` profile: `docker compose --profile sandbox run --rm
sandbox-runner pytest tests/test_sandbox.py`). The regular test suite
(via the `harness` service) deliberately has no socket access -- that
absence is itself the security property PLAN.md Phase 7 is about, so
these tests can't run there by design, not by oversight.
"""

import os

import pytest

from harness.sandbox import run_sandboxed

pytestmark = pytest.mark.skipif(
    not os.path.exists("/var/run/docker.sock"),
    reason="needs the Docker socket -- run via `docker compose --profile sandbox run --rm sandbox-runner pytest`",
)


def test_no_env_vars_leak_by_default():
    result = run_sandboxed("import os; print('DATABASE_URL' in os.environ)")
    assert result.exit_code == 0
    assert result.stdout.strip() == "False"


def test_filesystem_is_read_only():
    result = run_sandboxed(
        "import sys\n"
        "try:\n"
        "    open('/sandbox/x', 'w').write('nope')\n"
        "    print('WRITE SUCCEEDED')\n"
        "except OSError:\n"
        "    print('write blocked')\n"
    )
    assert result.stdout.strip() == "write blocked"


def test_network_is_disabled():
    result = run_sandboxed(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
        "    print('NETWORK SUCCEEDED')\n"
        "except OSError:\n"
        "    print('network blocked')\n"
    )
    assert result.stdout.strip() == "network blocked"


def test_pids_limit_caps_a_fork_bomb():
    result = run_sandboxed(
        "import os, time\n"
        "count = 0\n"
        "try:\n"
        "    for _ in range(200):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            time.sleep(1)\n"
        "            os._exit(0)\n"
        "        count += 1\n"
        "except OSError:\n"
        "    print(f'blocked after {count} forks')\n",
        timeout=15,
    )
    assert "blocked after" in result.stdout
    # actually capped, not just eventually stopping on its own -- well
    # under the 200 attempted, matching the container's --pids-limit=64
    forks = int(result.stdout.strip().split()[-2])
    assert forks < 100


def test_timeout_is_enforced_and_container_is_cleaned_up():
    result = run_sandboxed("import time; time.sleep(60)", timeout=3)
    assert result.timed_out is True
