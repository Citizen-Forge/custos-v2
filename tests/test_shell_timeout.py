"""A shell command that hangs is a tool result, not a crashed run.

Regression for 2026-09-13: shell_exec had a hardcoded timeout of 120s and
let subprocess.TimeoutExpired escape the tool node, so one slow command --
typically an edit followed by `npm test` -- killed the whole graph run.
After MAX_TICKET_FAILURES the dispatcher parked the ticket for a human;
~29 Silent Run tickets were stranded this way. The timeout is now
configurable (SHELL_TIMEOUT) and, on expiry, the command's process group
is killed and a readable message is returned to the agent.
"""

import time

from harness import tools


def test_normal_command_returns_output(tmp_path):
    assert "hello" in tools._run_shell("echo hello", str(tmp_path))


def test_timeout_is_returned_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SHELL_TIMEOUT", 1)
    start = time.monotonic()
    out = tools._run_shell("sleep 30", str(tmp_path))
    assert "timed out" in out
    assert time.monotonic() - start < 15


def test_timeout_kills_the_process_group(tmp_path, monkeypatch):
    """The shell's own child must die too: shell=True puts the real work in
    a grandchild, so killing only the direct child would orphan a running
    test runner."""
    monkeypatch.setattr(tools, "SHELL_TIMEOUT", 1)
    marker = tmp_path / "survived"
    tools._run_shell(f"sh -c 'sleep 3; touch {marker}' & wait", str(tmp_path))
    time.sleep(4)
    assert not marker.exists()
