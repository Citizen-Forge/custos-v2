"""
run_self_mod_deploy.py's own hard, non-LLM-dependent precondition:
_has_clean_sandbox_result is the one thing standing between a
reviewer's "allow" verdict and this script actually touching the real
tree. What's proven here is that check's logic specifically -- the
real git/docker orchestration around it is proven live instead (see
PLAN.md), the same way sandbox.py's own Docker orchestration is
live-tested rather than unit-tested.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
os.environ.setdefault("REPO_HOST_PATH", "/tmp/unused-in-this-test")

from run_self_mod_deploy import _has_clean_sandbox_result  # noqa: E402


def _proposal(**overrides) -> dict:
    base = {"sandbox_tests_passed": 10, "sandbox_tests_failed": 0, "sandbox_exit_code": 0}
    base.update(overrides)
    return base


def test_clean_run_passes():
    assert _has_clean_sandbox_result(_proposal()) is True


def test_any_failed_test_blocks_deployment_regardless_of_verdict():
    assert _has_clean_sandbox_result(_proposal(sandbox_tests_failed=1)) is False


def test_incomplete_run_blocks_deployment():
    """tests_passed=None means the sandbox never completed a real run
    (diff didn't apply, build failed) -- not the same as "ran and found
    zero tests," and must not be treated as safe."""
    assert _has_clean_sandbox_result(_proposal(sandbox_tests_passed=None, sandbox_tests_failed=None)) is False


def test_nonzero_exit_code_blocks_deployment_even_with_zero_failed_count():
    """A crash after pytest's own summary line printed (or a summary
    that only ever showed 0 failed because collection itself blew up)
    shouldn't be trusted just because sandbox_tests_failed happens to
    read 0."""
    assert _has_clean_sandbox_result(_proposal(sandbox_exit_code=2)) is False
