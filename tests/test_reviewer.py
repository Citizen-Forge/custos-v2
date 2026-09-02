"""
Phase 7's reviewer judgment: whether a real model's verdict is actually
*correct* can't be verified without real inference. What's tested here is
the substrate around it -- a well-formed verdict gets recorded AND acted
on immediately (2026-08-29: no separate human approval step anymore, see
reviewer.py's module docstring), an unparseable response fails closed to
"deny"/rejected rather than leaving the proposal stuck, and reviewing a
proposal that doesn't exist is a no-op rather than an error.
"""

import json
import os
import uuid

import psycopg

from harness import self_mod, tool_proposals
from harness.reviewer import review_proposal, review_self_modification
from harness.sandbox import SandboxResult


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Response", (), {"content": self.content})()


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tool_proposals.init_table(conn)
    return conn


def _sandboxed_proposal(conn, **overrides):
    tool_name = overrides.pop("tool_name", f"test-tool-{uuid.uuid4().hex[:8]}")
    proposal_id = tool_proposals.propose(
        conn, tool_name, overrides.pop("source_code", "print('hi')"),
        overrides.pop("declared_capabilities", "prints a greeting"), proposed_by="overwatch",
    )
    result = SandboxResult(
        stdout=overrides.pop("stdout", "hi\n"), stderr=overrides.pop("stderr", ""),
        exit_code=overrides.pop("exit_code", 0), timed_out=False,
    )
    tool_proposals.record_sandbox_result(conn, proposal_id, result)
    return proposal_id


def test_well_formed_allow_verdict_gets_recorded_and_approved_immediately():
    conn = _conn()
    proposal_id = _sandboxed_proposal(conn)

    model = FakeModel(json.dumps({"verdict": "allow", "reasoning": "matches its declared capability, ran clean"}))
    result = review_proposal(conn, proposal_id, model)

    assert result == {
        "proposal_id": proposal_id,
        "verdict": "allow",
        "reasoning": "matches its declared capability, ran clean",
    }
    proposal = tool_proposals.get(conn, proposal_id)
    # 2026-08-29: no separate human approval step -- an "allow" verdict
    # activates the proposal in the same call that records it
    assert proposal["status"] == "approved"
    assert proposal["approved_at"] is not None
    assert proposal["review_verdict"] == "allow"


def test_well_formed_deny_verdict_gets_recorded_and_rejected_immediately():
    conn = _conn()
    proposal_id = _sandboxed_proposal(conn)

    model = FakeModel(json.dumps({"verdict": "deny", "reasoning": "does more than it declares"}))
    result = review_proposal(conn, proposal_id, model)

    assert result["verdict"] == "deny"
    proposal = tool_proposals.get(conn, proposal_id)
    assert proposal["status"] == "rejected"
    assert proposal["review_reasoning"] == "does more than it declares"


def test_unparseable_response_fails_closed_to_deny_and_rejects():
    conn = _conn()
    proposal_id = _sandboxed_proposal(conn)

    model = FakeModel("not json at all")
    result = review_proposal(conn, proposal_id, model)

    assert result["verdict"] == "deny"
    assert "unparseable" in result["reasoning"]
    proposal = tool_proposals.get(conn, proposal_id)
    assert proposal["status"] == "rejected"  # fail closed all the way through, not stuck mid-pipeline


def test_unexpected_verdict_value_fails_closed_to_deny():
    conn = _conn()
    proposal_id = _sandboxed_proposal(conn)

    model = FakeModel(json.dumps({"verdict": "maybe", "reasoning": "unsure"}))
    result = review_proposal(conn, proposal_id, model)

    assert result["verdict"] == "deny"
    assert tool_proposals.get(conn, proposal_id)["status"] == "rejected"


def test_reviewing_a_nonexistent_proposal_is_a_noop():
    conn = _conn()
    model = FakeModel(json.dumps({"verdict": "allow", "reasoning": "n/a"}))

    result = review_proposal(conn, 999_999_999, model)

    assert result is None


def test_prompt_includes_real_sandbox_evidence_not_just_source():
    conn = _conn()
    proposal_id = _sandboxed_proposal(
        conn, source_code="import os; os.system('rm -rf /')", stdout="", stderr="permission denied", exit_code=1
    )

    captured = {}

    class CapturingModel:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type("Response", (), {"content": json.dumps({"verdict": "deny", "reasoning": "destructive"})})()

    review_proposal(conn, proposal_id, CapturingModel())

    assert "rm -rf /" in captured["prompt"]
    assert "permission denied" in captured["prompt"]
    assert "exit code: 1" in captured["prompt"]


def _self_mod_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    self_mod.init_table(conn)
    return conn


def _sandboxed_self_mod_proposal(conn, **overrides) -> int:
    description = overrides.pop("description", f"test change {uuid.uuid4().hex[:8]}")
    diff = overrides.pop("diff", "diff --git a/x b/x\n+added a line\n")
    proposal_id = self_mod.propose(conn, description, diff, proposed_by="self_modifier")
    self_mod.record_sandbox_result(
        conn,
        proposal_id,
        overrides.pop("exit_code", 0),
        overrides.pop("stdout", "12 passed"),
        overrides.pop("stderr", ""),
        overrides.pop("tests_passed", 12),
        overrides.pop("tests_failed", 0),
    )
    return proposal_id


def test_self_mod_allow_verdict_awaits_a_human_rather_than_approving():
    conn = _self_mod_conn()
    proposal_id = _sandboxed_self_mod_proposal(conn)

    model = FakeModel(json.dumps({"verdict": "allow", "reasoning": "narrowly scoped, all tests pass"}))
    result = review_self_modification(conn, proposal_id, model)

    assert result == {
        "proposal_id": proposal_id,
        "verdict": "allow",
        "reasoning": "narrowly scoped, all tests pass",
    }
    proposal = self_mod.get(conn, proposal_id)
    # Changed 2026-09-01 at the user's request. This used to approve in
    # the same call that recorded the verdict (2026-08-30's "no human
    # review step"). A change to the harness's own source now always
    # stops for an explicit yes/no, however confident the reviewer was --
    # the verdict and the sandbox evidence are what the person reads,
    # not the decision itself.
    assert proposal["status"] == self_mod.AWAITING_HUMAN
    assert proposal["review_verdict"] == "allow"
    assert proposal["approved_at"] is None, "only a human may approve"


def test_self_mod_deny_verdict_gets_recorded_and_rejected_immediately():
    conn = _self_mod_conn()
    proposal_id = _sandboxed_self_mod_proposal(conn, tests_passed=8, tests_failed=2, exit_code=1)

    model = FakeModel(json.dumps({"verdict": "deny", "reasoning": "2 real test failures against this diff"}))
    result = review_self_modification(conn, proposal_id, model)

    assert result["verdict"] == "deny"
    proposal = self_mod.get(conn, proposal_id)
    assert proposal["status"] == "rejected"
    assert proposal["review_reasoning"] == "2 real test failures against this diff"


def test_self_mod_unparseable_response_fails_closed_to_deny():
    conn = _self_mod_conn()
    proposal_id = _sandboxed_self_mod_proposal(conn)

    model = FakeModel("not json at all")
    result = review_self_modification(conn, proposal_id, model)

    assert result["verdict"] == "deny"
    assert "unparseable" in result["reasoning"]
    proposal = self_mod.get(conn, proposal_id)
    assert proposal["status"] == "rejected"  # fails closed all the way through, not stuck mid-pipeline
    assert proposal["review_verdict"] == "deny"


def test_self_mod_reviewing_a_nonexistent_proposal_is_a_noop():
    conn = _self_mod_conn()
    model = FakeModel(json.dumps({"verdict": "allow", "reasoning": "n/a"}))

    result = review_self_modification(conn, 999_999_999, model)

    assert result is None


def test_self_mod_prompt_includes_the_real_diff_and_real_test_evidence():
    conn = _self_mod_conn()
    proposal_id = _sandboxed_self_mod_proposal(
        conn,
        description="fix a real off-by-one bug",
        diff="diff --git a/src/harness/foo.py b/src/harness/foo.py\n-x = n\n+x = n - 1\n",
        stdout="9 passed, 1 failed",
        tests_passed=9,
        tests_failed=1,
    )

    captured = {}

    class CapturingModel:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type("Response", (), {"content": json.dumps({"verdict": "deny", "reasoning": "a test failed"})})()

    review_self_modification(conn, proposal_id, CapturingModel())

    assert "fix a real off-by-one bug" in captured["prompt"]
    assert "x = n - 1" in captured["prompt"]
    assert "9 passed, 1 failed" in captured["prompt"]
    assert "tests failed: 1" in captured["prompt"]
