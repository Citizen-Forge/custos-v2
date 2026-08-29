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

from harness import tool_proposals
from harness.reviewer import review_proposal
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
