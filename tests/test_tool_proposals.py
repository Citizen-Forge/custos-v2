"""
Phase 7: the proposal/review-gate lifecycle. Runs against the real
Postgres service, same style as prompts.py's tests -- proves the state
machine, and specifically that nothing here ever auto-activates a
proposal (unlike seat creation): sandboxing and reviewing both leave
status short of 'approved', which only a direct, human-triggered
`approve` call can reach.
"""

import os
import uuid

import psycopg

from harness import tool_proposals as tp
from harness.sandbox import SandboxResult


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tp.init_table(conn)
    return conn


def test_full_lifecycle_requires_explicit_human_approval_at_the_end():
    conn = _conn()
    tool_name = f"test-tool-{uuid.uuid4().hex[:8]}"

    proposal_id = tp.propose(
        conn, tool_name, "def run(): return 1", "no network, no filesystem writes", proposed_by="overwatch"
    )
    assert tp.get(conn, proposal_id)["status"] == "pending"

    tp.record_sandbox_result(conn, proposal_id, SandboxResult(stdout="1", stderr="", exit_code=0, timed_out=False))
    after_sandbox = tp.get(conn, proposal_id)
    assert after_sandbox["status"] == "sandboxed"
    assert after_sandbox["sandbox_stdout"] == "1"

    tp.record_review(conn, proposal_id, "allow", "stays within declared capabilities")
    after_review = tp.get(conn, proposal_id)
    assert after_review["status"] == "reviewed"  # NOT approved -- a favorable review alone changes nothing
    assert after_review["review_verdict"] == "allow"

    tp.approve(conn, proposal_id)
    approved = tp.get(conn, proposal_id)
    assert approved["status"] == "approved"
    assert approved["approved_at"] is not None


def test_rejected_proposal_never_reaches_approved():
    conn = _conn()
    tool_name = f"test-tool-{uuid.uuid4().hex[:8]}"
    proposal_id = tp.propose(conn, tool_name, "os.system('rm -rf /')", "claims: none", proposed_by="overwatch")

    tp.record_sandbox_result(
        conn, proposal_id, SandboxResult(stdout="", stderr="blocked", exit_code=1, timed_out=False)
    )
    tp.record_review(conn, proposal_id, "deny", "attempts destructive filesystem access outside declared scope")
    tp.reject(conn, proposal_id)

    rejected = tp.get(conn, proposal_id)
    assert rejected["status"] == "rejected"
    assert rejected["approved_at"] is None


def test_list_by_status_filters_correctly():
    conn = _conn()
    tool_name = f"test-tool-{uuid.uuid4().hex[:8]}"
    proposal_id = tp.propose(conn, tool_name, "code", "caps", proposed_by="overwatch")

    pending = tp.list_by_status(conn, "pending")
    assert any(p["id"] == proposal_id for p in pending)

    tp.record_sandbox_result(conn, proposal_id, SandboxResult(stdout="", stderr="", exit_code=0, timed_out=False))

    still_in_pending = tp.list_by_status(conn, "pending")
    assert not any(p["id"] == proposal_id for p in still_in_pending)
    sandboxed = tp.list_by_status(conn, "sandboxed")
    assert any(p["id"] == proposal_id for p in sandboxed)
