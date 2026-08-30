"""
Phase 7's self-modification lifecycle -- tests the DB primitives
directly (record_review/approve/reject/mark_deployed), same style as
test_tool_proposals.py: proves the state machine itself, not the
judgment that drives it (reviewer.review_self_modification, tested in
test_reviewer.py, is what actually calls approve/reject automatically
off a real verdict -- see self_mod.py's module docstring for the full
picture, including run_self_mod_deploy.py's own hard, non-LLM-dependent
gate before deployed is ever reachable for real)."""

import os
import uuid

import psycopg

from harness import self_mod


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    self_mod.init_table(conn)
    return conn


def test_full_lifecycle_from_propose_through_approve():
    conn = _conn()
    description = f"test change {uuid.uuid4().hex[:8]}"

    proposal_id = self_mod.propose(conn, description, "diff --git a/x b/x\n", proposed_by="self_modifier")
    assert self_mod.get(conn, proposal_id)["status"] == "pending"

    self_mod.record_sandbox_result(conn, proposal_id, 0, "12 passed", "", 12, 0)
    after_sandbox = self_mod.get(conn, proposal_id)
    assert after_sandbox["status"] == "sandboxed"
    assert after_sandbox["sandbox_tests_passed"] == 12
    assert after_sandbox["sandbox_tests_failed"] == 0

    self_mod.record_review(conn, proposal_id, "allow", "tests pass, change is narrowly scoped")
    after_review = self_mod.get(conn, proposal_id)
    # record_review alone only ever reaches 'reviewed' -- it's
    # reviewer.review_self_modification (test_reviewer.py) that calls
    # approve/reject as a separate, deliberate follow-up call off a
    # real verdict, same shape as tool_proposals.py's own split between
    # "record the verdict" and "act on it"
    assert after_review["status"] == "reviewed"
    assert after_review["review_verdict"] == "allow"

    self_mod.approve(conn, proposal_id)
    approved = self_mod.get(conn, proposal_id)
    assert approved["status"] == "approved"
    assert approved["approved_at"] is not None
    assert approved["deployed_at"] is None


def test_rejected_proposal_never_reaches_approved_or_deployed():
    conn = _conn()
    description = f"test change {uuid.uuid4().hex[:8]}"
    proposal_id = self_mod.propose(conn, description, "diff --git a/x b/x\n", proposed_by="self_modifier")

    self_mod.record_sandbox_result(conn, proposal_id, 1, "", "3 failed", 5, 3)
    self_mod.record_review(conn, proposal_id, "deny", "3 real test failures against this exact diff")
    self_mod.reject(conn, proposal_id)

    rejected = self_mod.get(conn, proposal_id)
    assert rejected["status"] == "rejected"
    assert rejected["approved_at"] is None
    assert rejected["deployed_at"] is None


def test_mark_deployed_only_reachable_after_approval():
    conn = _conn()
    description = f"test change {uuid.uuid4().hex[:8]}"
    proposal_id = self_mod.propose(conn, description, "diff --git a/x b/x\n", proposed_by="self_modifier")
    self_mod.record_sandbox_result(conn, proposal_id, 0, "1 passed", "", 1, 0)
    self_mod.record_review(conn, proposal_id, "allow", "fine")
    self_mod.approve(conn, proposal_id)

    self_mod.mark_deployed(conn, proposal_id)

    deployed = self_mod.get(conn, proposal_id)
    assert deployed["status"] == "deployed"
    assert deployed["deployed_at"] is not None


def test_list_by_status_filters_correctly():
    conn = _conn()
    description = f"test change {uuid.uuid4().hex[:8]}"
    proposal_id = self_mod.propose(conn, description, "diff --git a/x b/x\n", proposed_by="self_modifier")

    pending = self_mod.list_by_status(conn, "pending")
    assert any(p["id"] == proposal_id for p in pending)

    self_mod.record_sandbox_result(conn, proposal_id, 0, "", "", 1, 0)

    still_pending = self_mod.list_by_status(conn, "pending")
    assert not any(p["id"] == proposal_id for p in still_pending)
    sandboxed = self_mod.list_by_status(conn, "sandboxed")
    assert any(p["id"] == proposal_id for p in sandboxed)


def test_sandbox_result_distinguishes_incomplete_run_from_zero_tests():
    """A diff that never even applied, or a build that never completed,
    should be recorded as tests_passed/failed=None -- distinct from a
    real run that happened to touch zero tests -- so the reviewer
    prompt can tell "couldn't evaluate this" from "evaluated, found
    nothing.\""""
    conn = _conn()
    description = f"test change {uuid.uuid4().hex[:8]}"
    proposal_id = self_mod.propose(conn, description, "not a valid diff", proposed_by="self_modifier")

    self_mod.record_sandbox_result(conn, proposal_id, 1, "", "patch did not apply", None, None)

    result = self_mod.get(conn, proposal_id)
    assert result["sandbox_tests_passed"] is None
    assert result["sandbox_tests_failed"] is None
    assert result["status"] == "sandboxed"
