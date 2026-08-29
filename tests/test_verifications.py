"""
Storage layer for the acceptance-criteria verification loop. Runs against
real Postgres, same style as test_tool_proposals.py -- proves the
summary() aggregation meta_agent.py relies on, and that re-verifying the
same issue updates in place (idempotent) rather than accumulating
duplicate rows.
"""

import os
import uuid

import psycopg

from harness import verifications as v


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    v.init_table(conn)
    return conn


def test_record_and_get_for_issue_round_trips():
    conn = _conn()
    issue_id = f"test-issue-{uuid.uuid4().hex[:8]}"

    v.record(conn, issue_id, "some-seat", "pass", "meets the stated criteria")

    result = v.get_for_issue(conn, issue_id)
    assert result["verdict"] == "pass"
    assert result["reasoning"] == "meets the stated criteria"
    assert result["seat_id"] == "some-seat"


def test_get_for_issue_returns_none_when_unverified():
    conn = _conn()
    assert v.get_for_issue(conn, f"never-verified-{uuid.uuid4().hex[:8]}") is None


def test_re_recording_the_same_issue_updates_in_place_not_a_duplicate():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    issue_id = f"test-issue-{uuid.uuid4().hex[:8]}"

    v.record(conn, issue_id, seat_id, "fail", "first pass judgment")
    v.record(conn, issue_id, seat_id, "pass", "corrected judgment")

    result = v.get_for_issue(conn, issue_id)
    assert result["verdict"] == "pass"
    assert result["reasoning"] == "corrected judgment"
    matching_rows = [r for r in v.list_for_seat(conn, seat_id) if r["issue_id"] == issue_id]
    assert len(matching_rows) == 1  # updated in place, not a second row


def test_summary_aggregates_pass_fail_and_collects_fail_reasons():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    v.record(conn, f"issue-a-{uuid.uuid4().hex[:8]}", seat_id, "pass", "met criteria")
    v.record(conn, f"issue-b-{uuid.uuid4().hex[:8]}", seat_id, "fail", "missed the file-size limit")
    v.record(conn, f"issue-c-{uuid.uuid4().hex[:8]}", seat_id, "fail", "output format didn't match spec")

    result = v.summary(conn, seat_id)
    assert result["verified_total"] == 3
    assert result["passed"] == 1
    assert result["failed"] == 2
    assert "missed the file-size limit" in result["fail_reasons"]
    assert "output format didn't match spec" in result["fail_reasons"]


def test_summary_is_empty_not_an_error_for_an_unverified_seat():
    conn = _conn()
    result = v.summary(conn, f"never-verified-seat-{uuid.uuid4().hex[:8]}")
    assert result == {
        "seat_id": result["seat_id"],
        "verified_total": 0,
        "passed": 0,
        "failed": 0,
        "fail_reasons": [],
    }
