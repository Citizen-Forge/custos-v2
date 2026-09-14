"""
The acceptance-criteria verification loop's judgment substrate -- runs
against a real Beads ticket (like test_welfare_behaviors.py), since the
point is proving verify_ticket reads and reacts to real ticket state, not
a fake. Whether a real model's pass/fail judgment is actually *correct*
needs real inference to validate (see PLAN.md's real-model session log).
"""

import json
import os

import psycopg

from harness import beads, verifications
from harness.verifier import MAX_VERIFIER_REWORKS, verify_ticket


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Response", (), {"content": self.content})()


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    verifications.init_table(conn)
    return conn


def test_ticket_with_no_acceptance_criteria_is_not_a_candidate():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("no criteria set", "x")
    beads.claim(issue["id"])
    beads.close(issue["id"])

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None
    assert verifications.get_for_issue(conn, issue["id"]) is None


def test_ticket_not_yet_closed_is_not_a_candidate():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("still open", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    # deliberately not closed

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None


def test_well_formed_pass_verdict_gets_recorded():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create(
        "print hello world", "write a script that prints hello world", acceptance_criteria="output is exactly 'hello world'"
    )
    beads.claim(issue["id"])
    beads.append_note(issue["id"], "wrote the script, ran it, output was 'hello world'")
    beads.close(issue["id"], reason="done, verified output matches")

    model = FakeModel(json.dumps({"verdict": "pass", "reasoning": "close reason confirms exact output match"}))
    result = verify_ticket(conn, issue["id"], model)

    # no Custos seat assignment in this test (assigned_seat is unset), so
    # this falls back to Beads' own `assignee` field, which is always
    # populated with the acting actor (DEFAULT_ACTOR here) -- "unknown"
    # is the harder-to-hit fallback for the case where BOTH are somehow
    # empty, not the common case. beads.create()'s own response is a
    # leaner shape than beads.show()'s (no `assignee` key at all) -- look
    # the real value up via show(), the same call verify_ticket itself
    # makes internally, rather than trusting create()'s response shape.
    real_assignee = beads.show(issue["id"])["assignee"]
    assert result == {
        "issue_id": issue["id"],
        "seat_id": real_assignee,
        "verdict": "pass",
        "reasoning": "close reason confirms exact output match",
    }
    stored = verifications.get_for_issue(conn, issue["id"])
    assert stored["verdict"] == "pass"


def test_already_verified_ticket_is_skipped_idempotent():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("already checked", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "pass", "already judged once")

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None
    # untouched -- the original verdict, not overwritten by this no-op call
    assert verifications.get_for_issue(conn, issue["id"])["reasoning"] == "already judged once"


def test_unparseable_response_fails_closed_to_fail():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("bad response test", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])

    result = verify_ticket(conn, issue["id"], FakeModel("not json at all"))

    assert result["verdict"] == "fail"
    assert "unparseable" in result["reasoning"]
    assert verifications.get_for_issue(conn, issue["id"])["verdict"] == "fail"


def test_prompt_includes_real_ticket_evidence():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create(
        "evidence check", "the description text", acceptance_criteria="the specific criteria text"
    )
    beads.claim(issue["id"])
    beads.append_note(issue["id"], "the specific notes text")
    beads.close(issue["id"], reason="the specific close reason text")

    captured = {}

    class CapturingModel:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type("Response", (), {"content": json.dumps({"verdict": "pass", "reasoning": "n/a"})})()

    verify_ticket(conn, issue["id"], CapturingModel())

    assert "the specific criteria text" in captured["prompt"]
    assert "the specific notes text" in captured["prompt"]
    assert "the specific close reason text" in captured["prompt"]


# -- the rework loop --------------------------------------------------
#
# A fail used to be recorded and then parked (reopened only when the test
# suite was objectively broken, otherwise just flagged), so the verifier's
# finding sat unactioned on the board. It now goes back to an agent with
# the finding attached, bounded by MAX_VERIFIER_REWORKS.


def test_failed_verification_requeues_with_the_finding():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("rework me", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "completion_summary", "I did the thing")
    beads.set_metadata(issue["id"], "work_commit", "abc123")
    beads.close(issue["id"], reason="done")

    model = FakeModel(json.dumps({"verdict": "fail", "reasoning": "the thing was not done"}))
    result = verify_ticket(conn, issue["id"], model)

    assert result["verdict"] == "fail"
    current = beads.show(issue["id"])
    assert current["status"] == "open", "a failed ticket must go back in the queue"
    assert beads.is_flagged_for_human(current) is False, "requeued work must be dispatchable"
    meta = current.get("metadata") or {}
    assert int(meta.get("rework_count")) == 1
    assert "the thing was not done" in (meta.get("rework_reason") or "")
    assert not meta.get("completion_summary"), "the old claim must not re-close the ticket"
    assert verifications.get_for_issue(conn, issue["id"])["verdict"] == "fail"


def test_requeue_is_bounded_then_flags_for_a_human():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("exhausted", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "rework_count", str(MAX_VERIFIER_REWORKS))
    beads.close(issue["id"], reason="done")

    result = verify_ticket(
        conn, issue["id"], FakeModel(json.dumps({"verdict": "fail", "reasoning": "still wrong"}))
    )

    assert result["verdict"] == "fail"
    assert beads.is_flagged_for_human(beads.show(issue["id"])) is True


def test_legacy_verdict_is_trusted_not_rejudged():
    """A verdict recorded before work_commit existed (NULL) must be left
    alone -- re-judging every settled ticket would reopen it and re-block
    its dependents."""
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("legacy pass", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "pass", "judged before work_commit existed")

    assert verify_ticket(conn, issue["id"], FakeModel("should never be called")) is None


def test_legacy_verdict_is_rejudged_when_requeued():
    """A requeued ticket's old verdict must not block judging its new work,
    even though that verdict predates the work_commit column."""
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("legacy rework", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "rework_count", "1")
    beads.set_metadata(issue["id"], "work_commit", "newsha")
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "fail", "judged before work_commit existed")

    result = verify_ticket(
        conn, issue["id"], FakeModel(json.dumps({"verdict": "pass", "reasoning": "now it does"}))
    )

    assert result["verdict"] == "pass"
