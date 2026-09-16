"""The escalation queue and its bounded resolution budget.

Tickets other agents park via flag_for_human (the `human` label) are the
product-owner's escalation queue. Most are resolvable by re-scoping or
requeueing; the per-ticket budget is what stops an unresolvable one from
cycling forever instead of reaching a person.
"""

import os

import psycopg
import pytest

from harness import beads, escalations, verifications


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()
    # pending() reads a ticket's last verdict to decide whether a CLOSED
    # escalation is still worth reconsidering.
    conn = _conn()
    verifications.init_table(conn)
    conn.close()


def _conn():
    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


def _escalated(seat):
    project = beads.create("esc proj", "d", issue_type="epic", priority=1)
    story = beads.create("esc story", "d", parent=project["id"], acceptance_criteria="do it")
    beads.assign_to_seat(story["id"], seat)
    beads.claim(story["id"], actor=seat)
    beads.flag_for_human(story["id"], "needs a decision")
    return beads.show(story["id"])


def test_pending_includes_a_live_escalation():
    story = _escalated("esc-a")
    assert story["id"] in {i["id"] for i in escalations.pending(_conn())}


def test_a_ticket_closed_without_a_verdict_is_not_pending():
    """Closed is not on its own enough to re-open the queue: a ticket
    dismissed by hand was answered, not left hanging."""
    story = _escalated("esc-b")
    beads.dismiss_human(story["id"], "resolved by hand")

    assert story["id"] not in {i["id"] for i in escalations.pending(_conn())}


def test_a_closed_ticket_with_a_failing_verdict_is_pending():
    """The case no agent can answer: the attempt refused because the
    ticket's named deliverable is already in the project, so the ticket is
    closed and verifier-exhausted and nothing an agent does can move it.
    Someone with the authority to accept the delivery has to see it."""
    story = _escalated("esc-e")
    beads.close(story["id"], reason="already delivered")
    verifications.record(_conn(), story["id"], "esc-e", "fail", "nothing left to do")

    assert story["id"] in {i["id"] for i in escalations.pending(_conn())}


def test_a_closed_ticket_that_passed_is_not_pending():
    story = _escalated("esc-f")
    beads.close(story["id"], reason="done")
    verifications.record(_conn(), story["id"], "esc-f", "pass", "the criteria are met")

    assert story["id"] not in {i["id"] for i in escalations.pending(_conn())}


def test_requeue_spends_one_attempt_and_reopens():
    story = _escalated("esc-c")

    n = escalations.requeue(_conn(), story["id"], "cause fixed; try again")

    assert n == 1
    current = beads.show(story["id"])
    assert current["status"] == "open", "a requeued escalation must be dispatchable"
    assert beads.is_flagged_for_human(current) is False, "the human flag must clear"
    assert int((current.get("metadata") or {}).get("escalation_attempts")) == 1


def test_capped_escalations_drop_out_of_the_queue():
    story = _escalated("esc-d")
    beads.set_metadata(story["id"], "escalation_attempts", str(escalations.MAX_ESCALATION_ATTEMPTS))

    assert story["id"] not in {i["id"] for i in escalations.pending(_conn())}


def test_a_closed_failing_escalation_still_respects_the_budget():
    """Re-opening the queue to closed tickets must not make it unbounded:
    the budget still applies, or the same ticket is reconsidered for ever."""
    story = _escalated("esc-g")
    beads.close(story["id"], reason="already delivered")
    verifications.record(_conn(), story["id"], "esc-g", "fail", "nothing left to do")
    beads.set_metadata(story["id"], "escalation_attempts", str(escalations.MAX_ESCALATION_ATTEMPTS))

    assert story["id"] not in {i["id"] for i in escalations.pending(_conn())}
