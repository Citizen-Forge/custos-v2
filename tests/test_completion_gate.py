"""A ticket must not close on silence.

work_one_ticket used to call beads.close() whenever the graph finished
without refusing, so an agent that talked for a while and stopped was
recorded as success. On 2026-09-01 four Silent Run and Custos stories
were closed exactly that way -- no notes, no summary, no code anywhere in
the workspace -- including "Ship movement over system-scale distances".

Closure now requires an explicit complete_ticket claim, and stories carry
acceptance criteria so verifier.py actually runs on them (it returns None
for any ticket without criteria -- which was every one of 128).
"""

import pytest
from fastapi.testclient import TestClient

from harness import beads, worker
from harness.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()


class StubRuntime:
    """Stands in for a seat runtime. The graph is a no-op, so these tests
    exercise the close/flag decision rather than any model behaviour."""

    def __init__(self, seat_id):
        self.seat_id = seat_id
        self.system_prompt = None
        self.who = seat_id
        self.graph = self

    def get_state(self, config):
        class S:
            values = {"already": "resumed"}
        return S()

    def invoke(self, *a, **k):
        return None


class CapturingRuntime:
    """A fresh-start runtime (no existing graph state) that records the
    initial state worker.work_one_ticket passes to the graph, so the opening
    prompt can be asserted on."""

    def __init__(self, seat_id, captured):
        self.seat_id = seat_id
        self.system_prompt = None
        self.who = seat_id
        self.captured = captured
        self.graph = self

    def get_state(self, config):
        class S:
            values = None  # falsy -> worker takes the "starting thread" path

        return S()

    def invoke(self, state, config=None):
        self.captured["state"] = state
        return None


def _assigned_story(seat_id):
    project = beads.create("gate proj", "d", issue_type="epic", priority=1)
    story = beads.create("gate story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], seat_id)
    beads.claim(story["id"], actor=seat_id)
    return beads.show(story["id"])


def test_no_completion_claim_does_not_close():
    story = _assigned_story("gate-seat")

    outcome = worker.work_one_ticket(StubRuntime("gate-seat"), story)

    assert outcome == "unclaimed"
    current = beads.show(story["id"])
    assert current["status"] != "closed", "silence must never count as success"


def test_no_completion_claim_flags_for_a_human():
    story = _assigned_story("gate-seat-2")

    worker.work_one_ticket(StubRuntime("gate-seat-2"), story)

    assert beads.is_flagged_for_human(beads.show(story["id"])) is True


def test_a_claimed_completion_closes_with_its_summary():
    story = _assigned_story("gate-seat-3")
    beads.set_metadata(story["id"], "completion_summary", "wrote src/sim/tick.ts and tests")

    outcome = worker.work_one_ticket(StubRuntime("gate-seat-3"), beads.show(story["id"]))

    assert outcome == "closed"
    current = beads.show(story["id"])
    assert current["status"] == "closed"
    assert "tick.ts" in (current.get("close_reason") or "")


def test_close_blocked_by_an_open_blocker_is_not_a_failure(monkeypatch):
    """bd refuses to close a ticket whose blocker is still open -- which
    happens while a re-opened blocker is mid-rework. That is not a crash (the
    work is done and committed), so it must not burn the retry budget and
    park the ticket for a human. Regression for workspace-9jg.11.6."""
    story = _assigned_story("blocked-close-seat")
    beads.set_metadata(story["id"], "completion_summary", "did the work")

    def refuse(*a, **k):
        raise beads.BeadsError("cannot close: blocked by open issues [x]")

    monkeypatch.setattr(beads, "close", refuse)

    outcome = worker.work_one_ticket(StubRuntime("blocked-close-seat"), beads.show(story["id"]))

    assert outcome == "blocked"
    # Park it so it cannot pollute next_assigned_ticket for later tests
    # (the close path is stubbed out in this test).
    beads.flag_for_human(story["id"], "test cleanup")


def test_refusal_still_wins_over_the_completion_gate():
    """refuse_ticket must keep parking a ticket for a human rather than
    falling through to the unclaimed path."""
    story = _assigned_story("gate-seat-4")
    beads.flag_for_human(story["id"], "genuinely ambiguous")

    outcome = worker.work_one_ticket(StubRuntime("gate-seat-4"), beads.show(story["id"]))

    assert outcome == "flagged"
    assert beads.show(story["id"])["status"] != "closed"


def test_rework_reason_is_in_the_initial_prompt():
    """A verifier-requeued ticket carries the finding that failed it; the
    fresh attempt's opening prompt must include it, or the agent re-submits
    the same work and the rework loop burns its budget for nothing."""
    story = _assigned_story("rework-seat")
    beads.set_metadata(story["id"], "rework_reason", "the suite never asserted determinism")

    captured = {}
    worker.work_one_ticket(CapturingRuntime("rework-seat", captured), beads.show(story["id"]))

    prompt = captured["state"]["messages"][-1][1]
    assert "the suite never asserted determinism" in prompt


def test_acceptance_criteria_are_in_the_initial_prompt():
    """The criteria live in the ticket's metadata, and the agent has no
    tool that reads a ticket -- only its own workspace. The escalation role
    sets them THERE and tells the agent to "read them before implementing",
    so if they never reach the opening prompt the agent refuses instead,
    every time, until its escalation budget is gone. Found live 2026-09-15:
    6.3, 6.5 and 13.5 parked with "no quotable acceptance criteria ...
    anywhere in the turn"."""
    story = _assigned_story("criteria-seat")
    beads.set_acceptance_criteria(story["id"], "a saturated sink stops absorbing")

    captured = {}
    worker.work_one_ticket(
        CapturingRuntime("criteria-seat", captured), beads.show(story["id"])
    )

    prompt = captured["state"]["messages"][-1][1]
    assert "a saturated sink stops absorbing" in prompt


# -- acceptance criteria ---------------------------------------------


def test_story_creation_accepts_acceptance_criteria():
    project = client.post(
        "/projects", json={"name": "ac proj", "description": "d", "priority": 2}
    ).json()
    epic = client.post(
        f"/projects/{project['id']}/epics", json={"title": "ac epic", "description": "d"}
    ).json()

    story = client.post(
        f"/epics/{epic['id']}/stories",
        json={
            "title": "ac story",
            "description": "d",
            "acceptance_criteria": "tick loop is deterministic under a fixed seed",
        },
    ).json()

    assert beads.acceptance_criteria(beads.show(story["id"])) == (
        "tick loop is deterministic under a fixed seed"
    )


def test_acceptance_criteria_stays_optional():
    """Existing callers omit it; they must keep working."""
    project = client.post(
        "/projects", json={"name": "ac2 proj", "description": "d", "priority": 2}
    ).json()
    epic = client.post(
        f"/projects/{project['id']}/epics", json={"title": "ac2 epic", "description": "d"}
    ).json()

    response = client.post(
        f"/epics/{epic['id']}/stories", json={"title": "ac2 story", "description": "d"}
    )

    assert response.status_code == 200


def test_a_story_with_criteria_is_a_verifier_candidate():
    """verifier.verify_ticket returns None for any ticket with no
    criteria -- which is why 128 stories produced 0 verifications."""
    project = beads.create("ac3 proj", "d", issue_type="epic", priority=1)
    story = beads.create("ac3 story", "d", parent=project["id"],
                         acceptance_criteria="does the thing")

    assert beads.acceptance_criteria(beads.show(story["id"])) == "does the thing"
