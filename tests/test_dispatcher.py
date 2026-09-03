"""Capacity-driven dispatch and the speciality-decline path.

Written against the real failure this replaced (2026-08-31): seats existed
with tickets assigned to them and nothing ran, because the only worker
process was bound to a different seat id via SEAT_ID and no process was
ever created for a product-owner-created seat.
"""

import pytest

from harness import beads, dispatcher
from harness.routing import RoutingTable


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()


def _dispatcher(max_agents=1):
    return dispatcher.Dispatcher("postgresql://unused", RoutingTable({}), max_agents=max_agents)


# -- what counts as dispatchable work ---------------------------------


def test_projects_and_epics_are_not_dispatchable():
    """Projects and epics are Beads issues too and appear in `bd ready`
    (verified live -- a project came back as a ready ticket). Assigning
    one to a seat would hand an agent a whole project as a task."""
    project = beads.create("dispatch proj", "d", issue_type="epic", priority=1)
    story = beads.create("dispatch story", "d", parent=project["id"])

    assert dispatcher.dispatchable(beads.show(project["id"])) is False
    assert dispatcher.dispatchable(beads.show(story["id"])) is True


def test_next_assigned_ticket_finds_assigned_work():
    project = beads.create("assigned proj", "d", issue_type="epic", priority=1)
    story = beads.create("assigned story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "some-seat")

    issue, seat_id = dispatcher.next_assigned_ticket()

    assert issue is not None
    assert seat_id == "some-seat"
    assert issue["id"] == story["id"]


def test_unassigned_work_is_detected():
    project = beads.create("unassigned proj", "d", issue_type="epic", priority=1)
    beads.create("unassigned story", "d", parent=project["id"])

    assert dispatcher.has_unassigned_work() is True


# -- capacity ---------------------------------------------------------


def test_capacity_respects_max_agents():
    d = _dispatcher(max_agents=2)
    assert d.capacity() == 2


def test_start_agent_refuses_beyond_max():
    d = _dispatcher(max_agents=1)
    # Occupy the only slot without actually running a model.
    with d._lock:
        d._running["seat-a"] = {"ticket_id": "x", "title": "x", "started_at": 0}

    assert d.capacity() == 0
    assert d.start_agent("seat-b", {"id": "y", "title": "y"}) is False


def test_start_agent_refuses_second_ticket_for_same_seat():
    """Two threads on one seat would race to claim and resume the same
    LangGraph thread."""
    d = _dispatcher(max_agents=4)
    with d._lock:
        d._running["seat-a"] = {"ticket_id": "x", "title": "x", "started_at": 0}

    assert d.start_agent("seat-a", {"id": "z", "title": "z"}) is False


def test_tick_reports_at_capacity_without_touching_beads():
    d = _dispatcher(max_agents=1)
    with d._lock:
        d._running["seat-a"] = {"ticket_id": "x", "title": "x", "started_at": 0}

    assert d.tick() == "at capacity"


# -- speciality decline -----------------------------------------------


def test_decline_returns_ticket_to_the_pool():
    project = beads.create("decline proj", "d", issue_type="epic", priority=1)
    story = beads.create("decline story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "wrong-seat")
    beads.claim(story["id"], actor="wrong-seat")

    beads.release_to_pool(story["id"], "wrong-seat", "not my speciality")

    current = beads.show(story["id"])
    assert beads.assigned_seat(current) is None, "should no longer be assigned"
    assert current["status"] == "open", "must reopen -- bd ready only returns open issues"
    assert "wrong-seat" in beads.declined_by(current)


def test_decline_is_not_the_human_escalation_path():
    """refuse_ticket labels 'human' and parks the ticket for a person;
    worker._next_ticket skips flagged issues so they are never reclaimed.
    A speciality decline must stay out of that queue."""
    project = beads.create("decline2 proj", "d", issue_type="epic", priority=1)
    story = beads.create("decline2 story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "seat-x")

    beads.release_to_pool(story["id"], "seat-x", "wrong specialist")

    assert beads.is_flagged_for_human(beads.show(story["id"])) is False


def test_declines_accumulate_across_seats():
    project = beads.create("decline3 proj", "d", issue_type="epic", priority=1)
    story = beads.create("decline3 story", "d", parent=project["id"])

    beads.release_to_pool(story["id"], "seat-a", "nope")
    beads.release_to_pool(story["id"], "seat-b", "also nope")

    declined = beads.declined_by(beads.show(story["id"]))
    assert declined == ["seat-a", "seat-b"]


def test_declined_ticket_is_dispatchable_again():
    """The whole point: it goes back in the pool for someone else."""
    project = beads.create("decline4 proj", "d", issue_type="epic", priority=1)
    story = beads.create("decline4 story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "seat-a")
    beads.claim(story["id"], actor="seat-a")
    beads.release_to_pool(story["id"], "seat-a", "nope")

    ready_ids = {i["id"] for i in beads.ready()}
    assert story["id"] in ready_ids


# -- claiming ---------------------------------------------------------


def test_start_agent_claims_the_ticket():
    """Regression: the first cut never claimed. `bd ready` only returns
    open issues, so an unclaimed ticket stayed in the pool while an agent
    worked it, and running_agents() (which reads in_progress) reported
    nothing running."""
    project = beads.create("claim proj", "d", issue_type="epic", priority=1)
    story = beads.create("claim story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "claim-seat")

    d = _dispatcher(max_agents=1)
    # Real claim, but no model work: the agent thread will fail to
    # connect and release its slot, which is fine -- the claim is what
    # this asserts, and it happens synchronously before the thread runs.
    d.start_agent("claim-seat", beads.show(story["id"]))

    current = beads.show(story["id"])
    assert current["status"] == "in_progress"
    assert story["id"] not in {i["id"] for i in beads.ready()}


def test_orphaned_in_progress_work_is_picked_up_again():
    """A ticket left in_progress by a crashed agent never reappears in
    `bd ready`, so dispatch has to look for it explicitly or it strands."""
    project = beads.create("orphan proj", "d", issue_type="epic", priority=1)
    story = beads.create("orphan story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "orphan-seat")
    beads.claim(story["id"], actor="orphan-seat")

    issue, seat_id = dispatcher.next_assigned_ticket()

    assert issue is not None and issue["id"] == story["id"]
    assert seat_id == "orphan-seat"


def test_human_flagged_work_is_not_treated_as_an_orphan():
    project = beads.create("orphan2 proj", "d", issue_type="epic", priority=1)
    story = beads.create("orphan2 story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "parked-seat")
    beads.claim(story["id"], actor="parked-seat")
    beads.flag_for_human(story["id"], "needs a call")

    issue, _ = dispatcher.next_assigned_ticket()

    assert issue is None or issue["id"] != story["id"]


# -- observability ----------------------------------------------------


def test_running_agents_excludes_human_flagged_work():
    project = beads.create("running proj", "d", issue_type="epic", priority=1)
    parked = beads.create("parked story", "d", parent=project["id"])
    beads.assign_to_seat(parked["id"], "seat-p")
    beads.claim(parked["id"], actor="seat-p")
    beads.flag_for_human(parked["id"], "needs a decision")

    ids = {a["ticket_id"] for a in dispatcher.running_agents()}
    assert parked["id"] not in ids, "parked for a human is not 'running'"


def test_running_agents_reports_claimed_work():
    project = beads.create("running2 proj", "d", issue_type="epic", priority=1)
    active = beads.create("active story", "d", parent=project["id"])
    beads.assign_to_seat(active["id"], "seat-r")
    beads.claim(active["id"], actor="seat-r")

    running = {a["ticket_id"]: a for a in dispatcher.running_agents()}
    assert active["id"] in running
    assert running[active["id"]]["seat_id"] == "seat-r"
