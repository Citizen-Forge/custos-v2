"""Holding a project out of dispatch.

Added after a 12-hour waste (2026-09-01): every Custos-improvement ticket
asks for changes to the harness's own source under /app/src, which agents
cannot reach -- permissions.check_within_workspace confines them to
/workspace by design. An agent spent half a day searching for a file it
could never open, and the product-owner had no way to know, so it would
have kept assigning.

The first cut of this had a worse bug than the one it fixed: it stopped
the dispatch loop at a held ticket instead of skipping it, so one held
assigned ticket starved every other project.
"""

import pytest

from harness import beads, dispatcher
from harness.routing import RoutingTable


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()
    dispatcher._hold_cache.update(at=-1e9, held={})


def _project_with_story(name, hold=None):
    project = beads.create(name, "d", issue_type="epic", priority=1)
    story = beads.create(f"{name} story", "d", parent=project["id"])
    if hold:
        beads.set_metadata(project["id"], dispatcher.HOLD_KEY, hold)
    return project, story


def test_held_project_is_reported_with_its_reason():
    project, story = _project_with_story("hold proj", hold="agents cannot reach this source")
    assert dispatcher.project_hold(story["id"]) == "agents cannot reach this source"


def test_unheld_project_has_no_hold():
    project, story = _project_with_story("nohold proj")
    assert dispatcher.project_hold(story["id"]) is None


def test_held_assigned_ticket_is_skipped_not_returned():
    """The regression that mattered: returning a held ticket from
    next_assigned_ticket deadlocked dispatch on it forever."""
    held_p, held_s = _project_with_story("held proj", hold="on hold")
    beads.assign_to_seat(held_s["id"], "seat-h")
    dispatcher._hold_cache.update(at=-1e9, held={})

    issue, _ = dispatcher.next_assigned_ticket()

    assert issue is None or issue["id"] != held_s["id"]


def test_a_held_ticket_does_not_starve_another_project():
    held_p, held_s = _project_with_story("starve held", hold="on hold")
    beads.assign_to_seat(held_s["id"], "seat-a")
    open_p, open_s = _project_with_story("starve open")
    beads.assign_to_seat(open_s["id"], "seat-b")
    dispatcher._hold_cache.update(at=-1e9, held={})

    issue, seat_id = dispatcher.next_assigned_ticket()

    assert issue is not None and issue["id"] == open_s["id"]
    assert seat_id == "seat-b"


def test_held_work_does_not_wake_the_product_owner():
    """Unassigned work in a held project must not look like something to
    broker -- otherwise the product-owner is woken to assign tickets no
    agent could start."""
    beads.ensure_initialized()
    held_p, held_s = _project_with_story("wake held", hold="on hold")
    dispatcher._hold_cache.update(at=-1e9, held={})

    unassigned_ids = {
        i["id"] for i in beads.ready()
        if dispatcher.dispatchable(i) and beads.assigned_seat(i) is None
    }
    assert held_s["id"] in unassigned_ids, "precondition: it is unassigned and ready"

    # It is present in the pool, but must not count as brokerable work.
    from harness import toolchain
    held = dispatcher.held_projects()
    assert toolchain.project_id_for(held_s["id"]) in held
