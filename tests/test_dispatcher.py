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


def test_next_assigned_ticket_finds_assigned_work(monkeypatch):
    # Constant order and no in-progress leftovers keeps this test about
    # *finding* assigned work; roadmap ordering has its own test, and the
    # shared test workspace carries leftovers that would otherwise decide
    # the min.
    monkeypatch.setattr(dispatcher, "_order", lambda issue: ())
    monkeypatch.setattr(beads, "in_progress", lambda: [])
    project = beads.create("assigned proj", "d", issue_type="epic", priority=1)
    story = beads.create("assigned story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "some-seat")
    # Pin the selector's ready pool to this test's ticket. The project holds
    # no other ticket, but earlier tests in the same session leave their own
    # ready assigned tickets in the shared DB, and a constant _order makes
    # them all tie -- so the min was decided by a leftover, not by the
    # ticket under test.
    monkeypatch.setattr(beads, "ready", lambda: [beads.show(story["id"])])

    issue, seat_id = dispatcher.next_assigned_ticket()

    assert issue is not None
    assert seat_id == "some-seat"
    assert issue["id"] == story["id"]


def test_unassigned_work_is_detected():
    project = beads.create("unassigned proj", "d", issue_type="epic", priority=1)
    beads.create("unassigned story", "d", parent=project["id"])

    assert dispatcher.has_unassigned_work() is True


# -- capacity ---------------------------------------------------------


def test_next_unassigned_ticket_prefers_higher_priority():
    project = beads.create("prio proj", "d", issue_type="epic", priority=0)
    low = beads.create("low", "d", parent=project["id"], priority=3)
    urgent = beads.create("urgent", "d", parent=project["id"], priority=0)
    try:
        picked = dispatcher.next_unassigned_ticket()
        assert picked is not None and picked["id"] == urgent["id"]
    finally:
        # Do not leave a high-priority unassigned ticket in the shared
        # workspace: tick() now routes to the product-owner when one
        # outranks the assigned work, which would break other tests.
        beads.close(urgent["id"])
        beads.close(low["id"])


# -- priority preemption of an assigned backlog ----------------------


def test_higher_priority_unassigned_work_preempts_assigned_backlog(monkeypatch):
    """A P0 ticket no seat has been earmarked for must not sit behind a
    seat's already-assigned P2 backlog. The project has exactly one seat, so
    dispatch assigns it deterministically -- no product-owner session -- and
    the next tick starts it ahead of the backlog. Regression for the Silent
    Run scaffolding starvation, 2026-09-04."""
    project = beads.create("preempt proj", "d", issue_type="epic", priority=0)
    backlog = beads.create("assigned p2", "d", parent=project["id"], priority=2)
    beads.assign_to_seat(backlog["id"], "busy-seat")
    urgent = beads.create("urgent p0", "d", parent=project["id"], priority=0)

    # Pin what each scan sees; the shared test workspace carries leftovers
    # from other tests (same reason test_toolchain.py pins next_assigned).
    monkeypatch.setattr(
        dispatcher, "next_assigned_ticket",
        lambda *a, **k: (beads.show(backlog["id"]), "busy-seat"),
    )
    monkeypatch.setattr(
        dispatcher, "next_unassigned_ticket", lambda *a, **k: beads.show(urgent["id"])
    )
    monkeypatch.setattr(
        dispatcher.Dispatcher, "wake_product_owner",
        lambda self: (_ for _ in ()).throw(AssertionError("no session expected")),
    )

    try:
        d = _dispatcher(max_agents=1)
        assert d.tick() == "assigned"
        assert beads.assigned_seat(beads.show(urgent["id"])) == "busy-seat", (
            "the front ticket goes to the project's one seat without a model call"
        )
        assert beads.show(backlog["id"])["status"] == "open", "assigned P2 must not be claimed"
    finally:
        beads.close(urgent["id"])


# -- deterministic assignment (no product-owner session for the easy case)


def test_project_seat_is_the_single_seat(monkeypatch):
    _project(monkeypatch, [
        _issue("proj-x.1", seat="seat-a"),
        _issue("proj-x.2", seat=None),
    ])
    assert dispatcher.project_seat("proj-x") == "seat-a"


def test_project_seat_is_none_with_no_seat(monkeypatch):
    _project(monkeypatch, [_issue("proj-x.1", seat=None), _issue("proj-x.2", seat=None)])
    assert dispatcher.project_seat("proj-x") is None


def test_project_seat_is_none_with_more_than_one_seat(monkeypatch):
    _project(monkeypatch, [
        _issue("proj-x.1", seat="seat-a"),
        _issue("proj-x.2", seat="seat-b"),
    ])
    assert dispatcher.project_seat("proj-x") is None


def test_assign_front_skips_a_seat_that_declined_the_ticket(monkeypatch):
    """The product-owner's own rule -- don't hand a ticket back to the seat
    that declined it -- must hold on the deterministic path too, which then
    opens the door to a session."""
    declined = _issue("proj-x.1", seat=None)
    declined["metadata"] = {"declined_by": "seat-a"}
    _project(monkeypatch, [_issue("proj-x.2", seat="seat-a"), declined])

    assert dispatcher.assign_front(declined) is None


def test_a_single_seat_project_is_assigned_without_a_session(monkeypatch):
    issues = [
        _issue("proj-x.1", seat="seat-a", status="closed"),
        _issue("proj-x.2", seat=None),
    ]
    _project(monkeypatch, issues)
    calls = []
    monkeypatch.setattr(
        dispatcher.beads, "assign_to_seat", lambda tid, seat: calls.append((tid, seat))
    )
    monkeypatch.setattr(
        dispatcher.Dispatcher, "wake_product_owner",
        lambda self: (_ for _ in ()).throw(AssertionError("no session expected")),
    )

    d = _dispatcher(max_agents=1)

    assert d.tick() == "assigned"
    assert calls == [("proj-x.2", "seat-a")]


def test_a_multi_seat_project_falls_back_to_the_product_owner(monkeypatch):
    """Two established seats is a real choice about which specialist a piece
    of work belongs to, so that stays a product-owner judgment call."""
    issues = [
        _issue("proj-x.1", seat="seat-a", status="closed"),
        _issue("proj-x.2", seat="seat-b", status="closed"),
        _issue("proj-x.3", seat=None, priority=2),
    ]
    _project(monkeypatch, issues)
    monkeypatch.setattr(dispatcher.Dispatcher, "wake_product_owner", lambda self: "brokered")

    d = _dispatcher(max_agents=1)

    assert d.tick() == "brokered"


def test_equal_priority_assigned_work_is_not_preempted(monkeypatch):
    """At equal priority the assigned ticket keeps precedence: a seat's
    in-flight backlog is drained, not churned by same-priority brokering."""
    project = beads.create("nopreempt proj", "d", issue_type="epic", priority=1)
    backlog = beads.create("assigned a", "d", parent=project["id"], priority=2)
    beads.assign_to_seat(backlog["id"], "seat-q")
    other = beads.create("unassigned b", "d", parent=project["id"], priority=2)

    monkeypatch.setattr(
        dispatcher, "next_assigned_ticket",
        lambda *a, **k: (beads.show(backlog["id"]), "seat-q"),
    )
    monkeypatch.setattr(
        dispatcher, "next_unassigned_ticket", lambda *a, **k: beads.show(other["id"])
    )

    d = _dispatcher(max_agents=1)
    # No toolchain declared, so preflight passes and it tries to start the
    # assigned ticket -- a real claim, no model work (the thread fails to
    # connect and releases its slot, same as test_start_agent_claims).
    assert d.tick() in ("started", "could not start")
    assert beads.show(backlog["id"])["status"] == "in_progress"


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


def test_orphaned_in_progress_work_is_picked_up_again(monkeypatch):
    """A ticket left in_progress by a crashed agent never reappears in
    `bd ready`, so dispatch has to look for it explicitly or it strands."""
    monkeypatch.setattr(dispatcher, "_order", lambda issue: ())
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


def test_next_assigned_ticket_skips_seats_already_running(monkeypatch):
    """With MAX_RUNNING_AGENTS > 1 the selector must not keep returning a
    ticket whose seat is already working -- start_agent refuses that as
    already-running (one ticket per seat), tick() reports "could not
    start", and dispatch stalls at a single agent however high the cap is.
    Found live 2026-09-13."""
    monkeypatch.setattr(dispatcher, "held_projects", lambda: {})
    # One project each: a project runs one ticket at a time, so two in the
    # SAME project would be decided by that rule rather than by busy_seats,
    # which is what this is about.
    monkeypatch.setattr(dispatcher.beads, "in_progress", lambda: [
        {"id": "busy-a.1", "issue_type": "task", "metadata": {"assigned_seat": "seat-1"}},
        {"id": "busy-b.1", "issue_type": "task", "metadata": {"assigned_seat": "seat-2"}},
    ])
    monkeypatch.setattr(dispatcher.beads, "ready", lambda: [])

    issue, seat = dispatcher.next_assigned_ticket(busy_seats={"seat-1"})

    assert issue["id"] == "busy-b.1"
    assert seat == "seat-2"


def test_order_prefers_the_earlier_epic_over_a_better_story_priority(monkeypatch):
    """Roadmap order: a story under an earlier (higher-priority) epic sorts
    before a later epic's story, even when that later story has a better
    story-level priority. Silent Run's stories are all P2, so the epic is
    the only ordering signal. Found live 2026-09-14 -- the first epic sat
    at 1/6 while later epics were already in flight because selection used
    raw `bd` order."""
    monkeypatch.setattr(dispatcher.beads, "list_all", lambda: [
        {"id": "proj", "priority": 1},
        {"id": "proj.1", "priority": 0},
        {"id": "proj.1.1", "priority": 2},
        {"id": "proj.2", "priority": 1},
        {"id": "proj.2.1", "priority": 0},
    ])
    dispatcher._order_cache["at"] = -1e9  # force a recompute against the stub

    assert dispatcher._order({"id": "proj.1.1", "priority": 2}) < dispatcher._order(
        {"id": "proj.2.1", "priority": 0}
    )


def test_human_flagged_ready_work_is_not_dispatched(monkeypatch):
    """The in_progress path skipped parked tickets but the ready path did
    not, so a ticket reopened-and-flagged (the old failed-verification
    shape) could be picked back up. Both paths must skip them."""
    monkeypatch.setattr(dispatcher, "held_projects", lambda: {})
    monkeypatch.setattr(dispatcher.beads, "in_progress", lambda: [])
    monkeypatch.setattr(dispatcher.beads, "ready", lambda: [
        {"id": "parked.1", "issue_type": "task", "labels": ["human"],
         "metadata": {"assigned_seat": "seat-1"}},
    ])

    issue, seat = dispatcher.next_assigned_ticket(busy_seats=set())

    assert issue is None
    assert seat is None


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


# -- concurrency: one ticket per seat, several seats per project --------
#
# This used to run one ticket per PROJECT, which serialised a whole project
# even when a dozen specialist seats each had ready work. It is now one
# ticket per SEAT, so different seats run at once; dependency ordering is
# `bd ready`'s job (blocked tickets excluded, their blockers surfaced). A
# file collision between two concurrent tickets is handled by
# conflict-requeue on landing (verifier.land), not by serialising.
#
# These use synthetic issues rather than beads.create: bd does not give
# tickets a shared id root here, so created tickets do not land in one
# project and could not exercise the selection at all.


def _issue(ticket_id, *, priority=1, seat="seat-a", status="open", labels=None):
    return {
        "id": ticket_id,
        "title": ticket_id,
        "status": status,
        "priority": priority,
        "issue_type": "task",
        "labels": labels or [],
        "metadata": {"assigned_seat": seat} if seat else {},
    }


def _project(monkeypatch, issues):
    """Point the dispatcher at one synthetic project.

    `ready` mimics bd's: it returns only status=open, so a ticket standing
    in for a dependency-blocked one is simply not in it."""
    monkeypatch.setattr(beads, "list_all", lambda: list(issues))
    monkeypatch.setattr(beads, "ready", lambda: [i for i in issues if i["status"] == "open"])
    monkeypatch.setattr(
        beads, "in_progress", lambda: [i for i in issues if i["status"] == "in_progress"]
    )
    monkeypatch.setattr(dispatcher, "held_projects", lambda: {})
    monkeypatch.setattr(dispatcher, "_order", lambda issue: (issue["priority"], issue["id"]))
    monkeypatch.setattr(beads, "blocked_ids", lambda: set())
    dispatcher._open_parents_cache["at"] = -1e9  # recompute against the stub


def test_the_front_of_a_project_is_its_earliest_unfinished_ticket(monkeypatch):
    issues = [_issue("proj-x.2"), _issue("proj-x.1")]
    _project(monkeypatch, issues)

    assert dispatcher.roadmap_fronts() == {"proj-x": "proj-x.1"}


def test_the_front_descends_to_a_child_when_the_parent_has_open_work(monkeypatch):
    """A parent with open children cannot be closed (bd refuses), so it is
    not the actionable front -- its earliest open child is. Found live
    2026-09-19/20: workspace-o0n.3.2 and 4.2 both had a parent whose subtasks
    (later ids, so behind the front gate) never ran, so the parent was
    restarted and parked forever."""
    issues = [
        _issue("proj-x.1", seat="seat-a"),
        _issue("proj-x.1.1", seat="seat-a"),
    ]
    _project(monkeypatch, issues)

    assert dispatcher.roadmap_fronts() == {"proj-x": "proj-x.1.1"}
    picked, seat = dispatcher.next_assigned_ticket()
    assert picked["id"] == "proj-x.1.1"
    assert seat == "seat-a"


def test_a_parent_is_the_front_once_its_children_close(monkeypatch):
    issues = [
        _issue("proj-x.1", seat="seat-a"),
        _issue("proj-x.1.1", seat="seat-a", status="closed"),
    ]
    _project(monkeypatch, issues)

    assert dispatcher.roadmap_fronts() == {"proj-x": "proj-x.1"}


def test_a_dependency_blocked_ticket_is_not_the_front(monkeypatch):
    """Found the hard way, and it wedged the project dead: workspace-9jg's
    front was 2.5, blocked by 14.1, so the gate refused everything in the
    project -- 14.1 included -- and dispatch went silent.

    Blocked is not "cannot be actioned", it is "not actionable". The ticket
    to work is its blocker, which can be LATER in roadmap order."""
    blocked = _issue("proj-a.1")
    blocked["status"] = "blocked"
    blocker = _issue("proj-a.9")
    issues = [blocked, blocker]
    _project(monkeypatch, issues)

    assert dispatcher.roadmap_fronts() == {"proj-a": "proj-a.9"}

    picked, _ = dispatcher.next_assigned_ticket()

    assert picked["id"] == "proj-a.9", "the blocker takes its place"


def test_a_closed_ticket_never_holds_its_project_up(monkeypatch):
    """Verifier-exhausted tickets are closed AND human-labelled, and the
    escalation role skips closed tickets by design -- gating on them would
    stall the project for good rather than for a while."""
    issues = [_issue("proj-x.1", status="closed"), _issue("proj-x.2")]
    _project(monkeypatch, issues)

    assert dispatcher.roadmap_fronts() == {"proj-x": "proj-x.2"}


def test_a_closed_ticket_does_not_have_to_be_worked_first(monkeypatch):
    issues = [_issue("proj-x.1", status="closed"), _issue("proj-x.2")]
    _project(monkeypatch, issues)

    picked, seat = dispatcher.next_assigned_ticket()

    assert picked["id"] == "proj-x.2"
    assert seat == "seat-a"


def test_a_parked_ticket_does_not_stop_its_project(monkeypatch):
    """A parked (human) ticket is skipped and the project's next ready
    ticket is selected -- parking one ticket no longer halts the rest."""
    issues = [
        _issue("proj-a.1", labels=["human"], seat="seat-a"),
        _issue("proj-a.2", seat="seat-a"),
        _issue("proj-b.1", seat="seat-b"),
    ]
    _project(monkeypatch, issues)

    picked, seat = dispatcher.next_assigned_ticket()

    assert picked["id"] == "proj-a.2"
    assert seat == "seat-a"


def test_a_parked_ticket_is_not_selected(monkeypatch):
    issues = [_issue("proj-x.1", labels=["human"]), _issue("proj-x.2", priority=2)]
    _project(monkeypatch, issues)

    picked, _ = dispatcher.next_assigned_ticket()

    assert picked is not None and picked["id"] == "proj-x.2"


def test_an_orphan_is_resumed_before_fresh_work(monkeypatch):
    """A crashed run's leftover is picked up first -- `bd ready` never
    returns it, so nothing else would resume it."""
    issues = [_issue("proj-x.1"), _issue("proj-x.2", status="in_progress")]
    _project(monkeypatch, issues)

    picked, _ = dispatcher.next_assigned_ticket()

    assert picked["id"] == "proj-x.2"


def test_an_orphan_at_the_front_is_resumed(monkeypatch):
    """The other half: a crashed run at the front is exactly what should
    resume -- it never comes back through `bd ready`."""
    issues = [_issue("proj-x.1", status="in_progress")]
    _project(monkeypatch, issues)

    picked, seat = dispatcher.next_assigned_ticket()

    assert picked["id"] == "proj-x.1"
    assert seat == "seat-a"


def test_a_running_seat_is_not_given_a_second_ticket(monkeypatch):
    """One ticket per seat: with seat-a already running, seat-a's next
    ticket is skipped rather than started twice."""
    issues = [_issue("proj-x.1", seat="seat-a"), _issue("proj-x.2", seat="seat-a")]
    _project(monkeypatch, issues)

    picked, _ = dispatcher.next_assigned_ticket(busy_seats={"seat-a"})

    assert picked is None


def test_different_seats_are_both_eligible(monkeypatch):
    """The concurrency the per-seat rule buys: once seat-a is busy, seat-b's
    ticket is still selectable, so two seats work one project at once."""
    issues = [_issue("proj-x.1", seat="seat-a"), _issue("proj-x.2", seat="seat-b")]
    _project(monkeypatch, issues)

    first, _ = dispatcher.next_assigned_ticket()
    second, _ = dispatcher.next_assigned_ticket(busy_seats={"seat-a"})

    assert first["id"] == "proj-x.1"
    assert second["id"] == "proj-x.2"


def test_a_blocked_in_progress_orphan_is_not_resumed(monkeypatch):
    """A ticket left in_progress that has since gained an open blocker must
    not be resumed: it cannot close until the blocker lands, so resuming it
    only re-lands the same work. Found live 2026-09-21: workspace-o0n.15.4
    started and landed three times in twenty minutes while blocked."""
    issues = [_issue("proj-x.1", status="in_progress")]
    _project(monkeypatch, issues)
    monkeypatch.setattr(beads, "blocked_ids", lambda: {"proj-x.1"})

    picked, _ = dispatcher.next_assigned_ticket()

    assert picked is None


def test_unassigned_work_behind_the_front_is_not_brokered(monkeypatch):
    """The product-owner brokers for the front ticket and nothing else, or
    it would assign work the front gate then refuses to start."""
    issues = [_issue("proj-x.1", priority=0, seat="seat-a"), _issue("proj-x.2", seat=None)]
    _project(monkeypatch, issues)

    assert dispatcher.next_unassigned_ticket() is None


def test_the_gate_fails_open_when_the_front_cannot_be_determined(monkeypatch):
    """A stall the operator cannot see is worse than a ticket starting out
    of order -- same posture as project_hold and the toolchain preflight."""
    issues = [_issue("proj-x.2")]
    _project(monkeypatch, issues)
    monkeypatch.setattr(dispatcher, "roadmap_fronts", lambda: {})

    picked, _ = dispatcher.next_assigned_ticket()

    assert picked is not None and picked["id"] == "proj-x.2"


