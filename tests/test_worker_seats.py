"""
Proves the actual mechanism that makes specialization emerge from
product-owner assignment rather than being a free-for-all: a seat's
worker only ever claims tickets assigned to it (plus its own orphaned
in-progress work), never unassigned tickets or another seat's work.
"""

from harness import beads
from harness.worker import _next_ticket


def test_seat_only_claims_its_own_assigned_tickets(monkeypatch):
    beads.ensure_initialized()
    monkeypatch.setattr(beads, "in_progress", lambda: [])

    unassigned = beads.create("nobody's yet", "x")
    for_seat_a = beads.create("for seat A", "x")
    for_seat_b = beads.create("for seat B", "x")

    beads.assign_to_seat(for_seat_a["id"], "seat-a")
    beads.assign_to_seat(for_seat_b["id"], "seat-b")
    # Hermetic: the session's shared workspace carries ready tickets already
    # assigned to seat-a from earlier tests, and ready_for_seat would hand
    # one of those back. Pin the pool to this test's own two tickets.
    monkeypatch.setattr(
        beads, "ready",
        lambda: [beads.show(for_seat_a["id"]), beads.show(for_seat_b["id"])],
    )

    claimed = _next_ticket("seat-a")

    assert claimed is not None
    assert claimed["id"] == for_seat_a["id"]
    assert claimed["assignee"] == "seat-a"

    # seat-a's poll must not have touched the unassigned or seat-b tickets
    assert beads.show(unassigned["id"])["status"] == "open"
    assert beads.show(for_seat_b["id"])["status"] == "open"


def test_seat_with_no_assigned_work_gets_nothing():
    beads.ensure_initialized()
    beads.create("unassigned ticket", "x")  # exists, but not assigned to anyone

    assert _next_ticket("seat-with-nothing-assigned") is None


def test_seat_resumes_its_own_orphaned_work_before_claiming_new(monkeypatch):
    beads.ensure_initialized()

    orphaned = beads.create("was already claimed by seat-a", "x")
    beads.claim(orphaned["id"], actor="seat-a")  # simulates a crashed prior run

    fresh = beads.create("new work for seat-a", "x")
    beads.assign_to_seat(fresh["id"], "seat-a")
    # Hermetic: earlier tests leave in_progress seat-a tickets in the shared
    # workspace; without pinning, a different orphan could win.
    monkeypatch.setattr(beads, "in_progress", lambda: [beads.show(orphaned["id"])])

    claimed = _next_ticket("seat-a")

    assert claimed["id"] == orphaned["id"]  # orphaned work takes priority
