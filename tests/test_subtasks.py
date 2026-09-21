"""Subtasks must be closable by the agent, and a parent must not be
completable while any of its subtasks is still open. This is the wedge that
stranded workspace-o0n.3.2: the agent created four subtasks via
create_subtask, but had no tool to close them (bd close is denied for
agents), so the parent could never close and got parked for a human."""

from harness import beads, tools


def test_complete_ticket_refuses_while_a_subtask_is_open():
    beads.ensure_initialized()
    parent = beads.create("subtask guard parent", "d")
    child = beads.create("subtask guard child", "d", parent=parent["id"])

    msg = tools.complete_ticket.func(summary="did it", state={"ticket_id": parent["id"]})
    assert "open subtask" in msg
    assert child["id"] in msg
    assert "completion_summary" not in (beads.show(parent["id"]).get("metadata") or {})

    closed = tools.complete_subtask.func(
        subtask_id=child["id"], summary="child done", state={"ticket_id": parent["id"]}
    )
    assert "closed subtask" in closed
    assert beads.show(child["id"])["status"] == "closed"

    recorded = tools.complete_ticket.func(summary="did it", state={"ticket_id": parent["id"]})
    assert "completion recorded" in recorded


def test_complete_subtask_refuses_a_non_child():
    beads.ensure_initialized()
    parent = beads.create("subtask guard parent 2", "d")
    unrelated = beads.create("subtask guard unrelated", "d")

    msg = tools.complete_subtask.func(
        subtask_id=unrelated["id"], summary="x", state={"ticket_id": parent["id"]}
    )
    assert "not a subtask" in msg
    assert beads.show(unrelated["id"])["status"] != "closed"


def test_create_subtask_assigns_the_child_to_the_parents_seat():
    """A decomposed subtask must be pickable without the product-owner brokering
    it: create_subtask assigns it to the parent's own seat. Found live
    2026-09-21: 5.4's child 5.4.4 landed unassigned and the parent stalled for
    an hour because the dispatcher only brokers with spare capacity."""
    import re

    beads.ensure_initialized()
    project = beads.create("subtask seat proj", "d", issue_type="epic")
    parent = beads.create("subtask seat parent", "d", parent=project["id"])
    beads.assign_to_seat(parent["id"], "seat-subtask")

    msg = tools.create_subtask.func(
        title="child", description="d", state={"ticket_id": parent["id"]}
    )

    child_id = re.search(r"created subtask (\S+)", msg).group(1)
    assert beads.assigned_seat(beads.show(child_id)) == "seat-subtask"
