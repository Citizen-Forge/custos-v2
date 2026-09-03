"""A failing ticket must not hold the harness hostage.

On 2026-09-02 one ticket was started and failed 2901 times overnight. An
agent read `wiki/agents/<seat>` -- a habit from when its root was
/workspace, which had a wiki/ -- that path stopped existing under
per-project workspaces, read_file raised FileNotFoundError, the exception
propagated out of the tool node and killed the graph run, and the
dispatcher immediately restarted the same ticket. It held the only agent
slot all night and nothing else ran.

Two independent faults, so two independent fixes: a missing file is now a
tool result rather than a crash, and a ticket that keeps crashing stops
being retried.
"""

import pytest

from harness import beads, dispatcher, tools
from harness.permissions import PermissionDenied
from harness.routing import RoutingTable


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()


# -- file errors are results, not crashes -----------------------------


def test_missing_file_returns_a_message(tmp_path):
    bound = {t.name: t for t in tools.build_workspace_tools(str(tmp_path))}
    result = bound["read_file"].invoke({"path": "does/not/exist.ts"})
    assert isinstance(result, str)
    assert "no such file" in result


def test_missing_wiki_path_points_at_the_right_tool(tmp_path):
    """The exact call that caused the outage, with a hint so an agent
    does not simply try again."""
    bound = {t.name: t for t in tools.build_workspace_tools(str(tmp_path))}
    result = bound["read_file"].invoke({"path": "wiki/agents/orbital-transit-motion"})
    assert "read_wiki_page" in result


def test_reading_a_directory_is_not_a_crash(tmp_path):
    (tmp_path / "sub").mkdir()
    bound = {t.name: t for t in tools.build_workspace_tools(str(tmp_path))}
    assert "directory" in bound["read_file"].invoke({"path": "sub"})


def test_escapes_still_raise(tmp_path):
    """Containment is a security boundary and stays loud -- only ordinary
    filesystem errors were softened."""
    bound = {t.name: t for t in tools.build_workspace_tools(str(tmp_path))}
    with pytest.raises(PermissionDenied):
        bound["read_file"].invoke({"path": "../outside.txt"})


def test_write_to_an_unwritable_path_returns_a_message(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    bound = {t.name: t for t in tools.build_workspace_tools(str(tmp_path))}
    result = bound["write_file"].invoke({"path": "afile/nested.txt", "content": "y"})
    assert "could not write" in result


# -- a crashing ticket stops being retried ----------------------------


def _dispatcher():
    return dispatcher.Dispatcher("postgresql://unused", RoutingTable({}), max_agents=1)


def test_failures_accumulate_then_flag():
    project = beads.create("crash proj", "d", issue_type="epic", priority=1)
    story = beads.create("crash story", "d", parent=project["id"])
    d = _dispatcher()

    for _ in range(dispatcher.MAX_TICKET_FAILURES):
        d._record_outcome(story["id"], "failed")

    assert beads.is_flagged_for_human(beads.show(story["id"])), (
        "a ticket that keeps crashing must be escalated, not retried forever"
    )


def test_a_flagged_ticket_is_no_longer_dispatchable():
    """This is what actually breaks the loop: flagged issues are skipped
    by next_assigned_ticket, so the slot is released for other work."""
    project = beads.create("crash2 proj", "d", issue_type="epic", priority=1)
    story = beads.create("crash2 story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "seat-c")
    beads.claim(story["id"], actor="seat-c")
    d = _dispatcher()

    for _ in range(dispatcher.MAX_TICKET_FAILURES):
        d._record_outcome(story["id"], "failed")

    issue, _ = dispatcher.next_assigned_ticket()
    assert issue is None or issue["id"] != story["id"]


def test_one_failure_does_not_flag():
    """Transients happen -- a model 503, a bd timeout -- and should be
    retried rather than escalated."""
    project = beads.create("crash3 proj", "d", issue_type="epic", priority=1)
    story = beads.create("crash3 story", "d", parent=project["id"])
    d = _dispatcher()

    d._record_outcome(story["id"], "failed")

    assert not beads.is_flagged_for_human(beads.show(story["id"]))


def test_success_clears_the_failure_count():
    project = beads.create("crash4 proj", "d", issue_type="epic", priority=1)
    story = beads.create("crash4 story", "d", parent=project["id"])
    d = _dispatcher()

    d._record_outcome(story["id"], "failed")
    d._record_outcome(story["id"], "closed")
    d._record_outcome(story["id"], "failed")

    assert not beads.is_flagged_for_human(beads.show(story["id"])), (
        "an intermittent failure must not accumulate across successes"
    )
