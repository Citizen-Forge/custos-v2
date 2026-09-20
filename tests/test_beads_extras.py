"""
Phase 3 groundwork: search over past/current work, and epic/subtask
decomposition via Beads' native hierarchy. Runs against the real `bd`
CLI, same style as the rest of this suite -- these are thin wrappers, so
the only thing worth testing is that the real CLI shapes match what
beads.py assumes.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import pytest

from harness import beads
from harness.graph import build_graph_from_model


def test_add_dependency_blocks_bd_ready_until_the_blocker_closes():
    beads.ensure_initialized()
    blocker = beads.create("dep test scaffold", "sets things up")
    blocked = beads.create("dep test downstream", "needs the scaffold")

    result = beads.add_dependency(blocked["id"], blocker["id"])

    assert result["status"] == "added"
    ready_ids = {i["id"] for i in beads.ready()}
    assert blocked["id"] not in ready_ids
    assert blocker["id"] in ready_ids

    beads.close(blocker["id"])
    ready_ids = {i["id"] for i in beads.ready()}
    assert blocked["id"] in ready_ids, "must unblock once the blocker closes"


def test_add_dependency_rejects_a_nonexistent_blocker():
    """`bd dep add` itself does not validate either id -- confirmed live
    2026-09-04, it silently wires an edge to a typo'd id. An LLM tool
    call is exactly the caller likely to pass one, so the wrapper must
    catch it rather than wedge a ticket on a blocker that can never
    close."""
    beads.ensure_initialized()
    story = beads.create("dep test orphan blocked", "x")

    with pytest.raises(beads.BeadsError):
        beads.add_dependency(story["id"], "does-not-exist-xyz")


def test_search_finds_related_issues_by_keyword():
    beads.ensure_initialized()
    beads.create("fix login bug", "users cant log in with SSO")
    beads.create("improve login performance", "SSO login is slow on mobile")
    beads.create("unrelated: update readme", "typo fixes")

    results = beads.search("login")

    titles = {r["title"] for r in results}
    assert "fix login bug" in titles
    assert "improve login performance" in titles
    assert "unrelated: update readme" not in titles


def test_create_with_parent_produces_hierarchical_id():
    beads.ensure_initialized()
    epic = beads.create("epic: SSO overhaul", "top level", issue_type="epic")
    subtask = beads.create("fix SSO login bug", "sub task", parent=epic["id"])

    assert subtask["id"].startswith(epic["id"] + ".")


def test_reopen_releases_the_previous_claim():
    """A reopened ticket has to be claimable by the seat it is now
    assigned to.

    `bd --claim` sets assignee and status together and is only idempotent
    for the actor already holding the ticket, so an `open` ticket that
    keeps its old assignee can never be claimed by anyone else -- and the
    dispatcher retries that same highest-priority ticket every cycle
    rather than moving on, wedging the whole queue. Found live
    2026-09-15 on workspace-9jg.1.3. Reopening means "back in the pool",
    so the claim has to go with it."""
    beads.ensure_initialized()
    ticket = beads.create("reopen must release the claim", "x")

    beads.claim(ticket["id"], actor="seat-a")
    claimed = beads.show(ticket["id"])
    assert claimed["assignee"] == "seat-a"
    assert claimed["status"] == "in_progress"

    beads.reopen(ticket["id"], "rework: the criteria were wrong")

    reopened = beads.show(ticket["id"])
    assert reopened["status"] == "open"
    assert reopened.get("assignee") in (None, ""), "reopen must release the old holder"

    beads.claim(ticket["id"], actor="seat-b")
    assert beads.show(ticket["id"])["assignee"] == "seat-b"


class ProposesSubtaskThenDone:
    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"got: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_subtask",
                    "args": {"title": "part 2", "description": "the rest of the work"},
                    "id": "call-1",
                }
            ],
        )


def test_set_acceptance_checks_rejects_an_unknown_schema():
    """The product-owner is an LLM and periodically invents a check schema
    (found live 2026-09-20: {"kind": "command_exit_zero"} on
    workspace-o0n.4.2), which then fails every check as "unknown type None".
    Writing must reject it so the caller retries with a valid shape."""
    beads.ensure_initialized()
    issue = beads.create("checks schema", "d")
    with pytest.raises(beads.BeadsError):
        beads.set_acceptance_checks(issue["id"], [{"kind": "command_exit_zero"}])
    # A valid shape is accepted.
    beads.set_acceptance_checks(issue["id"], [{"type": "tests_at_least", "count": 1}])


def test_create_subtask_tool_parents_under_current_ticket():
    beads.ensure_initialized()
    parent = beads.create("big ticket", "turns out this is two pieces of work")
    thread_id = parent["id"]

    graph = build_graph_from_model(ProposesSubtaskThenDone(), InMemorySaver())
    graph.invoke(
        {"messages": [("user", "do the thing")], "ticket_id": thread_id, "turn_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )

    children = beads.search("part 2")
    assert any(c["id"].startswith(thread_id + ".") for c in children)


# -- acting as the ticket's holder ------------------------------------
#
# bd refuses close/reopen by an actor other than the assignee. The
# dispatcher claims a ticket as the seat while worker.close ran as the
# system, so a normal completion would have been refused -- these pin the
# assignee-resolving behaviour that keeps completion working.


def test_close_acts_as_the_assignee():
    beads.ensure_initialized()
    ticket = beads.create("close as assignee", "x")
    beads.claim(ticket["id"], actor="test-seat-close")

    beads.close(ticket["id"], reason="done")

    assert beads.show(ticket["id"])["status"] == "closed"


def test_reopen_acts_as_the_assignee():
    beads.ensure_initialized()
    ticket = beads.create("reopen as assignee", "x")
    beads.claim(ticket["id"], actor="test-seat-reopen")
    beads.close(ticket["id"], reason="done")

    beads.reopen(ticket["id"], "the criteria were wrong")

    assert beads.show(ticket["id"])["status"] == "open"


def test_create_stores_acceptance_checks():
    beads.ensure_initialized()
    checks = [{"type": "file_exists", "path": "project.godot"}]

    ticket = beads.create("checks ticket", "d", acceptance_checks=checks)

    assert beads.acceptance_checks(beads.show(ticket["id"])) == checks


def test_read_helpers_ask_for_unlimited_results(monkeypatch):
    """`bd ready` defaults to 100 results and `bd list` to 50 -- a silent
    cap that hides tickets past the boundary. Found live 2026-09-17 when
    the shared test workspace passed 100 ready tickets and newly created
    ones disappeared from `bd ready`. Guard that every read wrapper asks
    for the unlimited form."""
    calls = []

    def fake_run(args, actor=None):
        calls.append(args)
        return "[]"

    monkeypatch.setattr(beads, "_run", fake_run)
    beads.ready()
    beads.in_progress()
    beads.list_all()
    beads.list_top_level()
    beads.children_of("x")
    beads.list_by_assignee("someone")

    assert calls, "no read helper called bd"
    for args in calls:
        assert "--limit" in args, args
        assert args[args.index("--limit") + 1] == "0", args
