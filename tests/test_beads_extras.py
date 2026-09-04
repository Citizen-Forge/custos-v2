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
