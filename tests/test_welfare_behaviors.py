"""
Phase 4: refuse-work, handoff notes, and the bounded-workday nudge.

Runs against the real `bd` CLI (like test_worker_resume.py) since the
whole point is proving these tools actually mutate the real Beads issue,
not just that a fake tool function got called. Uses an InMemorySaver --
this is about the welfare mechanics, not durability (see
test_worker_resume.py for that proof) -- and a scripted model, since no
local model is reachable in this environment yet.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads
from harness.graph import HANDOFF_NUDGE, build_graph_from_model


class RefusesImmediately:
    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"stopped: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "refuse_ticket", "args": {"reason": "needs a human call"}, "id": "call-1"}
            ],
        )


def test_respond_to_human_closes_with_response_recorded():
    beads.ensure_initialized()
    issue = beads.create("respond test", "x")
    beads.claim(issue["id"])
    beads.flag_for_human(issue["id"], "which approach?")

    result = beads.respond_to_human(issue["id"], "go with option A")

    assert result["status"] == "closed"
    assert result["close_reason"] == "Responded"
    assert "go with option A" in result["notes"]


def test_dismiss_human_closes_with_reason_recorded():
    beads.ensure_initialized()
    issue = beads.create("dismiss test", "x")
    beads.claim(issue["id"])
    beads.flag_for_human(issue["id"], "not sure about this")

    result = beads.dismiss_human(issue["id"], reason="no longer needed")

    assert result["status"] == "closed"
    assert result["close_reason"] == "Dismissed"
    assert "no longer needed" in result["notes"]


def test_refuse_ticket_flags_the_real_issue_for_human_review():
    beads.ensure_initialized()
    issue = beads.create("refuse-work test", "should this even be attempted")
    thread_id = issue["id"]

    graph = build_graph_from_model(RefusesImmediately(), InMemorySaver())
    graph.invoke(
        {"messages": [("user", "do it")], "ticket_id": thread_id, "turn_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )

    updated = beads.show(thread_id)
    assert beads.is_flagged_for_human(updated)
    assert "needs a human call" in updated["notes"]

    # the whole point of the label: worker.py's orphan-resume must skip it
    assert beads.is_flagged_for_human(updated) is True


class WritesHandoffNoteThenDone:
    def invoke(self, messages):
        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "write_handoff_note", "args": {"note": "half done, needs part 2"}, "id": "call-1"}
            ],
        )


def test_handoff_note_appends_to_the_real_issue():
    beads.ensure_initialized()
    issue = beads.create("handoff-note test", "stop partway through")
    thread_id = issue["id"]

    graph = build_graph_from_model(WritesHandoffNoteThenDone(), InMemorySaver())
    result = graph.invoke(
        {"messages": [("user", "start work")], "ticket_id": thread_id, "turn_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )

    assert result["messages"][-1].content == "done"
    updated = beads.show(thread_id)
    assert "half done, needs part 2" in updated["notes"]


class ConsumesATurnThenRespondsToNudge:
    """State-driven, like the other scripted models in this test suite:
    behavior depends on the last message's content, not a call counter --
    so it behaves identically regardless of how many turns actually ran."""

    def invoke(self, messages):
        last = messages[-1] if messages else None
        if isinstance(last, HumanMessage) and last.content == HANDOFF_NUDGE:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_handoff_note", "args": {"note": "stopping here"}, "id": "call-handoff"}
                ],
            )
        if isinstance(last, ToolMessage) and "handoff note recorded" in last.content:
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[{"name": "remember_fact", "args": {"text": "consuming a turn"}, "id": "call-noop"}],
        )


def test_turn_budget_nudges_instead_of_force_stopping():
    beads.ensure_initialized()
    issue = beads.create("budget test", "keep going until nudged")
    thread_id = issue["id"]

    graph = build_graph_from_model(ConsumesATurnThenRespondsToNudge(), InMemorySaver(), turn_budget=2)
    result = graph.invoke(
        {"messages": [("user", "start")], "ticket_id": thread_id, "turn_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]
    assert any(isinstance(m, HumanMessage) and m.content == HANDOFF_NUDGE for m in messages)
    assert messages[-1].content == "done"
    # the nudge produced a real effect (a handoff note), not just a message
    # that the model was free to ignore -- soft nudge, but not a no-op
    assert "stopping here" in beads.show(thread_id)["notes"]
