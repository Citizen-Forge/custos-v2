"""
Graph-wiring smoke test using an in-memory checkpointer and a fake model.

This proves the ReAct loop assembles and runs end-to-end. It does NOT
prove the crash/resume durability guarantee — that requires a real
Postgres checkpointer surviving across two separate process invocations.
See scripts/enqueue_demo.py + PLAN.md's Phase 1 exit criteria for that.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads
from harness.graph import (
    COMPLETION_NUDGE,
    _repair_dangling_tool_calls,
    build_graph_from_model,
)


class FakeModel:
    def invoke(self, messages):
        return AIMessage(content="done")


def test_graph_runs_to_completion():
    checkpointer = InMemorySaver()
    graph = build_graph_from_model(FakeModel(), checkpointer)

    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="say hi")], "ticket_id": "test-1"},
        config,
    )

    assert result["messages"][-1].content == "done"
    assert graph.get_state(config).next == ()


class StopsThenCompletes:
    """Stops with prose first; only calls complete_ticket once nudged."""

    def invoke(self, messages):
        last = messages[-1] if messages else None
        if isinstance(last, HumanMessage) and last.content == COMPLETION_NUDGE:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "complete_ticket", "args": {"summary": "did the thing"}, "id": "c1"}
                ],
            )
        return AIMessage(content="I believe I am finished")


def test_stopping_without_a_completion_claim_gets_one_nudge():
    beads.ensure_initialized()
    issue = beads.create("completion nudge test", "do the thing")
    thread_id = issue["id"]

    graph = build_graph_from_model(StopsThenCompletes(), InMemorySaver(), completion_gate=True)
    graph.invoke(
        {"messages": [HumanMessage(content="go")], "ticket_id": thread_id, "turn_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )

    summary = (beads.show(thread_id).get("metadata") or {}).get("completion_summary")
    assert summary == "did the thing"


def test_repair_answers_only_the_calls_nothing_answered():
    """Idempotent, and it never disturbs a history that is already valid."""
    valid = [
        HumanMessage(content="go"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "a"}]),
        ToolMessage(content="contents", tool_call_id="a"),
        AIMessage(content="finished"),
    ]
    assert _repair_dangling_tool_calls(valid) == valid

    dangling = [
        HumanMessage(content="go"),
        AIMessage(content="", tool_calls=[
            {"name": "shell_exec", "args": {}, "id": "a"},
            {"name": "read_file", "args": {}, "id": "b"},
        ]),
        ToolMessage(content="ran", tool_call_id="a"),
    ]
    repaired = _repair_dangling_tool_calls(dangling)
    answered = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
    assert answered == ["a", "b"]
    # ...and repairing again changes nothing.
    assert _repair_dangling_tool_calls(repaired) == repaired


def test_a_thread_stopped_mid_tool_call_still_resumes():
    """The checkpoint shape a crash leaves behind must not poison the thread.

    A run killed between the model asking for tools and the results landing
    checkpoints an assistant message whose calls nothing answers, and the
    provider then rejects EVERY subsequent request for that thread with a 400
    -- so the ticket can never retry and ends up parked for a human. Found live
    2026-09-15 with ten threads in exactly this state ("blocked tickets building
    up again"). The model must be handed an answer for the missing call.
    """
    seen = {}

    class RecordingModel:
        def invoke(self, messages):
            seen["messages"] = list(messages)
            return AIMessage(content="recovered")

    graph = build_graph_from_model(RecordingModel(), InMemorySaver())
    config = {"configurable": {"thread_id": "poisoned-1"}}

    # as_node="tools" reproduces the real resume point: the checkpoint reads as
    # though the tool node had just run, so invoking resumes AT the model.
    graph.update_state(
        config,
        {
            "messages": [
                HumanMessage(content="do the thing"),
                AIMessage(content="", tool_calls=[
                    {"name": "shell_exec", "args": {"command": "ls"}, "id": "call-a"},
                ]),
            ],
            "ticket_id": "poisoned-1",
        },
        as_node="tools",
    )
    graph.invoke(None, config)
    result = graph.get_state(config).values

    answered = {m.tool_call_id for m in seen["messages"] if isinstance(m, ToolMessage)}
    assert "call-a" in answered, "the dangling call must be answered before the model sees it"
    # And the run got PAST the model instead of being rejected by the provider.
    assert result["messages"][-1].content == "recovered"
