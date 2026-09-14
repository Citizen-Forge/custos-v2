"""
Graph-wiring smoke test using an in-memory checkpointer and a fake model.

This proves the ReAct loop assembles and runs end-to-end. It does NOT
prove the crash/resume durability guarantee — that requires a real
Postgres checkpointer surviving across two separate process invocations.
See scripts/enqueue_demo.py + PLAN.md's Phase 1 exit criteria for that.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads
from harness.graph import COMPLETION_NUDGE, build_graph_from_model


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
