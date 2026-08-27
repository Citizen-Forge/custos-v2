"""
Graph-wiring smoke test using an in-memory checkpointer and a fake model.

This proves the ReAct loop assembles and runs end-to-end. It does NOT
prove the crash/resume durability guarantee — that requires a real
Postgres checkpointer surviving across two separate process invocations.
See scripts/enqueue_demo.py + PLAN.md's Phase 1 exit criteria for that.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness.graph import build_graph_from_model


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
