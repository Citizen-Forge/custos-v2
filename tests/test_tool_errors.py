"""A tool that raises must not kill the run: its error reaches the agent as a
ToolMessage it can read and correct.

Regression for workspace-o0n.7.1 (2026-09-21): the agent wrote to
`/tmp/heatsink_loader_block.gd`, `write_file`'s containment check raised
PermissionDenied, and the exception propagated out of the ToolNode and crashed
the graph -- three times, until the ticket was flagged for a human. Containment
still holds (the tool does not act); the run just survives it."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from harness.graph import build_graph_from_model
from harness.permissions import PermissionDenied


@tool
def boom() -> str:
    """Always raises, standing in for a tool whose guard rejects the call."""
    raise PermissionDenied("path escapes workspace: '/tmp/x'")


class CallsBoomThenDone:
    def __init__(self):
        self.pending = [{"name": "boom", "args": {}, "id": "c1"}]

    def invoke(self, messages):
        if self.pending:
            return AIMessage(content="", tool_calls=[self.pending.pop(0)])
        return AIMessage(content="done")


def test_a_raising_tool_returns_a_message_and_the_run_continues():
    graph = build_graph_from_model(CallsBoomThenDone(), InMemorySaver(), tools=[boom])

    result = graph.invoke(
        {"messages": [("user", "go")], "ticket_id": "boom-1", "turn_count": 0},
        {"configurable": {"thread_id": "boom-1"}},
    )

    contents = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("escapes workspace" in c for c in contents), contents
    assert result["messages"][-1].content == "done"
