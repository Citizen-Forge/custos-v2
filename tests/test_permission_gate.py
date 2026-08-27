"""
Proves the permission_gate node actually blocks execution -- a denied
tool call must never reach the real tool implementation, and the model
must see a "permission denied" ToolMessage instead of silence.

Uses an in-memory checkpointer (this test is about gating behavior, not
durability -- see test_worker_resume.py for the durability proof) and a
scripted fake classifier instead of a real model, since no local model is
reachable in this environment yet (see PLAN.md's open hardware question).
"""

import os

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness.classifier import Verdict
from harness.config import WORKSPACE_ROOT
from harness.graph import build_graph_from_model

TARGET_PATH = os.path.join(WORKSPACE_ROOT, "gate-test.txt")


class ProposesWrite:
    """Proposes writing a file once, then reports whatever the tool call
    resolved to (denied or actually executed) as its final answer."""

    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"got: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "write_file", "args": {"path": "gate-test.txt", "content": "hello"}, "id": "call-1"}
            ],
        )


def _always(decision: str, reason: str):
    def classify(tool_name, tool_args):
        return Verdict(decision=decision, reason=reason)

    return classify


def _remove_target():
    if os.path.exists(TARGET_PATH):
        os.remove(TARGET_PATH)


def test_denied_call_never_executes():
    _remove_target()
    graph = build_graph_from_model(
        ProposesWrite(), InMemorySaver(), classify=_always("deny", "no writes allowed")
    )
    result = graph.invoke(
        {"messages": [("user", "write a file")], "ticket_id": "gate-1"},
        {"configurable": {"thread_id": "gate-1"}},
    )

    assert not os.path.exists(TARGET_PATH)
    assert "permission denied" in result["messages"][-1].content


def test_allowed_call_executes():
    _remove_target()
    graph = build_graph_from_model(
        ProposesWrite(), InMemorySaver(), classify=_always("allow", "fine")
    )
    graph.invoke(
        {"messages": [("user", "write a file")], "ticket_id": "gate-2"},
        {"configurable": {"thread_id": "gate-2"}},
    )

    assert os.path.exists(TARGET_PATH)
    _remove_target()
