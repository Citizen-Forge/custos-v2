"""
End-to-end proof of PLAN.md's Phase 1 exit criteria, run against the real
Postgres checkpointer and the real `bd` CLI -- everything except the LLM
itself, which is swapped for a scripted fake since no local model is
reachable in this environment yet (see PLAN.md's open hardware question).
Requires DATABASE_URL and a real `bd` binary -- run via docker compose,
not bare pytest, same as tests/test_graph.py's InMemorySaver test but this
one needs the real Postgres service up.

Simulates a worker crash with `interrupt_after=["tools"]`: the graph stops
right after the tool executes, before the model's final answer -- state at
that point is exactly what a process kill would leave behind, since
LangGraph commits a checkpoint after every completed superstep, not just
at the end of `invoke()`. A brand new checkpointer connection + a brand
new model instance (simulating a full process restart) then resumes the
same thread_id and must pick up from the tool result, not redo it.
"""

import os

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver

from harness import beads
from harness.graph import build_graph_from_model


class ScriptedModel:
    """Response depends on conversation state, not call count -- so it
    behaves correctly whether this is a fresh process or a resumed one,
    exactly like a real LLM would (a real model doesn't know or care how
    many times *this Python object* has been called)."""

    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "remember_fact", "args": {"text": "phase1 e2e check"}, "id": "call-1"}
            ],
        )


def test_worker_resume_across_process_restart():
    conn_string = os.environ["DATABASE_URL"]
    beads.ensure_initialized()
    issue = beads.create("e2e resume test", "prove crash/resume works")
    thread_id = issue["id"]
    config = {"configurable": {"thread_id": thread_id}}

    # "process 1": run up to and including the tool call, then stop -- as
    # if the worker died right after the tool executed but before the
    # model produced its final answer.
    with PostgresSaver.from_conn_string(conn_string) as checkpointer_1:
        checkpointer_1.setup()
        graph_1 = build_graph_from_model(
            ScriptedModel(), checkpointer_1, interrupt_after=["tools"]
        )
        graph_1.invoke(
            {"messages": [("user", "do the thing")], "ticket_id": thread_id},
            config,
        )
        state_after_crash = graph_1.get_state(config)
        assert state_after_crash.next == ("agent",)
        assert any(isinstance(m, ToolMessage) for m in state_after_crash.values["messages"])

    # "process 2": brand new checkpointer connection + brand new model
    # instance, same thread_id -- simulates a full process restart.
    with PostgresSaver.from_conn_string(conn_string) as checkpointer_2:
        graph_2 = build_graph_from_model(ScriptedModel(), checkpointer_2)
        result = graph_2.invoke(None, config)
        assert result["messages"][-1].content == "done"
        assert graph_2.get_state(config).next == ()

    closed = beads.close(thread_id)
    assert closed["status"] == "closed"
