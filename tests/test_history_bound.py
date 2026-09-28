"""The history bound must count tool-call ARGUMENTS, not just message content.

A `write_file` carries the whole file in its args; counting only `content` let a
thread grow far past HISTORY_MAX_CHARS in real tokens, which then blew past a
provider's window (2026-09-21: a 330k-token request refused by the 30B
fallback's 64k context)."""

from langchain_core.messages import AIMessage, HumanMessage

from harness import graph as graph_mod


def test_the_bound_counts_tool_call_arguments(monkeypatch):
    monkeypatch.setattr(graph_mod, "HISTORY_MAX_CHARS", 500)
    big = "x" * 1000
    messages = [
        HumanMessage(content="brief"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "write_file", "args": {"path": "a.gd", "content": big}, "id": "c1"}
            ],
        ),
        HumanMessage(content="next"),
    ]

    bounded = graph_mod._bound_history(messages)

    assert all(not getattr(m, "tool_calls", None) for m in bounded), (
        "the huge-args tool call must be dropped"
    )
    assert bounded[-1].content == "next"
