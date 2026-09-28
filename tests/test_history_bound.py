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


def test_routing_bounds_each_provider_to_its_own_window():
    """A fallback with a smaller window gets a trimmed request while the primary
    keeps a large one -- the whole point of making the bound provider-aware."""
    from langchain_core.messages import AIMessage

    from harness import routing
    from harness.providers import ProviderConfig

    history = [HumanMessage(content="brief")] + [
        AIMessage(content="x" * 500) for _ in range(20)
    ]

    sizes = {}

    class Primary:
        def invoke(self, messages):
            sizes["primary"] = sum(
                len(str(getattr(m, "content", "") or "")) for m in messages
            )
            raise RuntimeError("400 exceed_context_size_error")  # content error: no cooldown

    class Fallback:
        def invoke(self, messages):
            sizes["fallback"] = sum(
                len(str(getattr(m, "content", "") or "")) for m in messages
            )
            return AIMessage(content="ok")

    def factory(cfg):
        return Primary() if cfg.name == "primary" else Fallback()

    table = routing.RoutingTable(
        {
            "r": [
                ProviderConfig(name="primary", base_url="x", model="m", context_chars=100000),
                ProviderConfig(name="fallback", base_url="x", model="m", context_chars=2000),
            ]
        }
    )
    routing.RoutedModel(
        "r", table, routing.ConcurrencyGate(), model_factory=factory
    ).invoke(history)

    assert sizes["primary"] > 9000, "the primary's generous window kept the whole history"
    assert sizes["fallback"] <= 2000 + len("brief"), "the fallback got a trimmed request"
