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


def test_routing_passes_a_string_prompt_through_unchanged():
    """The classifier calls invoke() with a plain prompt STRING, not a message
    list. Routing must pass it through -- feeding a string to bound_history
    char-splits it into one-character 'messages' and hands the model garbage
    (which made the classifier ramble and deny benign calls, 2026-09-28)."""
    from langchain_core.messages import AIMessage

    from harness import routing
    from harness.providers import ProviderConfig

    seen = {}

    class Rec:
        def invoke(self, messages):
            seen["arg"] = messages
            return AIMessage(content="ok")

    table = routing.RoutingTable({"r": [ProviderConfig(name="p", base_url="x", model="m")]})
    routing.RoutedModel(
        "r", table, routing.ConcurrencyGate(), model_factory=lambda cfg: Rec()
    ).invoke("classify this call")

    assert seen["arg"] == "classify this call"


# -- oversized briefs (2026-09-30: `bd prime` reached 1.69M chars) ------------

from langchain_core.messages import SystemMessage  # noqa: E402

from harness import context  # noqa: E402


def _prime(n_memories, body="lesson " * 40, tagged=()):
    memories = "".join(
        f"### mem-{i:03d}{'-workspace-o0n' if i in tagged else ''}\n{body}\n\n"
        for i in range(n_memories)
    )
    return (
        "[bd prime] header\n\n# Beads Workflow Context\n\nintro\n\n"
        f"## Persistent Memories ({n_memories})\n\n{memories}"
        "# SESSION CLOSE PROTOCOL\n\nalways run the gate\n"
    )


def test_cap_prime_keeps_workflow_sections_and_drops_memories_to_fit():
    text = _prime(500)
    capped = context.cap_prime(text, 10000)

    assert len(capped) <= 10000
    assert capped.startswith("[bd prime] header")
    assert capped.rstrip().endswith("always run the gate")
    assert "of 500 shown" in capped
    assert "bd memories <keyword>" in capped


def test_cap_prime_prefers_the_tickets_own_project():
    text = _prime(500, tagged={450, 460, 470})
    capped = context.cap_prime(text, 5000, project_id="workspace-o0n")

    for i in (450, 460, 470):
        assert f"### mem-{i:03d}-workspace-o0n" in capped


def test_cap_prime_leaves_a_small_prime_untouched():
    text = _prime(3)
    assert context.cap_prime(text, 10000) == text


def test_cap_brief_keeps_the_ticket_text_whole():
    ticket = f"{context.BRIEF_TICKET_MARKER} Fix the gate\n\nthe full ticket description" + "d" * 2000
    capped = context.cap_brief(_prime(500) + ticket, 12000)

    assert len(capped) <= 12000
    assert capped.endswith(ticket)


def test_bound_history_caps_an_oversized_brief_in_a_resumed_checkpoint():
    """17.1.5's checkpoint held a 1.69M-char brief; the old bound kept the
    lead-in verbatim and the fallback got a 449k-token request."""
    ticket = f"{context.BRIEF_TICKET_MARKER} Make the gate tractable\n\nticket body"
    history = [
        SystemMessage(content="system prompt"),
        HumanMessage(content=_prime(3000) + ticket),
        AIMessage(content="working on it"),
    ]

    bounded = context.bound_history(history, 150000)

    assert sum(context.message_size(m) for m in bounded) <= 150000
    assert bounded[1].content.endswith(ticket)
    assert bounded[-1].content == "working on it"
