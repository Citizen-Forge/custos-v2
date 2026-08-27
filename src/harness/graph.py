"""
The Phase 1 agent loop: agent -> permission_gate -> tools -> agent ...
until no more tool calls, compiled with a caller-supplied checkpointer.

Durability comes entirely from the checkpointer, not from anything special
in the graph shape — LangGraph persists state after every superstep, so a
process killed mid-`invoke` can resume the same `thread_id` later with
`graph.invoke(None, config)` and pick up from the last completed step. See
PLAN.md's Phase 1 exit criteria.

`permission_gate` sits between the model and real tool execution (the
PreToolUse-style boundary v1 had via a Claude Code hook -- there's no hook
contract to hang off of anymore, so it's a real node in this graph
instead). A denied tool call never reaches `tools`: the gate synthesizes a
ToolMessage explaining the denial and routes straight back to `agent`, so
the model sees the refusal as a normal tool result rather than the graph
silently doing nothing.

Known simplification: if a single AIMessage carries multiple tool calls
and any one is denied, every call in that batch is denied (the allowed
ones get "a sibling call in this batch was denied" instead of running).
Matches v1's own noted limitation that local-model tool-call translation
only reliably handles one call per turn anyway -- not a real loss of
capability today, worth revisiting if that changes.

`turn_budget` (Phase 4) is a *soft* nudge, not a hard cutoff: reaching it
appends one message asking the model to wrap up via `write_handoff_note`
and stop, rather than truncating the loop or force-terminating the
thread. Deliberate, per the welfare-essay behaviors PLAN.md commits to --
an agent that's over budget still gets to finish its thought and hand off
on its own terms rather than being cut off mid-turn.

`tools` is caller-supplied, not hardcoded to the general worker's
`ALL_TOOLS` -- the product-owner agent (product_owner.py) runs the same
graph shape with a completely different tool set (list seats, assign
tickets, request a new specialist), and future seats may too. Defaults to
`ALL_TOOLS` so every call site that predates this (tests, `build_graph`)
keeps working unchanged.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .providers import ProviderConfig, build_chat_model
from .state import HarnessState
from .tools import ALL_TOOLS

HANDOFF_NUDGE = (
    "You've reached your turn budget for this session. Please wrap up: call "
    "write_handoff_note with what's done and what's left, then stop."
)


def build_graph_from_model(model, checkpointer, tools=None, classify=None, interrupt_after=None, turn_budget=None):
    """Build the graph from an already-tool-bound model. Split out from
    `build_graph` so tests can pass a fake model without a real provider.

    `tools`: the toolset available to this graph's `tools` node. Defaults
    to `ALL_TOOLS` (the general worker's set) when omitted.

    `classify` is an optional `(tool_name, tool_args) -> classifier.Verdict`
    callable (see classifier.build_classifier). When omitted, the gate node
    allows everything -- used by tests that predate/don't exercise gating
    (tests/test_graph.py, tests/test_worker_resume.py) so they don't need a
    classifier model.

    `turn_budget` is an optional int: when the running turn count hits it
    exactly, one HANDOFF_NUDGE message is appended before that call (once,
    not repeated on every subsequent turn). `None` disables it entirely --
    existing tests that don't pass it are unaffected.

    `interrupt_after` is test-only (see tests/test_worker_resume.py) --
    production never sets it.
    """
    tools = ALL_TOOLS if tools is None else tools

    def call_model(state: HarnessState):
        turn_count = state.get("turn_count", 0) + 1
        messages = state["messages"]
        extra = []
        if turn_budget is not None and turn_count == turn_budget:
            nudge = HumanMessage(content=HANDOFF_NUDGE)
            messages = messages + [nudge]
            extra = [nudge]
        return {"messages": [*extra, model.invoke(messages)], "turn_count": turn_count}

    def permission_gate(state: HarnessState):
        if classify is None:
            return {}

        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        verdicts = {call["id"]: classify(call["name"], call["args"]) for call in tool_calls}

        if all(v.decision == "allow" for v in verdicts.values()):
            return {}

        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"permission denied: {verdicts[call['id']].reason}"
                        if verdicts[call["id"]].decision == "deny"
                        else "not executed: a sibling tool call in this batch was denied"
                    ),
                    tool_call_id=call["id"],
                )
                for call in tool_calls
            ]
        }

    def route_after_gate(state: HarnessState) -> str:
        return "agent" if isinstance(state["messages"][-1], ToolMessage) else "tools"

    builder = StateGraph(HarnessState)
    builder.add_node("agent", call_model)
    builder.add_node("permission_gate", permission_gate)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "permission_gate", END: END})
    builder.add_conditional_edges("permission_gate", route_after_gate, {"tools": "tools", "agent": "agent"})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def build_graph(provider_cfg: ProviderConfig, checkpointer, tools=None, classifier=None, turn_budget=None):
    tools = ALL_TOOLS if tools is None else tools
    model = build_chat_model(provider_cfg).bind_tools(tools)
    return build_graph_from_model(model, checkpointer, tools=tools, classify=classifier, turn_budget=turn_budget)
