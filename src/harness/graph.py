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
"""

from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .providers import ProviderConfig, build_chat_model
from .state import HarnessState
from .tools import ALL_TOOLS


def build_graph_from_model(model, checkpointer, classify=None, interrupt_after=None):
    """Build the graph from an already-tool-bound model. Split out from
    `build_graph` so tests can pass a fake model without a real provider.

    `classify` is an optional `(tool_name, tool_args) -> classifier.Verdict`
    callable (see classifier.build_classifier). When omitted, the gate node
    allows everything -- used by tests that predate/don't exercise gating
    (tests/test_graph.py, tests/test_worker_resume.py) so they don't need a
    classifier model.

    `interrupt_after` is test-only (see tests/test_worker_resume.py) --
    production never sets it.
    """

    def call_model(state: HarnessState):
        return {"messages": [model.invoke(state["messages"])]}

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
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "permission_gate", END: END})
    builder.add_conditional_edges("permission_gate", route_after_gate, {"tools": "tools", "agent": "agent"})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def build_graph(provider_cfg: ProviderConfig, checkpointer, classifier=None):
    model = build_chat_model(provider_cfg).bind_tools(ALL_TOOLS)
    return build_graph_from_model(model, checkpointer, classify=classifier)
