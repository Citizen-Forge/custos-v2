"""
The Phase 1 agent loop: a plain ReAct graph (model -> tools -> model ...
until no more tool calls), compiled with a caller-supplied checkpointer.

Durability comes entirely from the checkpointer, not from anything special
in the graph shape — LangGraph persists state after every superstep, so a
process killed mid-`invoke` can resume the same `thread_id` later with
`graph.invoke(None, config)` and pick up from the last completed step. See
PLAN.md's Phase 1 exit criteria.
"""

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .providers import ProviderConfig, build_chat_model
from .state import HarnessState
from .tools import ALL_TOOLS


def build_graph_from_model(model, checkpointer, interrupt_after=None):
    """Build the graph from an already-tool-bound model. Split out from
    `build_graph` so tests can pass a fake model without a real provider.
    `interrupt_after` is test-only (see tests/test_worker_resume.py) --
    production never sets it."""

    def call_model(state: HarnessState):
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(HarnessState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def build_graph(provider_cfg: ProviderConfig, checkpointer):
    model = build_chat_model(provider_cfg).bind_tools(ALL_TOOLS)
    return build_graph_from_model(model, checkpointer)
