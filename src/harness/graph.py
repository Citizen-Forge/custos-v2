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

import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from . import permissions
from .classifier import Verdict
from .providers import ProviderConfig, build_chat_model
from .state import HarnessState
from .tools import ALL_TOOLS

HANDOFF_NUDGE = (
    "You've reached your turn budget for this session. Please wrap up: call "
    "write_handoff_note with what's done and what's left, then stop."
)

# The tools that legitimately end a ticket. complete_ticket records the
# work; refuse_ticket escalates to a human; decline_ticket hands it back.
# A run that stops without one of these has neither done the work nor said
# why not, which is what worker.work_one_ticket then flags for a human.
TERMINAL_TOOL_NAMES = {"complete_ticket", "refuse_ticket", "decline_ticket"}

# One extra turn, offered when the model stops without a terminal call.
# Off the back of the strongest remaining failure mode on the board
# (2026-09-13: ~13 of 16 parked Silent Run tickets read "agent stopped
# without calling complete_ticket"): the model finishes its reasoning,
# writes a summary in prose, and ends the turn without ever calling the
# tool. One nudge converts most of those into a real completion claim.
# Deliberately ONE, not a loop: a model that ignores the nudge or answers
# it with more tool calls must not be given an unbounded chance to spin.
COMPLETION_NUDGE = (
    "Your turn ended without finishing this ticket: you have not called "
    "complete_ticket. If the work is genuinely done, call complete_ticket NOW with a "
    "concrete summary of what you changed and how you checked it. If it is not done and "
    "you cannot finish it, call refuse_ticket with the reason, or decline_ticket if it "
    "belongs to another specialist. Do not simply stop again."
)


def _terminal_called(messages: list) -> bool:
    """True once the model has called complete_ticket/refuse_ticket/
    decline_ticket at any point in this thread."""
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") in TERMINAL_TOOL_NAMES:
                return True
    return False


def _stopped_without_terminal(messages: list) -> bool:
    """The model's last turn ended on prose, with no tool call and no
    terminal call already made."""
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return False
    if getattr(last, "tool_calls", None):
        return False
    return not _terminal_called(messages)


def _repair_dangling_tool_calls(messages: list) -> list:
    """Make a checkpointed history valid for the provider again.

    A run that dies -- or is killed -- between the model returning tool_calls
    and the tool results being recorded leaves an assistant message whose calls
    nothing answers. The provider rejects the ENTIRE request over that
    ("an assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'"), and the malformed history is exactly
    what got checkpointed, so every resume fails identically. A transient crash
    therefore becomes a ticket permanently parked for a human -- the queue
    quietly grows instead of draining.

    Found live 2026-09-15: ten threads were in this state at once (five ticket
    threads and five harness threads -- triage, dispatch, overwatch, reflect),
    which is what "blocked tickets building up again" turned out to be.

    The repair answers the missing calls in place, saying they never returned.
    That keeps every other message in the thread -- the work the agent actually
    did -- instead of throwing the thread away, and it is idempotent: a history
    that is already valid comes back unchanged.
    """
    repaired: list = []
    index = 0
    while index < len(messages):
        message = messages[index]
        repaired.append(message)
        index += 1
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            continue
        # The contiguous run of tool messages answering this assistant turn.
        answered: set = set()
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            answered.add(getattr(messages[index], "tool_call_id", None))
            repaired.append(messages[index])
            index += 1
        for call in calls:
            if call.get("id") not in answered:
                repaired.append(
                    ToolMessage(
                        content=(
                            "not executed: the run stopped before this tool call "
                            "returned, so it has no result. Re-issue it if you still "
                            "need it."
                        ),
                        tool_call_id=call["id"],
                    )
                )
    return repaired


# The provider refuses a request that outgrows its window, and refuses it on
# EVERY retry, because the request is rebuilt from the same checkpoint each
# time -- so an over-long thread parks its ticket exactly the way a malformed
# one does, and the human queue fills up with work that could not have failed
# for a content reason.
#
# Measured live 2026-09-15: the threads that failed had 600-830 KB of
# checkpointed messages while a 195 KB one ran fine, and the SAME message list
# both succeeded and failed depending only on how much output budget the
# request reserved -- which is the signature of input + output exceeding the
# window rather than of a bad message shape.
#
# Tool results are what make threads grow (one `shell_exec` can return tens of
# KB), so each is capped as well as the total. Both limits are deliberately
# generous: this is a backstop against a request the provider will refuse, not
# a context-management policy.
HISTORY_MAX_CHARS = int(os.environ.get("HISTORY_MAX_CHARS", "400000"))
TOOL_MESSAGE_MAX_CHARS = int(os.environ.get("TOOL_MESSAGE_MAX_CHARS", "20000"))


def _cap_tool_message(message: ToolMessage) -> ToolMessage:
    """Shorten one tool result, marking the cut so the model knows it is partial."""
    content = getattr(message, "content", None)
    if not isinstance(content, str) or len(content) <= TOOL_MESSAGE_MAX_CHARS:
        return message
    return ToolMessage(
        content=(
            content[:TOOL_MESSAGE_MAX_CHARS]
            + f"\n... [truncated at {TOOL_MESSAGE_MAX_CHARS} characters]"
        ),
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
    )


def _bound_history(messages: list) -> list:
    """Drop the oldest messages until the request fits, cutting only at boundaries.

    The retained window must never BEGIN on a tool message: a tool reply whose
    assistant message was dropped is invalid on its own, which is the exact
    failure this exists to prevent. Keeping a suffix (rather than a prefix plus
    a suffix) is what makes that a one-line guarantee -- every assistant
    tool_calls inside a suffix still has its replies directly after it.

    A history that already fits is returned unchanged.
    """
    capped = [
        _cap_tool_message(m) if isinstance(m, ToolMessage) else m for m in messages
    ]
    total = sum(len(str(getattr(m, "content", "") or "")) for m in capped)
    if total <= HISTORY_MAX_CHARS:
        return capped

    # Keep the LEAD-IN, which for a ticket thread is [system prompt, brief].
    # Dropping the brief is not a tidy truncation: it removes the only copy of
    # the ticket text, and agents then refuse with "no ticket text reached this
    # turn". Found live 2026-09-15, after the first version of this bound
    # shipped -- 6.1, 6.3, 6.5 and 9.4 were all parked saying exactly that.
    # Only system/human messages are kept, never an assistant turn, so this can
    # never separate a tool_calls message from its replies.
    head: list = []
    for message in capped[:2]:
        if getattr(message, "type", None) in ("system", "human"):
            head.append(message)
        else:
            break
    body = capped[len(head):]

    kept: list = []
    used = 0
    for message in reversed(body):
        size = len(str(getattr(message, "content", "") or ""))
        if kept and used + size > HISTORY_MAX_CHARS:
            break
        kept.append(message)
        used += size
    kept.reverse()
    while kept and isinstance(kept[0], ToolMessage):
        kept.pop(0)
    return head + kept


def build_graph_from_model(model, checkpointer, tools=None, classify=None, interrupt_after=None, turn_budget=None, workspace_root=None, completion_gate=False):
    """Build the graph from an already-tool-bound model. Split out from
    `build_graph` so tests can pass a fake model without a real provider.

    `tools`: the toolset available to this graph's `tools` node. Defaults
    to `ALL_TOOLS` (the general worker's set) when omitted.

    `classify` is an optional `(tool_name, tool_args) -> classifier.Verdict`
    callable (see classifier.build_classifier). When omitted, the gate node
    allows everything -- used by tests that predate/don't exercise gating
    (tests/test_graph.py, tests/test_worker_resume.py) so they don't need a
    classifier model.

    `workspace_root` enables the static allow-list fast path
    (permissions.is_statically_safe): ordinary build/test/git commands
    scoped to the workspace skip `classify` entirely. `None` disables it,
    so every call is classified as before.

    `turn_budget` is an optional int: when the running turn count hits it
    exactly, one HANDOFF_NUDGE message is appended before that call (once,
    not repeated on every subsequent turn). `None` disables it entirely --
    existing tests that don't pass it are unaffected.

    `completion_gate` (worker only, off otherwise): when the model stops
    without calling complete_ticket/refuse_ticket/decline_ticket, offer one
    COMPLETION_NUDGE turn before ending. Opt-in rather than inferred from
    the toolset so the graph stays a plain ReAct loop for every caller that
    is not a ticket worker -- scripted test models that end with a tool call
    should not suddenly grow an extra turn.

    `interrupt_after` is test-only (see tests/test_worker_resume.py) --
    production never sets it.
    """
    tools = ALL_TOOLS if tools is None else tools
    has_completion_gate = completion_gate

    def call_model(state: HarnessState):
        turn_count = state.get("turn_count", 0) + 1
        nudged = state.get("completion_nudged", False)
        # Repaired here rather than at each resume site so EVERY caller is
        # covered -- the worker, product-owner triage, overwatch and
        # reflection all resume checkpointed threads through this node. Then
        # bounded, so a long thread cannot outgrow the provider's window and
        # 400 on every retry.
        messages = _bound_history(_repair_dangling_tool_calls(state["messages"]))
        extra = []
        if turn_budget is not None and turn_count == turn_budget:
            nudge = HumanMessage(content=HANDOFF_NUDGE)
            messages = messages + [nudge]
            extra = [nudge]
        if has_completion_gate and not nudged and _stopped_without_terminal(messages):
            nudge = HumanMessage(content=COMPLETION_NUDGE)
            messages = messages + [nudge]
            extra = [*extra, nudge]
            nudged = True
        return {
            "messages": [*extra, model.invoke(messages)],
            "turn_count": turn_count,
            "completion_nudged": nudged,
        }

    def permission_gate(state: HarnessState):
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return {}

        verdicts = {}
        for call in tool_calls:
            forbidden = permissions.forbidden_reason(call["name"], call["args"])
            if forbidden:
                # Hard denial, ahead of both the allow-list and the
                # classifier: these commands move the integration branch or
                # mutate the board outside the harness's own pathways, so
                # they are not a judgment the classifier gets to make.
                verdicts[call["id"]] = Verdict("deny", forbidden)
            elif permissions.is_statically_safe(call["name"], call["args"], workspace_root):
                verdicts[call["id"]] = Verdict("allow", "allow-listed dev command")
            elif classify is not None:
                verdicts[call["id"]] = classify(call["name"], call["args"])
            else:
                verdicts[call["id"]] = Verdict("allow", "no classifier configured")

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

    def after_agent(state: HarnessState) -> str:
        """Where the agent goes when it stops making tool calls.

        Normally END. With a completion gate, a run that stops without a
        terminal call gets ONE extra turn (see COMPLETION_NUDGE) before
        ending -- after which it ends regardless, so a model that answers
        the nudge with more tool calls cannot loop here."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "permission_gate"
        if not has_completion_gate or state.get("completion_nudged", False):
            return END
        if _terminal_called(state["messages"]):
            return END
        return "agent"

    builder = StateGraph(HarnessState)
    builder.add_node("agent", call_model)
    builder.add_node("permission_gate", permission_gate)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, {"permission_gate": "permission_gate", "agent": "agent", END: END})
    builder.add_conditional_edges("permission_gate", route_after_gate, {"tools": "tools", "agent": "agent"})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


def build_graph(provider_cfg: ProviderConfig, checkpointer, tools=None, classifier=None, turn_budget=None, workspace_root=None, completion_gate=False):
    tools = ALL_TOOLS if tools is None else tools
    model = build_chat_model(provider_cfg).bind_tools(tools)
    return build_graph_from_model(
        model, checkpointer, tools=tools, classify=classifier,
        turn_budget=turn_budget, workspace_root=workspace_root,
        completion_gate=completion_gate,
    )
