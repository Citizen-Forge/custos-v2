"""
Phase 7's overwatch agent: the piece PLAN.md described as "write new
tools, extend the harness's own capabilities" but never built the
judgment for -- the containment/proposal substrate (sandbox.py,
tool_proposals.py) existed first specifically so this agent's output is
never trusted by default, no matter how good its judgment turns out to
be (PLAN.md Phase 7's "promotion gate").

Same tool-calling LangGraph shape as product_owner.py (one thread per
session, a handful of tools, run to a final "I'm done" message) but a
narrower, lower-trust toolset: it can only ever *propose* a tool, never
register/activate one, and -- critically -- `propose_tool` does NOT run
the sandbox itself. This module runs inside the `harness` service, the
same one whose tickets execute arbitrary `shell_exec` calls; PLAN.md
Phase 7 is explicit that the Docker socket sandboxing needs must live
*only* in the dedicated `sandbox-runner` service, never here, or a
worker with both capabilities could just `docker run` its way out.
Calling `sandbox.run_sandboxed` from this module was tried and caught by
the test suite (`KeyError` on `SANDBOX_SCRATCH_*`, since those env vars
and the socket mount are deliberately absent from `harness`) -- a real
design bug, not a config gap, fixed by splitting proposal creation
(here) from the actual sandbox run (`scripts/run_sandbox_for_proposals.py`,
invoked only via the `sandbox-runner` service, same pattern as
`tests/test_sandbox.py`). This means the overwatch agent does NOT see
its own code's real sandbox output within the same session/turn the way
first drafted -- an inherent, necessary consequence of the privilege
boundary, not an oversight.

Signal source for "what's missing": real refusal reasons across the
current seat roster (outcomes.py's per-actor refused_reasons, rolled up
here across every seat) -- a seat that keeps getting flagged for human
review for the same kind of reason is exactly the "no tool covers this"
signal PLAN.md's original framing was after. Thin today because refusal
data is thin (most real runs so far have succeeded, not refused) -- an
explicit `brief` argument lets a session be pointed at a specific gap
directly rather than only relying on the automatic scan, same relief
valve product_owner.request_new_seat gives a human/product-owner for a
gap the automatic path hasn't surfaced yet.
"""

import uuid

from langchain_core.tools import tool

from . import outcomes, seats, tool_proposals
from .graph import build_graph_from_model

ROLE = "overwatch"

SYSTEM_PROMPT = """You are the overwatch agent for an autonomous software delivery system. \
Your job: notice when the existing toolset can't cover work that's being asked of it, and \
propose a new tool to close that gap. Use list_capability_gaps to see real evidence (seats \
that keep getting flagged for human review, and why) and list_pending_proposals to avoid \
duplicating a proposal already in review. When you have a concrete gap and a specific tool \
that would close it, write the tool as a single self-contained Python script (it will later \
run sandboxed by a separate process: no network, no filesystem beyond its own directory, no \
secrets, a strict resource limit, and a short timeout -- write it accordingly) and call \
propose_tool. You will NOT see the sandbox output in this session -- a separate process runs \
it, and a separate reviewer agent judges the result, both deliberately outside your own \
privileges. Do not propose a tool for a gap you don't have real evidence for. When there's \
nothing concrete to propose, say so and stop rather than inventing a speculative tool."""


def build_tools(conn):
    @tool
    def list_capability_gaps() -> str:
        """List seats that have been flagged for human review recently, with why -- the real
        evidence for 'no existing tool covers this kind of work.'"""
        roster = seats.list_all(conn)
        lines = []
        for s in roster:
            o = outcomes.summary(s["seat_id"])
            if o["refused_reasons"]:
                for reason in o["refused_reasons"]:
                    lines.append(f"{s['seat_id']} ({s['specialty']}): {reason}")
        return "\n".join(lines) if lines else "no refusals on record -- no concrete gap evidence yet"

    @tool
    def list_pending_proposals() -> str:
        """List tool proposals still awaiting review or a human decision -- check before proposing
        something that's already in the pipeline."""
        pending = tool_proposals.list_by_status(conn, "pending")
        sandboxed = tool_proposals.list_by_status(conn, "sandboxed")
        reviewed = tool_proposals.list_by_status(conn, "reviewed")
        all_open = pending + sandboxed + reviewed
        if not all_open:
            return "no open proposals"
        return "\n".join(f"#{p['id']} {p['tool_name']} ({p['status']}): {p['declared_capabilities']}" for p in all_open)

    @tool
    def propose_tool(tool_name: str, source_code: str, declared_capabilities: str) -> str:
        """Propose a new tool. Does NOT run it -- sandboxing happens later in a separate,
        more privileged process (this agent never gets Docker access, deliberately). You
        will not see the sandbox output or the reviewer's verdict in this session."""
        proposal_id = tool_proposals.propose(conn, tool_name, source_code, declared_capabilities, proposed_by=ROLE)
        return (
            f"proposal #{proposal_id} recorded (status: pending). "
            f"Sandboxing and review happen in a later, separate step -- nothing applied automatically."
        )

    return [list_capability_gaps, list_pending_proposals, propose_tool]


def run_overwatch_session(agent_model, tools, checkpointer, brief: str | None = None, thread_id: str | None = None) -> dict:
    """One overwatch pass. `agent_model` must already be tool-bound to
    `tools` (matches product_owner.run_triage_session's contract).
    `brief` optionally points the session at a specific gap rather than
    relying only on the automatic list_capability_gaps scan -- useful
    while refusal data is thin, and mirrors the relief valve
    product_owner.request_new_seat already gives a human for the same
    reason."""
    thread_id = thread_id or f"overwatch-{uuid.uuid4().hex[:8]}"
    graph = build_graph_from_model(agent_model, checkpointer, tools=tools)
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    if state.values:
        result = graph.invoke(None, config)
    else:
        user_message = brief or "Scan for capability gaps and propose a tool if you find real evidence for one."
        result = graph.invoke(
            {
                "messages": [("system", SYSTEM_PROMPT), ("user", user_message)],
                "ticket_id": thread_id,
                "turn_count": 0,
            },
            config,
        )

    return {"thread_id": thread_id, "final_message": result["messages"][-1].content}
