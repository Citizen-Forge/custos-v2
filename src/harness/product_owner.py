"""
The product-owner agent: a tool-calling LangGraph loop, not a rule table
-- assignment, and "does a specialist for this exist yet," are judgment
calls the product-owner makes over live data (the seat roster + outcomes
+ unassigned tickets), not decisions this module makes on its own.
PLAN.md's "emergent, not hardcoded" design goal for how work gets divided
among seats.

One LangGraph thread per triage *session*, not per ticket like the
worker's: a single pass naturally inspects and acts on several tickets,
and possibly creates a new seat, before it's done.
"""

import uuid

from langchain_core.tools import tool

from . import beads, meta_agent, outcomes, seats
from .graph import build_graph_from_model

ROLE = "product_owner"

SYSTEM_PROMPT = """You are the product-owner for an autonomous software delivery system. \
Your job each session: look at unassigned ready tickets and the current specialist seat \
roster, and assign each ticket to the seat best suited for it. If no existing seat is a \
good fit for a ticket, create a new specialist seat for that kind of work rather than \
forcing a mismatch -- specialization should emerge from real needs, not be forced upfront. \
Use the tools available to inspect the roster and outcomes before deciding. When you've \
handled everything you can, stop."""


def build_tools(conn, requesting_model):
    """`requesting_model` is what request_new_seat delegates to (via
    meta_agent.create_specialist_seat) to draft a new seat's prompt --
    pass a *tool-free* model reference here (e.g. a RoutedModel
    constructed with tools=None), since that call just wants a plain
    JSON response, not another round of tool orchestration."""

    @tool
    def list_seats() -> str:
        """List active specialist seats with their recent track record -- check this before assigning work."""
        roster = seats.list_all(conn)
        if not roster:
            return "no seats exist yet"
        lines = []
        for s in roster:
            o = outcomes.summary(s["seat_id"])
            lines.append(
                f"{s['seat_id']}: {s['specialty']} "
                f"(closed={o['closed']}, refused={o['refused']}, still_open={o['still_open']})"
            )
        return "\n".join(lines)

    @tool
    def list_unassigned_tickets() -> str:
        """List ready tickets not yet assigned to any seat."""
        tickets = beads.unassigned_ready()
        if not tickets:
            return "no unassigned tickets"
        return "\n".join(f"{t['id']}: {t['title']} -- {t.get('description', '')}" for t in tickets)

    @tool
    def assign_ticket(issue_id: str, seat_id: str) -> str:
        """Assign an unassigned ticket to an existing seat by id."""
        if not seats.get(conn, seat_id):
            return f"error: no seat {seat_id!r} -- create it first with request_new_seat"
        beads.assign_to_seat(issue_id, seat_id, actor=ROLE)
        return f"assigned {issue_id} to {seat_id}"

    @tool
    def request_new_seat(specialty_description: str) -> str:
        """Create a brand-new specialist seat for work no existing seat covers.
        Goes active immediately -- no approval step, unlike revising an existing seat's prompt."""
        result = meta_agent.create_specialist_seat(
            conn, specialty_description, requested_by=ROLE, model=requesting_model
        )
        if result is None:
            return "failed to create a new seat -- try a more specific specialty description"
        return f"created seat {result['seat_id']}"

    return [list_seats, list_unassigned_tickets, assign_ticket, request_new_seat]


def run_triage_session(agent_model, tools, checkpointer, thread_id: str | None = None) -> dict:
    """One triage pass. `agent_model` must already be tool-bound to
    `tools` (matches `build_graph_from_model`'s existing contract) --
    build it via `RoutedModel(ROLE, routing, gate, tools=tools)` or
    `plain_model.bind_tools(tools)`."""
    thread_id = thread_id or f"triage-{uuid.uuid4().hex[:8]}"
    graph = build_graph_from_model(agent_model, checkpointer, tools=tools)
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    if state.values:
        result = graph.invoke(None, config)
    else:
        result = graph.invoke(
            {
                "messages": [("system", SYSTEM_PROMPT), ("user", "Triage the current queue.")],
                "ticket_id": thread_id,
                "turn_count": 0,
            },
            config,
        )

    return {"thread_id": thread_id, "final_message": result["messages"][-1].content}
