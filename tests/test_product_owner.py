"""
Product-owner tools tested directly (each is a real Beads/seats mutation,
not a fake), plus one full triage session through the actual LangGraph
loop with a scripted model -- proving the graph wiring works with a
completely different tool set than the general worker's ALL_TOOLS, not
just that build_graph_from_model's `tools` parameter exists.
"""

import json
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads, seats
from harness.product_owner import build_tools, run_triage_session


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Response", (), {"content": self.content})()


def _conn():
    import os

    import psycopg

    from harness import prompts

    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    seats.init_table(conn)
    return conn


def test_list_seats_tool_includes_outcomes():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "frontend specialist", created_by="test")

    list_seats, _, _, _ = build_tools(conn, requesting_model=None)
    result = list_seats.invoke({})

    assert seat_id in result
    assert "frontend specialist" in result
    assert "closed=" in result


def test_list_unassigned_tickets_tool():
    beads.ensure_initialized()
    unassigned = beads.create("needs triage", "x")

    _, list_unassigned, _, _ = build_tools(_conn(), requesting_model=None)
    result = list_unassigned.invoke({})

    assert unassigned["id"] in result
    assert "needs triage" in result


def test_assign_ticket_tool_requires_existing_seat():
    conn = _conn()
    beads.ensure_initialized()
    ticket = beads.create("some work", "x")

    _, _, assign_ticket, _ = build_tools(conn, requesting_model=None)
    result = assign_ticket.invoke({"issue_id": ticket["id"], "seat_id": "does-not-exist"})

    assert "error" in result.lower()
    assert beads.assigned_seat(beads.show(ticket["id"])) is None  # not assigned


def test_assign_ticket_tool_assigns_when_seat_exists():
    conn = _conn()
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "some specialty", created_by="test")
    ticket = beads.create("some work", "x")

    _, _, assign_ticket, _ = build_tools(conn, requesting_model=None)
    result = assign_ticket.invoke({"issue_id": ticket["id"], "seat_id": seat_id})

    assert seat_id in result
    assert beads.assigned_seat(beads.show(ticket["id"])) == seat_id


def test_request_new_seat_tool_delegates_to_meta_agent():
    conn = _conn()
    new_seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    requesting_model = FakeModel(json.dumps({"seat_id": new_seat_id, "system_prompt": "specialize in Y"}))

    _, _, _, request_new_seat = build_tools(conn, requesting_model)
    result = request_new_seat.invoke({"specialty_description": "does Y"})

    assert new_seat_id in result
    assert seats.get(conn, new_seat_id) is not None


class AssignsOneTicketThenStops:
    """State-driven: lists unassigned tickets, assigns the first one it
    sees to a known seat, then stops on the next turn."""

    def __init__(self, target_ticket_id, target_seat_id):
        self.target_ticket_id = target_ticket_id
        self.target_seat_id = target_seat_id

    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done triaging")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "assign_ticket",
                    "args": {"issue_id": self.target_ticket_id, "seat_id": self.target_seat_id},
                    "id": "call-1",
                }
            ],
        )


def test_full_triage_session_assigns_a_ticket():
    conn = _conn()
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "handles this kind of work", created_by="test")
    ticket = beads.create("needs the specialist", "x")

    tools = build_tools(conn, requesting_model=None)
    model = AssignsOneTicketThenStops(ticket["id"], seat_id)

    result = run_triage_session(model, tools, InMemorySaver())

    assert result["final_message"] == "done triaging"
    assert beads.assigned_seat(beads.show(ticket["id"])) == seat_id
