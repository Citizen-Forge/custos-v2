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

    list_seats, _, _, _, _, _, _, _ = build_tools(conn, requesting_model=None)
    result = list_seats.invoke({})

    assert seat_id in result
    assert "frontend specialist" in result
    assert "closed=" in result


def test_list_unassigned_tickets_tool():
    beads.ensure_initialized()
    unassigned = beads.create("needs triage", "x")

    _, list_unassigned, _, _, _, _, _, _ = build_tools(_conn(), requesting_model=None)
    result = list_unassigned.invoke({})

    assert unassigned["id"] in result
    assert "needs triage" in result


def test_assign_ticket_tool_requires_existing_seat():
    conn = _conn()
    beads.ensure_initialized()
    ticket = beads.create("some work", "x")

    _, _, assign_ticket, _, _, _, _, _ = build_tools(conn, requesting_model=None)
    result = assign_ticket.invoke({"issue_id": ticket["id"], "seat_id": "does-not-exist"})

    assert "error" in result.lower()
    assert beads.assigned_seat(beads.show(ticket["id"])) is None  # not assigned


def test_assign_ticket_tool_assigns_when_seat_exists():
    conn = _conn()
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "some specialty", created_by="test")
    ticket = beads.create("some work", "x")

    _, _, assign_ticket, _, _, _, _, _ = build_tools(conn, requesting_model=None)
    result = assign_ticket.invoke({"issue_id": ticket["id"], "seat_id": seat_id})

    assert seat_id in result
    assert beads.assigned_seat(beads.show(ticket["id"])) == seat_id


def test_list_projects_tool_shows_priority_ordering():
    beads.ensure_initialized()
    _, _, _, _, list_projects, create_project, _, _ = build_tools(_conn(), requesting_model=None)

    create_project.invoke({"name": "low priority idea", "description": "someday", "priority": 4})
    create_project.invoke({"name": "urgent idea", "description": "now", "priority": 0})

    result = list_projects.invoke({})

    assert "urgent idea" in result
    assert "low priority idea" in result
    # priority 0 (highest) sorts before priority 4 -- matches bd list --sort priority
    assert result.index("urgent idea") < result.index("low priority idea")


def test_create_project_tool_creates_a_top_level_issue_with_priority():
    beads.ensure_initialized()
    _, _, _, _, _, create_project, _, _ = build_tools(_conn(), requesting_model=None)

    result = create_project.invoke({"name": "new project", "description": "the goal", "priority": 1})

    project_id = result.split()[2]
    shown = beads.show(project_id)
    assert shown["title"] == "new project"
    assert shown["priority"] == 1
    assert "." not in shown["id"]  # top-level, no parent


def test_create_epic_tool_requires_a_project_parent():
    beads.ensure_initialized()
    _, _, _, _, _, create_project, create_epic, _ = build_tools(_conn(), requesting_model=None)

    project_result = create_project.invoke({"name": "a project", "description": "x", "priority": 2})
    project_id = project_result.split()[2]

    epic_result = create_epic.invoke({"project_id": project_id, "title": "big idea", "description": "the epic goal"})

    epic_id = epic_result.split()[2]
    shown = beads.show(epic_id)
    assert shown["title"] == "big idea"
    assert shown["issue_type"] == "epic"
    assert project_id in shown["id"]  # parented under the project


def test_add_subtask_to_epic_tool_parents_under_the_real_epic():
    beads.ensure_initialized()
    _, _, _, _, _, create_project, create_epic, add_subtask_to_epic = build_tools(_conn(), requesting_model=None)

    project_id = create_project.invoke({"name": "a project", "description": "x", "priority": 2}).split()[2]
    epic_result = create_epic.invoke({"project_id": project_id, "title": "big idea", "description": "the overall goal"})
    epic_id = epic_result.split()[2]

    subtask_result = add_subtask_to_epic.invoke(
        {"epic_id": epic_id, "title": "first concrete piece", "description": "do the first part"}
    )

    assert epic_id in subtask_result
    subtask_id = subtask_result.split()[2]
    shown = beads.show(subtask_id)
    assert shown["title"] == "first concrete piece"
    assert epic_id in shown["id"]  # Beads' hierarchical id convention, e.g. epic-id.1


def test_request_new_seat_tool_delegates_to_meta_agent():
    conn = _conn()
    new_seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    requesting_model = FakeModel(json.dumps({"seat_id": new_seat_id, "system_prompt": "specialize in Y"}))

    _, _, _, request_new_seat, _, _, _, _ = build_tools(conn, requesting_model)
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
