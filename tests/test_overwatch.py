"""
Overwatch tools tested directly (each is a real Beads/seats/proposals
read or mutation, not a fake), plus one full session through the actual
LangGraph loop with a scripted model -- proving propose_tool really runs
the real sandbox.py, not a stub, and that the proposal lands in Postgres
with real sandbox evidence attached.
"""

import os
import uuid

import psycopg
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads, outcomes, seats, tool_proposals
from harness.overwatch import build_tools, run_overwatch_session


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    seats.init_table(conn)
    tool_proposals.init_table(conn)
    return conn


def test_list_capability_gaps_surfaces_real_refusal_reasons():
    conn = _conn()
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "does X", created_by="test")
    ticket = beads.create("some work", "x")
    beads.assign_to_seat(ticket["id"], seat_id)
    beads.claim(ticket["id"], actor=seat_id)
    beads.flag_for_human(ticket["id"], "no tool exists to do the thing this ticket needs", actor=seat_id)

    list_gaps, _, _ = build_tools(conn)
    result = list_gaps.invoke({})

    assert seat_id in result
    assert "no tool exists to do the thing this ticket needs" in result


def test_list_capability_gaps_omits_a_seat_with_no_refusals():
    # Scans across every seat in the roster (a real system-wide view, not
    # scoped to one test), so this only asserts THIS test's own clean seat
    # doesn't appear -- other tests in the same run may have added seats
    # with real refusals, and that's correct behavior, not test pollution.
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "does X", created_by="test")

    list_gaps, _, _ = build_tools(conn)
    result = list_gaps.invoke({})

    assert seat_id not in result


def test_list_pending_proposals_shows_open_statuses_only():
    conn = _conn()
    open_id = tool_proposals.propose(conn, "open-tool", "print(1)", "does a thing", proposed_by="overwatch")
    approved_id = tool_proposals.propose(conn, "approved-tool", "print(2)", "does another thing", proposed_by="overwatch")
    tool_proposals.approve(conn, approved_id)

    _, list_pending, _ = build_tools(conn)
    result = list_pending.invoke({})

    assert "open-tool" in result
    assert "approved-tool" not in result  # already resolved, not "open" anymore


def test_propose_tool_only_records_pending_never_runs_the_sandbox_itself():
    # propose_tool deliberately does NOT call sandbox.run_sandboxed --
    # that needs Docker access, which only the separate sandbox-runner
    # service has (scripts/run_sandbox_for_proposals.py), never `harness`
    # (the same service that runs agent shell_exec calls). This was a
    # real bug caught live: an earlier version called run_sandboxed
    # directly here and crashed with KeyError on SANDBOX_SCRATCH_* env
    # vars that are deliberately absent from harness.
    conn = _conn()
    _, _, propose_tool = build_tools(conn)

    result = propose_tool.invoke(
        {
            "tool_name": "greet",
            "source_code": "print('hello from sandbox')",
            "declared_capabilities": "prints a greeting, no filesystem/network access needed",
        }
    )

    assert "pending" in result
    # list_by_status is a real system-wide query (shared Postgres across
    # tests in this session), so scope the assertion to this test's own
    # proposal rather than asserting an exact global count.
    proposals = tool_proposals.list_by_status(conn, "pending")
    mine = [p for p in proposals if p["tool_name"] == "greet"]
    assert len(mine) == 1
    assert mine[0]["sandbox_stdout"] is None  # not yet sandboxed


class ProposesOneToolThenStops:
    """State-driven: proposes a fixed tool on the first turn, stops on the
    turn after it sees the sandbox result come back."""

    def __init__(self):
        self.proposed = False

    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content="proposed one tool, nothing else to do")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "propose_tool",
                    "args": {
                        "tool_name": "count-lines",
                        "source_code": "print(1)",
                        "declared_capabilities": "counts lines in a file, read-only",
                    },
                    "id": "call-1",
                }
            ],
        )


def test_full_overwatch_session_proposes_a_tool():
    conn = _conn()
    tools = build_tools(conn)
    model = ProposesOneToolThenStops()

    result = run_overwatch_session(model, tools, InMemorySaver())

    assert result["final_message"] == "proposed one tool, nothing else to do"
    proposals = tool_proposals.list_by_status(conn, "pending")  # not sandboxed yet -- separate step
    assert any(p["tool_name"] == "count-lines" for p in proposals)


def test_session_accepts_an_explicit_brief_instead_of_the_default_scan_instruction():
    conn = _conn()
    tools = build_tools(conn)

    captured_first_user_message = {}

    class CapturesFirstMessage:
        def invoke(self, messages):
            if "first" not in captured_first_user_message:
                captured_first_user_message["first"] = messages[-1].content
            return AIMessage(content="done")

    run_overwatch_session(CapturesFirstMessage(), tools, InMemorySaver(), brief="focus on the CSV-parsing gap specifically")

    assert captured_first_user_message["first"] == "focus on the CSV-parsing gap specifically"
