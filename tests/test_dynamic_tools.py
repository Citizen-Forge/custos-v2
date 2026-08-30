"""
Tool activation: an `approved` tool_proposals row becomes a real,
callable langchain tool -- see dynamic_tools.py's module docstring for
the design (mechanical trust established once at proposal-review time,
real-time LLM classifier gating every call after that, same as every
built-in tool). What's proven here: only `approved` proposals are
picked up, a real subprocess actually runs and its output comes back,
a nonzero exit/timeout/bad-args are reported rather than raised, and
the tool's name/description come from the proposal.
"""

import os
import uuid

import psycopg

from harness import dynamic_tools, tool_proposals


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tool_proposals.init_table(conn)
    return conn


def _approved_proposal(conn, source_code: str, tool_name: str | None = None, declared="a test tool") -> int:
    tool_name = tool_name or f"test-tool-{uuid.uuid4().hex[:8]}"
    proposal_id = tool_proposals.propose(conn, tool_name, source_code, declared, proposed_by="test")
    tool_proposals.record_review(conn, proposal_id, "allow", "test")
    tool_proposals.approve(conn, proposal_id)
    return proposal_id


ECHO_ARGS_SCRIPT = """
import sys
print("args:", sys.argv[1:])
"""


def test_build_dynamic_tools_returns_empty_list_when_nothing_approved():
    conn = _conn()
    tool_name = f"never-approved-{uuid.uuid4().hex[:8]}"
    tool_proposals.propose(conn, tool_name, "print('hi')", "unused", proposed_by="test")

    tools = dynamic_tools.build_dynamic_tools(conn)

    assert not any(t.name == tool_name for t in tools)


def test_only_approved_status_proposals_are_included():
    conn = _conn()
    pending_name = f"pending-{uuid.uuid4().hex[:8]}"
    rejected_name = f"rejected-{uuid.uuid4().hex[:8]}"
    tool_proposals.propose(conn, pending_name, "print('hi')", "x", proposed_by="test")
    rejected_id = tool_proposals.propose(conn, rejected_name, "print('hi')", "x", proposed_by="test")
    tool_proposals.record_review(conn, rejected_id, "deny", "no")
    tool_proposals.reject(conn, rejected_id, "no")

    tools = dynamic_tools.build_dynamic_tools(conn)

    names = {t.name for t in tools}
    assert pending_name not in names
    assert rejected_name not in names


def test_approved_tool_actually_runs_as_a_real_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(dynamic_tools, "WORKSPACE_ROOT", str(tmp_path))
    conn = _conn()
    tool_name = f"echo-args-{uuid.uuid4().hex[:8]}"
    _approved_proposal(conn, ECHO_ARGS_SCRIPT, tool_name=tool_name, declared="echoes its argv back")

    tools = dynamic_tools.build_dynamic_tools(conn)
    tool = next(t for t in tools if t.name == tool_name)

    assert tool.description == "echoes its argv back"
    result = tool.invoke({"cli_args": "--foo bar baz"})
    assert "['--foo', 'bar', 'baz']" in result


def test_nonzero_exit_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(dynamic_tools, "WORKSPACE_ROOT", str(tmp_path))
    conn = _conn()
    tool_name = f"fails-{uuid.uuid4().hex[:8]}"
    _approved_proposal(conn, "import sys; print('broken'); sys.exit(3)", tool_name=tool_name)

    tools = dynamic_tools.build_dynamic_tools(conn)
    tool = next(t for t in tools if t.name == tool_name)

    result = tool.invoke({"cli_args": ""})
    assert "exit 3" in result
    assert "broken" in result


def test_timeout_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(dynamic_tools, "WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(dynamic_tools, "TIMEOUT_SECONDS", 1)
    conn = _conn()
    tool_name = f"hangs-{uuid.uuid4().hex[:8]}"
    _approved_proposal(conn, "import time; time.sleep(30)", tool_name=tool_name)

    tools = dynamic_tools.build_dynamic_tools(conn)
    tool = next(t for t in tools if t.name == tool_name)

    result = tool.invoke({"cli_args": ""})
    assert "timed out" in result


def test_unparseable_args_are_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(dynamic_tools, "WORKSPACE_ROOT", str(tmp_path))
    conn = _conn()
    tool_name = f"bad-args-{uuid.uuid4().hex[:8]}"
    _approved_proposal(conn, "print('unreachable')", tool_name=tool_name)

    tools = dynamic_tools.build_dynamic_tools(conn)
    tool = next(t for t in tools if t.name == tool_name)

    result = tool.invoke({"cli_args": "unbalanced 'quote"})
    assert "could not parse args" in result


def test_two_proposals_never_collide_on_the_same_materialized_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dynamic_tools, "WORKSPACE_ROOT", str(tmp_path))
    conn = _conn()
    shared_name = f"shared-name-{uuid.uuid4().hex[:8]}"
    _approved_proposal(conn, "print('first')", tool_name=shared_name)
    # A second proposal reusing the same tool_name (e.g. a superseding
    # approval) must not clobber the first proposal's own materialized
    # source underneath an already-built tool.
    second_id = tool_proposals.propose(conn, shared_name, "print('second')", "x", proposed_by="test")
    tool_proposals.record_review(conn, second_id, "allow", "test")
    tool_proposals.approve(conn, second_id)

    tools = dynamic_tools.build_dynamic_tools(conn)
    matching = [t for t in tools if t.name == shared_name]

    assert len(matching) == 2
    outputs = {t.invoke({"cli_args": ""}).strip() for t in matching}
    assert outputs == {"first", "second"}
