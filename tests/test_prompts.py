"""
Phase 5 groundwork: versioned system prompts with a pending/approve
workflow. Runs against the real Postgres service (needs DATABASE_URL --
via docker compose, same as test_worker_resume.py).
"""

import os
import uuid

import psycopg

from harness import prompts


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    return conn


def test_propose_then_approve_lifecycle():
    conn = _conn()
    role = f"test-role-{uuid.uuid4().hex[:8]}"

    assert prompts.get_active(conn, role) is None

    v1 = prompts.propose(conn, role, "be helpful", reason="initial")
    assert v1 == 1
    assert prompts.get_active(conn, role) is None  # proposed, not yet active
    assert [p["version"] for p in prompts.pending(conn, role)] == [1]

    prompts.approve(conn, role, v1)
    assert prompts.get_active(conn, role) == "be helpful"
    assert prompts.pending(conn, role) == []


def test_approving_a_new_version_supersedes_the_old_one():
    conn = _conn()
    role = f"test-role-{uuid.uuid4().hex[:8]}"

    v1 = prompts.propose(conn, role, "v1 text")
    prompts.approve(conn, role, v1)
    assert prompts.get_active(conn, role) == "v1 text"

    v2 = prompts.propose(conn, role, "v2 text")
    assert v2 == 2
    assert prompts.get_active(conn, role) == "v1 text"  # v2 still just pending

    prompts.approve(conn, role, v2)
    assert prompts.get_active(conn, role) == "v2 text"
