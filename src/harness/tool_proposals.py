"""
Phase 7: pending tool proposals -- candidate code, its declared
capabilities, its sandbox run results, and a (separate) reviewer agent's
verdict, all tracked before anything is ever added to a seat's real tool
list. Unlike `meta_agent.create_specialist_seat` (a new seat goes active
immediately -- low blast radius, just a name and a prompt), a new *tool*
never auto-activates at any stage: `approve` is a distinct, human-only
step, deliberate given generated code would run at `shell_exec`-level
trust once promoted. Mirrors `prompts.py`'s pending/approve shape,
extended with what a code proposal specifically needs reviewed against.

Lifecycle: pending (just proposed) -> sandboxed (ran in sandbox.py's
container, results attached) -> reviewed (a reviewer agent's verdict
attached -- still not applied) -> approved (human-only) or rejected.
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_proposals (
            id SERIAL PRIMARY KEY,
            tool_name TEXT NOT NULL,
            source_code TEXT NOT NULL,
            declared_capabilities TEXT NOT NULL,
            proposed_by TEXT NOT NULL,
            sandbox_stdout TEXT,
            sandbox_stderr TEXT,
            sandbox_exit_code INT,
            review_verdict TEXT,
            review_reasoning TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            approved_at TIMESTAMPTZ
        )
        """
    )


def propose(conn, tool_name: str, source_code: str, declared_capabilities: str, proposed_by: str) -> int:
    row = conn.execute(
        "INSERT INTO tool_proposals (tool_name, source_code, declared_capabilities, proposed_by) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (tool_name, source_code, declared_capabilities, proposed_by),
    ).fetchone()
    return row[0]


def record_sandbox_result(conn, proposal_id: int, result) -> None:
    """`result` is a `sandbox.SandboxResult`."""
    conn.execute(
        "UPDATE tool_proposals SET sandbox_stdout=%s, sandbox_stderr=%s, sandbox_exit_code=%s, "
        "status='sandboxed' WHERE id=%s",
        (result.stdout, result.stderr, result.exit_code, proposal_id),
    )


def record_review(conn, proposal_id: int, verdict: str, reasoning: str) -> None:
    """`verdict` is the reviewer agent's own call ('allow'/'deny') -- recorded, never applied on its own."""
    conn.execute(
        "UPDATE tool_proposals SET review_verdict=%s, review_reasoning=%s, status='reviewed' WHERE id=%s",
        (verdict, reasoning, proposal_id),
    )


def approve(conn, proposal_id: int) -> None:
    conn.execute(
        "UPDATE tool_proposals SET status='approved', approved_at=now() WHERE id=%s",
        (proposal_id,),
    )


def reject(conn, proposal_id: int, reason: str | None = None) -> None:
    conn.execute(
        "UPDATE tool_proposals SET status='rejected', review_reasoning=COALESCE(%s, review_reasoning) WHERE id=%s",
        (reason, proposal_id),
    )


def get(conn, proposal_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, tool_name, source_code, declared_capabilities, proposed_by, "
        "sandbox_stdout, sandbox_stderr, sandbox_exit_code, review_verdict, review_reasoning, "
        "status, created_at, approved_at FROM tool_proposals WHERE id = %s",
        (proposal_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_by_status(conn, status: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, tool_name, source_code, declared_capabilities, proposed_by, "
        "sandbox_stdout, sandbox_stderr, sandbox_exit_code, review_verdict, review_reasoning, "
        "status, created_at, approved_at FROM tool_proposals WHERE status = %s ORDER BY created_at",
        (status,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "tool_name": row[1],
        "source_code": row[2],
        "declared_capabilities": row[3],
        "proposed_by": row[4],
        "sandbox_stdout": row[5],
        "sandbox_stderr": row[6],
        "sandbox_exit_code": row[7],
        "review_verdict": row[8],
        "review_reasoning": row[9],
        "status": row[10],
        "created_at": row[11],
        "approved_at": row[12],
    }
