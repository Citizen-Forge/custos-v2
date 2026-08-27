"""
Phase 5 groundwork: versioned system prompts per role, with a
pending/approve workflow. This is the actual substrate a meta-agent needs
to "tune" anything -- without it, there's no persistent, editable prompt
to propose changes to in the first place.

A proposed prompt never takes effect on its own; only `approve` flips it
active. Matches v1's existing "autonomy off by default for every role
except product-owner" pattern (PLAN.md Phase 5).

Lives in the same Postgres instance as the checkpointer, its own table
and connection -- similar split to what queue_store.py used to have
before Beads replaced it as the actual work queue (PLAN.md Phase 1);
prompts have no Beads equivalent, they aren't work items.
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_prompts (
            id SERIAL PRIMARY KEY,
            role TEXT NOT NULL,
            version INT NOT NULL,
            text TEXT NOT NULL,
            reason TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            approved_at TIMESTAMPTZ,
            UNIQUE (role, version)
        )
        """
    )


def get_active(conn, role: str) -> str | None:
    row = conn.execute(
        "SELECT text FROM system_prompts WHERE role = %s AND status = 'active' ORDER BY version DESC LIMIT 1",
        (role,),
    ).fetchone()
    return row[0] if row else None


def propose(conn, role: str, text: str, reason: str = "") -> int:
    next_version = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM system_prompts WHERE role = %s", (role,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO system_prompts (role, version, text, reason, status) VALUES (%s, %s, %s, %s, 'pending')",
        (role, next_version, text, reason),
    )
    return next_version


def approve(conn, role: str, version: int) -> None:
    conn.execute(
        "UPDATE system_prompts SET status = 'superseded' WHERE role = %s AND status = 'active'",
        (role,),
    )
    conn.execute(
        "UPDATE system_prompts SET status = 'active', approved_at = now() WHERE role = %s AND version = %s",
        (role, version),
    )


def pending(conn, role: str | None = None) -> list[dict]:
    if role:
        rows = conn.execute(
            "SELECT role, version, text, reason, created_at FROM system_prompts "
            "WHERE role = %s AND status = 'pending' ORDER BY version",
            (role,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT role, version, text, reason, created_at FROM system_prompts "
            "WHERE status = 'pending' ORDER BY role, version"
        ).fetchall()
    return [{"role": r[0], "version": r[1], "text": r[2], "reason": r[3], "created_at": r[4]} for r in rows]
