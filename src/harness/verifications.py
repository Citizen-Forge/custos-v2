"""
Storage for the acceptance-criteria verification loop (added 2026-08-29,
replaces the originally-planned "Laurels" human-feedback surface -- user's
call: for this project's kind of work, an automated pass/fail against a
ticket's own stated acceptance criteria is a better positive-feedback
signal than a human rating, and it's the mechanism that actually gives
meta-agent (Phase 5) real quality data instead of just closed/refused
counts. Mirrors tool_proposals.py's shape: a dedicated small table, not
folded into Beads metadata, because this needs to be queried/aggregated
by seat (verifier.py, outcomes.py) the way tool_proposals is queried by
status.
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verifications (
            id SERIAL PRIMARY KEY,
            issue_id TEXT NOT NULL UNIQUE,
            seat_id TEXT NOT NULL,
            verdict TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def record(conn, issue_id: str, seat_id: str, verdict: str, reasoning: str) -> int:
    row = conn.execute(
        "INSERT INTO verifications (issue_id, seat_id, verdict, reasoning) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (issue_id) DO UPDATE SET seat_id = EXCLUDED.seat_id, verdict = EXCLUDED.verdict, "
        "reasoning = EXCLUDED.reasoning, verified_at = now() "
        "RETURNING id",
        (issue_id, seat_id, verdict, reasoning),
    ).fetchone()
    return row[0]


def get_for_issue(conn, issue_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, issue_id, seat_id, verdict, reasoning, verified_at FROM verifications WHERE issue_id = %s",
        (issue_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_for_seat(conn, seat_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, issue_id, seat_id, verdict, reasoning, verified_at FROM verifications "
        "WHERE seat_id = %s ORDER BY verified_at",
        (seat_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def summary(conn, seat_id: str) -> dict:
    """Pass/fail counts + real failure reasoning for a seat -- the actual
    quality signal meta_agent.py's harder judgment case (real problems,
    not just refusals) needs. Empty/zero when nothing's been verified
    yet, not an error -- a seat with no acceptance-criteria tickets is a
    real, valid state, not a broken one."""
    verifications = list_for_seat(conn, seat_id)
    passed = [v for v in verifications if v["verdict"] == "pass"]
    failed = [v for v in verifications if v["verdict"] == "fail"]
    return {
        "seat_id": seat_id,
        "verified_total": len(verifications),
        "passed": len(passed),
        "failed": len(failed),
        "fail_reasons": [v["reasoning"] for v in failed],
    }


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "issue_id": row[1],
        "seat_id": row[2],
        "verdict": row[3],
        "reasoning": row[4],
        "verified_at": row[5],
    }
