"""
An open-ended registry of named agent "seats" -- specialists the
product-owner creates (via the meta-agent, see meta_agent.create_
specialist_seat) as work demands, not a fixed predefined roster.

A seat_id doubles as the `role` string routing.py/prompts.py/outcomes.py
already accept -- those were built role-open-ended from Phase 2/5
specifically so a growing, emergent seat roster wouldn't need a parallel
prompt/outcome system of its own.

Deliberately thin: specialization is meant to emerge from what a seat
actually gets assigned and how it performs (outcomes.py), not from a
rigid taxonomy enforced here. `specialty` is free text the product-owner
(or the meta-agent, on its behalf) writes when creating the seat -- a
description, not an enum, so nothing here constrains what kinds of
specialists can exist.
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seats (
            seat_id TEXT PRIMARY KEY,
            specialty TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            status TEXT NOT NULL DEFAULT 'active'
        )
        """
    )


def create(conn, seat_id: str, specialty: str, created_by: str) -> None:
    conn.execute(
        "INSERT INTO seats (seat_id, specialty, created_by) VALUES (%s, %s, %s) "
        "ON CONFLICT (seat_id) DO NOTHING",
        (seat_id, specialty, created_by),
    )


def get(conn, seat_id: str) -> dict | None:
    row = conn.execute(
        "SELECT seat_id, specialty, created_by, created_at, status FROM seats WHERE seat_id = %s",
        (seat_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_all(conn, status: str = "active") -> list[dict]:
    rows = conn.execute(
        "SELECT seat_id, specialty, created_by, created_at, status FROM seats "
        "WHERE status = %s ORDER BY created_at",
        (status,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def retire(conn, seat_id: str) -> None:
    conn.execute("UPDATE seats SET status = 'retired' WHERE seat_id = %s", (seat_id,))


def _row_to_dict(row) -> dict:
    return {
        "seat_id": row[0],
        "specialty": row[1],
        "created_by": row[2],
        "created_at": row[3],
        "status": row[4],
    }
