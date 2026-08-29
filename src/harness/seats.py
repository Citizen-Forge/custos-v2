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

`display_name`/`pronouns` (added Phase 4, 2026-08-29) are deliberately
separate from `seat_id`: PLAN.md's welfare-essay-derived design goal is
an agent that chooses its own identity, not just a functional label a
human/product-owner assigned it (`seat_id` stays lowercase-hyphenated and
functional -- e.g. "workspace-implement-verify" -- since routing/prompts/
outcomes all key off it and changing that shape would ripple everywhere;
`display_name` is the actual chosen name, e.g. "Sable"). Both nullable:
a seat created before this existed, or by any path that doesn't ask, has
no chosen identity yet rather than a fabricated placeholder one.
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
    # ALTER ... ADD COLUMN IF NOT EXISTS rather than folding into the
    # CREATE TABLE above -- this table already exists in the live
    # database, and CREATE TABLE IF NOT EXISTS is a no-op against an
    # existing table regardless of column differences. This is the
    # standard safe way to evolve a live Postgres schema without a
    # migration framework: idempotent, doesn't touch existing rows'
    # other columns, and existing rows just get NULL for the new ones.
    conn.execute("ALTER TABLE seats ADD COLUMN IF NOT EXISTS display_name TEXT")
    conn.execute("ALTER TABLE seats ADD COLUMN IF NOT EXISTS pronouns TEXT")


def create(conn, seat_id: str, specialty: str, created_by: str, display_name: str | None = None, pronouns: str | None = None) -> None:
    conn.execute(
        "INSERT INTO seats (seat_id, specialty, created_by, display_name, pronouns) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (seat_id) DO NOTHING",
        (seat_id, specialty, created_by, display_name, pronouns),
    )


def get(conn, seat_id: str) -> dict | None:
    row = conn.execute(
        "SELECT seat_id, specialty, created_by, created_at, status, display_name, pronouns FROM seats WHERE seat_id = %s",
        (seat_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_all(conn, status: str = "active") -> list[dict]:
    rows = conn.execute(
        "SELECT seat_id, specialty, created_by, created_at, status, display_name, pronouns FROM seats "
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
        "display_name": row[5],
        "pronouns": row[6],
    }
