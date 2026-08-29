"""
System-wide settings -- currently just the cost slider (user's own
architecture call, 2026-08-29): one value from "slow/free" (0) to
"fast/costly" (100) that steers which provider tier the product-owner
should reach for, replacing the idea of a human pre-deciding one fixed
ordered fallback chain per role. Deliberately a plain key-value table,
not a dataclass of named fields -- cheap to add more settings later
without a schema migration each time.
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def get(conn, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (key,)).fetchone()
    return row[0] if row else default


def set(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (key, value),
    )


COST_SLIDER_KEY = "cost_slider"
DEFAULT_COST_SLIDER = 0  # slow/free by default -- matches "only local configured" reality


def get_cost_slider(conn) -> int:
    raw = get(conn, COST_SLIDER_KEY)
    return int(raw) if raw is not None else DEFAULT_COST_SLIDER


def set_cost_slider(conn, value: int) -> None:
    if not (0 <= value <= 100):
        raise ValueError(f"cost slider must be 0-100, got {value}")
    set(conn, COST_SLIDER_KEY, str(value))
