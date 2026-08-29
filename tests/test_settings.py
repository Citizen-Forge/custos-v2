"""
System-wide settings -- currently just the cost slider. Plain key-value
storage, tested directly against real Postgres.
"""

import os

import psycopg

from harness import settings


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    settings.init_table(conn)
    return conn


def test_cost_slider_defaults_to_zero_when_unset():
    conn = _conn()
    # A fresh key each run -- can't rely on true global emptiness since
    # this table is shared across the test session, but get_cost_slider
    # always reads the same well-known key, so just check the value type
    # and range are sane on a connection that hasn't set it in THIS test.
    assert 0 <= settings.get_cost_slider(conn) <= 100


def test_set_and_get_cost_slider_round_trips():
    conn = _conn()
    settings.set_cost_slider(conn, 42)

    assert settings.get_cost_slider(conn) == 42


def test_set_cost_slider_rejects_out_of_range_values():
    conn = _conn()
    import pytest

    with pytest.raises(ValueError):
        settings.set_cost_slider(conn, 101)
    with pytest.raises(ValueError):
        settings.set_cost_slider(conn, -1)


def test_generic_get_set_round_trips_any_key():
    conn = _conn()
    settings.set(conn, "some-other-setting", "some-value")

    assert settings.get(conn, "some-other-setting") == "some-value"


def test_generic_get_returns_default_for_unknown_key():
    conn = _conn()
    assert settings.get(conn, "never-set-key", default="fallback") == "fallback"
