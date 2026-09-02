"""Self-modification stops for an explicit human yes/no.

Until 2026-09-01 the reviewing agent's own "allow" verdict approved a
change to the harness's source, and deployment followed. At the user's
request a favourable review now parks the proposal at `awaiting_human`
instead: the verdict and the sandbox evidence are what a person reads,
not the decision itself.
"""

import pytest

from harness import self_mod


@pytest.fixture
def conn():
    import os

    import psycopg

    c = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    self_mod.init_table(c)
    yield c
    c.close()


def _sandboxed(conn, clean=True):
    pid = self_mod.propose(conn, "widen the tick loop", "diff --git a/x b/x", "self_modifier")
    self_mod.record_sandbox_result(
        conn, pid, 0 if clean else 1, "out", "err", 10 if clean else 8, 0 if clean else 2
    )
    return pid


def test_a_favourable_review_awaits_a_human(conn):
    pid = _sandboxed(conn)
    self_mod.record_review(conn, pid, "allow", "looks correct")
    self_mod.await_human(conn, pid)

    assert self_mod.get(conn, pid)["status"] == "awaiting_human"


def test_awaiting_human_is_not_approved(conn):
    """The distinction that matters: parked is not deployable."""
    pid = _sandboxed(conn)
    self_mod.await_human(conn, pid)

    assert self_mod.get(conn, pid)["status"] != "approved"
    assert pid not in {p["id"] for p in self_mod.list_by_status(conn, "approved")}


def test_a_human_yes_makes_it_deployable(conn):
    pid = _sandboxed(conn)
    self_mod.await_human(conn, pid)

    self_mod.approve(conn, pid)

    assert self_mod.get(conn, pid)["status"] == "approved"


def test_a_human_no_rejects_it(conn):
    pid = _sandboxed(conn)
    self_mod.await_human(conn, pid)

    self_mod.reject(conn, pid, "not worth the risk")

    assert self_mod.get(conn, pid)["status"] == "rejected"


def test_awaiting_human_queue_is_listable(conn):
    """The dashboard's decision queue reads exactly this."""
    pid = _sandboxed(conn)
    self_mod.await_human(conn, pid)

    assert pid in {p["id"] for p in self_mod.list_by_status(conn, "awaiting_human")}


def test_proposal_carries_its_ticket(conn):
    pid = self_mod.propose(conn, "d", "diff", "self_modifier", ticket_id="workspace-r2w.11.1")
    assert self_mod.get(conn, pid)["ticket_id"] == "workspace-r2w.11.1"
    assert pid in {p["id"] for p in self_mod.for_ticket(conn, "workspace-r2w.11.1")}


def test_deployed_since_counts_only_recent(conn):
    """The rate limit reads this; a fresh proposal has not deployed."""
    pid = _sandboxed(conn)
    before = self_mod.deployed_since(conn, 24 * 3600)
    self_mod.approve(conn, pid)
    self_mod.mark_deployed(conn, pid)

    assert self_mod.deployed_since(conn, 24 * 3600) == before + 1
