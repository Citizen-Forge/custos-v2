"""
API layer over the harness's existing state. Uses FastAPI's TestClient
(in-process, no running server needed) against the real Postgres/Beads
services -- same integration style as the rest of this suite.
"""

import os
import uuid

import psycopg
from fastapi.testclient import TestClient

from harness import beads, prompts
from harness.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_list_ready_tickets_reflects_real_beads_state():
    beads.ensure_initialized()
    issue = beads.create("api test ready ticket", "x")

    ids = {t["id"] for t in client.get("/tickets", params={"status": "ready"}).json()}
    assert issue["id"] in ids


def test_list_human_tickets_only_shows_flagged_ones():
    beads.ensure_initialized()
    normal = beads.create("api test normal", "x")
    beads.claim(normal["id"])

    flagged = beads.create("api test flagged", "x")
    beads.claim(flagged["id"])
    beads.flag_for_human(flagged["id"], "needs review")

    ids = {t["id"] for t in client.get("/tickets", params={"status": "human"}).json()}
    assert flagged["id"] in ids
    assert normal["id"] not in ids

    # in_progress listing should exclude the flagged one -- same filter
    # worker.py's own orphan-resume relies on, exercised here via the API.
    in_progress_ids = {t["id"] for t in client.get("/tickets", params={"status": "in_progress"}).json()}
    assert normal["id"] in in_progress_ids
    assert flagged["id"] not in in_progress_ids


def test_get_ticket_by_id():
    beads.ensure_initialized()
    issue = beads.create("api test show", "x")

    response = client.get(f"/tickets/{issue['id']}")
    assert response.json()["id"] == issue["id"]


def test_get_unknown_ticket_returns_404():
    assert client.get("/tickets/does-not-exist-xyz").status_code == 404


def test_list_tickets_rejects_unknown_status():
    assert client.get("/tickets", params={"status": "bogus"}).status_code == 400


def test_respond_endpoint_closes_the_ticket():
    beads.ensure_initialized()
    issue = beads.create("api respond test", "x")
    beads.claim(issue["id"])
    beads.flag_for_human(issue["id"], "needs a call")

    response = client.post(f"/tickets/{issue['id']}/respond", json={"response": "proceed"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "closed"
    assert "proceed" in data["notes"]


def test_dismiss_endpoint_closes_the_ticket():
    beads.ensure_initialized()
    issue = beads.create("api dismiss test", "x")
    beads.claim(issue["id"])
    beads.flag_for_human(issue["id"], "needs a call")

    response = client.post(f"/tickets/{issue['id']}/dismiss", json={"reason": "not needed"})

    assert response.status_code == 200
    assert response.json()["status"] == "closed"


def test_approve_prompt_endpoint_activates_it():
    role = f"api-test-role-{uuid.uuid4().hex[:8]}"
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    version = prompts.propose(conn, role, "test prompt text")

    response = client.post(f"/prompts/{role}/{version}/approve")

    assert response.status_code == 200
    assert prompts.get_active(conn, role) == "test prompt text"


def test_pending_prompts_endpoint_filters_by_role():
    role = f"api-test-role-{uuid.uuid4().hex[:8]}"
    other_role = f"api-test-role-{uuid.uuid4().hex[:8]}"
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    prompts.propose(conn, role, "mine")
    prompts.propose(conn, other_role, "not mine")

    response = client.get("/prompts/pending", params={"role": role})

    texts = {p["text"] for p in response.json()}
    assert "mine" in texts
    assert "not mine" not in texts


def test_seats_endpoint_includes_outcomes():
    import uuid

    from harness import seats as seats_module

    seat_id = f"api-test-seat-{uuid.uuid4().hex[:8]}"
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    seats_module.init_table(conn)
    seats_module.create(conn, seat_id, "test specialty", created_by="test")

    roster = client.get("/seats").json()

    entry = next(s for s in roster if s["seat_id"] == seat_id)
    assert entry["specialty"] == "test specialty"
    assert "closed" in entry["outcomes"]


def test_outcomes_endpoint():
    beads.ensure_initialized()
    actor = f"api-test-actor-{uuid.uuid4().hex[:8]}"
    issue = beads.create("api outcomes test", "x")
    beads.claim(issue["id"], actor=actor)
    beads.close(issue["id"])

    data = client.get(f"/outcomes/{actor}").json()

    assert data["total"] == 1
    assert data["closed"] == 1
