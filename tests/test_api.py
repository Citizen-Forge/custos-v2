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
from harness import tool_proposals as tp
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
    assert entry["verification"]["verified_total"] == 0  # nothing verified yet, not an error


def test_tool_proposals_endpoint_filters_by_status_and_approve_works():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tp.init_table(conn)
    tool_name = f"api-test-tool-{uuid.uuid4().hex[:8]}"
    proposal_id = tp.propose(conn, tool_name, "code", "no network", proposed_by="overwatch")
    tp.record_review(conn, proposal_id, "allow", "looks fine")

    reviewed = client.get("/tool-proposals", params={"status": "reviewed"}).json()
    assert any(p["id"] == proposal_id for p in reviewed)

    response = client.post(f"/tool-proposals/{proposal_id}/approve")
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert tp.get(conn, proposal_id)["approved_at"] is not None


def test_tool_proposals_reject_endpoint():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tp.init_table(conn)
    tool_name = f"api-test-tool-{uuid.uuid4().hex[:8]}"
    proposal_id = tp.propose(conn, tool_name, "os.system('rm -rf /')", "claims none", proposed_by="overwatch")

    response = client.post(f"/tool-proposals/{proposal_id}/reject", json={"reason": "dangerous"})

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_outcomes_endpoint():
    beads.ensure_initialized()
    actor = f"api-test-actor-{uuid.uuid4().hex[:8]}"
    issue = beads.create("api outcomes test", "x")
    beads.claim(issue["id"], actor=actor)
    beads.close(issue["id"])

    data = client.get(f"/outcomes/{actor}").json()

    assert data["total"] == 1
    assert data["closed"] == 1


def test_projects_endpoint_returns_the_full_tree():
    beads.ensure_initialized()
    project = beads.create("api-test project", "goal", issue_type="epic", priority=1)
    epic = beads.create("api-test epic", "epic goal", issue_type="epic", parent=project["id"])
    beads.create("api-test story", "story goal", parent=epic["id"])

    tree = client.get("/projects").json()

    found = next(p for p in tree if p["id"] == project["id"])
    assert found["priority"] == 1
    found_epic = next(e for e in found["epics"] if e["id"] == epic["id"])
    assert len(found_epic["stories"]) == 1
    assert found_epic["stories"][0]["title"] == "api-test story"


def test_avatar_endpoint_404s_when_no_generated_avatar_exists():
    response = client.get("/avatars/never-generated-seat")
    assert response.status_code == 404


def test_avatar_endpoint_serves_a_real_generated_file(monkeypatch, tmp_path):
    from harness import avatar as avatar_module

    monkeypatch.setattr(avatar_module, "WORKSPACE_ROOT", str(tmp_path))
    avatar_dir = tmp_path / "avatars"
    avatar_dir.mkdir()
    (avatar_dir / "some-seat.png").write_bytes(b"fake png bytes")

    response = client.get("/avatars/some-seat")

    assert response.status_code == 200
    assert response.content == b"fake png bytes"
    assert response.headers["content-type"] == "image/png"


def test_wiki_list_and_get_endpoints():
    from harness import wiki

    slug = f"api-test-page-{uuid.uuid4().hex[:8]}"
    wiki.write_page(slug, "# real content")

    listing = client.get("/wiki").json()
    assert slug in listing["pages"]

    page = client.get(f"/wiki/{slug}").json()
    assert page["content"] == "# real content"


def test_wiki_get_endpoint_404s_for_missing_page():
    response = client.get("/wiki/never-written-anywhere")
    assert response.status_code == 404


def test_cost_slider_get_and_put_round_trip():
    response = client.put("/settings/cost-slider", json={"value": 55})
    assert response.status_code == 200
    assert response.json()["value"] == 55

    response = client.get("/settings/cost-slider")
    assert response.json()["value"] == 55


def test_cost_slider_rejects_out_of_range_value():
    response = client.put("/settings/cost-slider", json={"value": 200})
    assert response.status_code == 422


def test_avatar_style_get_and_put_round_trip():
    response = client.put("/settings/avatar-style", json={"value": "bottts"})
    assert response.status_code == 200
    assert response.json()["value"] == "bottts"

    response = client.get("/settings/avatar-style")
    assert response.json()["value"] == "bottts"


def test_avatar_style_rejects_empty_value():
    response = client.put("/settings/avatar-style", json={"value": "  "})
    assert response.status_code == 422


def test_model_registry_endpoint_lists_configured_providers(monkeypatch):
    monkeypatch.delenv("MODEL_REGISTRY", raising=False)
    monkeypatch.setenv("LOCAL_MODEL_NAME", "test-model-for-api")

    response = client.get("/settings/model-registry")

    assert response.status_code == 200
    assert any(p["model"] == "test-model-for-api" for p in response.json())


def test_health_is_reachable_without_a_token_even_when_auth_is_on(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-test-token")
    assert client.get("/health").json() == {"status": "ok"}


def test_protected_endpoint_rejects_missing_token_when_auth_is_on(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-test-token")
    response = client.get("/tickets", params={"status": "ready"})
    assert response.status_code == 401


def test_protected_endpoint_rejects_wrong_token_when_auth_is_on(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-test-token")
    response = client.get(
        "/tickets", params={"status": "ready"}, headers={"Authorization": "Bearer wrong-token"}
    )
    assert response.status_code == 401


def test_protected_endpoint_accepts_correct_token_when_auth_is_on(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret-test-token")
    response = client.get(
        "/tickets", params={"status": "ready"}, headers={"Authorization": "Bearer secret-test-token"}
    )
    assert response.status_code == 200


def test_protected_endpoint_has_no_auth_requirement_when_token_unset(monkeypatch):
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    response = client.get("/tickets", params={"status": "ready"})
    assert response.status_code == 200
