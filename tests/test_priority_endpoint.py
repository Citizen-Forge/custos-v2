"""PATCH /issues/{id}/priority and priority-at-creation.

Before this (2026-08-31) POST /projects was the only route in the whole
API that accepted a priority, and nothing could change one afterwards --
so epics all landed at bd's default and a backlog built through the API
could never be ordered through it. Ordering the harness's own
improvement epics required `docker exec ... bd update --priority`.
"""

from fastapi.testclient import TestClient

from harness import beads
from harness.api import app

client = TestClient(app)


def test_set_priority_changes_it():
    beads.ensure_initialized()
    issue = beads.create("priority probe", "x")

    response = client.patch(f"/issues/{issue['id']}/priority", json={"priority": 0})

    assert response.status_code == 200
    assert beads.show(issue["id"])["priority"] == 0


def test_set_priority_rejects_out_of_range():
    beads.ensure_initialized()
    issue = beads.create("priority range probe", "x")

    response = client.patch(f"/issues/{issue['id']}/priority", json={"priority": 9})

    assert response.status_code == 400
    assert "0-4" in response.json()["detail"]


def test_epic_and_story_accept_priority_at_creation():
    beads.ensure_initialized()
    project = client.post(
        "/projects", json={"name": "prio proj", "description": "d", "priority": 1}
    ).json()

    epic = client.post(
        f"/projects/{project['id']}/epics",
        json={"title": "prio epic", "description": "d", "priority": 0},
    ).json()
    story = client.post(
        f"/epics/{epic['id']}/stories",
        json={"title": "prio story", "description": "d", "priority": 3},
    ).json()

    assert beads.show(epic["id"])["priority"] == 0
    assert beads.show(story["id"])["priority"] == 3


def test_priority_is_optional_for_backwards_compatibility():
    """Existing callers (mcp-server, product_owner) omit it entirely."""
    beads.ensure_initialized()
    project = client.post(
        "/projects", json={"name": "optional prio proj", "description": "d", "priority": 2}
    ).json()

    response = client.post(
        f"/projects/{project['id']}/epics", json={"title": "no prio", "description": "d"}
    )

    assert response.status_code == 200


def test_projects_reports_snapshot_age_header():
    """The tree is served from a cache, so callers need a way to tell how
    stale it is without the body shape changing."""
    response = client.get("/projects")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert float(response.headers["X-Tree-Age-Seconds"]) >= 0


def test_writes_invalidate_the_cached_tree():
    """A create followed by a read must show the write immediately --
    otherwise the board looks like the write silently failed."""
    beads.ensure_initialized()
    client.get("/projects")  # prime the cache

    project = client.post(
        "/projects", json={"name": "cache bust proj", "description": "d", "priority": 2}
    ).json()

    ids = {p["id"] for p in client.get("/projects").json()}
    assert project["id"] in ids
