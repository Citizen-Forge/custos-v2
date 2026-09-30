"""External agent seats: a session outside Custos takes a ticket, pushes its
work, and submits it into the same verify-then-land path internal agents use.
Everything here runs against real bd, real Postgres and real git repos."""

import os
import subprocess

import psycopg
import pytest

from harness import beads, dispatcher, external, seats, workspaces

GIT = ["git", "-c", "user.name=ext", "-c", "user.email=ext@test"]


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(workspaces, "PROJECTS_ROOT", str(root))
    monkeypatch.setattr(external, "GIT_URL_TEMPLATE", str(root) + "/{project_id}")
    return root


@pytest.fixture
def conn():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as c:
        external.init_table(c)
        yield c


@pytest.fixture
def seat(conn):
    name = f"ext-{os.urandom(3).hex()}"
    token = external.register_seat(conn, name, "test session")
    external.invalidate_cache()
    return name, token


@pytest.fixture
def ticket():
    beads.ensure_initialized()
    project = beads.create("external proj", "d", issue_type="epic", priority=1)
    story = beads.create("external story", "write hello.txt", parent=project["id"])
    beads.assign_to_seat(story["id"], "internal-seat")
    yield project["id"], story["id"]
    try:
        beads.close(story["id"], reason="test cleanup", actor=None)
    except Exception:
        pass


def _clone_commit_push(brief, tmp_path, ref, filename="hello.txt"):
    work = tmp_path / f"clone-{os.urandom(3).hex()}"
    subprocess.run(["git", "clone", "-q", brief["git"]["clone_url"], str(work)], check=True)
    subprocess.run(["git", "checkout", "-q", brief["git"]["base_branch"]], cwd=work, check=True)
    (work / filename).write_text("hello\n")
    subprocess.run([*GIT, "add", "-A"], cwd=work, check=True)
    subprocess.run([*GIT, "commit", "-q", "-m", "external work"], cwd=work, check=True)
    return subprocess.run(["git", "push", "origin", f"HEAD:{ref}"], cwd=work, capture_output=True, text=True)


def test_registered_seat_is_external_and_out_of_the_active_roster(conn, seat):
    name, token = seat
    assert external.seat_for_token(conn, token) == name
    assert external.seat_for_token(conn, "wrong") is None
    assert name in external.seat_ids()
    assert name not in {s["seat_id"] for s in seats.list_all(conn)}


def test_an_internal_seat_cannot_be_turned_external(conn):
    seats.create(conn, "internal-only", "specialist", created_by="test")
    with pytest.raises(external.ExternalError):
        external.register_seat(conn, "internal-only", "hijack")


def test_request_claims_and_reseats_the_ticket(projects_root, seat, ticket):
    name, _ = seat
    project_id, story_id = ticket

    brief = external.request_ticket(name, project_id)

    assert brief["ticket"]["id"] == story_id
    current = beads.show(story_id)
    assert current["status"] == "in_progress"
    assert current["assignee"] == name
    assert beads.assigned_seat(current) == name
    assert current["metadata"][external.PREVIOUS_SEAT_KEY] == "internal-seat"
    assert external.lease_until(current) is not None
    assert brief["git"]["push_branch"] == f"external/{story_id}"

    # Asking again hands back the same ticket rather than a second one.
    assert external.request_ticket(name, project_id)["ticket"]["id"] == story_id


def test_dispatcher_does_not_treat_an_external_claim_as_an_orphan(projects_root, seat, ticket, monkeypatch):
    name, _ = seat
    project_id, story_id = ticket
    external.request_ticket(name, project_id)
    held = beads.show(story_id)
    monkeypatch.setattr(beads, "in_progress", lambda: [held])
    monkeypatch.setattr(beads, "ready", lambda: [])

    assert dispatcher.next_assigned_ticket()[0] is None


def test_pushes_are_confined_to_external_branches(projects_root, seat, ticket, tmp_path):
    name, _ = seat
    project_id, _ = ticket
    brief = external.request_ticket(name, project_id)

    # The checked-out integration branch is refused by git itself; any other
    # branch that is not external/* is refused by the hook.
    assert _clone_commit_push(brief, tmp_path, brief["git"]["base_branch"]).returncode != 0
    elsewhere = _clone_commit_push(brief, tmp_path, "feature/sneaky")
    assert elsewhere.returncode != 0
    assert "only accepted to refs/heads/external/" in elsewhere.stderr

    assert _clone_commit_push(brief, tmp_path, brief["git"]["push_branch"]).returncode == 0


def test_submit_without_a_push_is_refused(projects_root, conn, seat, ticket):
    name, _ = seat
    project_id, story_id = ticket
    external.request_ticket(name, project_id)
    with pytest.raises(external.ExternalError, match="nothing pushed"):
        external.submit_ticket(conn, name, story_id, "did it")


def test_submit_with_criteria_closes_and_queues_verification(projects_root, conn, seat, ticket, tmp_path):
    name, _ = seat
    project_id, story_id = ticket
    beads.set_acceptance_criteria(story_id, "hello.txt exists")
    brief = external.request_ticket(name, project_id)
    assert _clone_commit_push(brief, tmp_path, brief["git"]["push_branch"]).returncode == 0

    result = external.submit_ticket(conn, name, story_id, "added hello.txt; no tests apply")

    current = beads.show(story_id)
    assert current["status"] == "closed"
    assert current["metadata"]["completion_summary"].startswith("added hello.txt")
    base, tip = current["metadata"]["work_commit"].split("..")
    assert result["work_commit"] == f"{base}..{tip}"
    # The worktree the verifier judges now holds the pushed work.
    assert workspaces.head_for_ticket(story_id) == tip
    assert os.path.exists(os.path.join(workspaces.worktree_path_for(story_id), "hello.txt"))
    assert story_id in external.pending_verification(conn)
    # Not landed yet: that waits for the verifier's pass.
    assert not os.path.exists(os.path.join(workspaces.path_for(project_id), "hello.txt"))


def test_submit_without_criteria_lands_directly(projects_root, conn, seat, ticket, tmp_path):
    name, _ = seat
    project_id, story_id = ticket
    brief = external.request_ticket(name, project_id)
    assert _clone_commit_push(brief, tmp_path, brief["git"]["push_branch"]).returncode == 0

    result = external.submit_ticket(conn, name, story_id, "added hello.txt")

    assert result["state"].startswith("landed")
    assert beads.show(story_id)["status"] == "closed"
    assert os.path.exists(os.path.join(workspaces.path_for(project_id), "hello.txt"))


def test_only_the_holding_seat_can_submit(projects_root, conn, seat, ticket):
    name, _ = seat
    project_id, story_id = ticket
    external.request_ticket(name, project_id)
    other = external.register_seat(conn, f"ext-other-{os.urandom(2).hex()}", "other")
    other_name = external.seat_for_token(conn, other)
    with pytest.raises(external.ExternalError, match="not held by"):
        external.submit_ticket(conn, other_name, story_id, "not mine")


def test_release_returns_the_ticket_and_drops_the_pushed_branch(projects_root, seat, ticket, tmp_path):
    name, _ = seat
    project_id, story_id = ticket
    brief = external.request_ticket(name, project_id)
    assert _clone_commit_push(brief, tmp_path, brief["git"]["push_branch"]).returncode == 0

    external.release_ticket(name, story_id, "changed my mind")

    current = beads.show(story_id)
    assert current["status"] == "open"
    assert not current.get("assignee")
    assert beads.assigned_seat(current) == "internal-seat"
    repo = workspaces.path_for(project_id)
    assert external._branch_tip(repo, f"external/{story_id}") is None


def test_http_flow_needs_the_seat_token(projects_root, ticket):
    from fastapi.testclient import TestClient

    from harness.api import app

    client = TestClient(app)
    project_id, story_id = ticket
    registered = client.post("/external/seats", json={"seat_id": f"ext-http-{os.urandom(2).hex()}",
                                                      "description": "http test"})
    assert registered.status_code == 200
    token = registered.json()["token"]

    assert client.post("/external/tickets/request", json={"project_id": project_id}).status_code == 401
    assert client.post("/external/tickets/request", json={"project_id": project_id},
                       headers={"X-Custos-Seat-Token": "nope"}).status_code == 401

    headers = {"X-Custos-Seat-Token": token}
    got = client.post("/external/tickets/request", json={"project_id": project_id}, headers=headers)
    assert got.status_code == 200
    assert got.json()["ticket"]["id"] == story_id
    assert client.post(f"/external/tickets/{story_id}/heartbeat", headers=headers).status_code == 200
    # Submitting with nothing pushed is the caller's mistake: a 409, not a 500.
    refused = client.post(f"/external/tickets/{story_id}/submit", json={"summary": "x"}, headers=headers)
    assert refused.status_code == 409
    assert "nothing pushed" in refused.json()["detail"]


def test_an_expired_lease_is_swept_back_to_the_pool(projects_root, seat, ticket):
    name, _ = seat
    project_id, story_id = ticket
    external.request_ticket(name, project_id)

    released = external.sweep_expired(now=external.lease_until(beads.show(story_id)) + 1)

    assert story_id in released
    current = beads.show(story_id)
    assert current["status"] == "open"
    assert beads.assigned_seat(current) == "internal-seat"
