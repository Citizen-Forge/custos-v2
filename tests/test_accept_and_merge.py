"""Accepting a ticket's work must LAND it, not only close it.

A plain respond closes a ticket and records the decision, but landing only
ever followed a verifier pass, and the verifier defers to a human decision --
so workspace-o0n.16.1, accepted that way, sat unmerged on its branch."""

import os

import pytest

from harness import beads, verifier, workspaces


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(workspaces, "PROJECTS_ROOT", str(root))
    return root


def _ticket_with_work(title: str, filename: str = "accepted.gd", body: str = "accepted\n"):
    beads.ensure_initialized()
    project = beads.create(f"{title} proj", "d", issue_type="epic", priority=1)
    story = beads.create(title, "d", parent=project["id"])
    tree = workspaces.for_ticket(story["id"])
    with open(os.path.join(tree, filename), "w") as fh:
        fh.write(body)
    workspaces.commit_all_for_ticket(story["id"], f"{story['id']}: work")
    beads.claim(story["id"], actor="seat-accept")
    return project["id"], story["id"]


def _in_integration(project_id: str, filename: str) -> bool:
    return os.path.exists(os.path.join(workspaces.path_for(project_id), filename))


def test_accepting_a_flagged_ticket_closes_and_merges_it(projects_root):
    project_id, ticket_id = _ticket_with_work("accept flagged")
    beads.flag_for_human(ticket_id, "needs a person")
    assert not _in_integration(project_id, "accepted.gd")

    result = verifier.accept_and_merge(ticket_id, "good work, ship it")

    assert result == {"merged": True, "reason": ""}
    assert _in_integration(project_id, "accepted.gd"), "the work is on the integration branch"
    current = beads.show(ticket_id)
    assert current["status"] == "closed"
    assert not beads.is_flagged_for_human(current)
    assert beads.human_decision(current) == "answered"


def test_accepting_an_already_closed_ticket_merges_without_reopening(projects_root):
    """The 16.1 case: closed by a respond, never merged."""
    project_id, ticket_id = _ticket_with_work("accept closed", filename="late.gd")
    beads.flag_for_human(ticket_id, "needs a person")
    beads.respond_to_human(ticket_id, "accepted, earlier")
    assert not _in_integration(project_id, "late.gd")

    result = verifier.accept_and_merge(ticket_id, "and now merge it")

    assert result["merged"] is True
    assert _in_integration(project_id, "late.gd")
    assert beads.show(ticket_id)["status"] == "closed"


def test_no_branch_is_reported_not_claimed(projects_root):
    beads.ensure_initialized()
    project = beads.create("accept nobranch proj", "d", issue_type="epic", priority=1)
    story = beads.create("accept nobranch", "d", parent=project["id"])
    workspaces.ensure(project["id"])
    beads.claim(story["id"], actor="seat-accept")

    result = verifier.accept_and_merge(story["id"], "fine")

    assert result["merged"] is False
    assert "no branch" in result["reason"]


def test_a_conflict_goes_back_to_a_person_not_to_an_agent(projects_root):
    project_id, a = _ticket_with_work("accept conflict a", filename="shared.gd", body="a\n")
    # A second ticket branched from the same tip changes the same file.
    story_b = beads.create("accept conflict b", "d", parent=project_id)
    tree_b = workspaces.for_ticket(story_b["id"])
    with open(os.path.join(tree_b, "shared.gd"), "w") as fh:
        fh.write("b\n")
    workspaces.commit_all_for_ticket(story_b["id"], f"{story_b['id']}: b")
    beads.claim(story_b["id"], actor="seat-accept")
    assert verifier.accept_and_merge(a, "first")["merged"] is True

    result = verifier.accept_and_merge(story_b["id"], "second")

    assert result["merged"] is False
    current = beads.show(story_b["id"])
    assert beads.is_flagged_for_human(current), "flagged for a person"
    assert current["status"] == "closed", "not reopened for an agent to redo"


def test_the_api_accepts_and_merges(projects_root):
    from fastapi.testclient import TestClient

    from harness.api import app

    project_id, ticket_id = _ticket_with_work("accept api", filename="via_api.gd")
    beads.flag_for_human(ticket_id, "needs a person")

    response = TestClient(app).post(f"/tickets/{ticket_id}/accept", json={"response": "ok"})

    assert response.status_code == 200
    assert response.json() == {"merged": True, "reason": ""}
    assert _in_integration(project_id, "via_api.gd")


def _db():
    import psycopg

    from harness import verifications

    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    verifications.init_table(conn)
    return conn


def test_accepting_rejected_work_merges_the_judged_commit(projects_root):
    """The case accept exists for: the verifier failed the work, a person
    overrides it. Picking the ticket up for rework put its branch back on the
    integration tip, so the branch is EMPTY -- found live 2026-10-03, when
    workspace-o0n.1.5 "merged" that empty branch and reported merged=true.
    The work is the commit the verdict was recorded against."""
    from harness import verifications

    project_id, ticket_id = _ticket_with_work("accept rejected", filename="rejected.gd")
    repo = workspaces.path_for(project_id)
    judged = workspaces._git(["rev-parse", workspaces.ticket_branch(ticket_id)], repo).stdout.strip()
    conn = _db()
    verifications.record(conn, ticket_id, "seat-accept", "fail", "wrongly failed", work_commit=judged)
    workspaces.reset_for_attempt(ticket_id)
    assert workspaces.commits_ahead(project_id, workspaces.ticket_branch(ticket_id)) == 0

    result = verifier.accept_and_merge(ticket_id, "the verifier was wrong", conn=conn)
    conn.close()

    assert result == {"merged": True, "reason": ""}
    assert _in_integration(project_id, "rejected.gd"), "the judged work landed"
    assert judged[:12] in (beads.show(ticket_id).get("notes") or ""), "and the note says which commit"


def test_nothing_to_merge_is_not_reported_as_merged(projects_root):
    beads.ensure_initialized()
    project = beads.create("accept empty proj", "d", issue_type="epic", priority=1)
    story = beads.create("accept empty", "d", parent=project["id"])
    workspaces.ensure(project["id"])
    workspaces.for_ticket(story["id"])  # a branch, but no work on it
    beads.claim(story["id"], actor="seat-accept")
    conn = _db()

    result = verifier.accept_and_merge(story["id"], "fine", conn=conn)
    conn.close()

    assert result["merged"] is False
    assert "nothing to merge" in result["reason"]
