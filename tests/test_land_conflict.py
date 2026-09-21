"""A ticket that passed review but whose merge collides with work landed in
the meantime must be requeued, not stranded behind a human.

Every ticket forks a worktree from the integration tip when it starts, so
two independent tickets that touch the same file both land a change; the
second merge conflicts. Before this, `land` parked the second ticket for a
person and left its approved work on its own branch -- with concurrent
agents that would strand work routinely."""

import os

import pytest

from harness import beads, verifier, workspaces


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(workspaces, "PROJECTS_ROOT", str(root))
    return root


def test_land_requeues_a_conflicting_merge(projects_root):
    beads.ensure_initialized()
    project = beads.create("conflict land proj", "d", issue_type="epic")
    a = beads.create("conflict land a", "d", parent=project["id"])
    b = beads.create("conflict land b", "d", parent=project["id"])

    # Both branch from the same tip and change the SAME file.
    wa = workspaces.for_ticket(a["id"])
    wb = workspaces.for_ticket(b["id"])
    with open(os.path.join(wa, "shared.gd"), "w") as fh:
        fh.write("a\n")
    with open(os.path.join(wb, "shared.gd"), "w") as fh:
        fh.write("b\n")
    workspaces.commit_all_for_ticket(a["id"], f"{a['id']}: a")
    workspaces.commit_all_for_ticket(b["id"], f"{b['id']}: b")
    beads.claim(a["id"], actor="seat-conflict")
    beads.claim(b["id"], actor="seat-conflict")

    try:
        assert verifier.land(a["id"]) is True, "the first ticket lands"

        assert verifier.land(b["id"]) is False, "the second conflicts"

        current = beads.show(b["id"])
        assert current["status"] == "open", "requeued for rework, not left closed"
        assert beads.is_flagged_for_human(current) is False, "not parked for a human"
        assert int((current.get("metadata") or {}).get("rework_count") or 0) == 1
    finally:
        for tid in (a["id"], b["id"]):
            try:
                beads.close(tid, reason="test cleanup")
            except Exception:
                pass


def test_merge_discards_stray_edits_in_the_integration_checkout(projects_root):
    """Strays an agent wrote into the project root (the integration checkout)
    made `git merge` refuse -- "local changes would be overwritten" -- which
    parked workspace-o0n.7.3 for a human. The merge now discards them first."""
    a = workspaces.for_ticket("workspace-9jg.1.1")
    with open(os.path.join(a, "shared.gd"), "w") as fh:
        fh.write("committed\n")
    workspaces.commit_all_for_ticket("workspace-9jg.1.1", "workspace-9jg.1.1: a")

    repo = workspaces.path_for("workspace-9jg")
    stray = os.path.join(repo, "stray.gd")
    with open(stray, "w") as fh:
        fh.write("stray\n")

    ok, reason = workspaces.merge_to_integration("workspace-9jg.1.1")

    assert ok, reason
    assert not os.path.exists(stray), "the stray must be discarded"
