"""The verifier judges the code, not the agent's description of it.

verifier.py previously formed its verdict from the ticket's close reason
and notes -- an account of the work rather than the work. Each ticket's
output is now committed on its behalf when it claims completion, so there
is a diff attributable to exactly that ticket.
"""

import os

import pytest

from harness import beads, verifier, workspaces


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(workspaces, "PROJECTS_ROOT", str(root))
    return root


class FakeModel:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class R:
            pass

        r = R()
        r.content = self.content
        return r


def test_commit_all_records_a_ticket_diff(projects_root):
    path = workspaces.ensure("proj-x")
    open(os.path.join(path, "tick.ts"), "w").write("export const tick = () => 1;\n")

    sha = workspaces.commit_all("proj-x", "proj-x.1.1: added tick")

    assert sha, "a ticket that changed files must produce a commit"
    diff = workspaces.commit_diff("proj-x", sha)
    assert "tick.ts" in diff and "export const tick" in diff


def test_commit_all_returns_none_when_nothing_changed(projects_root):
    workspaces.ensure("proj-y")
    assert workspaces.commit_all("proj-y", "nothing") is None


def test_second_commit_diff_contains_only_the_second_ticket(projects_root):
    """Attribution is the point: ticket two's diff must not include
    ticket one's work."""
    path = workspaces.ensure("proj-z")
    open(os.path.join(path, "first.ts"), "w").write("// one\n")
    workspaces.commit_all("proj-z", "first")
    open(os.path.join(path, "second.ts"), "w").write("// two\n")
    sha = workspaces.commit_all("proj-z", "second")

    diff = workspaces.commit_diff("proj-z", sha)

    assert "second.ts" in diff
    assert "first.ts" not in diff


def test_diff_for_a_ticket_with_no_commit_is_empty():
    assert verifier._diff_for({"id": "x.1.1", "metadata": {}}) == ""


def test_verifier_prompt_carries_the_diff(projects_root, monkeypatch):
    beads.ensure_initialized()
    project = beads.create("vdiff proj", "d", issue_type="epic", priority=1)
    story = beads.create("vdiff story", "d", parent=project["id"],
                         acceptance_criteria="a tick function exists")
    path = workspaces.ensure(project["id"])
    open(os.path.join(path, "tick.ts"), "w").write("export const tick = () => 1;\n")
    sha = workspaces.commit_all(project["id"], "work")
    beads.set_metadata(story["id"], "work_commit", sha)
    beads.close(story["id"], reason="did the thing")

    model = FakeModel('{"verdict": "pass", "reasoning": "diff implements it"}')

    class FakeConn:
        pass

    monkeypatch.setattr(verifier.verifications, "get_for_issue", lambda c, i: None)
    monkeypatch.setattr(verifier.verifications, "record", lambda *a, **k: {"verdict": "pass"})

    verifier.verify_ticket(FakeConn(), story["id"], model)

    assert model.prompts, "the model should have been asked"
    assert "tick.ts" in model.prompts[0], "the diff must reach the verifier's prompt"
