"""Per-project workspaces, and the isolation they exist to provide.

Before this, WORKSPACE_ROOT was both the harness's own store and the
directory every agent worked in, so product code landed beside the Beads
issue database -- an agent's `git status` on 2026-09-01 reported
`M .beads/interactions.jsonl` as if it were a product change -- and a
second project would have written into the first one's files.
"""

import os

import pytest

from harness import permissions, tools, workspaces
from harness.permissions import PermissionDenied


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(workspaces, "PROJECTS_ROOT", str(root))
    return root


# -- containment ------------------------------------------------------


def test_sibling_workspace_is_not_reachable():
    """The bug that mattered: a string-prefix check let /projects/proj-a
    reach ../proj-abc, which would have meant no isolation at all between
    sibling projects."""
    with pytest.raises(PermissionDenied):
        permissions.check_within_workspace("../proj-abc/x", "/projects/proj-a")


def test_prefix_lookalike_directory_is_not_reachable():
    with pytest.raises(PermissionDenied):
        permissions.check_within_workspace("../workspace-evil/x", "/workspace")


def test_paths_inside_the_workspace_are_allowed():
    permissions.check_within_workspace("src/sim/tick.ts", "/projects/proj-a")
    permissions.check_within_workspace(".", "/projects/proj-a")


def test_absolute_escape_is_rejected():
    with pytest.raises(PermissionDenied):
        permissions.check_within_workspace("/etc/passwd", "/projects/proj-a")


def test_harness_store_is_not_reachable_from_a_project_workspace():
    """The point of the whole epic: an agent must not be able to reach
    the Beads issue database."""
    with pytest.raises(PermissionDenied):
        permissions.check_within_workspace("../../workspace/.beads", "/projects/proj-a")


# -- workspace creation -----------------------------------------------


def test_ensure_creates_a_workspace_with_a_repo(projects_root):
    path = workspaces.ensure("workspace-abc")
    assert os.path.isdir(path)
    assert os.path.isdir(os.path.join(path, ".git")), "each project gets its own repo"


def test_ensure_is_idempotent(projects_root):
    first = workspaces.ensure("workspace-abc")
    second = workspaces.ensure("workspace-abc")
    assert first == second


def test_ticket_resolves_to_its_project_workspace(projects_root):
    path = workspaces.for_ticket("workspace-9jg.1.5")
    assert path.endswith("workspace-9jg"), "story maps to its project, not its epic"


def test_two_projects_get_separate_directories(projects_root):
    a = workspaces.ensure("proj-a")
    b = workspaces.ensure("proj-b")
    assert a != b
    open(os.path.join(a, "only-in-a.txt"), "w").write("x")
    assert not os.path.exists(os.path.join(b, "only-in-a.txt"))


# -- tools bound to a workspace ---------------------------------------


def test_tools_are_bound_to_the_given_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    bound["write_file"].invoke({"path": "a/b.txt", "content": "hello"})

    assert (root / "a" / "b.txt").read_text() == "hello"


def test_bound_tools_reject_an_escape(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    with pytest.raises(PermissionDenied):
        bound["read_file"].invoke({"path": "../outside.txt"})


def test_two_agents_get_independently_rooted_tools(tmp_path):
    """An agent in project A must not reach project B, even though both
    tool sets are built from the same factory."""
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    (b / "secret.txt").write_text("B only")

    tools_a = {t.name: t for t in tools.build_workspace_tools(str(a))}

    with pytest.raises(PermissionDenied):
        tools_a["read_file"].invoke({"path": "../b/secret.txt"})


def test_build_agent_tools_includes_shared_and_bound(tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    names = {t.name for t in tools.build_agent_tools(str(root))}
    assert {"read_file", "write_file", "shell_exec"} <= names
    assert {"complete_ticket", "refuse_ticket", "decline_ticket"} <= names


# -- diff, for verification -------------------------------------------


def test_diff_reports_new_files(projects_root):
    path = workspaces.ensure("proj-diff")
    open(os.path.join(path, "new.ts"), "w").write("export const x = 1;\n")

    diff = workspaces.diff_since("proj-diff", None)

    assert "new.ts" in diff, "a ticket that only creates files must not look empty"


def test_diff_is_empty_for_an_untouched_workspace(projects_root):
    workspaces.ensure("proj-empty")
    assert workspaces.diff_since("proj-empty", None).strip() == ""
