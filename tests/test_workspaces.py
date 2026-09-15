"""Per-project workspaces, and the isolation they exist to provide.

Before this, WORKSPACE_ROOT was both the harness's own store and the
directory every agent worked in, so product code landed beside the Beads
issue database -- an agent's `git status` on 2026-09-01 reported
`M .beads/interactions.jsonl` as if it were a product change -- and a
second project would have written into the first one's files.
"""

import os
import subprocess

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


# -- harness/tool infrastructure ---------------------------------------


def test_infrastructure_paths_are_detected():
    assert permissions.is_infrastructure(".claude/settings.json")
    assert permissions.is_infrastructure(".beads/interactions.jsonl")
    assert permissions.is_infrastructure("nested/.git/config")


def test_ordinary_project_paths_are_not_infrastructure():
    assert not permissions.is_infrastructure("src/sim/tick.ts")
    assert not permissions.is_infrastructure("AGENTS.md")
    assert not permissions.is_infrastructure(".gitignore")  # a file, not the .git dir


# -- workspace creation -----------------------------------------------


def test_ensure_creates_a_workspace_with_a_repo(projects_root):
    path = workspaces.ensure("workspace-abc")
    assert os.path.isdir(path)
    assert os.path.isdir(os.path.join(path, ".git")), "each project gets its own repo"


def test_ensure_is_idempotent(projects_root):
    first = workspaces.ensure("workspace-abc")
    second = workspaces.ensure("workspace-abc")
    assert first == second


def test_ticket_resolves_to_its_own_worktree(projects_root):
    """A ticket is NOT rooted at the project directory. Every seat working
    a project shared that one checkout, and `git add -A` at the end of a
    ticket then swept the other agents' live files into its commit -- the
    failure the worktrees exist to fix (2026-09-15: 31 tickets on the human
    queue, ~14 of them rejected for judging another ticket's work)."""
    path = workspaces.for_ticket("workspace-9jg.1.5")
    assert path != workspaces.path_for("workspace-9jg"), "not the shared project tree"
    assert path.endswith(
        os.path.join("workspace-9jg", workspaces.WORKTREES_DIRNAME, "workspace-9jg.1.5")
    )
    assert os.path.exists(os.path.join(path, ".git")), "a real git worktree"


def test_two_tickets_in_one_project_get_separate_trees(projects_root):
    a = workspaces.for_ticket("workspace-9jg.1.1")
    b = workspaces.for_ticket("workspace-9jg.1.2")

    assert a != b
    open(os.path.join(a, "only-in-a.ts"), "w").write("a")
    assert not os.path.exists(os.path.join(b, "only-in-a.ts"))


def test_reopening_a_ticket_reuses_its_tree(projects_root):
    """Requeueing must not throw the ticket's files away -- the verifier's
    rework path resets the graph thread, not the worktree."""
    first = workspaces.for_ticket("workspace-9jg.1.1")
    open(os.path.join(first, "partial.ts"), "w").write("half done")

    assert workspaces.for_ticket("workspace-9jg.1.1") == first
    assert os.path.exists(os.path.join(first, "partial.ts"))


def test_a_ticket_commit_holds_only_that_tickets_files(projects_root):
    """The regression this change exists for.

    Two seats work the same project at once. In a shared tree `git add -A`
    committed both sets of files under whichever ticket finished first, and
    the verifier -- which reads that commit's diff -- failed the ticket for
    "the diff under review is not this ticket's work at all"."""
    a = workspaces.for_ticket("workspace-9jg.1.1")
    b = workspaces.for_ticket("workspace-9jg.1.2")
    open(os.path.join(a, "a.ts"), "w").write("a")
    open(os.path.join(b, "b.ts"), "w").write("b")

    sha = workspaces.commit_all_for_ticket("workspace-9jg.1.1", "workspace-9jg.1.1: did a")

    assert sha
    diff = workspaces.commit_diff("workspace-9jg", sha)
    assert "a.ts" in diff
    assert "b.ts" not in diff, "another ticket's live work must not be in this commit"


def test_ticket_work_reaches_the_integration_checkout(projects_root):
    worktree = workspaces.for_ticket("workspace-9jg.1.1")
    open(os.path.join(worktree, "shipped.ts"), "w").write("x")
    workspaces.commit_all_for_ticket("workspace-9jg.1.1", "workspace-9jg.1.1: shipped")

    ok, reason = workspaces.merge_to_integration("workspace-9jg.1.1")

    assert ok, reason
    assert os.path.exists(os.path.join(workspaces.path_for("workspace-9jg"), "shipped.ts"))


def test_a_new_ticket_branches_from_already_merged_work(projects_root):
    """A ticket has to start on top of what has already landed, or every
    ticket after the first would be built against the empty scaffold."""
    a = workspaces.for_ticket("workspace-9jg.1.1")
    open(os.path.join(a, "first.ts"), "w").write("x")
    workspaces.commit_all_for_ticket("workspace-9jg.1.1", "workspace-9jg.1.1: first")
    workspaces.merge_to_integration("workspace-9jg.1.1")

    b = workspaces.for_ticket("workspace-9jg.1.2")

    assert os.path.exists(os.path.join(b, "first.ts"))


def test_the_worktree_directory_is_never_committed(projects_root):
    """`.worktrees/` holds a whole checkout per ticket, so `git add -A` in
    the integration tree would otherwise commit every ticket's tree."""
    workspaces.for_ticket("workspace-9jg.1.1")
    assert workspaces.diff_since("workspace-9jg", None).strip() == ""


def test_a_project_with_no_commits_yet_still_gets_a_worktree(projects_root):
    """The one case that could otherwise still share a tree: a repo with no
    commit has no ref to branch from. ensure() only commits .gitignore when
    it has something to add to it, so a project that already carries the
    block reaches this with an empty history."""
    repo = workspaces.path_for("workspace-bare")
    os.makedirs(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    with open(os.path.join(repo, ".gitignore"), "w") as fh:
        fh.write(
            "# Added by the harness: never commit dependencies or build output.\n"
            "node_modules/\n.worktrees/\n"
        )

    path = workspaces.for_ticket("workspace-bare.1.1")

    assert path != repo, "isolation must hold from the first ticket too"
    assert os.path.exists(os.path.join(path, ".git"))


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
    assert {"read_file", "write_file", "shell_exec", "list_directory"} <= names
    assert {"complete_ticket", "refuse_ticket", "decline_ticket"} <= names


# -- list_directory hides harness/tool infrastructure ------------------


def test_list_directory_hides_infrastructure_but_shows_project_content(tmp_path):
    """Regression for the scaffolding-ticket stall, 2026-09-04: an agent
    listed its workspace, saw .claude/settings.json sitting right there,
    tried to read it, got denied, and its next turn came back completely
    empty. Hiding the temptation beats hoping a denial recovers well."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".claude").mkdir()
    (root / ".claude" / "settings.json").write_text("{}")
    (root / ".git").mkdir()
    (root / "src").mkdir()
    (root / "README.md").write_text("hi")
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    result = bound["list_directory"].invoke({"path": "."})

    assert "README.md" in result
    assert "src/" in result
    assert ".claude" not in result
    assert ".git" not in result
    assert "2 entries hidden" in result


def test_list_directory_on_an_empty_project_says_so(tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    assert bound["list_directory"].invoke({"path": "."}) == "(empty)"


def test_list_directory_rejects_an_escape(tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    with pytest.raises(PermissionDenied):
        bound["list_directory"].invoke({"path": ".."})


def test_read_file_on_infrastructure_returns_a_message_not_the_real_content(tmp_path):
    """Defense in depth: read_file had no sensitivity check at all
    before this -- is_statically_safe treats any in-workspace read_file
    call as safe and skips the classifier entirely, so the same path
    that gets denied via shell_exec would have been served straight
    from disk via read_file. This must not raise either -- an agent
    poking at one of these paths is ordinary orientation, not an escape
    attempt (see test_bound_tools_reject_an_escape for that case)."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".claude").mkdir()
    (root / ".claude" / "settings.json").write_text("super-secret-token")
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    result = bound["read_file"].invoke({"path": ".claude/settings.json"})

    assert "super-secret-token" not in result
    assert "infrastructure" in result


def test_write_file_on_infrastructure_does_not_touch_the_real_file(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".beads").mkdir()
    (root / ".beads" / "metadata.json").write_text("original")
    bound = {t.name: t for t in tools.build_workspace_tools(str(root))}

    result = bound["write_file"].invoke({"path": ".beads/metadata.json", "content": "clobbered"})

    assert "infrastructure" in result
    assert (root / ".beads" / "metadata.json").read_text() == "original"


# -- diff, for verification -------------------------------------------


def test_diff_reports_new_files(projects_root):
    path = workspaces.ensure("proj-diff")
    open(os.path.join(path, "new.ts"), "w").write("export const x = 1;\n")

    diff = workspaces.diff_since("proj-diff", None)

    assert "new.ts" in diff, "a ticket that only creates files must not look empty"


def test_diff_is_empty_for_an_untouched_workspace(projects_root):
    workspaces.ensure("proj-empty")
    assert workspaces.diff_since("proj-empty", None).strip() == ""
