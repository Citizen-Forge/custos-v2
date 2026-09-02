"""
One workspace per project.

Before this, WORKSPACE_ROOT (/workspace) was both the harness's own store
and the directory every agent worked in. That meant product code landed
beside the Beads issue database, and a second project would have written
straight into the first one's files. It showed: an agent's `git status`
on 2026-09-01 reported `M .beads/interactions.jsonl` -- the issue
database appearing as an uncommitted product change.

Now each project gets PROJECTS_ROOT/<project_id>, with its own git
repository, and that directory is the root an agent working one of its
tickets sees. The harness store stays where it is: the live issue
database is not something to move for tidiness, and leaving it put means
this change needs no migration of the one piece of state everything
depends on.

Isolation rests on permissions.check_within_workspace, which compares
path components rather than string prefixes -- a plain prefix test would
let /projects/proj-a reach ../proj-abc, i.e. no isolation at all between
sibling projects.
"""

import os
import subprocess

from .config import PROJECTS_ROOT


def project_id_for(ticket_id: str) -> str:
    """The project a ticket belongs to. Beads ids encode the hierarchy,
    so the root is the id up to the first dot -- same convention
    api._parent_id and toolchain.project_id_for rely on."""
    return ticket_id.split(".", 1)[0]


def path_for(project_id: str) -> str:
    return os.path.join(PROJECTS_ROOT, project_id)


def exists(project_id: str) -> bool:
    return os.path.isdir(path_for(project_id))


def ensure(project_id: str) -> str:
    """Create the workspace and its git repository if absent, and return
    its path. Idempotent -- safe to call on every ticket."""
    path = path_for(project_id)
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(
            ["git", "init", "-q"], cwd=path, capture_output=True, text=True, timeout=60
        )
        # The container sets a system-wide git identity (see Dockerfile),
        # so commits work without further configuration here.
    return path


def for_ticket(ticket_id: str) -> str:
    """The workspace an agent working this ticket should be rooted in."""
    return ensure(project_id_for(ticket_id))


def has_commits(project_id: str) -> bool:
    out = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=path_for(project_id), capture_output=True, text=True, timeout=60,
    )
    return out.returncode == 0


def diff_since(project_id: str, ref: str | None) -> str:
    """Changes in this project's workspace since `ref`, or the whole
    working tree's uncommitted diff when ref is None.

    This is what lets a verifier judge against the code a ticket actually
    produced rather than against the agent's description of it."""
    path = path_for(project_id)
    if not os.path.isdir(path):
        return ""
    args = ["git", "diff"]
    if ref:
        args.append(ref)
    tracked = subprocess.run(args, cwd=path, capture_output=True, text=True, timeout=120)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=path, capture_output=True, text=True, timeout=120,
    )
    parts = []
    if tracked.stdout.strip():
        parts.append(tracked.stdout)
    if untracked.stdout.strip():
        # New files do not appear in `git diff` until added, and a ticket
        # that only creates files would otherwise look like it produced
        # nothing at all.
        parts.append("new files:\n" + untracked.stdout)
    return "\n".join(parts)


def head(project_id: str) -> str | None:
    """Current commit, or None if the workspace has no commits yet."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path_for(project_id), capture_output=True, text=True, timeout=60,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def commit_all(project_id: str, message: str) -> str | None:
    """Commit everything currently in a project's workspace, returning the
    new commit sha, or None if there was nothing to commit.

    The harness does this on the ticket's behalf rather than requiring the
    agent to run git. That keeps the useful property -- each ticket's
    output is an inspectable diff attributable to exactly that ticket --
    without adding git tooling to the agent's surface, and without
    depending on an agent remembering to commit. It also gives the
    verifier something precise to judge: the diff of this commit is what
    this ticket did, rather than whatever happens to be lying around in
    the working tree from earlier work."""
    path = path_for(project_id)
    if not os.path.isdir(path):
        return None
    subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True, text=True, timeout=120)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=path, capture_output=True, text=True, timeout=120
    )
    if staged.returncode == 0:
        return None  # nothing staged -- the ticket changed no files
    done = subprocess.run(
        ["git", "commit", "-q", "-m", message[:2000]],
        cwd=path, capture_output=True, text=True, timeout=120,
    )
    if done.returncode != 0:
        return None
    return head(project_id)


def commit_diff(project_id: str, sha: str, max_chars: int = 20000) -> str:
    """The diff introduced by one commit -- what a ticket actually changed.

    Truncated: a verifier judging against acceptance criteria needs to see
    the shape of the work, not every line of a large generated file."""
    out = subprocess.run(
        ["git", "show", "--stat", "--patch", sha],
        cwd=path_for(project_id), capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        return ""
    text = out.stdout
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [diff truncated at {max_chars} characters]"
    return text
