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


# Dependency and build directories the harness must never commit.
# commit_all stages with `git add -A`, so anything not ignored here ends
# up in the ticket's commit. Found live 2026-09-09: Silent Run's
# .gitignore came from `bd init` and covered only Dolt files, so three
# consecutive commits added, removed and re-added the whole
# node_modules/@types tree. That buries the actual change -- and
# commit_diff truncates at 20k characters, so a verifier judging the
# diff can end up seeing nothing but dependency noise.
_IGNORE_BLOCK = """
# Added by the harness: never commit dependencies or build output.
node_modules/
dist/
build/
out/
.venv/
__pycache__/
*.pyc
.pytest_cache/
target/
"""


def _ensure_gitignore(path: str) -> None:
    """Append the harness's ignore block if it isn't already there, and
    commit it.

    Committing immediately matters: ensure() runs before every ticket, so
    an uncommitted .gitignore left lying in the workspace would be swept
    into the next ticket's commit by commit_all's `git add -A`, and would
    make commit_all report a change for a ticket that did nothing at all.
    That signal is the one the verifier relies on, so the workspace has to
    be genuinely clean when a ticket starts."""
    p = os.path.join(path, ".gitignore")
    try:
        existing = open(p).read() if os.path.isfile(p) else ""
        if "Added by the harness" in existing:
            return
        with open(p, "a") as fh:
            fh.write(_IGNORE_BLOCK)
    except OSError:
        return  # a workspace we cannot write to is not worth failing a ticket over

    subprocess.run(
        ["git", "add", ".gitignore"], cwd=path, capture_output=True, text=True, timeout=60
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "harness: ignore dependency and build directories"],
        cwd=path, capture_output=True, text=True, timeout=60,
    )


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
    _ensure_gitignore(path)
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
    """The diff a ticket actually produced.

    `sha` is either a single commit -- the usual case, the harness
    committed on the ticket's behalf -- or a `base..head` range, used when
    the agent ran git itself and so left commit_all nothing to stage. A
    range is diffed end-to-end rather than shown commit-by-commit, so the
    verifier sees one coherent change set either way.

    Truncated: a verifier judging against acceptance criteria needs to see
    the shape of the work, not every line of a large generated file."""
    args = (
        ["git", "diff", "--stat", "--patch", sha]
        if ".." in sha
        else ["git", "show", "--stat", "--patch", sha]
    )
    out = subprocess.run(
        args, cwd=path_for(project_id), capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        return ""
    text = out.stdout
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [diff truncated at {max_chars} characters]"
    return text


def run_tests(project_id: str, timeout: int = 600) -> dict | None:
    """Best-effort mechanical run of a project's own test suite.

    Returns None when there is nothing to run -- no package.json, or no
    `test` script in it. Absence of a suite is something the verifier's
    model should weigh against the acceptance criteria, not something to
    decide mechanically here.

    Otherwise returns {"ran", "passed", "failed", "exit", "tail"}.

    This exists because an exit code is not evidence that a suite ran.
    Node's test runner exits 0 when its file glob matches nothing. Found
    live 2026-09-09: workspace-9jg.1.6 shipped
    `node --test dist/test/*.js` alongside a tsconfig that only ever
    emitted src/, so `npm test` printed "# tests 0" and exited 0 -- a
    green command that verified nothing, on the ticket 73 others were
    blocked behind.

    Deliberately does NOT install dependencies: a missing node_modules
    should not be reported as the ticket's failure, and a verifier step
    should not reach the network.
    """
    path = path_for(project_id)
    pkg = os.path.join(path, "package.json")
    if not os.path.isfile(pkg):
        return None
    try:
        import json as _json

        with open(pkg) as fh:
            if not (_json.load(fh).get("scripts") or {}).get("test"):
                return None
    except Exception:
        return None

    try:
        out = subprocess.run(
            ["npm", "test"], cwd=path, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ran": 0, "passed": 0, "failed": 0, "exit": -1, "tail": "npm test timed out"}

    text = (out.stdout or "") + (out.stderr or "")

    def _count(label: str) -> int:
        for line in text.splitlines():
            if line.startswith(f"# {label} "):
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return 0
        return 0

    return {
        "ran": _count("tests"),
        "passed": _count("pass"),
        "failed": _count("fail"),
        "exit": out.returncode,
        "tail": "\n".join(text.splitlines()[-15:]),
    }
