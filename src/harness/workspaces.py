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

One project is still shared by every seat working one of its tickets, and
that is not enough on its own: commit_all stages with `git add -A`, so with
more than one agent running it sweeps the OTHER agents' half-written files
into whichever ticket happens to finish first. The commit is titled
`<ticket-id>: ...` and its content is several tickets' work, which is what
the verifier then judges -- it fails the ticket for "the diff under review
is not this ticket's work at all", the fail requeues, and after
MAX_VERIFIER_REWORKS the ticket parks. Found live 2026-09-15: 31 tickets on
the human queue and ~14 of them rejected in exactly those words.

So each ticket now gets its own `git worktree` (for_ticket), on its own
branch off the project's integration branch, and merges back when the
ticket completes (merge_to_integration). One writer per tree, so a
ticket's commit contains that ticket's work. The project directory stays
the integration checkout -- it is what the verifier reads and what a new
worktree branches from, so a ticket still starts on top of everything
already merged.
"""

import os
import subprocess

try:  # Linux only; the harness runs in a container, but tests can run anywhere.
    import fcntl
except ImportError:  # pragma: no cover - not reachable in the deployment
    fcntl = None  # type: ignore[assignment]

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
.beads/
.worktrees/
"""

# Added to workspaces that already carry _IGNORE_BLOCK. Kept separate
# because the block's marker check below short-circuits, so every existing
# project would otherwise never learn about the worktree directory -- and
# `git add -A` would then commit every ticket's other checkout into the
# integration branch.
_WORKTREES_IGNORE = """
# Added by the harness: per-ticket git worktrees live here.
.worktrees/
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
        if "Added by the harness" not in existing:
            addition = _IGNORE_BLOCK
        elif ".worktrees/" not in existing:
            addition = _WORKTREES_IGNORE
        else:
            return
        with open(p, "a") as fh:
            fh.write(addition)
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


# -- per-ticket worktrees ----------------------------------------------
#
# One working tree per ticket, so no two agents ever write into the same
# checkout. See the module docstring for the live failure this exists to
# stop.

WORKTREES_DIRNAME = ".worktrees"


def integration_branch() -> str:
    """The branch tickets branch from and merge back into. Pinned by name
    rather than taken from the repo's HEAD, because agents run `git
    checkout -b` themselves and a stray branch left checked out in the
    project directory would otherwise silently become the integration
    point for every later ticket."""
    return os.environ.get("HARNESS_INTEGRATION_BRANCH", "master")


def _git(args: list[str], cwd: str, timeout: int = 120):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def _current_branch(path: str) -> str | None:
    out = _git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    if out.returncode != 0:
        return None
    name = out.stdout.strip()
    # A detached HEAD reports "HEAD", which is not a branch to merge into.
    return name if name and name != "HEAD" else None


def _branch_exists(path: str, branch: str) -> bool:
    return _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], path).returncode == 0


def integration_ref(project_id: str) -> str | None:
    """The ref a new ticket branches from, or None while the project has
    no commits yet (nothing to branch from)."""
    path = path_for(project_id)
    if not os.path.isdir(path):
        return None
    if _branch_exists(path, integration_branch()):
        return integration_branch()
    # A repo created before this pin, or by a git whose default branch is
    # named otherwise: fall back to whatever it is actually on.
    return _current_branch(path) if has_commits(project_id) else None


def worktrees_root(project_id: str) -> str:
    return os.path.join(path_for(project_id), WORKTREES_DIRNAME)


def worktree_path_for(ticket_id: str) -> str:
    return os.path.join(worktrees_root(project_id_for(ticket_id)), ticket_id)


def ticket_branch(ticket_id: str) -> str:
    return f"ticket/{ticket_id}"


def _advance_unworked_tree(ticket_id: str, repo: str, worktree: str) -> None:
    """Move an untouched ticket tree up to the integration tip.

    A tree is branched from the tip when it is first created, which is what
    makes it contain everything already merged. But a tree can come to
    exist without ever being worked -- created for a dispatch that then
    died before the agent ran -- and reusing it leaves the ticket on a base
    that predates every ticket that has landed since. With one ticket at a
    time per project, that is the difference between a first attempt seeing
    the previous ticket's work and not seeing it at all, which is the whole
    point of serialising.

    Only a branch with NOTHING of its own is moved, and only while the tree
    is clean. Either qualification failing means the tree is a real attempt
    -- in progress, or finished and awaiting rework -- and it is left
    exactly where that attempt left it. Deliberately conservative: this
    must never be the thing that discards an agent's work."""
    branch = ticket_branch(ticket_id)
    if not _branch_exists(repo, branch):
        return
    base = integration_ref(project_id_for(ticket_id))
    if base is None:
        return

    branch_head = _git(["rev-parse", branch], repo).stdout.strip()
    tip = _git(["rev-parse", base], repo).stdout.strip()
    if not branch_head or not tip or branch_head == tip:
        return
    own_commits = _git(["rev-list", "--count", f"{base}..{branch}"], repo).stdout.strip()
    if own_commits != "0":
        return  # a previous attempt's work: not ours to move
    if _git(["status", "--porcelain"], worktree).stdout.strip():
        return  # uncommitted work in the tree: in use, leave it alone

    _git(["reset", "--hard", base], worktree)
    _git(["clean", "-qfd"], worktree)


def for_ticket(ticket_id: str) -> str:
    """The working tree an agent on this ticket is rooted in, created on
    first use and re-used afterwards.

    Re-used rather than recreated so a resumed or verifier-requeued ticket
    keeps the work already in its tree -- `_reset_thread` starts the agent
    over, it does not throw the ticket's files away. A tree with no work of
    its own is the exception: see _advance_unworked_tree.

    Always returns a worktree: a project with no commits yet gets an empty
    base commit so it has a ref to branch from. Anything that stops the
    worktree being created is raised rather than quietly handing back the
    shared project tree, which would reinstate the exact bug this
    replaces."""
    project_id = project_id_for(ticket_id)
    repo = ensure(project_id)
    worktree = worktree_path_for(ticket_id)
    if os.path.exists(os.path.join(worktree, ".git")):
        _advance_unworked_tree(ticket_id, repo, worktree)
        return worktree

    base = integration_ref(project_id)
    if base is None:
        # A project with no commits has no ref to branch from, which would
        # otherwise be the one case that still shares a tree between two
        # concurrent tickets. Give it an empty base instead, so isolation
        # holds from the very first ticket.
        _git(
            ["commit", "-q", "--allow-empty", "-m", "harness: base commit for ticket worktrees"],
            repo,
        )
        base = integration_ref(project_id)
    if base is None:
        raise RuntimeError(f"could not establish a base branch for {project_id}")

    os.makedirs(worktrees_root(project_id), exist_ok=True)
    # A directory left behind by a worktree git no longer knows about
    # (removed by hand, or a container recreate) makes `worktree add` fail.
    _git(["worktree", "prune"], repo)

    branch = ticket_branch(ticket_id)
    args = (
        ["worktree", "add", worktree, branch]
        if _branch_exists(repo, branch)
        else ["worktree", "add", "-b", branch, worktree, base]
    )
    out = _git(args, repo)
    if out.returncode != 0:
        raise RuntimeError(
            f"could not create a worktree for {ticket_id}: {out.stderr.strip()}"
        )
    return worktree


def head_at(path: str) -> str | None:
    out = _git(["rev-parse", "HEAD"], path)
    return out.stdout.strip() if out.returncode == 0 else None


def head(project_id: str) -> str | None:
    """Current commit of a project's integration checkout, or None if the
    workspace has no commits yet."""
    path = path_for(project_id)
    if not os.path.isdir(path):
        return None
    return head_at(path)


def head_for_ticket(ticket_id: str) -> str | None:
    """Where a ticket's own tree stands -- the base its run started from
    when read at the start of a run."""
    worktree = worktree_path_for(ticket_id)
    if os.path.exists(os.path.join(worktree, ".git")):
        return head_at(worktree)
    return head(project_id_for(ticket_id))


def reset_ticket_to_integration(ticket_id: str) -> str | None:
    """Move a ticket's tree onto the current integration tip, returning the
    commit its branch was on, or None if it had no branch.

    A requeued ticket's branch is frozen at the commit it forked from, so
    re-running the agent on top of it still merges into a conflict over
    whatever moved underneath in the meantime -- the barrel file in
    particular, which is exactly what parked these tickets in the first
    place. Starting from the tip is what makes the next attempt mergeable.

    The abandoned commits stay in the worktree's reflog, which is what the
    returned sha is for -- log it. Deliberately NOT kept as a branch or a
    tag: commits_for_ticket reads `git log --all`, so an archived ref would
    put the rejected attempt straight back into the diff the verifier
    judges, and the ticket would be re-failed on its own dead work."""
    project_id = project_id_for(ticket_id)
    repo = path_for(project_id)
    branch = ticket_branch(ticket_id)
    if not _branch_exists(repo, branch):
        return None
    base = integration_ref(project_id)
    if base is None:
        return None

    previous = _git(["rev-parse", branch], repo).stdout.strip()
    worktree = worktree_path_for(ticket_id)
    if os.path.exists(os.path.join(worktree, ".git")):
        _git(["reset", "--hard", base], worktree)
        # Untracked leftovers are a previous attempt's scratch, not work --
        # `-d` without `-x`, so ignored trees like node_modules stay put.
        _git(["clean", "-qfd"], worktree)
    else:
        _git(["branch", "-f", branch, base], repo)
    return previous


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


def _commit_all_at(path: str, message: str) -> str | None:
    """Commit everything currently in ONE working tree, returning the new
    commit sha, or None if there was nothing to commit.

    The harness does this on the ticket's behalf rather than requiring the
    agent to run git. That keeps the useful property -- each ticket's
    output is an inspectable diff attributable to exactly that ticket --
    without adding git tooling to the agent's surface, and without
    depending on an agent remembering to commit. It also gives the
    verifier something precise to judge: the diff of this commit is what
    this ticket did, rather than whatever happens to be lying around in
    the working tree from earlier work.

    `git add -A` is only safe because `path` is a per-ticket worktree: run
    in a tree two agents share, it stages their work as well as this
    ticket's."""
    if not os.path.isdir(path):
        return None
    subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True, text=True, timeout=120)
    # The harness's OWN Beads store lives inside the project workspace and
    # `bd init` does not ignore it, so `git add -A` stages its bookkeeping
    # (interactions.jsonl and friends) into every ticket's commit. That
    # directly corrupts the evidence the verifier judges: found live
    # 2026-09-15, workspace-9jg.2.3 failed because its commits "touch only
    # .beads/interactions.jsonl and two one-line scaffolding" files -- true,
    # and it hid whether any product work existed at all. Unstaged here as
    # well as ignored because .beads is already TRACKED in every existing
    # workspace, and .gitignore does not untrack a file.
    for ignore in (".beads", WORKTREES_DIRNAME):
        subprocess.run(
            ["git", "reset", "-q", "--", ignore],
            cwd=path, capture_output=True, text=True, timeout=120,
        )
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
    return head_at(path)


def commit_all(project_id: str, message: str) -> str | None:
    """Commit a project's integration checkout. See _commit_all_at."""
    return _commit_all_at(path_for(project_id), message)


def commit_all_for_ticket(ticket_id: str, message: str) -> str | None:
    """Commit the ticket's own worktree. This is the one a ticket's
    completion goes through -- it can only see that ticket's files."""
    return _commit_all_at(worktree_path_for(ticket_id), message)


def _merge_lock(project_id: str):
    """Serialise merges into the integration branch.

    Three agents finish tickets concurrently and each merges back, and git
    will not run two merges in one repository at once -- they share an
    index, so the loser sees a corrupted tree. Held on a file inside the
    (gitignored) worktrees directory so it never shows up as a pending
    change. Degrades to a no-op if fcntl is unavailable, which is a
    dev-on-Windows case only; the deployment is Linux."""
    import contextlib

    @contextlib.contextmanager
    def _lock():
        if fcntl is None:  # pragma: no cover - deployment is Linux
            yield
            return
        os.makedirs(worktrees_root(project_id), exist_ok=True)
        fd = os.open(os.path.join(worktrees_root(project_id), ".merge.lock"),
                     os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    return _lock()


def merge_to_integration(ticket_id: str) -> tuple[bool, str]:
    """Merge the ticket's branch into the project's integration branch.

    Until this runs the ticket's work exists only on its own branch, so
    nothing else in the project can see it -- and the verifier, which runs
    the project's test suite against the integration checkout, would be
    judging a tree without the change under review.

    Returns (ok, reason); a conflict is reported rather than resolved, so
    the caller can park the ticket for a person instead of guessing."""
    project_id = project_id_for(ticket_id)
    repo = path_for(project_id)
    base = integration_ref(project_id)
    if base is None:
        return True, ""
    branch = ticket_branch(ticket_id)
    if not _branch_exists(repo, branch):
        return True, ""  # the ticket produced no branch: nothing to merge

    with _merge_lock(project_id):
        # The merge has to happen in the integration checkout, because that
        # is the tree the project's own test suite runs in.
        if _current_branch(repo) != base:
            out = _git(["checkout", "-q", base], repo)
            if out.returncode != 0:
                return False, f"could not check out {base}: {out.stderr.strip()}"
        out = _git(
            ["merge", "--no-ff", "-m", f"{ticket_id}: merge into {base}", branch], repo
        )
        if out.returncode != 0:
            _git(["merge", "--abort"], repo)
            return False, (out.stderr.strip() or out.stdout.strip()).replace("\n", " ")[:400]
        return True, ""


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


def commits_for_ticket(project_id: str, ticket_id: str) -> list[str]:
    """Commit shas whose subject begins `<ticket_id>: ` -- i.e. the commits
    the harness made on this ticket's behalf (worker.commit_all uses that
    prefix). Oldest first. Empty when the workspace or git history is
    absent, or when the agent committed itself without the harness's
    prefix."""
    path = path_for(project_id)
    if not os.path.isdir(path):
        return []
    out = subprocess.run(
        ["git", "log", "--all", "--reverse", "--format=%H%x1f%s"],
        cwd=path, capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        return []
    prefix = f"{ticket_id}:"
    shas = []
    for line in out.stdout.splitlines():
        sha, _, subject = line.partition("\x1f")
        if sha and subject.startswith(prefix):
            shas.append(sha)
    return shas


def diff_for_ticket(project_id: str, ticket_id: str, work_commit: str | None = None,
                    max_chars: int = 20000) -> str:
    """The diff attributable to a ticket.

    Prefers the commits the harness made for this ticket, found by subject,
    so a missing or misattributed work_commit cannot hide the work. Falls
    back to the recorded work_commit when no such commit exists (e.g. the
    agent ran git itself). Found live 2026-09-14: four tickets (1.6, 13.1,
    13.4, 1.5) were failed by the verifier on empty or wrong diffs because
    work_commit had never been recorded or pointed at another seat's
    commit -- a false fail that parked real, present work.

    Each of the ticket's commits is diffed SEPARATELY rather than diffing
    the range from the first to the last. The workspace is shared by every
    seat on a project, so other tickets commit in between, and a range
    diff silently drags all of them in. Found live 2026-09-15:
    workspace-9jg.7.3's real change is two files (commit f75c3dc), but the
    range first^..last spanned 137 other commits and 206 files / 66,418
    insertions -- so the verifier saw other tickets' work, failed it as
    "not isolated" and "a huge multi-seat change", and the ticket parked
    once its rework budget was gone. Seventeen tickets were closed that
    way and had never once been judged on their own work."""
    shas = commits_for_ticket(project_id, ticket_id)
    if shas:
        parts: list[str] = []
        remaining = max_chars
        for sha in shas:
            if remaining <= 0:
                break
            chunk = commit_diff(project_id, sha, remaining)
            if chunk:
                parts.append(chunk)
                remaining -= len(chunk)
        if parts:
            return "\n".join(parts)
    if work_commit:
        return commit_diff(project_id, work_commit, max_chars)
    return ""


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


# Files acceptance criteria are usually written against -- "contains a
# package.json", "a README states how to build". Small, and their current
# contents answer such a criterion outright.
_CRITERIA_FILES = (
    "package.json", "tsconfig.json", "README.md", "readme.md",
    "pyproject.toml", "setup.cfg", "Makefile", "Cargo.toml", "go.mod",
)


def criteria_file_snapshot(project_id: str, max_chars: int = 6000) -> str:
    """Current contents of the project's configuration and readme files.

    A diff says what a ticket *changed*; it does not say what the project
    now *is*. Found live 2026-09-09: the verifier was shown a diff in
    which tsconfig's `"module": "commonjs"` was replaced by `"ESNext"`,
    correctly understood that the bad line had been removed, and still
    failed the ticket because it "cannot be verified as met without
    seeing the corrected tsconfig in the final state" -- which nothing
    ever gave it. Later commits by other tickets widen that gap further,
    since the reviewed commit is no longer HEAD.

    Returns "" when the workspace has none of these files."""
    path = path_for(project_id)
    parts, used = [], 0
    for name in _CRITERIA_FILES:
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        try:
            body = open(p).read()
        except OSError:
            continue
        if len(body) > 2000:
            body = body[:2000] + "\n... [truncated]"
        block = f"--- {name} (current contents) ---\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "".join(parts)
