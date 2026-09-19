"""Minimal Phase 1 tool set: shell exec, file read/write, Beads memory,
and Phase 4's refuse-work / handoff-note primitives."""

import os
import signal
import subprocess
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from . import beads, permissions, slack, wiki
from .config import WORKSPACE_ROOT
from .state import HarnessState


def _prompt_conn():
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


# How long a single agent shell command may run before it is killed and the
# timeout handed back as an ordinary tool result. Hardcoded at 120s
# originally, which killed real runs: an agent that edits a file and then
# runs `npm test` in one command routinely exceeds two minutes on a cold
# TypeScript build. subprocess.TimeoutExpired propagated out of the tool
# and took the whole graph run down; after MAX_TICKET_FAILURES the
# dispatcher parked the ticket for a human. A command that never finishes
# is information the agent should see and react to, not a reason to
# discard the run -- same reasoning as read_file/write_file returning
# their errors instead of raising. Matches workspaces.run_tests' 600s,
# since it is often the same suite being run by hand.
SHELL_TIMEOUT = int(os.environ.get("SHELL_TIMEOUT", "600"))


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the shell AND its children. shell=True puts the real work in a
    child of the shell, so proc.kill() alone can orphan a running test
    runner; on POSIX the whole session started via start_new_session is
    killed instead."""
    try:
        os.killpg(os.getpgid(proc.pid), getattr(signal, "SIGKILL", signal.SIGTERM))
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_shell(command: str, workspace_root: str) -> str:
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=workspace_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=SHELL_TIMEOUT)
        return out or ""
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            out, _ = proc.communicate(timeout=10)
        except Exception:  # noqa: BLE001 -- a final drain failure must
            # still return the timeout to the agent, not raise.
            out = ""
        return (out or "") + (
            f"\n[command timed out after {SHELL_TIMEOUT}s and was killed. "
            "If it was a test or build, run a narrower target or split the "
            "command; if it needs longer, raise SHELL_TIMEOUT.]"
        )


@tool
def remember_fact(text: str) -> str:
    """Persist a durable insight via Beads (`bd remember`) so it survives
    across sessions and tickets, not just this conversation."""
    result = beads.remember(text)
    return f"remembered: {result.get('key', text)}"


@tool
def search_related_work(query: str) -> str:
    """Search past and current Beads issues (title/description keyword
    match, includes closed ones) for work related to what you're about to
    do -- check before starting in case it's already been done, is in
    progress elsewhere, or there's useful prior context in a closed
    issue's notes."""
    results = beads.search(query)
    if not results:
        return "no related issues found"
    return "\n".join(f"{r['id']} [{r['status']}]: {r['title']}" for r in results)


@tool
def create_subtask(title: str, description: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Break a piece of this ticket's work out into its own subtask,
    parented under the current ticket in Beads' dependency graph -- use
    this when a ticket turns out to be bigger than one sitting of work, so
    the pieces are individually trackable/resumable rather than all living
    inside one giant thread.

    This ticket cannot be completed while any of its subtasks is still
    open. Close each one with `complete_subtask` (passing the id returned
    here) once its work is done, before calling `complete_ticket`."""
    subtask = beads.create(title, description, parent=state["ticket_id"])
    return f"created subtask {subtask['id']}: {subtask['title']}"


@tool
def complete_subtask(
    subtask_id: str, summary: str, state: Annotated[HarnessState, InjectedState]
) -> str:
    """Close a subtask you created with `create_subtask`, once its work is
    done -- the counterpart that tool is missing on its own. The parent
    ticket cannot be completed until every subtask is closed, so close each
    one here as you finish it.

    `subtask_id` must be a subtask of the ticket you are working (the id
    `create_subtask` returned); this refuses to close anything else."""
    parent = state["ticket_id"]
    children = {c["id"]: c for c in beads.children_of(parent)}
    child = children.get(subtask_id)
    if child is None:
        return (
            f"{subtask_id!r} is not a subtask of {parent} -- pass an id returned by "
            "create_subtask, or call complete_ticket to finish the ticket itself."
        )
    if child.get("status") == "closed":
        return f"{subtask_id} is already closed"
    beads.close(subtask_id, reason=summary[:500])
    return f"closed subtask {subtask_id}"


@tool
def refuse_ticket(reason: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Decline this ticket instead of attempting it -- for work that's
    outside scope, ambiguous enough to need a human call, or that seems
    like it shouldn't be done at all. Flags the ticket for human review
    (`bd human list`) rather than silently retrying or force-completing
    it. Use this instead of guessing when you're genuinely unsure whether
    the work should happen."""
    beads.flag_for_human(state["ticket_id"], reason)
    return f"flagged for human review: {reason}"


@tool
def complete_ticket(summary: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Declare this ticket finished, describing what you actually did and
    where the work lives -- files written, commands run, how you checked
    it works.

    You MUST call this before your work counts as complete. A ticket you
    stop working on without calling this is not closed; it is flagged for
    a human, because a ticket that ends with nothing recorded is
    indistinguishable from one where nothing was done.

    Be concrete and honest. This summary is the evidence a separate
    verifier judges against the ticket's acceptance criteria, so an
    inflated one fails verification rather than passing quietly. If you
    could not finish, use refuse_ticket or decline_ticket instead."""
    ticket_id = state["ticket_id"]
    # A ticket with open subtasks cannot be closed in Beads (bd refuses),
    # and the harness would otherwise park it for a human at the very end
    # of the run. Refuse HERE instead -- while the agent can still act --
    # so it closes them with complete_subtask and completes properly.
    open_subtasks = [
        c["id"] for c in beads.children_of(ticket_id) if c.get("status") != "closed"
    ]
    if open_subtasks:
        return (
            "cannot complete: this ticket still has open subtask(s) "
            + ", ".join(open_subtasks)
            + ". Finish each and close it with complete_subtask, then call "
            "complete_ticket again."
        )
    beads.set_metadata(ticket_id, "completion_summary", summary[:2000])
    beads.append_note(ticket_id, f"completed: {summary}")
    return "completion recorded -- the ticket will close if nothing else intervenes"


@tool
def decline_ticket(reason: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Hand this ticket back because it isn't your speciality -- another
    specialist should do it. Use this when the work itself is perfectly
    reasonable and clearly needs doing, but sits outside what you're for.

    Different from refuse_ticket: that one escalates to a human because
    the work is ambiguous or shouldn't happen at all. This one just says
    "not me", puts the ticket back in the pool, and lets the
    product-owner route it to a better-suited agent (or create one).
    Don't use it to avoid work you could reasonably do."""
    ticket_id = state["ticket_id"]
    # The seat holding a ticket is its assigned seat -- read it rather
    # than threading a seat_id through HarnessState, which is scoped to
    # one ticket thread and carries no seat identity.
    seat_id = beads.assigned_seat(beads.show(ticket_id)) or "unknown"
    beads.release_to_pool(ticket_id, seat_id, reason)
    return f"declined and returned to the pool: {reason}"


@tool
def scan_team_channel() -> str:
    """Check recent team-channel activity (Slack, if configured) for context another agent
    or a human may have left -- worth checking early on a ticket in case there's a relevant
    heads-up, decision, or in-progress conversation you'd otherwise miss. Returns nothing
    (not an error) if no team channel is configured."""
    messages = slack.recent_messages()
    if not messages:
        return "no team channel configured, or nothing recent"
    return "\n".join(messages)


@tool
def post_to_team(message: str) -> str:
    """Say something to the team channel -- a heads-up, something you hit that others will
    hit too, a question, or just something you felt like sharing. Reaches the humans and is
    visible to other agents via scan_team_channel. Not a status report: the board already
    tracks what got done."""
    if not slack.post_message(message):
        return "no team channel configured, so nothing was posted"
    return "posted to the team channel"


@tool
def suggest_prompt_change(new_prompt: str, reason: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Propose a revision to your OWN system prompt -- the standing instructions you work
    from on every ticket. Use it when something about how you have been told to work got in
    your way, or when you have learned something that should have been in there.

    Give the full replacement text, not a diff, and say plainly what you changed and why.

    This does not take effect on its own. It is recorded as a pending revision for review --
    deliberately, since an agent silently rewriting its own standing instructions is a
    different thing from an agent doing its work."""
    from . import prompts

    seat_id = beads.assigned_seat(beads.show(state["ticket_id"])) or "unknown"
    conn = _prompt_conn()
    try:
        prompts.init_table(conn)
        version = prompts.propose(conn, seat_id, new_prompt, reason=reason)
    finally:
        conn.close()
    return f"recorded as pending revision v{version} for {seat_id} -- it will not take effect until reviewed"


@tool
def read_wiki_page(slug: str) -> str:
    """Read a page from the project wiki (human-facing documentation, distinct from Beads
    notes) -- e.g. 'agents/some-seat-id' for that seat's own profile, or a topic doc.
    Returns a clear message rather than an error if the page doesn't exist yet."""
    content = wiki.read_page(slug)
    if content is None:
        return f"no wiki page at {slug!r} yet"
    return content


@tool
def write_wiki_page(slug: str, content: str) -> str:
    """Write (or overwrite) a page in the project wiki -- for human-facing documentation,
    not internal ticket notes (use write_handoff_note for those). Markdown, e.g.
    'agents/some-seat-id' for a profile page, or a topic like 'deployment-notes'."""
    path = wiki.write_page(slug, content)
    return f"wrote wiki page {slug} ({path})"


@tool
def list_wiki_pages() -> str:
    """List every page that currently exists in the project wiki."""
    pages = wiki.list_pages()
    if not pages:
        return "wiki is empty"
    return "\n".join(pages)


@tool
def write_handoff_note(note: str, state: Annotated[HarnessState, InjectedState]) -> str:
    """Record a note for whoever (or whatever future session of yourself)
    picks this ticket up next -- what's done, what's left, anything
    non-obvious. Call this before wrapping up, especially if asked to stop
    partway through rather than finishing."""
    beads.append_note(state["ticket_id"], note)
    return "handoff note recorded"


def build_workspace_tools(workspace_root: str) -> list:
    """The three tools that touch the filesystem, bound to one root.

    A factory rather than module-level definitions because the root is
    now per-ticket: an agent working a Silent Run story must be rooted in
    Silent Run's workspace, not in a process-wide constant that happens
    to point at the harness's own store. Closing over the root keeps the
    binding explicit -- an ambient/contextvar approach would silently
    fall back to the default root if it ever failed to propagate into a
    tool-execution thread, and a sandbox boundary should not fail quietly.
    """

    @tool
    def shell_exec(command: str) -> str:
        """Run a shell command in the workspace and return its combined output."""
        # Gating happens one layer up, in graph.py's permission_gate node --
        # every call reaches here already allowed (statically-safe fast path
        # or classifier-approved). No redundant check here: unlike file paths,
        # there's no workspace-independent hard invariant for shell commands
        # to enforce, and re-gating on the same static safe-set would silently
        # break any command the classifier explicitly approved.
        #
        # A timeout is returned as text, never raised. See _run_shell and
        # SHELL_TIMEOUT: an exception here used to kill the run and strand
        # the ticket behind a human flag.
        return _run_shell(command, workspace_root)

    @tool
    def read_file(path: str) -> str:
        """Read a text file's contents, relative to the workspace root."""
        # Containment failures still raise: an escape attempt is a
        # security event and should be loud. A missing file is not --
        # it is ordinary, and must come back as something the agent can
        # read and react to.
        #
        # This crashed a run 2901 times in one night (2026-09-02): an
        # agent read `wiki/agents/<seat>` out of habit from when its root
        # was /workspace, that path stopped existing under per-project
        # workspaces, FileNotFoundError propagated out of the tool node
        # and killed the graph, and the dispatcher restarted the same
        # ticket forever.
        permissions.check_readable(path, workspace_root)
        if permissions.is_infrastructure(path):
            return (
                f"{path!r} is harness/tool infrastructure ({', '.join(sorted(permissions.HIDDEN_FROM_LISTING))}), "
                "not part of your project -- there's nothing there relevant to your ticket."
            )
        resolved = os.path.abspath(os.path.join(workspace_root, path))
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            hint = ""
            if path.strip("/").startswith("wiki/"):
                hint = (
                    " The wiki is not in your workspace -- use read_wiki_page("
                    f"{path.strip('/')[len('wiki/'):]!r}) instead."
                )
            return f"no such file: {path!r} (relative to your workspace root).{hint}"
        except IsADirectoryError:
            return f"{path!r} is a directory, not a file"
        except OSError as e:
            return f"could not read {path!r}: {e}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write text content to a file, relative to the workspace root."""
        permissions.check_within_workspace(path, workspace_root)
        if permissions.is_infrastructure(path):
            return (
                f"{path!r} is harness/tool infrastructure ({', '.join(sorted(permissions.HIDDEN_FROM_LISTING))}), "
                "not part of your project -- nothing there should be written to."
            )
        resolved = os.path.abspath(os.path.join(workspace_root, path))
        try:
            os.makedirs(os.path.dirname(resolved) or workspace_root, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            # Same reasoning as read_file: a failed write is information
            # for the agent, not a reason to kill the run.
            return f"could not write {path!r}: {e}"
        return f"wrote {len(content)} bytes to {path}"

    @tool
    def list_directory(path: str = ".") -> str:
        """List a directory's contents, relative to the workspace root.
        Omits harness/tool infrastructure (.git, .beads, .claude, .codex,
        .agents) -- it isn't your project's own content, and at least one
        of those (.claude/settings.json) may carry credentials. Use this
        instead of `ls` to orient yourself, so you don't spend a turn
        finding out the hard way that something is off-limits."""
        permissions.check_readable(path, workspace_root)
        resolved = os.path.abspath(os.path.join(workspace_root, path))
        try:
            entries = sorted(os.listdir(resolved))
        except FileNotFoundError:
            return f"no such directory: {path!r} (relative to your workspace root)"
        except NotADirectoryError:
            return f"{path!r} is a file, not a directory"
        except OSError as e:
            return f"could not list {path!r}: {e}"

        visible = [e for e in entries if e not in permissions.HIDDEN_FROM_LISTING]
        hidden = len(entries) - len(visible)
        lines = [
            f"{name}/" if os.path.isdir(os.path.join(resolved, name)) else name
            for name in visible
        ] or ["(empty)"]
        if hidden:
            noun = "entry" if hidden == 1 else "entries"
            lines.append(f"({hidden} {noun} hidden -- harness/tool infrastructure, not your project's content)")
        return "\n".join(lines)

    return [shell_exec, read_file, write_file, list_directory]


# Tools that do not touch the project workspace and so need no binding.
SHARED_TOOLS = [
    remember_fact,
    post_to_team,
    search_related_work,
    scan_team_channel,
    read_wiki_page,
    write_wiki_page,
    list_wiki_pages,
    create_subtask,
    complete_subtask,
    refuse_ticket,
    decline_ticket,
    complete_ticket,
    write_handoff_note,
]


def build_agent_tools(workspace_root: str | None = None) -> list:
    """The full tool set for an agent rooted at `workspace_root`."""
    return build_workspace_tools(workspace_root or WORKSPACE_ROOT) + SHARED_TOOLS


# Default-rooted set, for callers that predate per-project workspaces
# (graph.build_graph's default, product-owner tests).
ALL_TOOLS = build_agent_tools()
