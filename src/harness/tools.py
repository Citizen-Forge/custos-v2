"""Minimal Phase 1 tool set: shell exec, file read/write, Beads memory,
and Phase 4's refuse-work / handoff-note primitives."""

import os
import subprocess
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from . import beads, permissions, slack, wiki
from .config import WORKSPACE_ROOT
from .state import HarnessState


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
    inside one giant thread."""
    subtask = beads.create(title, description, parent=state["ticket_id"])
    return f"created subtask {subtask['id']}: {subtask['title']}"


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
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return (result.stdout or "") + (result.stderr or "")

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
        permissions.check_within_workspace(path, workspace_root)
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

    return [shell_exec, read_file, write_file]


# Tools that do not touch the project workspace and so need no binding.
SHARED_TOOLS = [
    remember_fact,
    search_related_work,
    scan_team_channel,
    read_wiki_page,
    write_wiki_page,
    list_wiki_pages,
    create_subtask,
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
