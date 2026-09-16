"""
The product-owner agent: a tool-calling LangGraph loop, not a rule table
-- assignment, and "does a specialist for this exist yet," are judgment
calls the product-owner makes over live data (the seat roster + outcomes
+ unassigned tickets), not decisions this module makes on its own.
PLAN.md's "emergent, not hardcoded" design goal for how work gets divided
among seats.

One LangGraph thread per triage *session*, not per ticket like the
worker's: a single pass naturally inspects and acts on several tickets,
and possibly creates a new seat, before it's done.

Projects (added 2026-08-29, user's own architecture call): v1 had
separate agent rosters per project; v2 deliberately does NOT -- given
this harness's real hard concurrency constraint (Phase 1/PLAN.md: mostly
one local model, one thing happening at a time), running several
independent per-project agent teams would just mean context-switching
overhead with nothing to show for it. Instead: ONE product-owner, ONE
shared seat pool, MANY projects -- each with its own backlog/board (a
Beads issue hierarchy: project -> epic -> story/subtask, all via Beads'
native --parent/--priority rather than a parallel table, see
beads.list_top_level) -- and the product-owner itself decides which
project's work to actually pull from next, weighted by priority and
whatever a human has said matters most right now. This is deliberately
NOT "split effort evenly across all projects" -- time-slice toward the
top-priority project, only reaching into a lower one when the top is
exhausted or blocked, and widen that fan-out later if real concurrency
increases (more providers, faster local hardware) rather than assuming
it today.
"""

import uuid

from langchain_core.tools import tool

from . import beads, escalations, meta_agent, model_registry, outcomes, seats, settings
from .graph import build_graph_from_model

ROLE = "product_owner"

DEFAULT_BRIEF = "Triage the current queue."

# A session, not another standing role: the same product-owner, pointed at
# the escalation queue instead of the unassigned pool. Most escalations are
# the product-owner's to resolve -- fix an underspecified ticket, name a
# missing consumer, reassign to the right specialist, or requeue work whose
# crash cause is fixed -- and only the ones it cannot decide should keep a
# human's attention.
ESCALATION_BRIEF = """Tickets other agents parked for a human are waiting. Call \
`list_escalations`, then take exactly one action on each listed ticket, in order, and do not \
invent unrelated work.

Choose the action from what the escalation says:
- Crashed or stopped (run failed, no completion claim) with a cause that is infrastructure, not \
the work: `requeue_escalation` with a short directive for the next attempt.
- Underspecified (no acceptance criteria, a missing consumer, an unstated decision): fix that \
first -- `set_acceptance_criteria`, `split_escalated_ticket`, or `assign_ticket` to a better suit \
-- then `requeue_escalation` with a directive.
- A genuine product decision you can make: make it, recording it with \
`set_acceptance_criteria` or `resolve_escalation` rather than leaving it.
- Wrong specialist: `assign_ticket` (or `request_new_seat`) then `requeue_escalation`.
- The ticket's own named deliverable is ALREADY in the project, and the attempt refused because \
there is nothing honest left to do (it will say so, naming the file and the commit): \
`resolve_escalation`, recording what is present and where. Do NOT requeue that one -- the next \
attempt finds exactly the same thing and refuses again, which is now a dead end: the project \
will not move past its front ticket until this is answered. Accept the delivery when the named \
artifact really is there; requeue only if something the criteria ask for is genuinely missing.

Leave a ticket untouched only if it truly needs a person and you cannot decide it. Do not requeue \
a ticket whose project is on dispatch hold. Stop once every listed escalation has been actioned."""

SYSTEM_PROMPT = """You are the product-owner for an autonomous software delivery system. \
Several kinds of session you run:

1. Triage: look at unassigned ready tickets and the current specialist seat roster, and \
assign each ticket to the seat best suited for it. If no existing seat is a good fit, \
create a new specialist seat for that kind of work rather than forcing a mismatch -- \
specialization should emerge from real needs, not be forced upfront.

2. Idea decomposition: given a rough idea, decide first whether it's a brand-new project or \
belongs inside an existing one (check list_projects). For a new project, create_project once \
for the overall goal with a priority reflecting how it compares to what already exists. Then \
break it into epics (create_epic, parented under the project) and, within each epic, concrete \
individually-workable stories (add_subtask_to_epic) -- real tasks a seat could actually pick up \
and finish, not vague restatements of the idea. Match the depth of breakdown to the idea's \
actual size -- a small idea might just be one epic with a couple of stories, not three empty \
tiers for their own sake.

If any story in the breakdown must exist before others can even be attempted -- project \
scaffolding the whole codebase needs, a shared data model other stories import, an API \
contract other stories implement against -- create that story first and call add_dependency \
for every story it blocks, using the real ids both calls just returned. Do NOT just describe \
the ordering in a story's title or description: dispatch only ever reads `bd ready`, which \
only reads dependency edges, never prose, so an undeclared prerequisite gets handed out to a \
seat right alongside everything else -- wasted work against a project that cannot build or \
test yet.

3. Prioritization: there is deliberately only one shared pool of work capacity across every \
project (see list_projects for current priorities and any human guidance recorded there). Do \
NOT spread assignment evenly across all projects -- work the highest-priority project's ready \
tickets first, and only pull from a lower-priority project when the higher one has nothing \
ready to assign or is genuinely blocked. Within a project, drain one epic before starting the \
next: prefer the highest-priority epic that still has ready stories, and do not begin another \
epic's stories while that one still has unassigned or in-progress stories. The roadmap is a \
sequence of epics, not a set to run in parallel -- a board showing an early epic half-done \
while later epics are in flight is a prioritization failure. If a human's guidance changes \
what's most urgent, that should visibly change what you assign next, not just be acknowledged \
and ignored.

Use the tools available to inspect the roster, outcomes, and project priorities before \
deciding. When you've handled everything you can, stop."""


def build_tools(conn, requesting_model):
    """`requesting_model` is what request_new_seat delegates to (via
    meta_agent.create_specialist_seat) to draft a new seat's prompt --
    pass a *tool-free* model reference here (e.g. a RoutedModel
    constructed with tools=None), since that call just wants a plain
    JSON response, not another round of tool orchestration."""

    @tool
    def list_seats() -> str:
        """List active specialist seats with their recent track record -- check this before assigning work."""
        roster = seats.list_all(conn)
        if not roster:
            return "no seats exist yet"
        lines = []
        for s in roster:
            o = outcomes.summary(s["seat_id"])
            lines.append(
                f"{s['seat_id']}: {s['specialty']} "
                f"(closed={o['closed']}, refused={o['refused']}, still_open={o['still_open']})"
            )
        return "\n".join(lines)

    @tool
    def list_unassigned_tickets() -> str:
        """List ready tickets not yet assigned to any seat."""
        tickets = beads.unassigned_ready()
        if not tickets:
            return "no unassigned tickets"
        return "\n".join(f"{t['id']}: {t['title']} -- {t.get('description', '')}" for t in tickets)

    @tool
    def assign_ticket(issue_id: str, seat_id: str) -> str:
        """Assign an unassigned ticket to an existing seat by id."""
        if not seats.get(conn, seat_id):
            return f"error: no seat {seat_id!r} -- create it first with request_new_seat"
        beads.assign_to_seat(issue_id, seat_id, actor=ROLE)
        return f"assigned {issue_id} to {seat_id}"

    @tool
    def request_new_seat(specialty_description: str) -> str:
        """Create a brand-new specialist seat for work no existing seat covers.
        Goes active immediately -- no approval step, unlike revising an existing seat's prompt."""
        result = meta_agent.create_specialist_seat(
            conn, specialty_description, requested_by=ROLE, model=requesting_model
        )
        if result is None:
            return "failed to create a new seat -- try a more specific specialty description"
        return f"created seat {result['seat_id']}"

    @tool
    def check_model_options() -> str:
        """See the current system-wide cost-slider setting (0=slow/free, 100=fast/costly)
        and which configured model providers fit within it. Scaffolding today (only a local
        provider is typically configured) -- visibility, not yet automatic per-call routing."""
        slider = settings.get_cost_slider(conn)
        registry = model_registry.load_registry()
        eligible = model_registry.providers_at_or_below(registry, slider)
        lines = [f"cost slider: {slider}/100"]
        for p in registry:
            fits = "eligible" if p in eligible else "too costly for current slider"
            lines.append(f"- {p.name} (cost_tier={p.cost_tier}, model={p.model}): {fits}")
        return "\n".join(lines)

    @tool
    def list_projects() -> str:
        """List top-level projects sorted by priority (0=highest) -- check this before
        deciding what to work on next or where a new idea belongs. Every project's backlog
        lives underneath it as epics (create_epic) and stories (add_subtask_to_epic)."""
        top_level = beads.list_top_level()
        if not top_level:
            return "no projects exist yet"
        return "\n".join(
            f"{p['id']} [priority {p['priority']}, {p['status']}]: {p['title']} -- {p.get('description', '')}"
            for p in top_level
        )

    @tool
    def create_project(name: str, description: str, priority: int) -> str:
        """Create a new top-level project for a genuinely new body of work. priority is
        0-4 (0=highest) -- set it relative to whatever list_projects already shows, reflecting
        any human guidance about what matters most right now. Epics (create_epic) and
        stories (add_subtask_to_epic) nest under this."""
        project = beads.create(name, description, issue_type="epic", priority=priority)
        return f"created project {project['id']} (priority {priority}): {project['title']}"

    @tool
    def create_epic(project_id: str, title: str, description: str) -> str:
        """Create an epic under an existing project (see list_projects/create_project) --
        one coherent slice of that project's work. add_subtask_to_epic calls will hang off
        of this. Use once per epic, not per story."""
        epic = beads.create(title, description, issue_type="epic", parent=project_id)
        return f"created epic {epic['id']} under {project_id}: {epic['title']}"

    @tool
    def add_subtask_to_epic(epic_id: str, title: str, description: str,
                            acceptance_criteria: str = "") -> str:
        """Add one concrete, individually-workable story under an epic (created via
        create_epic). Call this once per real piece of work the epic decomposes into.

        ALWAYS give acceptance_criteria: what must be concretely true for this to be done,
        checkable by someone who didn't do the work. A story without criteria is never
        verified by anyone (verifier.py skips it), so the working agent's own claim of
        success becomes the only record that it happened."""
        subtask = beads.create(title, description, parent=epic_id,
                               acceptance_criteria=acceptance_criteria or None)
        return f"created subtask {subtask['id']} under {epic_id}: {subtask['title']}"

    @tool
    def add_dependency(blocked_id: str, blocker_id: str) -> str:
        """Declare that `blocked_id` cannot start until `blocker_id` is done -- e.g. every
        other story in a fresh project depends on its scaffolding story, or a story that
        implements an API depends on the story that defines its contract.

        Call this during breakdown for every real prerequisite you can see, right after
        creating both ends. Writing "depends on X" into a description does nothing on its
        own -- only a real dependency edge keeps `bd ready` (what dispatch actually reads)
        from handing the blocked story to a seat before its prerequisite exists."""
        try:
            beads.add_dependency(blocked_id, blocker_id, actor=ROLE)
        except Exception as e:
            return f"error: {e}"
        return f"{blocked_id} now depends on {blocker_id} -- blocked until it closes"

    @tool
    def list_escalations() -> str:
        """Tickets other agents escalated and parked for a human, each with the reason they gave."""
        items = escalations.pending(conn)
        if not items:
            return "no escalations pending"
        return "\n".join(
            f"{i['id']} [{i['status']}] {i['title']}\n  reason: {(i.get('notes') or '').strip()[:600]}"
            for i in items
        )

    @tool
    def set_acceptance_criteria(issue_id: str, criteria: str) -> str:
        """Set or replace a ticket's acceptance criteria -- the usual fix when an escalation
        says the ticket had none or was underspecified."""
        beads.set_acceptance_criteria(issue_id, criteria, actor=ROLE)
        return f"set acceptance criteria on {issue_id}"

    @tool
    def split_escalated_ticket(issue_id: str, title: str, description: str,
                               acceptance_criteria: str = "") -> str:
        """Split an oversized or multi-part escalated ticket by adding a child subtask under it,
        so the parent can be narrowed and the piece handed out separately."""
        sub = beads.create(title, description, parent=issue_id,
                           acceptance_criteria=acceptance_criteria or None)
        return f"created subtask {sub['id']} under {issue_id}: {sub['title']}"

    @tool
    def requeue_escalation(issue_id: str, directive: str) -> str:
        """Resolve an escalation by requeuing the ticket with a directive the next attempt sees.
        Bounded: after the escalation budget is spent the ticket is left for a human."""
        try:
            n = escalations.requeue(conn, issue_id, directive)
        except Exception as e:
            return f"error: {e}"
        return f"requeued {issue_id} (escalation attempt {n} of {escalations.MAX_ESCALATION_ATTEMPTS})"

    @tool
    def resolve_escalation(issue_id: str, resolution: str) -> str:
        """Close an escalated ticket with a decision recorded -- for escalations that are answered
        rather than requeued (accept a refusal, decide an ambiguity, cancel the work)."""
        beads.respond_to_human(issue_id, resolution, actor=ROLE)
        return f"resolved {issue_id}"

    return [
        list_seats,
        list_unassigned_tickets,
        assign_ticket,
        request_new_seat,
        check_model_options,
        list_projects,
        create_project,
        create_epic,
        add_subtask_to_epic,
        add_dependency,
    ]


def build_escalation_tools(conn, requesting_model):
    """The product-owner turned to the escalation queue: the assignment
    tools it still needs (who can take this, put it in the right hands)
    plus the escalation-specific actions.

    A separate factory from `build_tools`, not five more entries on its
    list: the triage/dispatch toolset's shape is fixed by its own callers,
    and the two briefs want different surfaces. Here the escalation tools
    are defined around the same `conn`."""
    base = build_tools(conn, requesting_model)
    keep = {"list_seats", "list_projects", "assign_ticket", "request_new_seat"}
    reused = [t for t in base if t.name in keep]

    @tool
    def list_escalations() -> str:
        """Tickets other agents escalated and parked for a human, each with the reason they gave."""
        items = escalations.pending(conn)
        if not items:
            return "no escalations pending"
        return "\n".join(
            f"{i['id']} [{i['status']}] {i['title']}\n  reason: {(i.get('notes') or '').strip()[:600]}"
            for i in items
        )

    @tool
    def set_acceptance_criteria(issue_id: str, criteria: str) -> str:
        """Set or replace a ticket's acceptance criteria -- the usual fix when an escalation
        says the ticket had none or was underspecified."""
        beads.set_acceptance_criteria(issue_id, criteria, actor=ROLE)
        return f"set acceptance criteria on {issue_id}"

    @tool
    def split_escalated_ticket(issue_id: str, title: str, description: str,
                               acceptance_criteria: str = "") -> str:
        """Split an oversized or multi-part escalated ticket by adding a child subtask under it,
        so the parent can be narrowed and the piece handed out separately."""
        sub = beads.create(title, description, parent=issue_id,
                           acceptance_criteria=acceptance_criteria or None)
        return f"created subtask {sub['id']} under {issue_id}: {sub['title']}"

    @tool
    def requeue_escalation(issue_id: str, directive: str) -> str:
        """Resolve an escalation by requeuing the ticket with a directive the next attempt sees.
        Bounded: after the escalation budget is spent the ticket is left for a human."""
        try:
            n = escalations.requeue(conn, issue_id, directive)
        except Exception as e:
            return f"error: {e}"
        return f"requeued {issue_id} (escalation attempt {n} of {escalations.MAX_ESCALATION_ATTEMPTS})"

    @tool
    def resolve_escalation(issue_id: str, resolution: str) -> str:
        """Close an escalated ticket with a decision recorded -- for escalations that are answered
        rather than requeued (accept a refusal, decide an ambiguity, cancel the work)."""
        beads.respond_to_human(issue_id, resolution, actor=ROLE)
        return f"resolved {issue_id}"

    return reused + [
        list_escalations,
        set_acceptance_criteria,
        split_escalated_ticket,
        requeue_escalation,
        resolve_escalation,
    ]


def run_triage_session(agent_model, tools, checkpointer, thread_id: str | None = None, brief: str | None = None) -> dict:
    """One product-owner session -- triage by default, or idea
    decomposition (and anything else the role covers) if `brief`
    overrides the default triage instruction. `agent_model` must already
    be tool-bound to `tools` (matches `build_graph_from_model`'s existing
    contract) -- build it via `RoutedModel(ROLE, routing, gate,
    tools=tools)` or `plain_model.bind_tools(tools)`."""
    thread_id = thread_id or f"triage-{uuid.uuid4().hex[:8]}"
    graph = build_graph_from_model(agent_model, checkpointer, tools=tools)
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    if state.values:
        result = graph.invoke(None, config)
    else:
        result = graph.invoke(
            {
                "messages": [("system", SYSTEM_PROMPT), ("user", brief or DEFAULT_BRIEF)],
                "ticket_id": thread_id,
                "turn_count": 0,
            },
            config,
        )

    return {"thread_id": thread_id, "final_message": result["messages"][-1].content}
