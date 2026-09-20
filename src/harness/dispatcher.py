"""
Capacity-driven dispatch: the product-owner brokers every ticket.

Replaces the old shape where one worker process was started per seat via
the SEAT_ID env var. That never worked in practice: docker-compose runs a
single worker with SEAT_ID defaulting to "worker", so any seat the
product-owner created had no process and simply never ran. Observed live
2026-08-31 -- seat `deterministic-tick` held 2 assigned tickets and
`orbital-transit-motion` 1, while the only running worker polled for seat
"worker", which had none, and idled forever.

The loop, once per cycle:

1. If the number of running agents is at MAX_RUNNING_AGENTS, do nothing.
2. Otherwise, if a ticket is already assigned to a seat and ready, start
   that seat on it.
3. Otherwise, if unassigned work exists, wake the product-owner to broker
   exactly one ticket -- pick it, decide whether an existing seat fits or
   a new specialist is needed, and assign it. The next cycle picks it up
   at step 2.

Two limits, deliberately separate -- conflating them would be wrong:

- MAX_RUNNING_AGENTS bounds how many agents are working at once.
- ProviderConfig.concurrency_limit bounds simultaneous LLM requests, and
  routing.ConcurrencyGate already enforces it with one semaphore per
  provider shared across roles.

Ten agents against a single inference slot is a valid configuration, not
a misconfiguration: the agents run, and their model calls queue in order
on the gate. At the default MAX_RUNNING_AGENTS=1 this is one ticket
crossing the board at a time, which is the intended behaviour.

An agent may hand a ticket back with `decline_ticket` if it isn't their
speciality (beads.release_to_pool) -- distinct from `refuse_ticket`,
which escalates to a human. A declined ticket returns to the pool with
the declining seat recorded, so the product-owner doesn't hand it
straight back.

Each agent runs in its own thread with its own checkpointer and database
connection rather than sharing a cached runtime. Building a runtime costs
a DB read and a graph compile per ticket, which is real but small; the
alternative -- caching graphs across threads -- would need the shared
PostgresSaver to be demonstrably thread-safe, which hasn't been
established here. Revisit if runtime build cost ever shows up as a
bottleneck.
"""

import logging
import os
import threading
import time

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from . import beads, reflection, seats, toolchain, verifications, workspaces
from .product_owner import ROLE as PRODUCT_OWNER_ROLE
from .product_owner import build_tools as build_product_owner_tools
from .product_owner import run_triage_session
from .providers import ProviderConfig
from .routing import ConcurrencyGate, RoutedModel, RoutingTable
from .verifier import build_model as build_verifier_model
from .verifier import verify_ticket
from .worker import build_seat_runtime, work_one_ticket

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dispatcher")

MAX_RUNNING_AGENTS = int(os.environ.get("MAX_RUNNING_AGENTS", "1"))

# How many times one ticket may crash before it stops being retried.
# Without this a ticket that fails deterministically is restarted every
# cycle forever: on 2026-09-02 a FileNotFoundError inside read_file
# killed the graph run, and the same ticket was started and failed 2901
# times overnight, holding the only agent slot the whole time and
# starving every other ticket. Three attempts is enough to ride out a
# transient (a model 503, a bd timeout) without burning a night on
# something that will never succeed.
MAX_TICKET_FAILURES = int(os.environ.get("MAX_TICKET_FAILURES", "3"))
POLL_SECONDS = int(os.environ.get("DISPATCH_POLL_INTERVAL", "10"))

# Only these are dispatchable work. Projects and epics are also Beads
# issues and also show up in `bd ready` (verified live: a project came
# back as a ready ticket), but they are containers -- assigning one to a
# seat would hand an agent an entire project as if it were a task.
# Stories are created with beads.create's default issue_type.
WORK_ITEM_TYPE = "task"

DISPATCH_BRIEF = """There is capacity for one more agent to start work.

Dispatch exactly ONE ticket, then stop.

1. `list_unassigned_tickets` shows what is waiting. `list_projects` shows
   how it is organised and at what priority -- prefer lower priority
   numbers (0 is highest), and prefer work that unblocks other work.
2. `list_seats` shows which specialists already exist. If one genuinely
   fits the ticket, `assign_ticket` it to them.
3. If no existing seat is a real fit, `request_new_seat` with a specific
   specialty description, then `assign_ticket` the ticket to the seat id
   that comes back. Do not create a new seat for work an existing seat
   could plausibly do -- a roster of near-duplicate specialists is worse
   than a slightly broad one.
4. If a ticket's notes say a seat already declined it, do not assign it
   back to that seat.

Assign exactly one ticket and stop. Do not decompose projects, create
epics, or add new work in this session -- this is dispatch only."""


def _provider(max_tokens: int) -> ProviderConfig:
    return ProviderConfig(
        name="dispatch-product-owner",
        base_url=os.environ.get(
            "SCHEDULER_MODEL_BASE_URL",
            os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        ),
        model=os.environ.get(
            "SCHEDULER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
        ),
        # api_key and extra_body fall back to LOCAL_* for the same reason
        # base_url and model already do: SCHEDULER_MODEL_* is an opt-out,
        # so moving the worker to a new provider must move this with it
        # whole rather than half. Missing until 2026-09-12 -- with the
        # worker on DeepSeek this config inherited an authenticated
        # base_url and no key, so wake_product_owner would have 401'd.
        api_key=os.environ.get("SCHEDULER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=max_tokens,
        # A thinking model requires its `reasoning_content` replayed on
        # the next turn and langchain_openai drops it -- and this is an
        # agentic tool loop, so it would fail on turn two. See
        # ProviderConfig.extra_body.
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get("SCHEDULER_MODEL_DISABLE_THINKING", os.environ.get("LOCAL_MODEL_DISABLE_THINKING"))
            else None
        ),
    )


def dispatchable(issue: dict) -> bool:
    return issue.get("issue_type") == WORK_ITEM_TYPE


# A project can be held out of dispatch with a stated reason, so work
# that exists and matters but cannot currently be done by an agent stops
# being handed out. Added after a real 12-hour waste (2026-09-01): every
# Custos-improvement ticket asks for changes to the harness's own source
# under /app/src, which agents cannot reach -- permissions.
# check_within_workspace confines them to /workspace by design -- so an
# agent spent half a day searching for a file it could never open. The
# product-owner had no way to know that, and would have kept assigning.
HOLD_KEY = "dispatch_hold"

# A project whose work means changing the harness's own source. Such a
# ticket can never be done by an ordinary agent -- agents are confined to
# their project workspace and the harness source is not in it -- so it is
# routed to the self-modification pipeline instead of being handed to a
# seat. Set as `target=harness` metadata on the project.
TARGET_KEY = "target"
HARNESS_TARGET = "harness"


def targets_harness(ticket_id: str) -> bool:
    from . import toolchain

    try:
        project = beads.show(toolchain.project_id_for(ticket_id))
    except Exception:
        return False
    return (project.get("metadata") or {}).get(TARGET_KEY) == HARNESS_TARGET


# Held reasons live on the project, but `bd list` does not return
# metadata (only `bd show` and `bd ready` do), so resolving them costs a
# show per project. Cached briefly: dispatch polls every 10s and holds
# change roughly never.
_HOLD_TTL = 60.0
_hold_cache: dict = {"at": -1e9, "held": {}}


def held_projects() -> dict[str, str]:
    """Project id -> hold reason, for every project currently held.

    Read straight out of the single `bd list --all`, which carries each
    issue's metadata (verified live -- both real holds, workspace-9jg and
    workspace-r2w, are visible in `bd list`, not only `bd show`). This used
    to be one `show` per project on top of the list: an N+1 that cost ~100s
    against the real 284-issue Dolt workspace at ~5s a call, and made the
    first dispatch tick after every restart or retry take over a minute."""
    now = time.monotonic()
    if now - _hold_cache["at"] < _HOLD_TTL:
        return _hold_cache["held"]

    held = {}
    for issue in beads.list_all():
        if "." in issue["id"]:
            continue
        reason = (issue.get("metadata") or {}).get(HOLD_KEY)
        if reason:
            held[issue["id"]] = reason
    _hold_cache.update(at=now, held=held)
    return held


def project_hold(ticket_id: str) -> str | None:
    """The reason this ticket's project is held out of dispatch, if it is.

    Fails open like the toolchain preflight: an unreadable project must
    not become a reason nothing ever runs."""
    from . import toolchain

    return held_projects().get(toolchain.project_id_for(ticket_id))


# A ticket with open children cannot be closed (bd refuses), so it is NOT the
# actionable front -- its children are. Without this the front stayed on the
# parent, the parent was restarted and parked forever, and its subtasks
# (later ids, so behind the front gate) were never dispatched at all:
# workspace-o0n.3.2 and 4.2 both wedged exactly this way (2026-09-19/20).
_OPEN_PARENTS_TTL = 60.0
_open_parents_cache: dict = {"at": -1e9, "ids": set()}


def _parents_with_open_children() -> set[str]:
    """Ids that have at least one non-closed direct child.

    Derived from the id hierarchy (`<parent>.<seg>`) -- the same convention
    toolchain.project_id_for and api._parent_id rely on -- so this is one
    `bd list --all`, cached briefly, rather than a `children_of` per node."""
    now = time.monotonic()
    if now - _open_parents_cache["at"] < _OPEN_PARENTS_TTL:
        return _open_parents_cache["ids"]
    ids: set[str] = set()
    for issue in beads.list_all():
        if issue.get("status") == "closed":
            continue
        iid = issue.get("id", "")
        if "." in iid:
            ids.add(iid.rsplit(".", 1)[0])
    _open_parents_cache.update(at=now, ids=ids)
    return ids


def _fronts_from(issues: list) -> dict[str, str]:
    """Project id -> the ticket it must work next, out of `issues`.

    The earliest ticket a project can ACTUALLY BE ACTIONED ON, by roadmap
    order. Nothing past it starts, because a ticket that cannot be actioned
    may be exactly what the ones after it need.

    Eligibility and order are two different things, and conflating them
    wedged a project dead:

    - Blocked-by-a-dependency is not "not actioned", it is not ACTIONABLE,
      and the ticket to work is the thing blocking it -- which can be
      LATER in roadmap order. Found live 2026-09-15: workspace-9jg's front
      was 2.5, blocked by 14.1, so the gate refused every ticket in the
      project -- 14.1 included -- and dispatch went silent. So callers
      pass bd's own answer to "can this be started": `bd ready` (open, no
      open blocker) plus the in_progress orphans. A blocked ticket is in
      neither, and its blocker takes its place.
    - Parked for a human IS the front, and holds the project, which is the
      point of the rule. `bd ready` returns human-labelled issues, so an
      open one is naturally in the running; a closed one never is, and
      verifier-exhausted tickets are closed, so they cannot stall anything
      for good.

    Fails open: a project with no issue in the pool has no front here, and
    _skip allows its candidates. A stall the operator cannot see is worse
    than a ticket starting out of order."""
    front: dict[str, tuple] = {}
    parents_with_children = _parents_with_open_children()
    for issue in issues:
        if issue.get("status") == "closed":
            continue
        if not dispatchable(issue):
            continue  # projects and epics are issues too, and are never worked
        if issue["id"] in parents_with_children:
            # Cannot be closed while its children are open, so it is not the
            # actionable front -- descend: its earliest open child is.
            continue
        project = toolchain.project_id_for(issue["id"])
        key = _order(issue)
        if project not in front or key < front[project][0]:
            front[project] = (key, issue["id"])
    return {project: value[1] for project, value in front.items()}


def roadmap_fronts() -> dict[str, str]:
    """The front of every project, from bd's own pools. See _fronts_from."""
    return _fronts_from(list(beads.in_progress()) + list(beads.ready()))


def next_assigned_ticket(
    busy_seats: set[str] | None = None,
    running_tickets: set[str] | None = None,
) -> tuple[dict | None, str | None]:
    """A ticket the product-owner has already assigned that can start now.

    Orphans first, for the reason worker.py's docstring spells out: `bd
    ready` only ever returns status=open issues, so a ticket left
    in_progress by a crashed agent never reappears there and would be
    stranded forever if this only looked at the ready pool. Human-flagged
    issues are skipped -- they are parked deliberately, not orphaned.

    `busy_seats` (optional) skips tickets whose seat already has a running
    agent. Without it, with MAX_RUNNING_AGENTS > 1, the selector returns
    the ticket it just claimed every cycle: start_agent refuses it as
    already running, tick() reports "could not start", and the
    dispatcher sits at one agent however high the cap is. Found live
    2026-09-13 while raising the cap to 3 -- the log showed a single
    'resuming thread' and nothing else for a full minute.

    `running_tickets` (optional) is what the dispatcher has actually got
    running in-process, and it is how a project stays serialised. Read
    from the dispatcher's own bookkeeping rather than from `bd`: an
    in_progress ticket that nothing is running is a crashed run's orphan,
    and treating THAT as "engaged" would let a stale orphan block the
    project's front ticket -- or, worse, let several orphans resume side
    by side, which is what happened live on restart when 1.2 and 1.3 both
    woke up in the same project."""
    held = held_projects()
    busy = busy_seats or set()
    # Fetched once and reused: these two lists ARE the pools below, and
    # their union is what the front is chosen from.
    in_progress_issues = beads.in_progress()
    ready_issues = beads.ready()
    fronts = _fronts_from(in_progress_issues + ready_issues)
    parents = _parents_with_open_children()
    running_projects = {
        toolchain.project_id_for(ticket_id) for ticket_id in (running_tickets or set())
    }

    def _skip(issue):
        project = toolchain.project_id_for(issue["id"])
        if project in held:
            return True
        if project in running_projects:
            return True  # this project is being worked right now
        front = fronts.get(project)
        # Anything past the front waits, including an orphan: the front
        # ticket may be exactly what it needs.
        return front is not None and issue["id"] != front

    def _pick(issues):
        """Best candidate by roadmap order among this pool. Choosing the min
        rather than bd's first result is what keeps dispatch depth-first by
        epic instead of interleaving epics."""
        best = None
        best_seat = None
        for issue in issues:
            if (
                not dispatchable(issue)
                or issue["id"] in parents
                or beads.is_flagged_for_human(issue)
                or _skip(issue)
            ):
                continue
            seat_id = beads.assigned_seat(issue)
            if not seat_id or seat_id in busy:
                continue
            if best is None or _order(issue) < _order(best):
                best, best_seat = issue, seat_id
        return best, best_seat

    orphaned = _pick(in_progress_issues)
    if orphaned[0] is not None:
        return orphaned
    return _pick(ready_issues)


def next_unassigned_ticket(running_tickets: set[str] | None = None) -> dict | None:
    """The highest-priority ready ticket no seat has been earmarked for
    yet, subject to the same dispatchability and dispatch-hold filters as
    next_assigned_ticket.

    `bd ready` is already priority-sorted (0 = highest), but this does
    not trust that -- it scans and keeps the lowest priority number seen,
    so a change to bd's default ordering cannot silently break the
    preemption check in tick()."""
    held = held_projects()
    ready_issues = beads.ready()
    fronts = _fronts_from(list(beads.in_progress()) + ready_issues)
    parents = _parents_with_open_children()
    running = {
        toolchain.project_id_for(ticket_id) for ticket_id in (running_tickets or set())
    }
    best: dict | None = None
    for issue in ready_issues:
        if not dispatchable(issue):
            continue
        if issue["id"] in parents:
            continue  # has open children -- its children are brokered, not it
        if beads.assigned_seat(issue) is not None:
            continue
        project_id = toolchain.project_id_for(issue["id"])
        if project_id in held or project_id in running:
            continue
        front = fronts.get(project_id)
        # The product-owner is brokering for the project's front ticket and
        # nothing else, or it would assign work the front gate then refuses
        # to start -- and the front is exactly what needs a seat.
        if front is not None and issue["id"] != front:
            continue
        if best is None or _priority(issue) < _priority(best):
            best = issue
    return best


def has_unassigned_work(running_tickets: set[str] | None = None) -> bool:
    """Held projects are excluded so the product-owner is not woken to
    broker work no agent could start. A project's non-front tickets are
    excluded for the same reason: assigning one would only queue behind
    the front gate."""
    return next_unassigned_ticket(running_tickets) is not None


def project_seat(project_id: str) -> str | None:
    """The one seat already working this project, or None if that is not a
    single, unambiguous answer.

    Read from ticket metadata (`assigned_seat`), which `bd list --all`
    returns. Exactly one seat across the project's stories is the common
    case -- a project's specialist. Zero means the project has never been
    seated (creating one is the product-owner's job), and more than one is
    a real choice about which specialist a piece of work belongs to; both
    are left to the product-owner."""
    seen: set[str] = set()
    for issue in beads.list_all():
        if issue.get("issue_type") != WORK_ITEM_TYPE:
            continue
        if toolchain.project_id_for(issue["id"]) != project_id:
            continue
        seat = beads.assigned_seat(issue)
        if seat:
            seen.add(seat)
            if len(seen) > 1:
                return None
    return seen.pop() if seen else None


def assign_front(issue: dict) -> str | None:
    """Assign a project's front ticket to its established seat, with no
    model call. Returns the seat id, or None when the choice needs the
    product-owner.

    The ticket itself is already deterministic -- next_unassigned_ticket
    picks the project's roadmpa front, highest priority, bd-ready. The only
    thing a roadmpa cannot answer is which seat, so this is a fast path for
    the case where the project already has exactly one: assigning it is not
    a judgment call, and skipping the session removes minutes of latency
    and ~40 model calls between every ticket (observed live 2026-09-17).
    A seat that already declined this specific ticket is not eligible --
    the product-owner's own rule -- so that opens the door to a session."""
    seat_id = project_seat(toolchain.project_id_for(issue["id"]))
    if not seat_id:
        return None
    if seat_id in beads.declined_by(issue):
        return None
    try:
        beads.assign_to_seat(issue["id"], seat_id)
    except Exception:
        log.exception("could not auto-assign %s to %s", issue["id"], seat_id)
        return None
    return seat_id



def _priority(issue: dict) -> int:
    """bd-native priority, 0 = highest. A missing or non-int value sorts
    last, so a ticket that has never been triaged can never preempt one
    that has."""
    p = issue.get("priority")
    return p if isinstance(p, int) else 99


def _seg_key(segments: list[str]) -> tuple:
    """Natural id ordering, so `.2` precedes `.10` -- a plain string sort
    puts `.10` first."""
    return tuple((0, int(s)) if s.isdigit() else (1, s) for s in segments)


# Fresh work is chosen in roadmap order: the highest-priority project, then
# its highest-priority epic, then the story. Without this, dispatch drains
# `bd`'s raw ordering, which interleaves epics -- the board showed the first
# epic at 1/6 while later epics were already in flight (2026-09-14). The
# epic's OWN priority is the load-bearing part: Silent Run's stories are all
# P2, so only the epic they hang under distinguishes "finish epic 1 first".
# Cached because it needs the whole issue list (one `bd list --all`, ~5s).
_ORDER_TTL = 60.0
_order_cache: dict = {"at": -1e9, "keys": {}}


def _order_keys() -> dict:
    now = time.monotonic()
    if now - _order_cache["at"] < _ORDER_TTL:
        return _order_cache["keys"]

    issues = beads.list_all()
    prio = {i["id"]: (i.get("priority") if isinstance(i.get("priority"), int) else 9) for i in issues}
    keys = {}
    for issue in issues:
        segments = issue["id"].split(".")
        project_id = segments[0]
        epic_id = ".".join(segments[:2]) if len(segments) >= 2 else project_id
        own = issue.get("priority") if isinstance(issue.get("priority"), int) else 9
        keys[issue["id"]] = (
            prio.get(project_id, 9),          # project priority
            prio.get(epic_id, 9),             # epic priority
            own,                               # the story's own priority
            _seg_key(segments[:2]),            # epic id, naturally
            _seg_key(segments),                # story id, naturally
        )
    _order_cache.update(at=now, keys=keys)
    return keys


def _order(issue: dict) -> tuple:
    key = _order_keys().get(issue["id"])
    if key is not None:
        return key
    return (9, 9, _priority(issue), _seg_key([]), _seg_key(issue["id"].split(".")))


def _outranks(challenger: dict | None, incumbent: dict | None) -> bool:
    """True if `challenger` should be brokered ahead of the already-
    assigned `incumbent`. Strictly-better priority only: at equal
    priority the assigned ticket keeps precedence, so a seat's in-flight
    backlog is drained rather than churned by same-priority preemption."""
    if challenger is None:
        return False
    if incumbent is None:
        return True
    return _priority(challenger) < _priority(incumbent)


def running_agents() -> list[dict]:
    """What is actually being worked right now, derived from Beads rather
    than in-process bookkeeping so it stays true across a dispatcher
    restart and can be read by the API in a different process.

    Human-flagged issues are excluded: they are `in_progress` but parked
    for a person, not being worked (same reasoning as
    worker._next_ticket's own filter)."""
    out = []
    for issue in beads.in_progress():
        if beads.is_flagged_for_human(issue):
            continue
        out.append(
            {
                "seat_id": beads.assigned_seat(issue) or issue.get("assignee"),
                "ticket_id": issue["id"],
                "title": issue.get("title"),
                "started_at": issue.get("started_at"),
            }
        )
    return out


class Dispatcher:
    def __init__(
        self,
        conn_string: str,
        routing: RoutingTable,
        gate: ConcurrencyGate | None = None,
        max_agents: int = MAX_RUNNING_AGENTS,
    ):
        self.conn_string = conn_string
        self.routing = routing
        self.gate = gate or ConcurrencyGate()
        self.max_agents = max_agents
        self._running: dict[str, dict] = {}
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()
        # Built lazily and kept: verifier.build_model constructs a provider
        # client, and verify-on-close runs once per finished ticket.
        self._verifier_model = None

    def capacity(self) -> int:
        with self._lock:
            return self.max_agents - len(self._running)

    def in_flight(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._running)

    def _restore_integration_if_moved(self, issue: dict) -> None:
        """Put the integration branch back if an agent moved it itself.

        The gated merge (verifier.land) is the only legitimate writer; a
        ref that moved without it means unverified work reached the
        integration branch. The rogue commit stays on the ticket's own
        branch, so verification and landing still run normally afterwards.
        """
        try:
            rogue = workspaces.revert_unexpected_integration_move(
                workspaces.project_id_for(issue["id"])
            )
        except Exception:
            log.exception("could not check the integration tip for %s", issue["id"])
            return
        if not rogue:
            return
        log.error(
            "%s: the integration branch was moved outside the gated merge (to %s); "
            "restored the ref -- the work stays on %s",
            issue["id"], rogue[:12], workspaces.ticket_branch(issue["id"]),
        )
        try:
            beads.append_note(
                issue["id"],
                "the integration branch was advanced outside the verifier-gated merge "
                f"(commit {rogue[:12]}); the ref was restored. The work remains on "
                f"{workspaces.ticket_branch(issue['id'])} and is judged normally.",
            )
        except Exception:
            log.exception("could not annotate %s about the restored ref", issue["id"])

    def _agent_thread(self, seat_id: str, issue: dict) -> None:
        """One agent, one ticket. Whatever happens -- success, refusal,
        decline, crash -- the capacity slot is released in `finally`, so a
        failing agent can never wedge dispatch at max forever.

        There is deliberately NO stall/response timeout here (user's call,
        2026-08-31). This harness runs against slow local inference where a
        single response can legitimately take a very long time, so an agent
        holding a ticket for hours is expected behaviour, not a hang -- a
        timeout would kill real work to solve a problem this deployment
        does not have. Crashes are the case worth handling, and `finally`
        handles them. Do not add one back without asking."""
        try:
            # Each ticket is worked in its own project's workspace, moved
            # onto the integration tip as it stands now -- every ticket is
            # actioned from the state of the integration branch when it
            # starts. Safe because the merge is gated on verification, so
            # that branch holds accepted work and nothing else.
            workspace_root = workspaces.for_ticket(issue["id"])
            previous = workspaces.reset_for_attempt(issue["id"])
            if previous:
                log.info(
                    "%s: tree reset to the integration tip (was %s)",
                    issue["id"], previous[:12],
                )
            # And the integration checkout itself, before the agent touches
            # anything: a run that was killed outright never got to file
            # what it left in there, and absorb_stray_edits would otherwise
            # hand it to THIS ticket as its own work. See
            # workspaces.discard_stray_edits for the live case.
            try:
                stray = workspaces.discard_stray_edits(issue["id"])
                if stray:
                    log.warning(
                        "%s: cleared %d stray file(s) left in the project root by an "
                        "earlier run that never filed them: %s",
                        issue["id"], len(stray), ", ".join(stray[:8]),
                    )
            except Exception:
                log.exception("could not clear stray files for %s", issue["id"])
            self._restore_integration_if_moved(issue)
            with PostgresSaver.from_conn_string(self.conn_string) as checkpointer:
                checkpointer.setup()
                with psycopg.connect(self.conn_string, autocommit=True) as conn:
                    runtime = build_seat_runtime(
                        conn, checkpointer, self.routing, seat_id, self.gate,
                        workspace_root=workspace_root,
                    )
                outcome = work_one_ticket(runtime, issue)
            log.info("seat %r finished %s: %s", seat_id, issue["id"], outcome)
            self._record_outcome(issue["id"], outcome)

            # The agent gets a hard denial for moving the integration
            # branch (permissions.forbidden_reason) and this is the
            # mechanical backstop for an evasion: check BEFORE anything is
            # verified, so a ref the agent advanced is put back and the
            # normal verify-then-land path still runs.
            self._restore_integration_if_moved(issue)

            # Judge it before anything else in this project is dispatched.
            if outcome == "closed":
                self.verify_now(issue["id"])

            # The agent's own slot, before it goes back to sleep. Held
            # inside the capacity slot deliberately: it is part of the
            # cycle, not something squeezed in beside the next ticket.
            # A failed run does not get one -- there is nothing to
            # reflect on and the ticket is about to be retried.
            if outcome != "failed":
                reflection.reflect(self.conn_string, runtime, issue, outcome)
        except Exception:
            log.exception("seat %r crashed on %s", seat_id, issue["id"])
            self._record_outcome(issue["id"], "failed")
        finally:
            with self._lock:
                self._running.pop(seat_id, None)

    def _record_outcome(self, ticket_id: str, outcome: str) -> None:
        """Count consecutive failures per ticket, and stop retrying one
        that keeps dying. A ticket that has exhausted its attempts is
        flagged for a human rather than left to spin -- it is a real
        problem someone needs to see, and the alternative is a wedged
        dispatcher."""
        if outcome != "failed":
            with self._lock:
                self._failures.pop(ticket_id, None)
            return

        with self._lock:
            count = self._failures.get(ticket_id, 0) + 1
            self._failures[ticket_id] = count

        if count < MAX_TICKET_FAILURES:
            log.warning("%s failed (%s/%s)", ticket_id, count, MAX_TICKET_FAILURES)
            return

        log.error("%s failed %s times -- flagging for a human", ticket_id, count)
        try:
            beads.flag_for_human(
                ticket_id,
                f"agent run failed {count} times in a row -- see the harness log for the "
                f"traceback. Not retried further to avoid holding an agent slot.",
            )
        except Exception:
            log.exception("could not flag %s", ticket_id)

    def verifier_model(self):
        if self._verifier_model is None:
            self._verifier_model = build_verifier_model()
        return self._verifier_model

    def verify_now(self, ticket_id: str) -> dict | None:
        """Judge a ticket that has just closed, here and now.

        The verifier is otherwise a scheduler job on a 1800s cadence, and
        dispatch is serialised per project behind "the previous ticket is
        finished". Leaving verification to the schedule would therefore
        idle a project for up to half an hour between tickets -- so it
        runs here instead, as soon as the ticket closes.

        This is the same judgement, not a shortcut around it: the same
        verifier.verify_ticket, the same prompt, the same separate model
        call reading the measured evidence -- still a different agent
        from the one that did the work, which is the property that
        matters (see verifier.py's docstring). A fail still requeues the
        ticket with the finding, so the project stays engaged and gets
        worked again rather than the next ticket jumping the queue.

        A verification that errors is logged and left to the scheduled
        job: it must not fail the ticket, whose work is committed and
        merged, nor kill the dispatch thread."""
        try:
            with psycopg.connect(self.conn_string, autocommit=True) as conn:
                verifications.init_table(conn)
                result = verify_ticket(conn, ticket_id, self.verifier_model())
        except Exception:
            log.exception(
                "could not verify %s on close; the scheduled verifier will retry", ticket_id
            )
            return None
        if result:
            log.info(
                "verified %s: %s -- %s",
                ticket_id, result["verdict"], result["reasoning"][:200],
            )
        return result

    def start_agent(self, seat_id: str, issue: dict) -> bool:
        with self._lock:
            if len(self._running) >= self.max_agents:
                return False
            if seat_id in self._running:
                # One ticket at a time per seat. Two threads for the same
                # seat would race to claim and resume the same LangGraph
                # thread -- the race worker.py's docstring already warns
                # about for concurrent same-seat processes.
                return False
            self._running[seat_id] = {
                "ticket_id": issue["id"],
                "title": issue.get("title"),
                "started_at": time.time(),
            }

        # Claim synchronously, before the thread starts. `bd ready` only
        # returns open issues, so claiming is what takes this ticket out
        # of the pool -- without it the next cycle would see it as still
        # available, and running_agents() (which reads in_progress) would
        # report nothing running while an agent was mid-ticket.
        try:
            beads.claim(issue["id"], actor=seat_id)
        except Exception:
            log.exception("seat %r could not claim %s", seat_id, issue["id"])
            with self._lock:
                self._running.pop(seat_id, None)
            return False

        log.info("starting seat %r on %s: %s", seat_id, issue["id"], issue.get("title"))
        threading.Thread(
            target=self._agent_thread, args=(seat_id, issue), daemon=True
        ).start()
        return True

    def start_self_modification(self, issue: dict, seat_id: str) -> bool:
        """Work a harness-source ticket by proposing a change to the
        harness's own source, rather than by editing files in a project
        workspace.

        The agent still only proposes. The trusted loop in sandbox-runner
        (run_self_mod_loop.py) sandboxes, and deploys once reviewed --
        this side never touches Docker, so routing work here grants the
        agent nothing it did not already have."""
        with self._lock:
            if len(self._running) >= self.max_agents or seat_id in self._running:
                return False
            self._running[seat_id] = {
                "ticket_id": issue["id"],
                "title": issue.get("title"),
                "started_at": time.time(),
            }
        try:
            beads.claim(issue["id"], actor=seat_id)
        except Exception:
            log.exception("could not claim %s", issue["id"])
            with self._lock:
                self._running.pop(seat_id, None)
            return False

        log.info("routing %s to self-modification", issue["id"])
        threading.Thread(
            target=self._self_mod_thread, args=(seat_id, issue), daemon=True
        ).start()
        return True

    def _self_mod_thread(self, seat_id: str, issue: dict) -> None:
        from . import self_mod_ticket

        try:
            self_mod_ticket.work_ticket(self.conn_string, self.routing, self.gate, issue)
        except Exception:
            log.exception("self-modification failed for %s", issue["id"])
        finally:
            with self._lock:
                self._running.pop(seat_id, None)

    def wake_product_owner(self) -> str | None:
        """One dispatch-only product-owner session: pick a ticket, decide
        existing seat vs new specialist, assign it. Returns its closing
        message for the log, or None if it failed -- a product-owner error
        must not kill the loop, the next cycle just tries again."""
        try:
            with psycopg.connect(self.conn_string, autocommit=True) as conn:
                seats.init_table(conn)
                requesting_model = RoutedModel(PRODUCT_OWNER_ROLE, self.routing, self.gate)
                tools = build_product_owner_tools(conn, requesting_model)
                agent_model = RoutedModel(
                    PRODUCT_OWNER_ROLE, self.routing, self.gate, tools=tools
                )
                with PostgresSaver.from_conn_string(self.conn_string) as checkpointer:
                    checkpointer.setup()
                    result = run_triage_session(
                        agent_model,
                        tools,
                        checkpointer,
                        thread_id=f"dispatch-{int(time.time())}",
                        brief=DISPATCH_BRIEF,
                    )
            return result["final_message"]
        except Exception:
            log.exception("product-owner dispatch session failed")
            return None

    def _start_assigned(self, issue: dict, seat_id: str) -> str:
        """Run one already-assigned ticket, applying the preflight gates.

        Preflight exists because of a real failure -- agents were
        dispatched onto TypeScript tickets in an image with no Node,
        burned hours of inference, and produced work nothing could build
        or test. Better to refuse loudly than to look busy. Split out of
        tick() so the same gates apply on both paths that reach it: the
        normal one, and the fallback when nothing unassigned outranks
        this ticket."""
        if targets_harness(issue["id"]):
            # Harness-source work goes through self-modification, not
            # through a workspace agent that structurally cannot do it.
            return "self-mod" if self.start_self_modification(issue, seat_id) else "could not start"

        hold = project_hold(issue["id"])
        if hold:
            log.error("not starting %s: project on dispatch hold -- %s", issue["id"], hold)
            return "on hold"

        gaps = toolchain.check_ticket(issue["id"])
        if gaps:
            log.error(
                "not starting %s: project toolchain missing %s",
                issue["id"], ", ".join(gaps),
            )
            return "blocked on toolchain"
        return "started" if self.start_agent(seat_id, issue) else "could not start"

    def tick(self) -> str:
        """One cycle. Returns what it did, which makes the loop testable
        without running it."""
        if self.capacity() <= 0:
            return "at capacity"

        with self._lock:
            busy = set(self._running)
            running_tickets = {info["ticket_id"] for info in self._running.values()}
        assigned, seat_id = next_assigned_ticket(busy, running_tickets)

        # Already-assigned work is drained before the product-owner is
        # woken to broker more -- UNLESS something unassigned strictly
        # outranks the best assigned ticket. Without that exception a
        # freshly created high-priority ticket that no seat has been
        # earmarked for sits at the top of `bd ready` indefinitely while
        # lower-priority assigned work runs ahead of it: tick() keeps
        # taking the assigned branch and never reaches wake_product_owner.
        # Observed live 2026-09-04 -- the Silent Run scaffolding ticket
        # (P0, unassigned, rank 1 in `bd ready`) was starved for days
        # behind ~20 pre-assigned P2 stories.
        if assigned is not None:
            challenger = next_unassigned_ticket(running_tickets)
            if not _outranks(challenger, assigned):
                return self._start_assigned(assigned, seat_id)
            log.info(
                "unassigned %s (P%s) outranks assigned %s (P%s) -- brokering it first",
                challenger["id"], _priority(challenger),
                assigned["id"], _priority(assigned),
            )

        # The ticket is already deterministic (the project's roadmpa front,
        # highest priority, bd-ready). Whether a model is needed depends
        # only on the seat: a project with no seat yet (create a
        # specialist) or several (choose between them) is a judgment call,
        # but a project with exactly one seat already working it is not --
        # see assign_front.
        front = next_unassigned_ticket(running_tickets)
        if front is None:
            return "idle"
        picked = assign_front(front)
        if picked:
            log.info(
                "%s (front of %s) assigned to its project seat %s without a "
                "product-owner session",
                front["id"], toolchain.project_id_for(front["id"]), picked,
            )
            return "assigned"

        log.info("capacity free and unassigned work waiting -- waking product-owner")
        message = self.wake_product_owner()
        if message:
            log.info("product-owner: %s", message[:200])
        return "brokered"

    def run_forever(self) -> None:
        beads.ensure_initialized()
        log.info(
            "dispatcher started: max %s agent(s), polling every %ss",
            self.max_agents,
            POLL_SECONDS,
        )
        while True:
            try:
                outcome = self.tick()
            except Exception:
                log.exception("dispatch cycle failed")
                outcome = "error"
            if outcome in ("idle", "at capacity", "could not start", "error"):
                time.sleep(POLL_SECONDS)


def _routing_table() -> RoutingTable:
    from .worker import _routing_table_from_env

    table = _routing_table_from_env()
    table._chains.setdefault(PRODUCT_OWNER_ROLE, [_provider(4000)])
    return table


if __name__ == "__main__":
    Dispatcher(os.environ["DATABASE_URL"], _routing_table()).run_forever()
