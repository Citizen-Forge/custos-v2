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

from . import beads, seats, toolchain, workspaces
from .product_owner import ROLE as PRODUCT_OWNER_ROLE
from .product_owner import build_tools as build_product_owner_tools
from .product_owner import run_triage_session
from .providers import ProviderConfig
from .routing import ConcurrencyGate, RoutedModel, RoutingTable
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
        max_tokens=max_tokens,
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
    """Project id -> hold reason, for every project currently held."""
    now = time.monotonic()
    if now - _hold_cache["at"] < _HOLD_TTL:
        return _hold_cache["held"]

    held = {}
    for issue in beads.list_all():
        if "." in issue["id"]:
            continue
        try:
            project = beads.show(issue["id"])
        except Exception:
            continue
        reason = (project.get("metadata") or {}).get(HOLD_KEY)
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


def next_assigned_ticket() -> tuple[dict | None, str | None]:
    """A ticket the product-owner has already assigned that can start now.

    Orphans first, for the reason worker.py's docstring spells out: `bd
    ready` only ever returns status=open issues, so a ticket left
    in_progress by a crashed agent never reappears there and would be
    stranded forever if this only looked at the ready pool. Human-flagged
    issues are skipped -- they are parked deliberately, not orphaned."""
    held = held_projects()

    def _skip(issue):
        from . import toolchain

        return toolchain.project_id_for(issue["id"]) in held

    for issue in beads.in_progress():
        if not dispatchable(issue) or beads.is_flagged_for_human(issue) or _skip(issue):
            continue
        seat_id = beads.assigned_seat(issue)
        if seat_id:
            return issue, seat_id

    for issue in beads.ready():
        if not dispatchable(issue) or _skip(issue):
            continue
        seat_id = beads.assigned_seat(issue)
        if seat_id:
            return issue, seat_id
    return None, None


def has_unassigned_work() -> bool:
    """Held projects are excluded so the product-owner is not woken to
    broker work no agent could start."""
    from . import toolchain

    held = held_projects()
    return any(
        dispatchable(i)
        and beads.assigned_seat(i) is None
        and toolchain.project_id_for(i["id"]) not in held
        for i in beads.ready()
    )


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

    def capacity(self) -> int:
        with self._lock:
            return self.max_agents - len(self._running)

    def in_flight(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._running)

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
            # Each ticket is worked in its own project's workspace.
            workspace_root = workspaces.for_ticket(issue["id"])
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

    def tick(self) -> str:
        """One cycle. Returns what it did, which makes the loop testable
        without running it."""
        if self.capacity() <= 0:
            return "at capacity"

        issue, seat_id = next_assigned_ticket()
        if issue is not None:
            # Preflight: never start work the environment cannot support.
            # This exists because of a real failure -- agents were
            # dispatched onto TypeScript tickets in an image with no Node,
            # burned hours of inference, and produced work nothing could
            # build or test. Better to refuse loudly than to look busy.
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

        if not has_unassigned_work():
            return "idle"

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
