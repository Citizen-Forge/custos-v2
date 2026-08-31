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

from . import beads, seats
from .product_owner import ROLE as PRODUCT_OWNER_ROLE
from .product_owner import build_tools as build_product_owner_tools
from .product_owner import run_triage_session
from .providers import ProviderConfig
from .routing import ConcurrencyGate, RoutedModel, RoutingTable
from .worker import build_seat_runtime, work_one_ticket

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dispatcher")

MAX_RUNNING_AGENTS = int(os.environ.get("MAX_RUNNING_AGENTS", "1"))
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


def next_assigned_ticket() -> tuple[dict | None, str | None]:
    """A ticket the product-owner has already assigned that can start now.

    Orphans first, for the reason worker.py's docstring spells out: `bd
    ready` only ever returns status=open issues, so a ticket left
    in_progress by a crashed agent never reappears there and would be
    stranded forever if this only looked at the ready pool. Human-flagged
    issues are skipped -- they are parked deliberately, not orphaned."""
    for issue in beads.in_progress():
        if not dispatchable(issue) or beads.is_flagged_for_human(issue):
            continue
        seat_id = beads.assigned_seat(issue)
        if seat_id:
            return issue, seat_id

    for issue in beads.ready():
        if not dispatchable(issue):
            continue
        seat_id = beads.assigned_seat(issue)
        if seat_id:
            return issue, seat_id
    return None, None


def has_unassigned_work() -> bool:
    return any(
        dispatchable(i) and beads.assigned_seat(i) is None for i in beads.ready()
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
        failing agent can never wedge dispatch at max forever."""
        try:
            with PostgresSaver.from_conn_string(self.conn_string) as checkpointer:
                checkpointer.setup()
                with psycopg.connect(self.conn_string, autocommit=True) as conn:
                    runtime = build_seat_runtime(
                        conn, checkpointer, self.routing, seat_id, self.gate
                    )
                outcome = work_one_ticket(runtime, issue)
            log.info("seat %r finished %s: %s", seat_id, issue["id"], outcome)
        except Exception:
            log.exception("seat %r crashed on %s", seat_id, issue["id"])
        finally:
            with self._lock:
                self._running.pop(seat_id, None)

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
