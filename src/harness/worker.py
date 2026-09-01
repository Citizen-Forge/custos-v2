"""
Per-seat queue worker -- polls Beads (not a bespoke queue table) for work
assigned to ONE specific seat, advances the matching LangGraph thread one
`invoke` cycle through a routed model, and closes the Beads issue on
success. One process per active seat (`SEAT_ID` env var, default
"worker" for the original single-generalist bootstrap case).

A seat only claims tickets the product-owner has explicitly assigned to
it (`beads.ready_for_seat`), plus its own orphaned in-progress work --
NOT "any ready ticket" anymore. That's the actual mechanism that makes
specialization emerge from product-owner assignment rather than being a
free-for-all where the fastest generic worker grabs everything (see
product_owner.py and PLAN.md's seat-assignment design). Tickets with no
assigned_seat metadata yet just sit in `bd ready` until a product-owner
triage session assigns them -- no seat's worker will touch them.

Two Beads queries drive polling, because `bd ready` only ever returns
`status=open` issues (see beads.py's module docstring) -- an issue left
`in_progress` by a crashed run never reappears there:

1. This seat's own orphaned in-progress work first -- safe to just resume
   in the single-worker-per-seat scope (this process is the only writer
   for its own seat_id). Multiple concurrent processes for the *same*
   seat reclaiming the same issue is a real race this doesn't guard
   against yet -- separate from routing.py's per-provider concurrency
   cap, which already works today regardless.
2. Tickets assigned to this seat and still `ready` otherwise -- claimed
   before starting.

The Beads issue id doubles as the LangGraph `thread_id`, so "resume this
ticket" and "resume this graph thread" are the same operation.

A `RoutedModel` failure (routing.AllProvidersCoolingDown, or every
provider in a chain erroring out) is just another exception here -- it
falls into the same `except Exception` as any other failure, leaving the
ticket `in_progress` for the next poll to retry. That's deliberate: no
special retry/backoff logic is needed at the worker level, because
routing.py's cooldown *is* the backoff, and the worker's poll loop *is*
the retry -- reusing the exact mechanism that already makes crash-resume
safe.
"""

import logging
import os
import time

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from . import beads, prompts, seats, slack
from .classifier import build_classifier_from_model
from .dynamic_tools import build_dynamic_tools
from .graph import build_graph_from_model
from .providers import ProviderConfig
from .routing import ConcurrencyGate, RoutedModel, RoutingTable
from .tools import ALL_TOOLS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

POLL_INTERVAL_SECONDS = int(os.environ.get("WORKER_POLL_INTERVAL", "5"))

# The original single-generalist bootstrap seat -- exists so the system
# is usable before any product-owner triage has created specialist seats.
DEFAULT_SEAT_ID = "worker"


def _chain_from_env(env_prefix: str, default_base_url: str, default_model: str, max_tokens: int | None = None) -> list[ProviderConfig]:
    # env var (if set) overrides the caller's default, same precedence as
    # every other *_MODEL_* setting here.
    env_max_tokens = os.environ.get(f"{env_prefix}_MAX_TOKENS")
    resolved_max_tokens = int(env_max_tokens) if env_max_tokens else max_tokens

    chain = [
        ProviderConfig(
            name=f"{env_prefix.lower()}-primary",
            base_url=os.environ.get(f"{env_prefix}_MODEL_BASE_URL", default_base_url),
            model=os.environ.get(f"{env_prefix}_MODEL_NAME", default_model),
            api_key=os.environ.get(f"{env_prefix}_MODEL_API_KEY"),
            max_tokens=resolved_max_tokens,
        )
    ]
    fallback_base_url = os.environ.get(f"{env_prefix}_FALLBACK_BASE_URL")
    if fallback_base_url:
        chain.append(
            ProviderConfig(
                name=f"{env_prefix.lower()}-fallback",
                base_url=fallback_base_url,
                model=os.environ.get(f"{env_prefix}_FALLBACK_MODEL_NAME", "gemini-2.0-flash"),
                api_key=os.environ.get(f"{env_prefix}_FALLBACK_API_KEY"),
                concurrency_limit=int(os.environ.get(f"{env_prefix}_FALLBACK_CONCURRENCY", "4")),
                max_tokens=resolved_max_tokens,
            )
        )
    return chain


def _routing_table_from_env() -> RoutingTable:
    # CLASSIFIER_* defaults to whatever LOCAL_* resolves to, so an
    # unconfigured classifier chain quietly uses the same model as
    # workers rather than failing -- override CLASSIFIER_MODEL_* to point
    # it at a smaller/faster model (v1 used qwen2.5:3b-instruct for this).
    # default_role="worker": any seat_id not explicitly registered here
    # (i.e. every seat the product-owner creates at runtime) falls back
    # to the shared LOCAL_* chain -- see routing.py's module docstring
    # for why routing is the one place specialization isn't per-seat.
    local_base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
    local_model = os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
    return RoutingTable(
        {
            # Worker turns can legitimately need to write a lot (a diff, a
            # long explanation) -- generous but still bounded, since even
            # a "generous" cap beats no cap at all (see ProviderConfig's
            # max_tokens docstring for what unbounded actually cost us).
            DEFAULT_SEAT_ID: _chain_from_env("LOCAL", local_base_url, local_model, max_tokens=8000),
            # The classifier answers one short JSON verdict on EVERY
            # non-trivial tool call -- the highest-frequency call in the
            # whole system, so keep this tight specifically.
            "classifier": _chain_from_env("CLASSIFIER", local_base_url, local_model, max_tokens=1000),
        },
        default_role=DEFAULT_SEAT_ID,
    )


def _next_ticket(seat_id: str) -> dict | None:
    # Human-flagged issues (Phase 4's refuse_ticket tool) are excluded
    # here on purpose: they're `in_progress` but intentionally parked, not
    # orphaned by a crash. Without this filter a refused ticket would get
    # reclaimed and re-run (and presumably re-refused) every poll forever.
    orphaned = [
        i
        for i in beads.in_progress()
        if i.get("assignee") == seat_id and not beads.is_flagged_for_human(i)
    ]
    if orphaned:
        return orphaned[0]

    candidates = beads.ready_for_seat(seat_id)
    if not candidates:
        return None

    return beads.claim(candidates[0]["id"], actor=seat_id)


class SeatRuntime:
    """One seat instantiated as a working agent: its system prompt, tool
    set, routed model and compiled graph.

    Split out of `run` (2026-08-31) so a seat can be instantiated on
    demand rather than only at process start. That is what lets the
    dispatcher put a seat the product-owner just created straight to
    work, instead of the old shape where a seat with no process simply
    never ran -- the live failure this refactor exists to fix."""

    def __init__(self, seat_id: str, graph, system_prompt: str | None, who: str):
        self.seat_id = seat_id
        self.graph = graph
        self.system_prompt = system_prompt
        self.who = who


def build_seat_runtime(
    prompt_conn,
    checkpointer,
    routing: RoutingTable,
    seat_id: str,
    gate: ConcurrencyGate | None = None,
    turn_budget: int | None = None,
) -> SeatRuntime:
    """Build everything one seat needs to work tickets.

    Deliberately expensive and deliberately reused: this reads the seat's
    prompt and profile and compiles a graph, so callers should hold the
    result across tickets rather than rebuilding per ticket.

    Carries forward the same staleness tradeoff the old inline version
    documented -- a prompt revision or newly approved dynamic tool is
    picked up when a runtime is rebuilt, not mid-life."""
    gate = gate or ConcurrencyGate()

    prompts.init_table(prompt_conn)
    system_prompt = prompts.get_active(prompt_conn, seat_id)
    seats.init_table(prompt_conn)
    seat_record = seats.get(prompt_conn, seat_id)
    who = seat_record["display_name"] if seat_record and seat_record.get("display_name") else seat_id
    tools = ALL_TOOLS + build_dynamic_tools(prompt_conn)

    worker_model = RoutedModel(seat_id, routing, gate, tools=tools)
    classify = build_classifier_from_model(RoutedModel("classifier", routing, gate))
    graph = build_graph_from_model(
        worker_model, checkpointer, tools=tools, classify=classify, turn_budget=turn_budget
    )
    return SeatRuntime(seat_id, graph, system_prompt, who)


def work_one_ticket(runtime: SeatRuntime, issue: dict) -> str:
    """Advance one ticket as far as it goes this run.

    Returns what happened, so a caller driving many seats can act on it:
    'closed', 'flagged' (refused to a human), 'released' (declined as
    out-of-speciality and handed back to the pool) or 'failed'."""
    thread_id = issue["id"]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = runtime.graph.get_state(config)
        if state.values:
            log.info("resuming thread %s", thread_id)
            runtime.graph.invoke(None, config)
        else:
            log.info("starting thread %s", thread_id)
            slack.post_message(
                f":rocket: {runtime.who} is starting work on {thread_id}: {issue['title']}"
            )
            context = beads.prime()
            prompt = (
                f"{context}\n\n---\n\nTicket: {issue['title']}\n\n"
                f"{issue.get('description', '')}"
            )
            initial_messages = (
                [("system", runtime.system_prompt)] if runtime.system_prompt else []
            ) + [("user", prompt)]
            runtime.graph.invoke(
                {"messages": initial_messages, "ticket_id": thread_id, "turn_count": 0},
                config,
            )

        current = beads.show(thread_id)
        # refuse_ticket already flagged+annotated the issue -- don't
        # also close it, that would erase the "needs a human" signal
        # bd human list depends on.
        if beads.is_flagged_for_human(current):
            log.info("thread %s refused, left for human review", thread_id)
            return "flagged"
        # decline_ticket releases the ticket by clearing its seat
        # assignment. Closing it here would record work as done that
        # nobody actually did.
        if beads.assigned_seat(current) != runtime.seat_id:
            log.info("thread %s declined by %r, back in the pool", thread_id, runtime.seat_id)
            return "released"

        # An agent must explicitly claim completion (complete_ticket) and
        # say what it did. This used to close unconditionally whenever the
        # graph finished without refusing, which meant an agent that
        # talked for a while and stopped was recorded as success.
        #
        # That was not hypothetical: on 2026-09-01, four Silent Run and
        # Custos stories were closed this way with no notes, no summary
        # and no code anywhere in the workspace -- including "Ship
        # movement over system-scale distances". A ticket that ends with
        # nothing recorded is indistinguishable from one where nothing
        # happened, so it is now flagged for a human instead of closed.
        summary = (current.get("metadata") or {}).get("completion_summary")
        if not summary:
            log.warning("thread %s ended with no completion claim -- flagging", thread_id)
            beads.flag_for_human(
                thread_id,
                "agent stopped without calling complete_ticket -- no record of what, "
                "if anything, was done",
            )
            return "unclaimed"

        beads.close(thread_id, reason=summary[:500])
        log.info("thread %s complete", thread_id)
        return "closed"
    except Exception:
        log.exception("thread %s failed, left in_progress for retry", thread_id)
        return "failed"


def run(
    conn_string: str,
    routing: RoutingTable,
    seat_id: str = DEFAULT_SEAT_ID,
    gate: ConcurrencyGate | None = None,
    turn_budget: int | None = None,
) -> None:
    """Single-seat poll loop -- the original entrypoint, kept working.
    The dispatcher (dispatcher.py) is the multi-seat path."""
    beads.ensure_initialized()
    gate = gate or ConcurrencyGate()

    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        with psycopg.connect(conn_string, autocommit=True) as prompt_conn:
            runtime = build_seat_runtime(
                prompt_conn, checkpointer, routing, seat_id, gate, turn_budget
            )

        log.info("worker started for seat %r, polling every %ss", seat_id, POLL_INTERVAL_SECONDS)
        while True:
            issue = _next_ticket(seat_id)
            if issue is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            work_one_ticket(runtime, issue)


if __name__ == "__main__":
    conn_string = os.environ["DATABASE_URL"]
    turn_budget_env = os.environ.get("TURN_BUDGET")
    run(
        conn_string,
        _routing_table_from_env(),
        seat_id=os.environ.get("SEAT_ID", DEFAULT_SEAT_ID),
        turn_budget=int(turn_budget_env) if turn_budget_env else None,
    )
