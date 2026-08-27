"""
Phase 1/2 queue worker -- polls Beads (not a bespoke queue table) for
work, advances the matching LangGraph thread one `invoke` cycle through a
routed model (routing.py: fallback chain + per-provider concurrency cap),
and closes the Beads issue on success.

Two Beads queries drive polling, because `bd ready` only ever returns
`status=open` issues (see beads.py's module docstring) -- an issue left
`in_progress` by a crashed run never reappears there:

1. `beads.in_progress()` first -- orphaned work from a crashed run, safe
   to just resume in Phase 1's single-worker scope (this worker is the
   only writer). Multiple concurrent workers reclaiming the same
   `in_progress` issue is a real race this doesn't guard against yet --
   needed before Phase 2 concurrency trusts this with more than one
   worker process (note: this is about multiple *worker processes*, a
   separate concern from routing.py's per-provider concurrency cap, which
   already works today).
2. `beads.ready()` otherwise -- new work, claimed before starting.

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

from . import beads, prompts
from .classifier import build_classifier_from_model
from .graph import build_graph_from_model
from .providers import ProviderConfig
from .routing import ConcurrencyGate, RoutedModel, RoutingTable
from .tools import ALL_TOOLS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

POLL_INTERVAL_SECONDS = int(os.environ.get("WORKER_POLL_INTERVAL", "5"))

# Same string used as: the RoutingTable role, the Beads --actor on claim
# (so outcomes.py's per-actor tracking means something), and the
# prompts.py role whose active system prompt gets injected. One identity,
# not three coincidentally-matching strings.
WORKER_ROLE = "worker"


def _chain_from_env(env_prefix: str, default_base_url: str, default_model: str) -> list[ProviderConfig]:
    chain = [
        ProviderConfig(
            name=f"{env_prefix.lower()}-primary",
            base_url=os.environ.get(f"{env_prefix}_MODEL_BASE_URL", default_base_url),
            model=os.environ.get(f"{env_prefix}_MODEL_NAME", default_model),
            api_key=os.environ.get(f"{env_prefix}_MODEL_API_KEY"),
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
            )
        )
    return chain


def _routing_table_from_env() -> RoutingTable:
    # CLASSIFIER_* defaults to whatever LOCAL_* resolves to, so an
    # unconfigured classifier chain quietly uses the same model as the
    # worker rather than failing -- override CLASSIFIER_MODEL_* to point
    # it at a smaller/faster model (v1 used qwen2.5:3b-instruct for this).
    local_base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
    local_model = os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
    return RoutingTable(
        {
            "worker": _chain_from_env("LOCAL", local_base_url, local_model),
            "classifier": _chain_from_env("CLASSIFIER", local_base_url, local_model),
        }
    )


def _next_ticket() -> dict | None:
    # Human-flagged issues (Phase 4's refuse_ticket tool) are excluded
    # here on purpose: they're `in_progress` but intentionally parked, not
    # orphaned by a crash. Without this filter a refused ticket would get
    # reclaimed and re-run (and presumably re-refused) every poll forever.
    orphaned = [i for i in beads.in_progress() if not beads.is_flagged_for_human(i)]
    if orphaned:
        return orphaned[0]

    candidates = beads.ready()
    if not candidates:
        return None

    return beads.claim(candidates[0]["id"], actor=WORKER_ROLE)


def run(
    conn_string: str,
    routing: RoutingTable,
    gate: ConcurrencyGate | None = None,
    turn_budget: int | None = None,
) -> None:
    beads.ensure_initialized()
    gate = gate or ConcurrencyGate()

    worker_model = RoutedModel("worker", routing, gate, tools=ALL_TOOLS)
    classify = build_classifier_from_model(RoutedModel("classifier", routing, gate))

    with psycopg.connect(conn_string, autocommit=True) as prompt_conn:
        prompts.init_table(prompt_conn)
        system_prompt = prompts.get_active(prompt_conn, WORKER_ROLE)

    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        graph = build_graph_from_model(worker_model, checkpointer, classify=classify, turn_budget=turn_budget)

        log.info("worker started, polling every %ss", POLL_INTERVAL_SECONDS)
        while True:
            issue = _next_ticket()
            if issue is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            thread_id = issue["id"]
            config = {"configurable": {"thread_id": thread_id}}

            try:
                state = graph.get_state(config)
                if state.values:
                    log.info("resuming thread %s", thread_id)
                    graph.invoke(None, config)
                else:
                    log.info("starting thread %s", thread_id)
                    context = beads.prime()
                    prompt = (
                        f"{context}\n\n---\n\nTicket: {issue['title']}\n\n"
                        f"{issue.get('description', '')}"
                    )
                    initial_messages = (
                        [("system", system_prompt)] if system_prompt else []
                    ) + [("user", prompt)]
                    graph.invoke(
                        {"messages": initial_messages, "ticket_id": thread_id, "turn_count": 0},
                        config,
                    )

                # refuse_ticket already flagged+annotated the issue -- don't
                # also close it, that would erase the "needs a human" signal
                # bd human list depends on.
                current = beads.show(thread_id)
                if beads.is_flagged_for_human(current):
                    log.info("thread %s refused, left for human review", thread_id)
                else:
                    beads.close(thread_id)
                    log.info("thread %s complete", thread_id)
            except Exception:
                log.exception("thread %s failed, left in_progress for retry", thread_id)


if __name__ == "__main__":
    conn_string = os.environ["DATABASE_URL"]
    turn_budget_env = os.environ.get("TURN_BUDGET")
    run(conn_string, _routing_table_from_env(), turn_budget=int(turn_budget_env) if turn_budget_env else None)
