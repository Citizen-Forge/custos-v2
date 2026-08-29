"""
Long-running loop that periodically triggers product-owner triage,
meta-agent revision proposals (one per active seat), overwatch
capability-gap scanning, and the acceptance-criteria verifier --
replacing manual `docker compose run` invocations for each of these
(PLAN.md's standing "nothing is scheduled" gap).

Runs as a standard, on-by-default docker-compose service (`docker
compose up` includes it, same as harness/api) -- a deliberate reversal
of v1's "autonomy off by default" posture, per the user's own call
(2026-08-29): this harness is built around a local, unmetered model, so
the cost/risk calculus that justified gating recurring work behind
manual activation elsewhere doesn't really apply here. Scoped narrowly,
though: this is about whether recurring *work* happens automatically,
not whether generated *tool code* activates automatically --
prompts.py's revision-approval step and tool_proposals.py's
approve/reject gate are untouched, since those protect against a bad
change silently taking effect (a real safety boundary), not against
unwanted spend.

Runs each job sequentially every cycle, not concurrently or on
independent timers: this harness's whole design assumes single/low
concurrency against local inference (Phase 1) -- overlapping scheduled
jobs would just queue up on the same inference slot anyway, not actually
run in parallel, so a simple round-robin loop matches the real resource
shape instead of pretending otherwise. One misbehaving job (a real model
call erroring, a network blip) is caught and logged, never kills the
loop -- the next cycle tries again.

Env: SCHEDULER_INTERVAL_SECONDS (default 1800 -- real calls take minutes
each, see this session's own timing data, so anything much shorter would
just mean overlapping/backed-up cycles), SCHEDULER_MODEL_BASE_URL/NAME
(defaults to the same local model as LOCAL_MODEL_*).
"""

import logging
import os
import time

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from harness import beads, prompts, seats, settings, tool_proposals, verifications
from harness.meta_agent import propose_prompt_update
from harness.overwatch import ROLE as OVERWATCH_ROLE
from harness.overwatch import build_tools as build_overwatch_tools
from harness.overwatch import run_overwatch_session
from harness.product_owner import ROLE as PRODUCT_OWNER_ROLE
from harness.product_owner import build_tools as build_product_owner_tools
from harness.product_owner import run_triage_session
from harness.providers import ProviderConfig, build_chat_model
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable
from harness.verifier import verify_ticket

log = logging.getLogger("scheduler")
logging.basicConfig(level=logging.INFO)

DEFAULT_INTERVAL_SECONDS = 1800


def _provider(name: str, max_tokens: int) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url=os.environ.get(
            "SCHEDULER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
        ),
        model=os.environ.get("SCHEDULER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        max_tokens=max_tokens,
    )


def run_product_owner_job(conn_string: str) -> None:
    routing = RoutingTable({PRODUCT_OWNER_ROLE: [_provider("product-owner", 4000)]})
    gate = ConcurrencyGate()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)
        settings.init_table(conn)
        requesting_model = RoutedModel(PRODUCT_OWNER_ROLE, routing, gate)
        tools = build_product_owner_tools(conn, requesting_model)
        agent_model = RoutedModel(PRODUCT_OWNER_ROLE, routing, gate, tools=tools)
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_triage_session(agent_model, tools, checkpointer)
    log.info("product-owner: %s", result["final_message"][:200])


def run_overwatch_job(conn_string: str) -> None:
    routing = RoutingTable({OVERWATCH_ROLE: [_provider("overwatch", 6000)]})
    gate = ConcurrencyGate()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        seats.init_table(conn)
        tool_proposals.init_table(conn)
        tools = build_overwatch_tools(conn)
        agent_model = RoutedModel(OVERWATCH_ROLE, routing, gate, tools=tools)
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_overwatch_session(agent_model, tools, checkpointer)
    log.info("overwatch: %s", result["final_message"][:200])


def run_meta_agent_job(conn_string: str) -> None:
    model = build_chat_model(_provider("meta-agent", 4000))
    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)
        verifications.init_table(conn)
        for seat in seats.list_all(conn):
            result = propose_prompt_update(conn, seat["seat_id"], model)
            if result:
                log.info("meta-agent proposed a revision for %s (v%s)", seat["seat_id"], result["version"])


def run_verifier_job(conn_string: str) -> None:
    model = build_chat_model(_provider("verifier", 6000))
    beads.ensure_initialized()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        seats.init_table(conn)
        verifications.init_table(conn)
        for seat in seats.list_all(conn):
            for issue in beads.list_by_assignee(seat["seat_id"]):
                if issue.get("status") != "closed" or not beads.acceptance_criteria(issue):
                    continue
                result = verify_ticket(conn, issue["id"], model)
                if result:
                    log.info("verified %s: %s", issue["id"], result["verdict"])


JOBS = [
    ("product_owner", run_product_owner_job),
    ("overwatch", run_overwatch_job),
    ("meta_agent", run_meta_agent_job),
    ("verifier", run_verifier_job),
]


def run_one_cycle(conn_string: str, jobs=JOBS) -> None:
    """One pass through every job, in order. Split out from main()'s
    infinite loop specifically so it's testable without needing to
    actually run forever -- a test can call this once with fake jobs and
    assert on what happened."""
    for name, job in jobs:
        try:
            log.info("running job: %s", name)
            job(conn_string)
        except Exception:
            log.exception("job %s failed, continuing", name)


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    interval = int(os.environ.get("SCHEDULER_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)))
    log.info("scheduler started, running %d job(s) every %ss", len(JOBS), interval)
    while True:
        run_one_cycle(conn_string)
        time.sleep(interval)


if __name__ == "__main__":
    main()
