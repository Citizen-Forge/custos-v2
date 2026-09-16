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
manual activation elsewhere doesn't really apply here. Originally scoped
narrowly -- recurring *work* running automatically, not generated *tool
code*/prompt revisions activating automatically -- but the user extended
that same call later the same session, after watching the reviewer
correctly deny 4/5 real adversarial proposals (see PLAN.md): both
prompts.py's revision-approval step and tool_proposals.py's
approve/reject gate now activate on the agent's own verdict too, no
human step between judgment and effect (meta_agent.py/reviewer.py's own
docstrings have the detail).

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

from harness import beads, progress, prompts, seats, settings, slack, tool_proposals, verifications
from harness.meta_agent import propose_prompt_update
from harness.overwatch import ROLE as OVERWATCH_ROLE
from harness.overwatch import build_tools as build_overwatch_tools
from harness.overwatch import run_overwatch_session
from harness.product_owner import ESCALATION_BRIEF
from harness.product_owner import ROLE as PRODUCT_OWNER_ROLE
from harness.product_owner import build_escalation_tools
from harness.product_owner import build_tools as build_product_owner_tools
from harness.product_owner import run_triage_session
from harness.providers import ProviderConfig
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable
from harness.verifier import awaiting_verdict, verify_ticket

log = logging.getLogger("scheduler")
logging.basicConfig(level=logging.INFO)

DEFAULT_INTERVAL_SECONDS = 1800


def _provider(name: str, max_tokens: int) -> ProviderConfig:
    # Every field falls back to LOCAL_MODEL_* so that pointing the worker
    # at a new provider moves the scheduler's jobs with it by default --
    # SCHEDULER_MODEL_* exists to opt one of them back out, not to have to
    # be set. The api_key fallback was missing until 2026-09-12: with the
    # worker moved to DeepSeek, base_url and model followed correctly and
    # the key did not, so every verifier/product-owner/overwatch call
    # would have authenticated as "not-needed" and 401'd.
    return ProviderConfig(
        name=name,
        base_url=os.environ.get(
            "SCHEDULER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
        ),
        model=os.environ.get("SCHEDULER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("SCHEDULER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=max_tokens,
        # See ProviderConfig.extra_body: a thinking model demands its
        # `reasoning_content` be replayed on the next turn and
        # langchain_openai does not round-trip it. The verifier makes a
        # single call and would survive; product_owner and overwatch are
        # multi-turn tool loops and would 400 on their second turn.
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get("SCHEDULER_MODEL_DISABLE_THINKING", os.environ.get("LOCAL_MODEL_DISABLE_THINKING"))
            else None
        ),
    )


def _fallback_provider(name: str, max_tokens: int) -> ProviderConfig | None:
    """The second entry of every scheduler chain, or None when no
    fallback is configured.

    Mirrors worker.py's LOCAL_FALLBACK_* handling so that "the model the
    scheduler falls back to" and "the model the worker falls back to" are
    one setting, not two that can drift apart. SCHEDULER_FALLBACK_*
    overrides it per-service the same way SCHEDULER_MODEL_* overrides the
    primary.
    """
    base_url = os.environ.get("SCHEDULER_FALLBACK_BASE_URL", os.environ.get("LOCAL_FALLBACK_BASE_URL"))
    if not base_url:
        return None
    return ProviderConfig(
        name=f"{name}-fallback",
        base_url=base_url,
        model=os.environ.get(
            "SCHEDULER_FALLBACK_MODEL_NAME", os.environ.get("LOCAL_FALLBACK_MODEL_NAME", "gemini-2.0-flash")
        ),
        api_key=os.environ.get("SCHEDULER_FALLBACK_API_KEY", os.environ.get("LOCAL_FALLBACK_API_KEY")),
        concurrency_limit=int(
            os.environ.get("SCHEDULER_FALLBACK_CONCURRENCY", os.environ.get("LOCAL_FALLBACK_CONCURRENCY", "4"))
        ),
        max_tokens=max_tokens,
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get(
                "SCHEDULER_FALLBACK_DISABLE_THINKING", os.environ.get("LOCAL_FALLBACK_DISABLE_THINKING")
            )
            else None
        ),
    )


def _chain(name: str, max_tokens: int) -> list[ProviderConfig]:
    """Primary plus fallback, the shape RoutingTable expects.

    Added 2026-09-12, when the scheduler's primary moved off the LAN to a
    paid API: until then every job here was a ONE-ENTRY chain, so any
    refusal from the single provider -- an expired key, exhausted credits
    (402), an outage -- did not degrade the scheduler, it stopped it.
    That is worse here than in the worker, because these jobs are the
    system's own feedback loop: no verifier means tickets close unchecked
    and no product-owner means nothing gets dispatched at all, and neither
    failure is loud.
    """
    chain = [_provider(name, max_tokens)]
    fallback = _fallback_provider(name, max_tokens)
    if fallback:
        chain.append(fallback)
    return chain


def _routed(role: str, name: str, max_tokens: int, tools=None) -> RoutedModel:
    """A `.invoke()`-compatible model over the full chain, for the jobs
    that used to call build_chat_model directly and so could not fail
    over at all. Each job builds its own RoutingTable, which means
    cooldown state is per-job rather than shared -- acceptable because a
    cycle runs them in sequence, and deliberate rather than incidental:
    one job cooling a provider down should not blind the next job to a
    provider that may have recovered by the time it runs.
    """
    return RoutedModel(role, RoutingTable({role: _chain(name, max_tokens)}), ConcurrencyGate(), tools=tools)


def run_product_owner_job(conn_string: str) -> None:
    # Triage's entire tool set acts on the unassigned queue (list_seats,
    # list_unassigned_tickets, assign_ticket, request_new_seat), so with
    # nothing unassigned it can only report that it found nothing -- which
    # it did, every 30 minutes, for the 3.5h the front gate held
    # workspace-9jg (2026-09-16): "Triage complete -- nothing to do", while
    # the account drained.
    #
    # This is narrower than the cycle-wide _nothing_to_do on purpose, and it
    # cannot lose work: it skips only when the dispatcher's own brokering
    # queue is empty. A parked ticket is the case the two disagree on -- it
    # gives the escalation role work while leaving triage with none.
    from harness import dispatcher

    if not dispatcher.has_unassigned_work():
        log.info("product-owner: nothing unassigned to broker; skipping the session")
        return

    routing = RoutingTable({PRODUCT_OWNER_ROLE: _chain("product-owner", 4000)})
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
    routing = RoutingTable({OVERWATCH_ROLE: _chain("overwatch", 6000)})
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
    model = _routed("meta-agent", "meta-agent", 4000)
    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)
        verifications.init_table(conn)
        for seat in seats.list_all(conn):
            result = propose_prompt_update(conn, seat["seat_id"], model)
            if result:
                log.info("meta-agent activated a revision for %s (v%s)", seat["seat_id"], result["version"])


def run_verifier_job(conn_string: str) -> None:
    model = _routed("verifier", "verifier", 6000)
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


def run_escalations_job(conn_string: str) -> None:
    """Product-owner session over the escalation queue -- tickets other
    agents parked for a human. Skipped entirely when the queue is empty
    (no model call), and bounded per ticket by escalations.MAX_ESCALATION_ATTEMPTS."""
    from harness import escalations

    pending = escalations.pending()
    if not pending:
        log.info("escalations: none pending")
        return

    routing = RoutingTable({PRODUCT_OWNER_ROLE: _chain("product-owner", 6000)})
    gate = ConcurrencyGate()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)
        settings.init_table(conn)
        verifications.init_table(conn)
        requesting_model = RoutedModel(PRODUCT_OWNER_ROLE, routing, gate)
        tools = build_escalation_tools(conn, requesting_model)
        agent_model = RoutedModel(PRODUCT_OWNER_ROLE, routing, gate, tools=tools)
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_triage_session(
                agent_model, tools, checkpointer, brief=ESCALATION_BRIEF
            )
    log.info("escalations: %s", result["final_message"][:300])


def run_progress_job(conn_string: str) -> None:
    """Check whether running agents are actually getting anywhere.

    Cadence: this runs on the scheduler's own cycle (default 1800s)
    rather than carrying a separate hourly timer, because the frequency
    that matters is progress.STALL_AFTER_SECONDS (default 3600), not how
    often the check runs. The cheap signal -- newest checkpoint timestamp
    -- is one indexed query, so running it every cycle costs nothing, and
    the model is only consulted about an agent that has taken no graph
    step for an hour. Checking twice an hour while escalating after an
    hour is strictly better than checking hourly, at the same inference
    cost.

    Deliberately does NOT flag stalled tickets for human review by
    default. flag_for_human has a side effect that would be a nasty
    surprise here: both worker._next_ticket and
    dispatcher.next_assigned_ticket skip flagged issues, so flagging
    parks the ticket AND frees the seat -- killing an agent's work on a
    clock, which is precisely the timeout this design rejects. Set
    STALL_FLAGS_FOR_HUMAN=true to opt in."""
    model = _routed("progress", "progress", 2000)
    beads.ensure_initialized()
    flag = os.environ.get("STALL_FLAGS_FOR_HUMAN", "").lower() in ("1", "true", "yes")

    with psycopg.connect(conn_string, autocommit=True) as conn:
        reports = progress.check_running_agents(conn, conn_string, model=model)

    for report in reports:
        if report["verdict"] == "progressing":
            log.info(
                "progress: %s on %s is working (idle %ss)",
                report["seat_id"], report["ticket_id"], report["idle_seconds"],
            )
            continue

        log.warning(
            "progress: %s on %s looks %s -- %s",
            report["seat_id"], report["ticket_id"], report["verdict"], report["reasoning"],
        )
        slack.post_message(
            f":hourglass: {report['seat_id']} on {report['ticket_id']} "
            f"({report['title']}) looks {report['verdict']} -- {report['reasoning']}"
        )
        # Record it on the ticket once per verdict, so the board shows it
        # without appending a note every single cycle.
        marker = f"progress-check: {report['verdict']}"
        try:
            existing = beads.show(report["ticket_id"]).get("notes") or ""
            if marker not in existing:
                beads.append_note(report["ticket_id"], f"{marker} -- {report['reasoning']}")
            if flag:
                beads.flag_for_human(report["ticket_id"], f"{marker} -- {report['reasoning']}")
        except Exception:
            log.exception("could not annotate %s", report["ticket_id"])


# Jobs that cost a full agent session even when the answer is "nothing to
# do", so a cycle with no work skips them outright (see _nothing_to_do).
# Every job NOT listed here checks for work of its own before it reaches a
# model: escalations returns on an empty queue, the verifier only calls the
# model for a ticket awaiting a verdict, and progress only asks about an
# agent that has taken no graph step for an hour.
CYCLE_GATED_JOBS = {"product_owner", "overwatch", "meta_agent"}


def _nothing_to_do(conn_string: str) -> bool:
    """True when this cycle's model-backed jobs have nothing to act on.

    Three independent things give a cycle a reason to run, and all three
    are readable without a model call:

    - work the dispatcher could start, which the product-owner brokers;
    - a ticket parked for a human, which the escalation role answers;
    - a closed ticket carrying acceptance criteria that no verdict covers
      for the commit that closed it, which the verifier judges.

    When none of them holds, every remaining session can only report that
    it found nothing -- and each report is a full agent round trip. Found
    live 2026-09-16: one parked ticket held its whole project for 3.5h (46
    dispatchable tickets frozen behind the front gate) while the scheduler
    went on running product_owner, overwatch and meta_agent every 30
    minutes -- the product-owner logging "Triage complete -- nothing to do"
    each time, and the account draining for no work.

    Fails OPEN, the same posture as the dispatcher's toolchain preflight and
    for the same reason: if the board cannot be read, run the cycle. A cycle
    nobody needed is cheaper than a job that silently never runs again."""
    try:
        # Imported here rather than at module scope: this module is imported
        # by tests that only exercise run_one_cycle, and neither of these is
        # needed to reach it.
        from harness import dispatcher, escalations

        if dispatcher.has_unassigned_work():
            return False
        if escalations.pending():
            return False
        with psycopg.connect(conn_string, autocommit=True) as conn:
            seats.init_table(conn)
            verifications.init_table(conn)
            for seat in seats.list_all(conn):
                for issue in beads.list_by_assignee(seat["seat_id"]):
                    if awaiting_verdict(conn, issue):
                        return False
        return True
    except Exception:
        log.exception("could not tell whether this cycle has work; running it")
        return False


JOBS = [
    # Escalations first, and deliberately. A ticket parked for a human is
    # its project's FRONT, and the front gate holds every other ticket in
    # that project until it is dealt with (dispatcher._fronts_from). Running
    # this job LAST put five agent sessions' worth of latency and spend
    # between a blocked project and the one job that can clear it -- so a
    # project whose front escalated waited the better part of an hour to be
    # looked at, and its whole backlog waited with it. Found live
    # 2026-09-16: workspace-9jg.1.1 had been parked for 3.5h with 46
    # dispatchable tickets behind it, and the cycle never reached this job.
    ("escalations", run_escalations_job),
    ("product_owner", run_product_owner_job),
    ("overwatch", run_overwatch_job),
    ("meta_agent", run_meta_agent_job),
    ("verifier", run_verifier_job),
    ("progress", run_progress_job),
]


def run_one_cycle(conn_string: str, jobs=JOBS) -> None:
    """One pass through every job, in order. Split out from main()'s
    infinite loop specifically so it's testable without needing to
    actually run forever -- a test can call this once with fake jobs and
    assert on what happened.

    The cycle is asked once, up front, whether it has any work at all (see
    _nothing_to_do), and a cycle that does not skips its model-backed jobs.
    Asked only when the cycle actually contains one, so a caller passing
    its own jobs -- as the tests do -- never pays for the lookup."""
    idle = any(name in CYCLE_GATED_JOBS for name, _ in jobs) and _nothing_to_do(conn_string)
    for name, job in jobs:
        if idle and name in CYCLE_GATED_JOBS:
            log.info("job %s skipped: this cycle has nothing to do", name)
            continue
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
