"""
Phase 1 queue worker -- polls Beads (not a bespoke queue table) for work,
advances the matching LangGraph thread one `invoke` cycle, and closes the
Beads issue on success.

Two Beads queries drive polling, because `bd ready` only ever returns
`status=open` issues (see beads.py's module docstring) -- an issue left
`in_progress` by a crashed run never reappears there:

1. `beads.in_progress()` first -- orphaned work from a crashed run, safe
   to just resume in Phase 1's single-worker scope (this worker is the
   only writer). Multiple concurrent workers reclaiming the same
   `in_progress` issue is a real race this doesn't guard against yet --
   needed before Phase 2 concurrency trusts this with more than one
   worker, same caveat as queue_store.py's stale-lease logic had before
   this rewrite replaced it.
2. `beads.ready()` otherwise -- new work, claimed before starting.

The Beads issue id doubles as the LangGraph `thread_id`, so "resume this
ticket" and "resume this graph thread" are the same operation.
"""

import logging
import os
import time

from langgraph.checkpoint.postgres import PostgresSaver

from . import beads
from .classifier import build_classifier
from .graph import build_graph
from .providers import ProviderConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

POLL_INTERVAL_SECONDS = int(os.environ.get("WORKER_POLL_INTERVAL", "5"))


def _provider_from_env() -> ProviderConfig:
    return ProviderConfig(
        name="local",
        base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        model=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"),
        api_key=os.environ.get("LOCAL_MODEL_API_KEY"),
    )


def _classifier_provider_from_env() -> ProviderConfig:
    # v1 used a distinct, smaller/faster model for classification
    # (qwen2.5:3b-instruct) than for general work. Phase 2's routing is
    # where that split gets a real config surface; for now this defaults
    # to the same endpoint as the main model but is already a separate
    # env var so pointing it elsewhere doesn't require code changes.
    return ProviderConfig(
        name="classifier",
        base_url=os.environ.get("CLASSIFIER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")),
        model=os.environ.get("CLASSIFIER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("CLASSIFIER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
    )


def _next_ticket() -> dict | None:
    orphaned = beads.in_progress()
    if orphaned:
        return orphaned[0]

    candidates = beads.ready()
    if not candidates:
        return None

    return beads.claim(candidates[0]["id"])


def run(conn_string: str, provider_cfg: ProviderConfig, classifier_provider_cfg: ProviderConfig) -> None:
    beads.ensure_initialized()
    classifier = build_classifier(classifier_provider_cfg)

    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        graph = build_graph(provider_cfg, checkpointer, classifier=classifier)

        log.info(
            "worker started against %s, polling every %ss",
            provider_cfg.model,
            POLL_INTERVAL_SECONDS,
        )
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
                    graph.invoke(
                        {"messages": [("user", prompt)], "ticket_id": thread_id},
                        config,
                    )
                beads.close(thread_id)
                log.info("thread %s complete", thread_id)
            except Exception:
                log.exception("thread %s failed, left in_progress for retry", thread_id)


if __name__ == "__main__":
    conn_string = os.environ["DATABASE_URL"]
    run(conn_string, _provider_from_env(), _classifier_provider_from_env())
