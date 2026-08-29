"""
Standalone entrypoint for the product-owner. Deliberately NOT part of any
single seat's ticket worker loop -- a different kind of process, like
scripts/run_meta_agent.py. Meant to run periodically (cron/schedule, not
built here yet), each run doing one session -- triage by default, or
idea decomposition (a rough idea -> an epic + concrete subtasks, Phase 3)
if PRODUCT_OWNER_BRIEF is set.

    docker compose run --rm harness python scripts/run_product_owner.py
    # or, for idea decomposition instead of triage:
    docker compose run --rm -e PRODUCT_OWNER_BRIEF="..." harness python scripts/run_product_owner.py

Env: PRODUCT_OWNER_MODEL_BASE_URL/NAME/API_KEY (defaults to the same
local model as LOCAL_MODEL_*, though PLAN.md's original intent is a
frontier model here once one's configured), PRODUCT_OWNER_BRIEF (optional,
overrides the default "triage the queue" instruction).
"""

import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from harness import prompts, seats
from harness.product_owner import ROLE, build_tools, run_triage_session
from harness.providers import ProviderConfig
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable


def _routing_table_from_env() -> RoutingTable:
    base_url = os.environ.get(
        "PRODUCT_OWNER_MODEL_BASE_URL",
        os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
    )
    model = os.environ.get(
        "PRODUCT_OWNER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
    )
    api_key = os.environ.get("PRODUCT_OWNER_MODEL_API_KEY")
    return RoutingTable(
        {
            ROLE: [
                ProviderConfig(
                    name="product-owner",
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    # See ProviderConfig.max_tokens -- a triage session can
                    # run several tool calls; generous but still bounded.
                    max_tokens=int(os.environ.get("PRODUCT_OWNER_MAX_TOKENS", "4000")),
                )
            ]
        }
    )


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    routing = _routing_table_from_env()
    gate = ConcurrencyGate()
    brief = os.environ.get("PRODUCT_OWNER_BRIEF")

    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)

        # tools=None here: a plain reasoning caller for request_new_seat's
        # delegation to meta_agent.create_specialist_seat, distinct from
        # the tool-bound agent_model below (see product_owner.build_tools).
        requesting_model = RoutedModel(ROLE, routing, gate)
        tools = build_tools(conn, requesting_model)
        agent_model = RoutedModel(ROLE, routing, gate, tools=tools)

        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_triage_session(agent_model, tools, checkpointer, brief=brief)

    print(f"product-owner session {result['thread_id']}: {result['final_message']}")


if __name__ == "__main__":
    main()
