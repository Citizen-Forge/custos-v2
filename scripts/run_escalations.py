"""
Standalone entrypoint for the product-owner's escalation session.

    docker compose run --rm harness python scripts/run_escalations.py

Handles the tickets other agents parked for a human (the `human` label):
for each, the product-owner either fixes the specification and requeues it,
makes the product decision, or leaves it if it genuinely needs a person.
Bounded per ticket by escalations.MAX_ESCALATION_ATTEMPTS; the scheduler
runs the same session as its `escalations` job.

Env: SCHEDULER_MODEL_BASE_URL/NAME/API_KEY (defaults to LOCAL_MODEL_*) --
the same chain the scheduled job uses.
"""

import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from harness import escalations, prompts, seats, settings, verifications
from harness.product_owner import (
    ESCALATION_BRIEF,
    ROLE,
    build_escalation_tools,
    run_triage_session,
)
from harness.providers import ProviderConfig
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    provider = ProviderConfig(
        name="product-owner",
        base_url=os.environ.get(
            "SCHEDULER_MODEL_BASE_URL",
            os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        ),
        model=os.environ.get("SCHEDULER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("SCHEDULER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=int(os.environ.get("SCHEDULER_MAX_TOKENS", "6000")),
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get("SCHEDULER_MODEL_DISABLE_THINKING", os.environ.get("LOCAL_MODEL_DISABLE_THINKING"))
            else None
        ),
    )

    pending = escalations.pending()
    if not pending:
        print("no escalations pending")
        return
    print(f"{len(pending)} escalation(s) pending: {', '.join(i['id'] for i in pending)}")

    routing = RoutingTable({ROLE: [provider]})
    gate = ConcurrencyGate()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        prompts.init_table(conn)
        seats.init_table(conn)
        settings.init_table(conn)
        verifications.init_table(conn)
        requesting_model = RoutedModel(ROLE, routing, gate)
        tools = build_escalation_tools(conn, requesting_model)
        agent_model = RoutedModel(ROLE, routing, gate, tools=tools)
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_triage_session(agent_model, tools, checkpointer, brief=ESCALATION_BRIEF)

    print(result["final_message"])


if __name__ == "__main__":
    main()
