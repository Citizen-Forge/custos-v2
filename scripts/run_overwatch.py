"""
Standalone entrypoint for the Phase 7 overwatch agent's own pass.
Deliberately NOT part of any single seat's worker loop, same reasoning as
run_product_owner.py / run_meta_agent.py -- a system-level agent, not
ticket work. Proposes at most a handful of tools per session, each
immediately sandboxed for real evidence -- never registered/activated on
its own (see tool_proposals.py's lifecycle and PLAN.md Phase 7's
promotion gate).

    docker compose run --rm harness python scripts/run_overwatch.py
    # or point it at a specific gap directly, since automatic refusal
    # evidence is thin until real usage accumulates:
    docker compose run --rm -e OVERWATCH_BRIEF="..." harness python scripts/run_overwatch.py

Env: OVERWATCH_MODEL_BASE_URL/NAME/API_KEY (defaults to the same local
model as LOCAL_MODEL_*), OVERWATCH_BRIEF (optional, overrides the default
"scan for gaps" instruction with a specific one).
"""

import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from harness import seats, tool_proposals
from harness.overwatch import ROLE, build_tools, run_overwatch_session
from harness.providers import ProviderConfig
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable


def _routing_table_from_env() -> RoutingTable:
    base_url = os.environ.get(
        "OVERWATCH_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
    )
    model = os.environ.get("OVERWATCH_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"))
    api_key = os.environ.get("OVERWATCH_MODEL_API_KEY")
    return RoutingTable(
        {
            ROLE: [
                ProviderConfig(
                    name="overwatch",
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    # See ProviderConfig.max_tokens -- writing real tool
                    # source code needs more room than a short verdict.
                    max_tokens=int(os.environ.get("OVERWATCH_MAX_TOKENS", "6000")),
                )
            ]
        }
    )


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    routing = _routing_table_from_env()
    gate = ConcurrencyGate()
    brief = os.environ.get("OVERWATCH_BRIEF")

    with psycopg.connect(conn_string, autocommit=True) as conn:
        seats.init_table(conn)
        tool_proposals.init_table(conn)

        tools = build_tools(conn)
        agent_model = RoutedModel(ROLE, routing, gate, tools=tools)

        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_overwatch_session(agent_model, tools, checkpointer, brief=brief)

    print(f"overwatch session {result['thread_id']}: {result['final_message']}")


if __name__ == "__main__":
    main()
