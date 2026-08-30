"""
Standalone entrypoint for the Phase 7 self-modification agent.
Deliberately NOT part of the scheduler's automatic loop (unlike product-
owner/meta-agent/overwatch/verifier) -- this is new, unproven, and the
highest blast-radius agent in the system, so it stays manually invoked
until it's earned more trust. Proposes at most one real change per
session; sandboxing, review, and deployment all happen as later,
separate steps (see self_mod.py's lifecycle).

    docker compose run --rm harness python scripts/run_self_modifier.py
    # or point it at a specific improvement directly:
    docker compose run --rm -e SELF_MOD_BRIEF="..." harness python scripts/run_self_modifier.py

Env: SELF_MOD_MODEL_BASE_URL/NAME/API_KEY (defaults to the same local
model as LOCAL_MODEL_*), SELF_MOD_BRIEF (optional).
"""

import os

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from harness import self_mod
from harness.providers import ProviderConfig
from harness.routing import ConcurrencyGate, RoutedModel, RoutingTable
from harness.self_modifier import ROLE, build_tools, run_self_mod_session


def _routing_table_from_env() -> RoutingTable:
    base_url = os.environ.get(
        "SELF_MOD_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
    )
    model = os.environ.get("SELF_MOD_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"))
    api_key = os.environ.get("SELF_MOD_MODEL_API_KEY")
    return RoutingTable(
        {
            ROLE: [
                ProviderConfig(
                    name="self-modifier",
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    max_tokens=int(os.environ.get("SELF_MOD_MAX_TOKENS", "8000")),
                )
            ]
        }
    )


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    routing = _routing_table_from_env()
    gate = ConcurrencyGate()
    brief = os.environ.get("SELF_MOD_BRIEF")

    with psycopg.connect(conn_string, autocommit=True) as conn:
        self_mod.init_table(conn)

        tools = build_tools(conn)
        agent_model = RoutedModel(ROLE, routing, gate, tools=tools)

        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            result = run_self_mod_session(agent_model, tools, checkpointer, brief=brief)

    print(f"self-modifier session {result['thread_id']}: {result['final_message']}")


if __name__ == "__main__":
    main()
