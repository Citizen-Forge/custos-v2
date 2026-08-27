"""
Standalone entrypoint for the Phase 5 meta-agent. Deliberately NOT part
of the ticket worker's loop -- this is a system-level agent that reviews
other agents' work, distinct from the agents doing the work, per PLAN.md.
Meant to run periodically (cron/schedule, not built here yet), each run
proposing at most one pending prompt revision for review -- never
auto-applying (see prompts.py's approve()).

    docker compose run --rm harness python scripts/run_meta_agent.py

Env: META_AGENT_TARGET_ROLE (default "worker"), META_AGENT_MODEL_BASE_URL/
NAME/API_KEY (defaults to the same local model as LOCAL_MODEL_*).
"""

import os

import psycopg

from harness.meta_agent import propose_prompt_update
from harness.prompts import init_table
from harness.providers import ProviderConfig, build_chat_model


def main() -> None:
    role = os.environ.get("META_AGENT_TARGET_ROLE", "worker")
    conn_string = os.environ["DATABASE_URL"]

    provider_cfg = ProviderConfig(
        name="meta-agent",
        base_url=os.environ.get(
            "META_AGENT_MODEL_BASE_URL",
            os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        ),
        model=os.environ.get("META_AGENT_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("META_AGENT_MODEL_API_KEY"),
    )
    model = build_chat_model(provider_cfg)

    with psycopg.connect(conn_string, autocommit=True) as conn:
        init_table(conn)
        result = propose_prompt_update(conn, role, model)

    if result:
        print(f"proposed {result['role']} v{result['version']}: {result['reasoning']}")
        print("pending human approval -- see prompts.pending() / prompts.approve()")
    else:
        print(f"no change proposed for role {role!r}")


if __name__ == "__main__":
    main()
