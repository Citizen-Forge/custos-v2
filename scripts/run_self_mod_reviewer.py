"""
Standalone entrypoint for reviewing self-modification proposals. Same
shape as run_reviewer.py, separate script because the verdict here
never auto-approves (see reviewer.review_self_modification's
docstring) -- reviews every proposal currently sitting in `sandboxed`
status (real diff + real isolated test results attached, no verdict
yet). Never applies anything; approve/reject stay separate, human-only
API calls via POST /self-mod-proposals/{id}/approve|reject.

    docker compose run --rm harness python scripts/run_self_mod_reviewer.py

Env: SELF_MOD_REVIEWER_MODEL_BASE_URL/NAME/API_KEY (defaults to the same
local model as LOCAL_MODEL_*).
"""

import os

import psycopg

from harness import self_mod
from harness.providers import ProviderConfig, build_chat_model
from harness.reviewer import review_self_modification


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]

    provider_cfg = ProviderConfig(
        name="self-mod-reviewer",
        base_url=os.environ.get(
            "SELF_MOD_REVIEWER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
        ),
        model=os.environ.get("SELF_MOD_REVIEWER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("SELF_MOD_REVIEWER_MODEL_API_KEY"),
        max_tokens=int(os.environ.get("SELF_MOD_REVIEWER_MAX_TOKENS", "10000")),
    )
    model = build_chat_model(provider_cfg)

    with psycopg.connect(conn_string, autocommit=True) as conn:
        self_mod.init_table(conn)
        sandboxed = self_mod.list_by_status(conn, "sandboxed")

        if not sandboxed:
            print("no self-modification proposals awaiting review")
            return

        for proposal in sandboxed:
            result = review_self_modification(conn, proposal["id"], model)
            print(f"#{result['proposal_id']}: {result['verdict']} -- {result['reasoning']}")

    print("all verdicts recorded -- awaiting human approve/reject via the API")


if __name__ == "__main__":
    main()
