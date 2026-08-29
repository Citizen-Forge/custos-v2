"""
Standalone entrypoint for the Phase 7 reviewer agent. Deliberately NOT
part of any worker/overwatch loop -- a separate agent forming a verdict
on someone else's proposal is exactly the point (PLAN.md Phase 7: "review
is what catches a bad change before it's ever granted that sandbox's
privileges"). Reviews every proposal currently sitting in `sandboxed`
status (has real sandbox evidence, no verdict yet) -- never applies
anything; `approve`/`reject` stay separate, human-only API calls.

    docker compose run --rm harness python scripts/run_reviewer.py

Env: REVIEWER_MODEL_BASE_URL/NAME/API_KEY (defaults to the same local
model as LOCAL_MODEL_*).
"""

import os

import psycopg

from harness.providers import ProviderConfig, build_chat_model
from harness.reviewer import review_proposal
from harness.tool_proposals import init_table, list_by_status


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]

    provider_cfg = ProviderConfig(
        name="reviewer",
        base_url=os.environ.get(
            "REVIEWER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
        ),
        model=os.environ.get("REVIEWER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")),
        api_key=os.environ.get("REVIEWER_MODEL_API_KEY"),
        # Caught live 2026-08-29: an uncapped reviewer call generated 5000+
        # tokens (~10 min) and STILL hadn't reached its JSON verdict when
        # killed -- a first attempt at 3000 confirmed this model's own
        # reasoning alone can blow past that with zero actual output.
        # Bounded, but generous enough that reasoning has real room to
        # finish -- see ProviderConfig.max_tokens.
        max_tokens=int(os.environ.get("REVIEWER_MAX_TOKENS", "10000")),
    )
    model = build_chat_model(provider_cfg)

    with psycopg.connect(conn_string, autocommit=True) as conn:
        init_table(conn)
        sandboxed = list_by_status(conn, "sandboxed")

        if not sandboxed:
            print("no proposals awaiting review")
            return

        for proposal in sandboxed:
            result = review_proposal(conn, proposal["id"], model)
            print(f"#{result['proposal_id']} {proposal['tool_name']}: {result['verdict']} -- {result['reasoning']}")

    print("all verdicts recorded -- awaiting human approve/reject via the API/dashboard")


if __name__ == "__main__":
    main()
