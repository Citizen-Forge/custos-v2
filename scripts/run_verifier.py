"""
Standalone entrypoint for the acceptance-criteria verifier. Deliberately
NOT part of any worker's loop -- a separate agent judging someone else's
completed work is exactly the point (see verifier.py's docstring for why
this replaced the originally-planned human-feedback "Laurels" surface).
Checks every seat's closed tickets for ones with acceptance criteria set
and not yet verified -- verify_ticket itself is idempotent, so re-running
this is always safe.

    docker compose run --rm harness python scripts/run_verifier.py

Env: VERIFIER_MODEL_BASE_URL/NAME/API_KEY (defaults to the same local
model as LOCAL_MODEL_*), VERIFIER_MAX_TOKENS (default 6000 -- generous
for real reasoning over evidence, see ProviderConfig.max_tokens and the
reviewer.py lesson about this model's reasoning length).
"""

import os

import psycopg

from harness import beads, seats, verifications
from harness.verifier import build_model, verify_ticket


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]

    # The model is built by verifier.build_model so the scheduled path,
    # this manual entry point, and the dispatcher's verify-on-close step
    # cannot drift into judging tickets with differently-configured models.
    model = build_model()

    beads.ensure_initialized()
    with psycopg.connect(conn_string, autocommit=True) as conn:
        seats.init_table(conn)
        verifications.init_table(conn)

        roster = seats.list_all(conn)
        checked = 0
        verified = 0
        for seat in roster:
            for issue in beads.list_by_assignee(seat["seat_id"]):
                if issue.get("status") != "closed" or not beads.acceptance_criteria(issue):
                    continue
                checked += 1
                result = verify_ticket(conn, issue["id"], model)
                if result is None:
                    continue  # already verified -- verify_ticket's own idempotency check
                verified += 1
                print(f"{result['issue_id']} ({result['seat_id']}): {result['verdict']} -- {result['reasoning']}")

    if verified == 0:
        print(f"nothing new to verify ({checked} closed ticket(s) with acceptance criteria checked, all already verified)")
    else:
        print(f"verified {verified} of {checked} eligible closed ticket(s)")


if __name__ == "__main__":
    main()
