"""One-off manual probe: give the real acceptance-criteria verifier
(real model, real Beads ticket, not scripted) a genuinely unmet-criteria
case, then feed the resulting real fail data into the real meta-agent.
Per PLAN.md's open item: "the meta-agent's harder case (real failure
data, not just successes) was pending a real failing ticket."

Creates one real ticket whose stated acceptance criteria explicitly
require demonstrated test evidence, then closes it the way an agent
that skipped verification honestly would -- a real, unremarkable failure
shape (claims success, no test run shown), not a contrived crash. The
verifier's own "if the evidence is too thin to tell either way, fail
rather than assume" instruction is exactly what this exercises against
real model judgment, not an assumption about what it'll say.

    docker compose run --rm harness python scripts/probe_verifier_failure_case.py

Env: same LOCAL_MODEL_* defaults as run_verifier.py/run_meta_agent.py.
"""

import os

import psycopg

from harness import beads, prompts, seats, verifications
from harness.meta_agent import propose_prompt_update
from harness.providers import ProviderConfig, build_chat_model
from harness.verifier import verify_ticket

SEAT_ID = "workspace-implement-verify"


def _provider(name: str, max_tokens: int) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        model=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"),
        max_tokens=max_tokens,
    )


def main() -> None:
    conn_string = os.environ["DATABASE_URL"]
    beads.ensure_initialized()

    issue = beads.create(
        "validate email addresses",
        "Write /workspace/validate_email.py with a function validate_email(s) "
        "that returns True/False for whether s looks like a valid email address.",
        acceptance_criteria=(
            "validate_email() must be demonstrated, by an actual executed test run (not just "
            "written code), to reject 'not-an-email' as invalid and accept 'user@example.com' "
            "as valid. A claim of correctness without shown test output does not satisfy this."
        ),
    )
    beads.assign_to_seat(issue["id"], SEAT_ID)
    beads.claim(issue["id"], actor=SEAT_ID)
    beads.append_note(
        issue["id"],
        "Wrote /workspace/validate_email.py with a regex-based validate_email(). "
        "Didn't get a chance to run it before the session ended -- the regex looks "
        "correct on inspection, should handle both cases fine.",
        actor=SEAT_ID,
    )
    beads.close(
        issue["id"],
        reason="Implemented validate_email() per the ticket; regex-based, should be correct.",
    )
    print(f"created and closed real ticket {issue['id']} (no test evidence shown, per the criteria)")

    with psycopg.connect(conn_string, autocommit=True) as conn:
        verifications.init_table(conn)
        seats.init_table(conn)
        prompts.init_table(conn)

        verifier_model = build_chat_model(_provider("verifier-probe", 6000))
        verdict = verify_ticket(conn, issue["id"], verifier_model)
        if verdict is None:
            print("verifier declined to judge this ticket (unexpected -- check criteria/status)")
            return
        print(f"verifier verdict: {verdict['verdict']} -- {verdict['reasoning']}")

        print(f"\ncurrent verification summary for {SEAT_ID}: {verifications.summary(conn, SEAT_ID)}")

        meta_model = build_chat_model(_provider("meta-agent-probe", 4000))
        result = propose_prompt_update(conn, SEAT_ID, meta_model)
        if result:
            print(f"\nmeta-agent proposed {SEAT_ID} v{result['version']}: {result['reasoning']}")
        else:
            print(f"\nmeta-agent proposed no change for {SEAT_ID} despite the new fail data")


if __name__ == "__main__":
    main()
