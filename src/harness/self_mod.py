"""
Phase 7, completed 2026-08-30: self-modification of the harness's OWN
source, with the same two-layer guardrail shape as everything else in
this project -- mechanical isolation (a real, throwaway test run
against the proposed change) proves it safe enough to trust, and a
separate reviewer agent's verdict alone decides whether it gets
deployed. No human reviews this before it takes effect (user's own
explicit call, 2026-08-30, made the same session this module was
built): git history is the actual safety net here -- a bad "allow" is
a `git revert` away, not something a pre-deployment approval step was
ever going to catch better than a real, passing test suite already
does. `approve`/`reject` stay reachable directly (e.g. via the API) as
a human override path, same as tool_proposals.py/prompts.py -- what's
different is nothing waits for that override before a verdict takes
effect.

The harness's own source is still the "control plane" PLAN.md's Phase 7
three-zones design walled off -- "no agent, including overwatch, ever
writes here directly" -- and an earlier pass through this same phase
found and fixed a real gap where that boundary had accidentally been
open (`/app/src` used to be read-write). Nothing about self-
modification reverses that fix: `/app/src` stays read-only for the
live system; the self-modifier agent only ever writes to a separate,
isolated checkout (self_modifier.py), and the only thing with write
access to the real tree is scripts/run_self_mod_deploy.py's own
trusted orchestration code -- never an agent tool, never something a
proposal's own diff content can reach or influence beyond being the
patch that gets applied.

Lifecycle: pending (proposed, a real diff attached) -> sandboxed (a
real isolated docker-compose test run against the diff, pass/fail
counts attached) -> reviewed + approved/rejected in the same call
(reviewer.review_self_modification's own verdict) -> deployed (the
diff has actually been applied to the real tree, re-tested there as a
final hard gate, committed, and the affected services rebuilt -- see
run_self_mod_deploy.py for why that final test run is a second,
non-LLM-dependent check, not just a formality).
"""


def init_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS self_mod_proposals (
            id SERIAL PRIMARY KEY,
            description TEXT NOT NULL,
            diff TEXT NOT NULL,
            proposed_by TEXT NOT NULL,
            sandbox_stdout TEXT,
            sandbox_stderr TEXT,
            sandbox_exit_code INT,
            sandbox_tests_passed INT,
            sandbox_tests_failed INT,
            review_verdict TEXT,
            review_reasoning TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            approved_at TIMESTAMPTZ,
            deployed_at TIMESTAMPTZ
        )
        """
    )


def propose(conn, description: str, diff: str, proposed_by: str) -> int:
    row = conn.execute(
        "INSERT INTO self_mod_proposals (description, diff, proposed_by) VALUES (%s, %s, %s) RETURNING id",
        (description, diff, proposed_by),
    ).fetchone()
    return row[0]


def record_sandbox_result(
    conn, proposal_id: int, exit_code: int, stdout: str, stderr: str, tests_passed: int | None, tests_failed: int | None
) -> None:
    """`tests_passed`/`tests_failed` are None when the sandboxed test run
    itself couldn't complete (e.g. the diff didn't apply, the image
    failed to build) -- distinct from a completed run that found real
    failures (tests_failed > 0), which the reviewer needs to be able to
    tell apart."""
    conn.execute(
        "UPDATE self_mod_proposals SET sandbox_exit_code=%s, sandbox_stdout=%s, sandbox_stderr=%s, "
        "sandbox_tests_passed=%s, sandbox_tests_failed=%s, status='sandboxed' WHERE id=%s",
        (exit_code, stdout, stderr, tests_passed, tests_failed, proposal_id),
    )


def record_review(conn, proposal_id: int, verdict: str, reasoning: str) -> None:
    """Just records the verdict/reasoning -- reviewer.review_self_modification
    is what acts on it (calling approve()/reject() itself right after),
    same as tool_proposals.py's review_proposal. See module docstring for
    why self-modification's "allow" takes effect immediately rather than
    waiting on a human approval step."""
    conn.execute(
        "UPDATE self_mod_proposals SET review_verdict=%s, review_reasoning=%s, status='reviewed' WHERE id=%s",
        (verdict, reasoning, proposal_id),
    )


def approve(conn, proposal_id: int) -> None:
    conn.execute(
        "UPDATE self_mod_proposals SET status='approved', approved_at=now() WHERE id=%s",
        (proposal_id,),
    )


def reject(conn, proposal_id: int, reason: str | None = None) -> None:
    conn.execute(
        "UPDATE self_mod_proposals SET status='rejected', review_reasoning=COALESCE(%s, review_reasoning) WHERE id=%s",
        (reason, proposal_id),
    )


def mark_deployed(conn, proposal_id: int) -> None:
    """Called only by scripts/run_self_mod_deploy.py, only after the diff
    has actually been applied to the real tree, re-tested there, and
    committed -- see that script's module docstring."""
    conn.execute(
        "UPDATE self_mod_proposals SET status='deployed', deployed_at=now() WHERE id=%s",
        (proposal_id,),
    )


def get(conn, proposal_id: int) -> dict | None:
    """Returns the proposal row as a dict, or None if no proposal with that id exists."""
    row = conn.execute(
        "SELECT id, description, diff, proposed_by, sandbox_stdout, sandbox_stderr, sandbox_exit_code, "
        "sandbox_tests_passed, sandbox_tests_failed, review_verdict, review_reasoning, status, "
        "created_at, approved_at, deployed_at FROM self_mod_proposals WHERE id = %s",
        (proposal_id,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_by_status(conn, status: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, description, diff, proposed_by, sandbox_stdout, sandbox_stderr, sandbox_exit_code, "
        "sandbox_tests_passed, sandbox_tests_failed, review_verdict, review_reasoning, status, "
        "created_at, approved_at, deployed_at FROM self_mod_proposals WHERE status = %s ORDER BY created_at",
        (status,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "description": row[1],
        "diff": row[2],
        "proposed_by": row[3],
        "sandbox_stdout": row[4],
        "sandbox_stderr": row[5],
        "sandbox_exit_code": row[6],
        "sandbox_tests_passed": row[7],
        "sandbox_tests_failed": row[8],
        "review_verdict": row[9],
        "review_reasoning": row[10],
        "status": row[11],
        "created_at": row[12],
        "approved_at": row[13],
        "deployed_at": row[14],
    }
