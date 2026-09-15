"""
The acceptance-criteria verification loop's judgment: given a closed
ticket that had explicit acceptance criteria (beads.acceptance_criteria),
decide whether the actual work meets them. Same single-shot judgment
shape as reviewer.py -- gather real evidence (the ticket's own notes and
close reason, not a self-report from the seat that did the work), ask a
model, record a verdict. This is deliberately a SEPARATE agent's call
from the seat that did the work, not a self-grade -- same reasoning as
Phase 7's reviewer being separate from the overwatch agent that proposes
a tool.

This is the mechanism the user asked for in place of "Laurels" (a
human-feedback surface, deferred): automated positive/negative signal a
seat can actually earn without a human rating every ticket, and real
quality data for meta_agent.py to reason about beyond "was this ticket
refused."
"""

import json
import logging
import os

import psycopg

from . import beads, verifications, workspaces

log = logging.getLogger(__name__)

# How many times a failed verification sends a ticket back to an agent
# before it is parked for a human instead. The point of re-queuing is that
# the verifier's finding gets acted on, not merely recorded -- but a
# reopened ticket re-blocks its dependents (see the comment on the old
# fail branch), so the loop has to be finite. Two attempts rides out a
# verdict the agent can actually fix, then hands a genuine impasse to a
# person rather than churning forever.
MAX_VERIFIER_REWORKS = int(os.environ.get("MAX_VERIFIER_REWORKS", "2"))


def build_model():
    """The verifier's model, from VERIFIER_* falling back to LOCAL_*.

    Shared by the standalone entry point and by the dispatcher's
    verify-on-close step, so the two cannot drift into judging tickets
    with differently-configured models."""
    from .providers import ProviderConfig, build_chat_model

    provider_cfg = ProviderConfig(
        name="verifier",
        base_url=os.environ.get(
            "VERIFIER_MODEL_BASE_URL",
            os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        ),
        model=os.environ.get(
            "VERIFIER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
        ),
        # Falls back to LOCAL_MODEL_API_KEY for the same reason base_url
        # and model already fall back to their LOCAL_* counterparts: a
        # VERIFIER_* override is opt-out, so an unset key must not mean
        # "no key" when the inherited base_url is an authenticated one.
        api_key=os.environ.get("VERIFIER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=int(os.environ.get("VERIFIER_MAX_TOKENS", "6000")),
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get(
                "VERIFIER_MODEL_DISABLE_THINKING", os.environ.get("LOCAL_MODEL_DISABLE_THINKING")
            )
            else None
        ),
    )
    return build_chat_model(provider_cfg)


PROMPT = """You are verifying whether completed work actually meets its stated acceptance \
criteria. You are a SEPARATE reviewer, not the agent that did the work -- judge honestly \
from the evidence, don't assume good faith just because the work was marked complete.

Ticket: {title}
Description: {description}
Acceptance criteria: {acceptance_criteria}

How the assigned agent says it was completed (close reason): {close_reason}
Accumulated notes from the work: {notes}

The actual code change this ticket produced, as a unified diff:
{diff}

Read the diff as a diff. A line starting with `-` was REMOVED by this ticket and is NOT in \
the code any more; a line starting with `+` was ADDED and IS the current state. Never report \
a `-` line as a present-day problem -- if the ticket deleted a bad line, that is the fix, not \
the fault. When both a `-` and a `+` version of the same setting appear, only the `+` one is live.

The project's configuration and readme files AS THEY STAND RIGHT NOW:
{current_files}

Those are the live contents, read from disk just now -- not the diff, not a claim. Later \
tickets may have committed on top of the one under review, so this is the state that \
actually exists. If a criterion asks whether a file exists or what it contains, answer from \
this section; do not say a thing cannot be verified when its current contents are printed above.

Result of actually running the project's own test suite just now:
{test_result}

That test result is MEASURED, not claimed -- the suite was executed to produce it. Where it \
speaks to a criterion (do the tests run, do they pass), believe it over anything you infer \
from reading the diff. Do not assert that tests fail, or that the project does not build, when \
the measured result above says otherwise.

Weigh the diff above all else. It is what the ticket actually changed; the close reason and \
notes are the agent's own account of it and may be generous. If the diff is empty or does not \
plausibly implement the criteria, that is a fail no matter how confident the description sounds.

Decide pass or fail against the acceptance criteria specifically -- not whether the work is \
impressive, not whether you'd have done it differently, just whether the stated criteria are \
actually met based on the evidence given. If the evidence is too thin to tell either way, \
fail rather than assume -- an unverifiable claim of success is not the same as success.

Respond with strict JSON and nothing else: \
{{"verdict": "pass"|"fail", "reasoning": "<one or two sentences>"}}
"""


def _diff_for(issue: dict) -> str:
    """The diff this ticket produced.

    Prefers the commits the harness made for this ticket (found by their
    `<ticket-id>: ` subject), falling back to the recorded work_commit.
    Absent for a ticket that genuinely changed no files -- which is a
    meaningful signal to the verifier, so this returns empty rather than
    raising."""
    sha = (issue.get("metadata") or {}).get("work_commit")
    try:
        return workspaces.diff_for_ticket(
            workspaces.project_id_for(issue["id"]), issue["id"], sha
        )
    except Exception:
        return ""


def _tests_for(issue: dict) -> dict | None:
    """Run the project's own suite, or None if there isn't one to run."""
    try:
        return workspaces.run_tests(workspaces.project_id_for(issue["id"]))
    except Exception:
        log.exception("could not run tests for %s", issue.get("id"))
        return None


def _current_files_for(issue: dict) -> str:
    """Live contents of the project's criteria-bearing files."""
    try:
        return workspaces.criteria_file_snapshot(workspaces.project_id_for(issue["id"]))
    except Exception:
        log.exception("could not snapshot files for %s", issue.get("id"))
        return ""


def _describe_tests(tests: dict | None) -> str:
    if tests is None:
        return "(no runnable test script in this project)"
    if tests["exit"] == 0 and tests["ran"] == 0:
        return (
            "THE TEST COMMAND EXITED 0 BUT RAN ZERO TESTS. This is not a passing suite -- it is "
            "a command that verified nothing (e.g. a runner whose file glob matched no files). "
            f"Treat it as no test coverage at all.\n{tests['tail']}"
        )
    return (
        f"exit={tests['exit']}  tests_run={tests['ran']}  passed={tests['passed']}  "
        f"failed={tests['failed']}\n{tests['tail']}"
    )


def _reset_thread(conn, issue_id: str) -> None:
    """Drop the ticket's LangGraph thread so a requeued attempt genuinely
    starts over.

    Without this, re-queuing is a no-op that lies: the graph for that
    ticket has already reached END, so worker.work_one_ticket's resume
    returns immediately, and (before completion_summary is cleared) the
    ticket would just be re-closed with its old claim. A fresh thread with
    the verifier's finding in the opening prompt is what makes the agent
    re-attempt. Best-effort: the checkpointer tables are absent in some
    tests, and a failed reset must not abort the verdict itself."""
    try:
        with conn.cursor() as cur:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (issue_id,))
    except psycopg.errors.UndefinedTable:
        # Test databases have no checkpointer tables until a worker test
        # creates them. Absent tables mean nothing to reset, not a failure.
        log.debug("no checkpointer tables to reset for %s", issue_id)
    except Exception:
        log.exception("could not reset the graph thread for %s", issue_id)


def requeue_for_rework(conn, issue_id: str, reasoning: str, attempt: int) -> None:
    """Send a verification-failed ticket back to an agent, carrying the
    verifier's actual finding.

    Clearing completion_summary is load-bearing: worker.work_one_ticket
    closes a ticket as soon as that key is set, so leaving it would let the
    old claim close the ticket again the moment capacity frees, with the
    verifier's finding never read. work_commit is cleared so the verifier's
    idempotency (which is per-commit) judges the next attempt's diff
    instead of skipping it."""
    beads.set_metadata(issue_id, "rework_count", str(attempt))
    beads.set_metadata(issue_id, "rework_reason", reasoning[:4000])
    for key in ("completion_summary", "work_commit"):
        try:
            beads.unset_metadata(issue_id, key)
        except Exception:
            log.exception("could not clear %s on %s", key, issue_id)
    try:
        beads.remove_human_flag(issue_id)
    except Exception:
        log.exception("could not clear the human flag on %s", issue_id)
    beads.reopen(issue_id, f"verifier fail #{attempt}: {reasoning}")
    _reset_thread(conn, issue_id)


def verify_ticket(conn, issue_id: str, model) -> dict | None:
    """The recorded verdict dict, or None when the ticket isn't a candidate:
    no acceptance criteria, not closed, or already verified for this commit.

    The idempotency check is per-commit, not per-issue: a verdict already
    recorded for the commit under judgement is skipped, but a ticket whose
    work_commit has changed since (i.e. it was requeued after a fail and
    produced new work) is judged again. An unparseable model response
    fails closed to "fail", same posture as reviewer.py -- an unverifiable
    verdict is not a pass."""
    issue = beads.show(issue_id)
    criteria = beads.acceptance_criteria(issue)
    if not criteria:
        return None
    if issue.get("status") != "closed":
        return None
    current_commit = (issue.get("metadata") or {}).get("work_commit")
    rework_count = (issue.get("metadata") or {}).get("rework_count")
    existing = verifications.get_for_issue(conn, issue_id)
    if existing:
        if existing.get("work_commit") == current_commit:
            return None
        # A verdict recorded before work_commit existed (NULL) is trusted
        # as-is: re-judging every already-verified legacy ticket would
        # reopen settled work and re-block its dependents for no reason.
        # The one exception is a ticket that was explicitly requeued for
        # rework -- its old verdict must not stand in the way of judging
        # the new work.
        if existing.get("work_commit") is None and not rework_count:
            return None

    seat_id = beads.assigned_seat(issue) or issue.get("assignee") or "unknown"
    tests = _tests_for(issue)

    response = model.invoke(
        PROMPT.format(
            title=issue.get("title", ""),
            description=issue.get("description", ""),
            acceptance_criteria=criteria,
            close_reason=issue.get("close_reason") or "(none recorded)",
            notes=issue.get("notes") or "(none)",
            diff=_diff_for(issue) or "(no code change recorded for this ticket)",
            current_files=_current_files_for(issue) or "(no configuration or readme files found)",
            test_result=_describe_tests(tests),
        )
    )
    content = getattr(response, "content", response)

    try:
        data = json.loads(content)
        verdict = data["verdict"]
        reasoning = data.get("reasoning", "")
        if verdict not in ("pass", "fail"):
            raise ValueError(f"unexpected verdict: {verdict!r}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        verdict = "fail"
        reasoning = f"verifier response unparseable: {e}"

    # Mechanical override, applied only to the one case that admits no
    # judgement: the suite exited 0 while running nothing. A model
    # reading a well-written test file has no way to see that the file is
    # never executed, so this is checked rather than asked.
    if verdict == "pass" and tests and tests["exit"] == 0 and tests["ran"] == 0:
        verdict = "fail"
        reasoning = (
            "Mechanical check overrode a pass: the project's test command exited 0 but ran "
            "zero tests, so it is not evidence of anything. " + reasoning
        )

    verifications.record(conn, issue_id, seat_id, verdict, reasoning, work_commit=current_commit)

    # A fail must be actioned, not just recorded. The ticket goes back to
    # an agent with the verifier's finding attached (requeue_for_rework),
    # and only after MAX_VERIFIER_REWORKS attempts is it parked for a
    # human. This replaced the old split -- reopen when the test suite was
    # objectively broken, otherwise only flag -- which left judgement-call
    # fails sitting unactioned on the board forever. The cost of reopening
    # is real (it re-blocks dependents; a false fail once froze 73 of them
    # for sixteen hours, see the git history), which is exactly why the
    # loop is bounded rather than unbounded.
    if verdict == "fail":
        attempt = int((issue.get("metadata") or {}).get("rework_count") or 0) + 1
        try:
            if attempt <= MAX_VERIFIER_REWORKS:
                log.info("requeuing %s for rework attempt %s: %s", issue_id, attempt, reasoning[:200])
                requeue_for_rework(conn, issue_id, reasoning, attempt)
            else:
                beads.flag_for_human(
                    issue_id,
                    f"verification failed after {MAX_VERIFIER_REWORKS} rework attempt(s): {reasoning}",
                )
        except Exception:
            log.exception("could not act on failed verification for %s", issue_id)

    return {"issue_id": issue_id, "seat_id": seat_id, "verdict": verdict, "reasoning": reasoning}
