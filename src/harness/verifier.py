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

from . import beads, verifications, workspaces

log = logging.getLogger(__name__)

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
    """The diff this ticket produced, from the commit made on its behalf
    when it claimed completion (worker.work_one_ticket). Absent for
    tickets closed before that existed, and for tickets that changed no
    files at all -- both are meaningful signals to a verifier, so this
    returns empty rather than raising."""
    sha = (issue.get("metadata") or {}).get("work_commit")
    if not sha:
        return ""
    try:
        return workspaces.commit_diff(workspaces.project_id_for(issue["id"]), sha)
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


def verify_ticket(conn, issue_id: str, model) -> dict | None:
    """Returns the recorded verdict dict, or None if this ticket isn't a
    candidate for verification: no acceptance criteria were ever set
    (nothing to check against), it isn't closed yet (nothing to verify),
    or it's already been verified (idempotent -- re-running the verifier
    across the same tickets shouldn't re-judge them every time). An
    unparseable model response fails closed to "fail", same posture as
    reviewer.py -- an unverifiable verdict is not a pass."""
    issue = beads.show(issue_id)
    criteria = beads.acceptance_criteria(issue)
    if not criteria:
        return None
    if issue.get("status") != "closed":
        return None
    if verifications.get_for_issue(conn, issue_id):
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

    verifications.record(conn, issue_id, seat_id, verdict, reasoning)

    # What a fail verdict costs depends on whether we reopen. Reopening
    # re-blocks every dependent, which is right when the work genuinely
    # is not there and catastrophic when the verdict is wrong: on
    # 2026-09-09 a false fail on workspace-9jg.1.6 -- scaffolding whose
    # suite was passing 21 of 21 at the time -- reopened the ticket and
    # froze all 73 dependents for sixteen hours. The morning of the same
    # day, the opposite: real breakage stayed closed and released those
    # dependents onto a scaffold whose tests ran nothing.
    #
    # So gate the expensive half on evidence that does not depend on a
    # model's judgement. If the measured suite is broken or hollow, the
    # work objectively is not done -- reopen. If it genuinely passes, a
    # fail is a judgement call (a missing README, thin coverage); flag it
    # for a human, but let the dependents keep moving.
    if verdict == "fail":
        objectively_unfinished = tests is not None and (
            tests["exit"] != 0 or tests["ran"] == 0
        )
        try:
            if objectively_unfinished:
                beads.reopen(issue_id, f"verification failed: {reasoning}")
            beads.flag_for_human(issue_id, f"verification failed: {reasoning}")
        except Exception:
            log.exception("could not act on failed verification for %s", issue_id)

    return {"issue_id": issue_id, "seat_id": seat_id, "verdict": verdict, "reasoning": reasoning}
