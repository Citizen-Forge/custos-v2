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

from . import beads, verifications, workspaces

PROMPT = """You are verifying whether completed work actually meets its stated acceptance \
criteria. You are a SEPARATE reviewer, not the agent that did the work -- judge honestly \
from the evidence, don't assume good faith just because the work was marked complete.

Ticket: {title}
Description: {description}
Acceptance criteria: {acceptance_criteria}

How the assigned agent says it was completed (close reason): {close_reason}
Accumulated notes from the work: {notes}

The actual code change this ticket produced:
{diff}

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

    response = model.invoke(
        PROMPT.format(
            title=issue.get("title", ""),
            description=issue.get("description", ""),
            acceptance_criteria=criteria,
            close_reason=issue.get("close_reason") or "(none recorded)",
            notes=issue.get("notes") or "(none)",
            diff=_diff_for(issue) or "(no code change recorded for this ticket)",
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

    verifications.record(conn, issue_id, seat_id, verdict, reasoning)
    return {"issue_id": issue_id, "seat_id": seat_id, "verdict": verdict, "reasoning": reasoning}
