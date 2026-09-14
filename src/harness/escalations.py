"""
The escalation queue: tickets other agents parked for a human, and the
bounded policy for what the product-owner may do about them.

An agent that refuses work (`refuse_ticket`) or that keeps crashing is
parked by `beads.flag_for_human`, which applies the `human` label. That
label is also the escalation queue. Left alone, every one of these waits on
a person -- but most are resolvable by the product-owner role, which is
what this module exists to let it do:

- a ticket that crashed on infrastructure since fixed -> requeue it;
- a ticket escalated as underspecified (no criteria, no named consumer) ->
  fix the specification, then requeue;
- a ticket in the wrong specialist's hands -> reassign, then requeue;
- a genuine decision the product-owner can make -> make it.

Guardrail: each requeue counts against a small per-ticket budget
(`escalation_attempts`). Once `MAX_ESCALATION_ATTEMPTS` is spent the ticket
drops out of the queue and stays for a human -- if the same escalation keeps
returning, the product-owner's judgment is not converging and a person
should decide. Closed tickets are never in the queue: a closed,
verifier-exhausted ticket is not live work.
"""

import os

from . import beads, verifier

MAX_ESCALATION_ATTEMPTS = int(os.environ.get("MAX_ESCALATION_ATTEMPTS", "2"))


def attempts(issue: dict) -> int:
    try:
        return int((issue.get("metadata") or {}).get("escalation_attempts") or 0)
    except (TypeError, ValueError):
        return 0


def pending() -> list[dict]:
    """Live escalations the product-owner may still act on: human-labelled,
    not closed, and not past their attempt budget.

    `parked_for_human` already returns the `--long` shape (notes + metadata
    + labels) in one `bd` call."""
    out = []
    for issue in beads.parked_for_human():
        if issue.get("status") == "closed":
            continue
        if attempts(issue) >= MAX_ESCALATION_ATTEMPTS:
            continue
        out.append(issue)
    return out


def requeue(conn, issue_id: str, directive: str) -> int:
    """Send an escalated ticket back to an agent with `directive` in its
    opening prompt, clearing the human flag and the old completion claim.

    Reuses `verifier.requeue_for_rework` for the parts that must happen
    together (clear completion_summary/work_commit, drop the human flag,
    reset the LangGraph thread, reopen) so there is one implementation of
    "put this ticket back in the queue for a real new attempt" rather than
    two that can drift. The escalation budget is counted separately."""
    issue = beads.show(issue_id)
    n = attempts(issue) + 1
    verifier.requeue_for_rework(conn, issue_id, directive, attempt=n)
    beads.set_metadata(issue_id, "escalation_attempts", str(n))
    return n
