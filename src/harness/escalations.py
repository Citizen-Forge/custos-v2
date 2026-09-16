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

import logging
import os

from . import beads, verifications, verifier

log = logging.getLogger(__name__)

MAX_ESCALATION_ATTEMPTS = int(os.environ.get("MAX_ESCALATION_ATTEMPTS", "2"))

# Set once an escalation has been ANSWERED, so the queue cannot ask about
# it again. Cleared by requeue(), because a ticket genuinely re-escalated
# later does deserve another look.
RESOLVED_KEY = "escalation_resolved"


def attempts(issue: dict) -> int:
    try:
        return int((issue.get("metadata") or {}).get("escalation_attempts") or 0)
    except (TypeError, ValueError):
        return 0


def pending(conn=None) -> list[dict]:
    """Live escalations the product-owner may still act on: human-labelled,
    inside their attempt budget, and either still open or closed on a
    FAILING verdict.

    Closed+human tickets used to be skipped outright, on the reasoning that
    a closed, verifier-exhausted ticket is not live work. That left exactly
    one case with nobody able to answer it: an attempt that refused because
    the ticket's named deliverable is ALREADY in the project. Found live
    2026-09-15 -- workspace-9jg.2.2.2 "the ticket's named defect is ALREADY
    FIXED in HEAD", 2.4 "every artifact the ticket names ALREADY EXISTS AT
    HEAD and is byte-identical to disk". The agent cannot close its own
    ticket, and requeueing it just produces the same refusal, so the one
    action that helps -- accepting a delivery that is already there -- needs
    a caller with the authority to take it.

    A ticket closed on a PASS is still not here: it is done.

    `conn` is optional only so the two standalone entry points, which ask
    this question before opening their own connection, keep working; the
    verdict lookup is the sole reason it is needed at all.

    `parked_for_human` already returns the `--long` shape (notes + metadata
    + labels) in one `bd` call."""
    if conn is None:
        import os

        import psycopg

        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as opened:
            return pending(opened)

    out = []
    for issue in beads.parked_for_human(include_closed=True):
        if (issue.get("metadata") or {}).get(RESOLVED_KEY):
            continue  # already answered; see resolve()
        if attempts(issue) >= MAX_ESCALATION_ATTEMPTS:
            continue
        if issue.get("status") == "closed":
            try:
                verdict = verifications.get_for_issue(conn, issue["id"])
            except Exception:
                # No verdict table (a fresh database). A closed ticket
                # cannot then be told apart from one answered by hand, so
                # it is left out rather than guessed at -- the same
                # posture as verifier._reset_thread's absent-table case.
                log.warning("no verdicts readable; skipping closed tickets this pass")
                continue
            if not verdict or verdict.get("verdict") != "fail":
                continue
        out.append(issue)
    return out


def resolve(conn, issue_id: str, resolution: str, actor: str) -> None:
    """Answer an escalation, and take it out of the queue for good.

    `beads.respond_to_human` records the decision and closes the ticket,
    but the queue only ever asked "is this parked?" -- so an answered
    ticket came straight back on the next pass and could be requeued,
    undoing the decision it had just made. Found live 2026-09-15:
    workspace-9jg.2.4 was resolved as already-delivered and was still in
    pending() minutes later, with the product-owner about to look at it
    again."""
    beads.respond_to_human(issue_id, resolution, actor=actor)
    beads.set_metadata(issue_id, RESOLVED_KEY, "answered")


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
    try:
        # A fresh attempt is not an answered one: if it escalates again,
        # that is a new question and it should be considered.
        beads.unset_metadata(issue_id, RESOLVED_KEY)
    except Exception:
        log.exception("could not clear %s on %s", RESOLVED_KEY, issue_id)
    return n
