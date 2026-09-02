"""
Phase 5 groundwork: outcome signals sourced directly from Beads' own
audit trail (the --actor field beads.py already sets on every write), not
a separate metrics store. Simple counts, not a rigorous evaluation
framework -- enough for a meta-agent (or a human) to notice "this role
refuses a lot" or "most of this role's tickets are still open," not a
statistically sound signal.
"""

from datetime import datetime, timezone

from . import beads


def summary(actor: str, issues: list[dict] | None = None) -> dict:
    """Pass `issues` (every issue, any status) to compute this without a
    `bd` call of its own.

    Added because /seats was taking over a minute: it called this and
    queue_stats per seat, and between them they made four `bd` calls each
    -- two of which fetched the very same list_by_assignee data twice.
    With a handful of seats that is dozens of subprocess calls at seconds
    apiece. The caller can now fetch once and share."""
    if issues is None:
        issues = beads.list_by_assignee(actor)
    else:
        issues = [i for i in issues if i.get("assignee") == actor]
    closed = [i for i in issues if i["status"] == "closed"]
    refused = [i for i in issues if beads.is_flagged_for_human(i)]
    still_open = [
        i for i in issues if i["status"] != "closed" and not beads.is_flagged_for_human(i)
    ]
    return {
        "actor": actor,
        "total": len(issues),
        "closed": len(closed),
        "refused": len(refused),
        "still_open": len(still_open),
        "refused_reasons": [i["notes"] for i in refused if i.get("notes")],
    }


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def queue_stats(
    seat_id: str,
    issues: list[dict] | None = None,
    ready: list[dict] | None = None,
    in_progress: list[dict] | None = None,
) -> dict:
    """Real empirical timing (added 2026-08-29, PLAN.md's original Phase 6
    gap: "nothing to estimate from without real inference timing data" --
    now there is, from real tickets completed this session). Average
    completion time is measured from real closed tickets'
    started_at/closed_at, not a token-count-based theoretical estimate --
    this session's own data showed wildly variable real durations (one
    ticket took hours under heavy concurrent load, others took minutes),
    so a measured average is more honest than a computed one. None (not
    0) when there's no completion history yet -- an unknown wait is a
    real, different state from a zero-second wait, and the UI should be
    able to tell them apart rather than showing a misleadingly confident
    number."""
    if issues is None:
        issues = beads.list_by_assignee(seat_id)
    else:
        issues = [i for i in issues if i.get("assignee") == seat_id]
    durations = []
    for i in issues:
        if i["status"] != "closed":
            continue
        started = _parse_ts(i.get("started_at"))
        closed = _parse_ts(i.get("closed_at"))
        if started and closed:
            durations.append((closed - started).total_seconds())

    avg_seconds = sum(durations) / len(durations) if durations else None

    ready_issues = beads.ready() if ready is None else ready
    running = beads.in_progress() if in_progress is None else in_progress
    ready = len([i for i in ready_issues if beads.assigned_seat(i) == seat_id])
    in_progress = len(
        [i for i in running if i.get("assignee") == seat_id and not beads.is_flagged_for_human(i)]
    )

    return {
        "seat_id": seat_id,
        "ready": ready,
        "in_progress": in_progress,
        "avg_completion_seconds": avg_seconds,
        "sample_size": len(durations),
        # rough total wait for everything currently queued/running at this
        # seat, at its own historical pace -- None propagates rather than
        # silently becoming a confident-looking 0.
        "estimated_wait_seconds": (
            avg_seconds * (ready + in_progress) if avg_seconds is not None else None
        ),
    }
