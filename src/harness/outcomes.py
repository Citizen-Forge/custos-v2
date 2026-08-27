"""
Phase 5 groundwork: outcome signals sourced directly from Beads' own
audit trail (the --actor field beads.py already sets on every write), not
a separate metrics store. Simple counts, not a rigorous evaluation
framework -- enough for a meta-agent (or a human) to notice "this role
refuses a lot" or "most of this role's tickets are still open," not a
statistically sound signal.
"""

from . import beads


def summary(actor: str) -> dict:
    issues = beads.list_by_assignee(actor)
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
