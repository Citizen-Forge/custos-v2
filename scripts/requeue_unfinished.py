"""
One-off: requeue tickets parked because a run stopped without calling
complete_ticket.

The completion nudge (graph.py) now gives a fresh run one chance to finish
properly, but tickets parked before it need a fresh thread to benefit --
their graph already reached END, so clearing the flag alone would just
resume a finished thread, change nothing, and re-flag it. This resets the
thread, clears the human flag and reopens the ticket, with a short reason
the worker puts in the opening prompt.

    docker compose run --rm harness python scripts/requeue_unfinished.py --dry-run
    docker compose run --rm harness python scripts/requeue_unfinished.py

Only tickets whose notes name the completion gate are touched.
"""

import argparse
import os

import psycopg

from harness import beads
from harness.verifier import _reset_thread

MARKER = "stopped without calling complete_ticket"
REASON = (
    "Your previous run ended without calling complete_ticket, so nothing was "
    "recorded. If the work is already done, call complete_ticket now with a concrete "
    "summary; otherwise continue the work and complete it when it is finished."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print what would change, do nothing")
    args = parser.parse_args()

    beads.ensure_initialized()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        count = 0
        for issue in beads.parked_for_human():
            if MARKER not in (issue.get("notes") or ""):
                continue
            print(f"requeue {issue['id']}: {issue.get('title', '')}")
            if not args.dry_run:
                beads.set_metadata(issue["id"], "rework_reason", REASON)
                try:
                    beads.remove_human_flag(issue["id"])
                except Exception:
                    pass
                _reset_thread(conn, issue["id"])
                beads.reopen(issue["id"], "requeued: previous run stopped without completing")
            count += 1
        verb = "would requeue" if args.dry_run else "requeued"
        print(f"{verb} {count} ticket(s)")


if __name__ == "__main__":
    main()
