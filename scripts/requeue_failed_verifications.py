"""
One-off maintenance: requeue tickets a previous verifier run failed but
that were only parked for a human.

Before verifier.py grew its rework loop (2026-09-13), a fail was recorded
and the ticket flagged -- the verifier's finding sat on the board with
nothing set up to act on it. verifier.verify_ticket now requeues a fail
itself; this script exists to action the backlog that predates that.

    docker compose run --rm harness python scripts/requeue_failed_verifications.py --dry-run
    docker compose run --rm harness python scripts/requeue_failed_verifications.py

Only tickets whose notes name a verification failure are touched (the
"stopped without complete_ticket" and design-question parks are left
alone), and tickets already carrying a rework_count are skipped so
re-running this is safe.
"""

import argparse
import os

import psycopg

from harness import beads, verifications
from harness.verifier import requeue_for_rework

PARK_MARKERS = ("verification failed", "verifier response unparseable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print what would change, do nothing")
    args = parser.parse_args()

    beads.ensure_initialized()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        verifications.init_table(conn)
        count = 0
        for issue in beads.parked_for_human():
            if (issue.get("metadata") or {}).get("rework_count"):
                continue  # already in the rework loop
            notes = (issue.get("notes") or "").strip()
            if not any(marker in notes for marker in PARK_MARKERS):
                continue
            print(f"requeue {issue['id']}: {notes.splitlines()[0][:140] if notes else '(no notes)'}")
            if not args.dry_run:
                requeue_for_rework(conn, issue["id"], notes or "verification failed", attempt=1)
            count += 1
        verb = "would requeue" if args.dry_run else "requeued"
        print(f"{verb} {count} ticket(s)")


if __name__ == "__main__":
    main()
