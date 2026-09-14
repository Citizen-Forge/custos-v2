"""
Requeue a parked ticket by id: reset its graph thread (so the next run is a
real attempt rather than a resume of a finished one), clear the human flag,
and reopen it. Useful for a stale infra failure the cause of which has since
been fixed.

    docker compose run --rm harness python scripts/requeue_ticket.py <id> [<id> ...] [--reason "..."]

`--reason` becomes the ticket's rework_reason, which worker.work_one_ticket
puts in the next attempt's opening prompt.
"""

import argparse
import os

import psycopg

from harness import beads
from harness.verifier import _reset_thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--reason", default=None)
    args = parser.parse_args()

    beads.ensure_initialized()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        for issue_id in args.ids:
            if args.reason:
                beads.set_metadata(issue_id, "rework_reason", args.reason)
            try:
                beads.remove_human_flag(issue_id)
            except Exception:
                pass
            _reset_thread(conn, issue_id)
            beads.reopen(issue_id, args.reason or "requeued by operator")
            print(f"requeued {issue_id}")


if __name__ == "__main__":
    main()
