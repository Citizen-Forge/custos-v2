"""
One-off maintenance: requeue every ticket the shared-workspace fault parked.

Before workspaces.for_ticket gave each ticket its own worktree, every seat
on a project worked in ONE checkout, and commit_all stages with `git add
-A`. Whichever ticket finished first committed the other agents' live files
under its own subject; the verifier -- which judges that commit's diff --
failed the ticket for "the diff under review is not this ticket's work at
all", the fail requeued it, the rework budget ran out, and it parked. Two
further faults parked tickets over the same period: the history bound
dropped the ticket brief (fixed in 1d26390), and the acceptance criteria
never reached the agent's prompt.

Requeueing these is not just clearing the label:

- their verdicts were reached on diffs that were not their own, so the
  rows are moved to a backup table and removed. Otherwise
  verifier.verify_ticket's per-commit idempotency sees a verdict whose
  work_commit is NULL against a ticket whose work_commit is now also NULL,
  treats it as already judged, and the next completion is never verified
  at all -- the ticket would close ungraded;
- rework_count and escalation_attempts are reset, because the budget was
  spent on failures the ticket did not cause;
- completion_summary and work_commit are cleared, or worker.work_one_ticket
  re-closes the ticket with the OLD claim the moment the graph runs;
- the LangGraph thread is reset, so the next run is a real attempt rather
  than a resume of a thread that already reached END;
- the ticket's git worktree is moved back onto the current integration
  tip, because its branch is frozen at the commit it forked from and
  re-running on top of that still merges into the same conflict it was
  parked for. The abandoned commit is logged; it stays in the worktree's
  reflog.

    docker compose run --rm harness python scripts/requeue_parked_tickets.py --dry-run <id> ...
    docker compose run --rm harness python scripts/requeue_parked_tickets.py <id> ...
"""

import argparse
import os

import psycopg

from harness import beads, verifications, workspaces
from harness.verifier import _reset_thread

# Kept rather than dropped: the evidence is worth having if anyone wants to
# re-read what the contaminated runs actually concluded.
BACKUP_TABLE = "verifications_backup_workspace_fix_20260915"

REASON = (
    "Requeued by the operator after the harness fix of 2026-09-15. Until then every seat "
    "on this project shared a single working tree, so a ticket's commit could contain "
    "other tickets' files and verification frequently rejected this ticket for work that "
    "was not its own; the history bound could also drop this ticket's brief entirely, and "
    "the acceptance criteria never reached the agent's prompt. Each ticket now runs in its "
    "own worktree. Re-attempt this ticket against the acceptance criteria below, and do "
    "not assume the earlier rejection describes your own work."
)


def _backup_and_clear(conn, issue_ids: list[str]) -> int:
    """Move these tickets' verdicts to the backup table and delete them."""
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} (LIKE verifications INCLUDING ALL)")
        # Re-runnable: replace this batch rather than piling up duplicates.
        cur.execute(f"DELETE FROM {BACKUP_TABLE} WHERE issue_id = ANY(%s)", (issue_ids,))
        cur.execute(
            f"INSERT INTO {BACKUP_TABLE} SELECT * FROM verifications WHERE issue_id = ANY(%s)",
            (issue_ids,),
        )
        moved = cur.rowcount
        cur.execute("DELETE FROM verifications WHERE issue_id = ANY(%s)", (issue_ids,))
        return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="print what would change, do nothing")
    args = parser.parse_args()

    beads.ensure_initialized()
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        verifications.init_table(conn)
        for issue_id in args.ids:
            issue = beads.show(issue_id)
            print(
                f"{issue_id:22} status={issue.get('status'):11} "
                f"human={'human' in (issue.get('labels') or [])!s:5} "
                f"rework={((issue.get('metadata') or {}).get('rework_count'))} "
                f"esc={((issue.get('metadata') or {}).get('escalation_attempts'))}"
            )
            if args.dry_run:
                continue
            beads.set_metadata(issue_id, "rework_reason", REASON)
            # 0, not unset: the budgets were spent on failures this ticket
            # did not cause, and a restored budget must be visible to
            # verify_ticket rather than looking like "never attempted".
            beads.set_metadata(issue_id, "rework_count", "0")
            beads.set_metadata(issue_id, "escalation_attempts", "0")
            for key in ("completion_summary", "work_commit"):
                try:
                    beads.unset_metadata(issue_id, key)
                except Exception:
                    pass
            _reset_thread(conn, issue_id)
            previous = workspaces.reset_for_attempt(issue_id)
            if previous:
                print(f"    moved {workspaces.ticket_branch(issue_id)} to the tip "
                      f"(was {previous[:12]}; recoverable via the worktree reflog)")
            beads.reopen(
                issue_id,
                "requeued: parked by the shared-workspace diff fault (workspace fix 2026-09-15)",
            )
            # The flag comes off LAST. Between clearing it and reopening,
            # the ticket would be back in `in_progress` with no completion
            # claim, i.e. claimable by dispatch in a half-prepared state --
            # and the dispatcher skips human-labelled tickets, so leaving
            # the label on until everything else is done is what makes this
            # sequence safe to run against a live board.
            try:
                beads.remove_human_flag(issue_id)
            except Exception:
                pass

        if args.dry_run:
            print(f"would requeue {len(args.ids)} ticket(s), nothing changed")
            return
        moved = _backup_and_clear(conn, args.ids)
        print(f"requeued {len(args.ids)} ticket(s); {moved} stale verdict(s) moved to {BACKUP_TABLE}")


if __name__ == "__main__":
    main()
