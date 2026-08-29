"""
The only place a proposed tool's code actually executes. Deliberately
separate from overwatch.py's propose_tool (which only ever inserts a
`pending` row) -- this script needs real Docker access to spawn the
sandbox container, and PLAN.md Phase 7 is explicit that capability must
never live in the same service that runs agent-invoked `shell_exec`
(`harness`/`api`). Only ever invoked via the `sandbox-runner` compose
profile, same as tests/test_sandbox.py:

    docker compose --profile sandbox run --rm sandbox-runner \\
        python scripts/run_sandbox_for_proposals.py

Processes every proposal currently `pending`, moving each to `sandboxed`
with its real stdout/stderr/exit code attached (tool_proposals.py). Run
scripts/run_reviewer.py afterward (in the regular `harness` service --
reviewing needs no Docker access, only the recorded evidence) to get a
verdict on each.
"""

import os

import psycopg

from harness import tool_proposals
from harness.sandbox import run_sandboxed


def main() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        tool_proposals.init_table(conn)
        pending = tool_proposals.list_by_status(conn, "pending")

        if not pending:
            print("no proposals awaiting sandboxing")
            return

        for proposal in pending:
            result = run_sandboxed(proposal["source_code"])
            tool_proposals.record_sandbox_result(conn, proposal["id"], result)
            status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
            print(f"#{proposal['id']} {proposal['tool_name']}: sandboxed ({status})")


if __name__ == "__main__":
    main()
