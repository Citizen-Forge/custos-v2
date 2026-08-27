"""
Manual demo/proof for the durability guarantee in PLAN.md's Phase 1 exit
criteria. Run against the docker-compose stack:

    docker compose run --rm harness python scripts/enqueue_demo.py \\
        "demo ticket" "list the files in the workspace"
    docker compose up harness          # worker claims + starts the ticket
    # ...kill it mid-task (docker compose kill harness)...
    docker compose up harness          # restart -- should resume via
                                        # beads.in_progress(), not restart
"""

import sys

from harness import beads


def main() -> None:
    title = sys.argv[1] if len(sys.argv) > 1 else "demo ticket"
    description = sys.argv[2] if len(sys.argv) > 2 else "say hello"

    beads.ensure_initialized()
    issue = beads.create(title, description)
    print(f"created {issue['id']}: {issue['title']}")


if __name__ == "__main__":
    main()
