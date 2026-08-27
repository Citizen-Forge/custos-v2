"""
Thin subprocess wrapper around the `bd` CLI (gastownhall/beads) -- the
work-queue and cross-session-memory spine (PLAN.md's Phase 1 decision).

Every shape below was verified live against bd v1.2.2 running in this
repo's own container, not assumed from docs:
- `bd ready`, `bd update --claim`, `bd show`, `bd close`, `bd list` all
  return a JSON *array* even for a single issue -- index [0] when acting
  on one specific id.
- `bd create` and `bd remember` return a single JSON *object*, no array.
- `bd prime` ignores --json for its main body -- it returns a Markdown
  blob meant to be pasted straight into an agent's context, not parsed --
  used as-is when seeding a new thread's first message.
- `bd ready` only ever returns `status=open` issues with no blockers -- an
  issue claimed and left `in_progress` (e.g. by a crashed worker) drops
  out of `bd ready` entirely. Resuming orphaned work means polling
  `bd list --status=in_progress` separately -- see worker.py.
"""

import json
import subprocess

from .config import DEFAULT_ACTOR, WORKSPACE_ROOT


class BeadsError(Exception):
    pass


def _run(args: list[str], actor: str = DEFAULT_ACTOR) -> str:
    result = subprocess.run(
        ["bd", *args, "--json", "--actor", actor],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise BeadsError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def ensure_initialized() -> None:
    check = subprocess.run(["bd", "where"], cwd=WORKSPACE_ROOT, capture_output=True, text=True)
    if check.returncode == 0:
        return
    subprocess.run(
        ["bd", "init", "--skip-agents", "--skip-hooks", "--json"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def ready() -> list[dict]:
    return json.loads(_run(["ready"]))


def in_progress() -> list[dict]:
    return json.loads(_run(["list", "--status=in_progress"]))


def claim(issue_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    return json.loads(_run(["update", issue_id, "--claim"], actor=actor))[0]


def close(issue_id: str, reason: str | None = None) -> dict:
    args = ["close", issue_id]
    if reason:
        args += ["--reason", reason]
    return json.loads(_run(args))[0]


def create(title: str, description: str, issue_type: str = "task") -> dict:
    return json.loads(_run(["create", title, "-d", description, "--type", issue_type]))


def remember(text: str) -> dict:
    return json.loads(_run(["remember", text]))


def prime() -> str:
    result = subprocess.run(
        ["bd", "prime"], cwd=WORKSPACE_ROOT, capture_output=True, text=True, timeout=30
    )
    return result.stdout
