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

from .config import BD_TIMEOUT, DEFAULT_ACTOR, WORKSPACE_ROOT


class BeadsError(Exception):
    pass


class BeadsTimeout(BeadsError):
    """A `bd` call exceeded BD_TIMEOUT. Distinct from BeadsError so
    callers can tell "the workspace is too slow right now" apart from
    "bd rejected this command" -- the former is retryable and is what a
    growing backlog produces, the latter never is."""


def _run(args: list[str], actor: str = DEFAULT_ACTOR) -> str:
    try:
        result = subprocess.run(
            ["bd", *args, "--json", "--actor", actor],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=BD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise BeadsTimeout(f"`bd {args[0]}` exceeded BD_TIMEOUT ({BD_TIMEOUT}s)") from e
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


def assign_to_seat(issue_id: str, seat_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    """The product-owner's core assignment primitive -- earmarks a ready
    ticket for a specific seat without claiming it (status stays `open`;
    the seat's own worker process claims it on its next poll). Verified
    live: `--set-metadata` round-trips through `bd ready`/`bd show`/`bd
    list --metadata-field` correctly, which is what makes seat-scoped
    polling (worker.py's `_next_ticket`) possible without a second data
    store alongside Beads."""
    return json.loads(
        _run(["update", issue_id, "--set-metadata", f"assigned_seat={seat_id}"], actor=actor)
    )[0]


def assigned_seat(issue: dict) -> str | None:
    return (issue.get("metadata") or {}).get("assigned_seat")


def set_acceptance_criteria(issue_id: str, criteria: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Same `--set-metadata` mechanism as assign_to_seat -- no second data
    store needed for this either. A ticket with no acceptance criteria
    set is simply not a candidate for the verification loop (verifier.py
    skips it), not an error."""
    return json.loads(
        _run(["update", issue_id, "--set-metadata", f"acceptance_criteria={criteria}"], actor=actor)
    )[0]


def acceptance_criteria(issue: dict) -> str | None:
    return (issue.get("metadata") or {}).get("acceptance_criteria")


def set_metadata(issue_id: str, key: str, value: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Set one metadata key -- same `--set-metadata` mechanism
    assign_to_seat and set_acceptance_criteria already use, generalised so
    callers don't each need their own wrapper."""
    return json.loads(
        _run(["update", issue_id, "--set-metadata", f"{key}={value}"], actor=actor)
    )[0]


def declined_by(issue: dict) -> list[str]:
    """Seats that have already declined this ticket as out-of-speciality."""
    raw = (issue.get("metadata") or {}).get("declined_by") or ""
    return [s for s in raw.split(",") if s]


def release_to_pool(issue_id: str, seat_id: str, reason: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Hand a ticket back to the unassigned pool because the seat holding
    it is the wrong specialist for it.

    Deliberately NOT flag_for_human: that labels the issue `human` and
    parks it for a person, and worker._next_ticket skips flagged issues
    on purpose so they're never reclaimed. Routing "wrong specialist"
    down that path would quietly fill a human's queue with work another
    agent could pick up.

    Three things have to happen together or the ticket strands: clear the
    seat assignment (so ready_for_seat stops offering it back to the same
    seat), reopen it (`bd ready` only ever returns status=open -- see this
    module's docstring -- so a claimed ticket left in_progress would
    vanish from the pool entirely), and record the decline so the
    product-owner doesn't immediately reassign it to a seat that already
    said no."""
    declined = declined_by(show(issue_id))
    if seat_id not in declined:
        declined.append(seat_id)
    append_note(issue_id, f"declined by {seat_id}: {reason}", actor=actor)
    return json.loads(
        _run(
            [
                "update", issue_id,
                "--unset-metadata", "assigned_seat",
                "--set-metadata", f"declined_by={','.join(declined)}",
                "--status", "open",
            ],
            actor=actor,
        )
    )[0]


def unassigned_ready() -> list[dict]:
    """Ready issues with no seat assignment yet -- exactly what the
    product-owner's triage pass looks at."""
    return [i for i in ready() if not assigned_seat(i)]


def ready_for_seat(seat_id: str) -> list[dict]:
    return [i for i in ready() if assigned_seat(i) == seat_id]


def list_by_assignee(actor: str) -> list[dict]:
    """Every issue ever assigned to `actor`, any status including closed
    -- Phase 5's outcome tracking reads this directly rather than
    maintaining a separate metrics store."""
    return json.loads(_run(["list", "--assignee", actor, "--all"]))


def claim(issue_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    return json.loads(_run(["update", issue_id, "--claim"], actor=actor))[0]


def show(issue_id: str) -> dict:
    return json.loads(_run(["show", issue_id]))[0]


def close(issue_id: str, reason: str | None = None) -> dict:
    args = ["close", issue_id]
    if reason:
        args += ["--reason", reason]
    return json.loads(_run(args))[0]


def create(
    title: str,
    description: str,
    issue_type: str = "task",
    parent: str | None = None,
    acceptance_criteria: str | None = None,
    priority: int | None = None,
) -> dict:
    args = ["create", title, "-d", description, "--type", issue_type]
    if parent:
        args += ["--parent", parent]
    if priority is not None:
        # bd's own native priority field (0-4, 0=highest) -- checked live
        # against bd v1.2.2's own --help. Reused as-is for the projects
        # concept (2026-08-29) rather than inventing a parallel priority
        # scheme: a "project" is just a top-level Beads issue with this
        # field set, ordered via `bd list --sort priority`.
        args += ["--priority", str(priority)]
    if acceptance_criteria:
        # bd create has its own --metadata flag (a JSON object string) --
        # checked live against bd v1.2.2's own --help rather than assumed
        # (a first draft of this guessed it didn't exist and planned a
        # wasteful second `bd update` round trip instead). Distinct from
        # `--set-metadata key=value` (bd update's flag, used by
        # assign_to_seat/set_acceptance_criteria below) -- this one takes
        # the whole metadata object as JSON, for create specifically.
        args += ["--metadata", json.dumps({"acceptance_criteria": acceptance_criteria})]
    return json.loads(_run(args))


def children_of(issue_id: str) -> list[dict]:
    """Direct children of an issue (`--parent`, checked live against bd
    v1.2.2's own --help) -- what the board UI walks to render a
    project's epics, and each epic's stories, without needing a second
    data store to track the tree shape (Beads' own hierarchy already is
    the tree)."""
    return json.loads(_run(["list", "--all", "--parent", issue_id, "--sort", "priority"]))


def list_top_level(issue_type: str | None = None) -> list[dict]:
    """Root issues only (`--no-parent`, checked live against bd v1.2.2's
    own --help) -- the projects concept (2026-08-29) deliberately reuses
    Beads' native hierarchy rather than a parallel table: a project is a
    top-level issue, an epic is its child, a story/subtask is the
    grandchild (create_subtask/add_subtask_to_epic already produce this
    shape). Sorted by priority (0=highest) so the highest-priority
    project/epic naturally comes first -- what the product-owner's
    time-slicing logic reads to decide what to work next."""
    args = ["list", "--all", "--no-parent", "--sort", "priority"]
    if issue_type:
        args += ["--type", issue_type]
    return json.loads(_run(args))


def list_all() -> list[dict]:
    """Every issue in the workspace, any status, in ONE `bd` call.

    Exists because walking the hierarchy with `children_of` per node is
    an N+1: api.list_projects used to cost 1 + one call per project +
    one per epic, and each `bd` invocation was measured at ~5s against a
    real Dolt-backed workspace -- ~85s for a 14-epic tree, against a
    dashboard polling every 5s.

    Note `bd list` does NOT return a parent field (verified live against
    bd v1.2.2: the keys are comment_count, created_at, created_by,
    dependency_count, dependent_count, description, id, issue_type,
    owner, priority, status, title, updated_at), so callers rebuild the
    hierarchy from the dotted id convention instead -- see
    api._tree_from_flat and the test that guards that assumption."""
    return json.loads(_run(["list", "--all"]))


def update_priority(issue_id: str, priority: int, actor: str = DEFAULT_ACTOR) -> dict:
    """Set an existing issue's priority (0-4, 0=highest -- bd's own
    range, per `bd update --help` on v1.2.2).

    `create` could already set a priority, but nothing could change one
    afterwards, and create_epic/create_story never accepted one at all --
    so every epic landed at bd's default and a backlog could be built
    through the API but never ordered. Ordering the harness's own
    improvement epics required shelling into the container to run `bd
    update --priority` by hand, which is what prompted this."""
    if not 0 <= priority <= 4:
        raise BeadsError(f"priority must be 0-4 (0=highest), got {priority}")
    return json.loads(
        _run(["update", issue_id, "--priority", str(priority)], actor=actor)
    )[0]


def search(query: str, status: str = "all", limit: int = 10) -> list[dict]:
    """Verified live: keyword/substring search over title+description
    (plus a long list of filter flags -- status/label/date/etc, not used
    here). NOT semantic/embedding search -- see PLAN.md Phase 3 for why
    that's an acceptable v1 tradeoff (Qdrant dropped, not carried into
    v2)."""
    return json.loads(_run(["search", query, "--status", status, "--limit", str(limit)]))


def remember(text: str) -> dict:
    return json.loads(_run(["remember", text]))


def flag_for_human(issue_id: str, reason: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Phase 4 refuse-work primitive: labels the issue `human` (verified
    live -- `bd human list` picks up anything with this label) and records
    why, instead of silently retrying or force-completing. Deliberately
    does NOT close or otherwise change status -- worker.py must exclude
    human-labeled issues from its own resume polling, or a refused ticket
    would just get reclaimed and re-refused forever."""
    return json.loads(
        _run(["update", issue_id, "--add-label", "human", "--notes", reason], actor=actor)
    )[0]


def is_flagged_for_human(issue: dict) -> bool:
    return "human" in (issue.get("labels") or [])


def append_note(issue_id: str, text: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Verified live: --append-notes accumulates (newline-joined) rather
    than overwriting, unlike the plain --notes flag that flag_for_human
    uses for its one-shot reason."""
    return json.loads(_run(["update", issue_id, "--append-notes", text], actor=actor))[0]


def respond_to_human(issue_id: str, response: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Resolve a human-flagged issue with a response, closing it -- the
    completion half of refuse_ticket's loop (flag_for_human -> a human
    reviews it -> this).

    NOT implemented via `bd human respond`: verified live against bd
    v1.2.2 that the real subcommand hard-fails with "storage is nil" on
    an embedded (non-server) Dolt backend -- reproduced with and without
    --json, and `bd human dismiss` fails the same way, while `bd human
    list` (the read path) works fine. This composes the same documented
    effect ("adds the response as a comment[-equivalent note] and closes
    with reason 'Responded'") out of append_note + close, both already
    verified working, rather than depending on the broken subcommand."""
    append_note(issue_id, f"human response: {response}", actor=actor)
    return close(issue_id, reason="Responded")


def dismiss_human(issue_id: str, reason: str | None = None, actor: str = DEFAULT_ACTOR) -> dict:
    """See respond_to_human's docstring -- same "bd human dismiss is
    broken on embedded Dolt" workaround, composed from close() alone."""
    if reason:
        append_note(issue_id, f"dismissed: {reason}", actor=actor)
    return close(issue_id, reason="Dismissed")


def prime() -> str:
    result = subprocess.run(
        ["bd", "prime"], cwd=WORKSPACE_ROOT, capture_output=True, text=True, timeout=30
    )
    return result.stdout
