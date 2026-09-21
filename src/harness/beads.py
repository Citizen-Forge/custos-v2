"""
Thin subprocess wrapper around the `bd` CLI (gastownhall/beads) -- the
work-queue and cross-session-memory spine (PLAN.md's Phase 1 decision).

Every shape below was verified live against bd v1.2.2 running in this
repo's own container, not assumed from docs:
- `bd ready`, `bd update --claim`, `bd show`, `bd close`, `bd list` all
  return a JSON *array* even for a single issue -- index [0] when acting
  on one specific id.
- `bd create` and `bd remember` return a single JSON *object*, no array;
  so does `bd dep add`.
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
    # `--limit 0` is not optional: `bd ready` defaults to 100 and `bd list`
    # to 50, and a silent cap is exactly the "green command that verified
    # nothing" class of bug this module exists to avoid -- found live
    # 2026-09-17, after the tests' shared workspace passed 100 ready
    # tickets and fresh tickets stopped appearing in `bd ready` at all.
    return json.loads(_run(["ready", "--limit", "0"]))


def in_progress() -> list[dict]:
    return json.loads(_run(["list", "--status=in_progress", "--limit", "0"]))


def blocked_map() -> dict[str, list[str]]:
    """Ticket id -> ids of its OPEN blockers (`bd blocked`).

    `bd ready` already excludes blocked tickets from the fresh-work pool, but
    an in_progress ticket that is later blocked still shows up as an orphan --
    and the board has no way to say *why* it is stuck. Found live 2026-09-21:
    workspace-o0n.15.4 was in_progress, gained open blockers, and the
    dispatcher resumed + re-landed it three times in twenty minutes because it
    could never close. Fails open to {}: a broken query must not stall
    dispatch or blank the board."""
    try:
        return {i["id"]: i.get("blocked_by", []) for i in json.loads(_run(["blocked"]))}
    except Exception:
        return {}


def blocked_ids() -> set[str]:
    return set(blocked_map())


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


# Machine-checkable acceptance criteria: a JSON list of checks the verifier
# can evaluate deterministically, before it spends a model call. Free-text
# `acceptance_criteria` stays for the judgment calls; these cover the
# mechanical half ("file X exists", "the suite runs at least N tests", "the
# README contains Y"). See harness/acceptance.py for the schema.
CHECKS_KEY = "acceptance_checks"


def set_acceptance_checks(issue_id: str, checks: list, actor: str = DEFAULT_ACTOR) -> dict:
    """Store the structured checks as JSON in one metadata field, the same
    mechanism as set_acceptance_criteria.

    Validates each check's `type` against acceptance.CHECK_TYPES. Without
    this, the product-owner (an LLM) periodically invents a schema --
    observed live 2026-09-20 on workspace-o0n.4.2, which carried
    {"kind": "command_exit_zero"} -- and every check then fails as "unknown
    check type None": a false fail with no model call. Failing loudly here
    lets the caller retry with a valid shape instead."""
    from . import acceptance  # local import: acceptance imports beads

    for check in checks or []:
        if not isinstance(check, dict):
            raise BeadsError(f"acceptance check must be an object, got {type(check).__name__}")
        kind = check.get("type")
        if kind not in acceptance.CHECK_TYPES:
            raise BeadsError(
                f"unknown acceptance check type {kind!r}; valid types: "
                + ", ".join(acceptance.CHECK_TYPES)
            )
    return set_metadata(issue_id, CHECKS_KEY, json.dumps(checks), actor=actor)


def acceptance_checks(issue: dict) -> list:
    """The ticket's structured checks, or [] when it has none / they are
    unparseable. Malformed JSON reads as no checks rather than raising --
    the verifier falls back to judging the free-text criteria, which is the
    safe direction (a missing check means less pre-filtering, not a pass)."""
    raw = (issue.get("metadata") or {}).get(CHECKS_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return data if isinstance(data, list) else []


def set_metadata(issue_id: str, key: str, value: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Set one metadata key -- same `--set-metadata` mechanism
    assign_to_seat and set_acceptance_criteria already use, generalised so
    callers don't each need their own wrapper."""
    return json.loads(
        _run(["update", issue_id, "--set-metadata", f"{key}={value}"], actor=actor)
    )[0]


def unset_metadata(issue_id: str, key: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Remove one metadata key. Used by the verifier's rework requeue to
    clear `completion_summary`/`work_commit` from a failed attempt, so the
    next dispatch cannot mistake the old completion claim for a new one."""
    return json.loads(
        _run(["update", issue_id, "--unset-metadata", key], actor=actor)
    )[0]


def remove_human_flag(issue_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Drop the `human` label. A ticket requeued for rework must not stay
    parked -- both dispatch and worker._next_ticket skip flagged issues, so
    leaving the label on would re-open a ticket nothing was allowed to
    touch."""
    return json.loads(
        _run(["update", issue_id, "--remove-label", "human"], actor=actor)
    )[0]


def add_dependency(blocked_id: str, blocker_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    """Record that `blocked_id` cannot be dispatched until `blocker_id`
    closes -- `bd dep add <blocked> <blocker>`, bd's own phrasing (the
    first id "depends on" the second).

    This is what makes `bd ready` actually enforce an ordering declared
    at breakdown time, rather than leaving it as prose in a description
    that no scan reads. Found live 2026-09-04: a scaffolding ticket's
    "everything else depends on this" was pure description text with no
    graph edge behind it, so `bd ready` handed out every downstream
    story right alongside it -- dispatcher.py's next_unassigned_ticket
    preemption logic covers the same gap at dispatch time, but a real
    edge is what stops `bd ready` from ever offering the blocked work in
    the first place.

    Both ids are checked with `show` first: `bd dep add` does NOT
    validate that either side exists -- confirmed live 2026-09-04, it
    silently wired an edge to a nonexistent id and reported "added". An
    LLM tool call is exactly the caller likely to pass a typo'd or
    hallucinated id, and an edge to one that can never close would wedge
    the blocked ticket forever with no obvious cause. `show` already
    raises BeadsError for a missing id, the same failure shape every
    other caller here expects."""
    show(blocked_id)
    show(blocker_id)
    return json.loads(
        _run(["dep", "add", blocked_id, blocker_id], actor=actor)
    )


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
    return json.loads(_run(["list", "--assignee", actor, "--all", "--limit", "0"]))


def claim(issue_id: str, actor: str = DEFAULT_ACTOR) -> dict:
    return json.loads(_run(["update", issue_id, "--claim"], actor=actor))[0]


def show(issue_id: str) -> dict:
    return json.loads(_run(["show", issue_id]))[0]


def _actor_for_issue(issue_id: str, actor: str | None) -> str:
    """The actor to act as: an explicit one, else the issue's current
    assignee, else DEFAULT_ACTOR.

    bd enforces that only the assignee may close/reopen a claimed issue --
    checked live against the shipped bd: `cannot close <id>: assignee is
    "seat-x", actor is "custos-worker"; reclaim or use --force`. The
    dispatcher claims a ticket AS the seat (start_agent, `beads.claim(actor=
    seat_id)`), while worker.close ran as the system, so a normal completion
    hit that refusal. Resolving the assignee keeps completion working and
    records the actor that actually held the ticket, rather than reaching
    for --force and erasing who did the work."""
    if actor:
        return actor
    try:
        assignee = (show(issue_id).get("assignee") or "").strip()
    except Exception:
        assignee = ""
    return assignee or DEFAULT_ACTOR


def close(issue_id: str, reason: str | None = None, actor: str | None = None) -> dict:
    args = ["close", issue_id]
    if reason:
        args += ["--reason", reason]
    return json.loads(_run(args, actor=_actor_for_issue(issue_id, actor)))[0]


def reopen(issue_id: str, reason: str, actor: str | None = None) -> dict:
    """Put a wrongly-closed ticket back into the queue.

    Closing is what releases a ticket's dependents, so a ticket that
    closed but did not actually satisfy its acceptance criteria has to go
    back to open -- otherwise everything blocked on it proceeds against
    work that was never done. Found live 2026-09-09: workspace-9jg.1.6
    closed, failed verification, and stayed closed anyway -- releasing 73
    dependents onto a scaffold whose test suite ran zero tests.

    Reopening also has to RELEASE the claim, which is why `--assignee ""`
    is here. `bd --claim` sets assignee *and* status together and is only
    idempotent for the actor already holding the ticket, so an `open`
    ticket that keeps its old assignee can never be claimed by the seat
    it is now assigned to. That is worse than one stranded ticket: the
    dispatcher's selector keeps returning the same highest-priority
    ticket, the claim throws "already claimed by <other seat>", and it
    retries that ticket every cycle instead of falling through to the
    rest of the queue -- so a single reopened ticket whose seat later
    changed wedges dispatch entirely. Found live 2026-09-15:
    workspace-9jg.1.3 (assignee deterministic-tick, assigned_seat
    crew-routing-ts) had stalled the dispatcher for ~16h with 83 open
    tickets and three free agent slots, and workspace-9jg.4.2 was queued
    to stall it next. Reopening means "back in the pool", so the previous
    holder goes with it; the dispatcher re-claims under the assigned seat."""
    return json.loads(
        _run(
            [
                "update", issue_id,
                "--status", "open",
                "--append-notes", reason,
                "--assignee", "",
            ],
            actor=_actor_for_issue(issue_id, actor),
        )
    )[0]


def create(
    title: str,
    description: str,
    issue_type: str = "task",
    parent: str | None = None,
    acceptance_criteria: str | None = None,
    priority: int | None = None,
    acceptance_checks: list | None = None,
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
    metadata = {}
    if acceptance_criteria:
        metadata["acceptance_criteria"] = acceptance_criteria
    if acceptance_checks:
        metadata[CHECKS_KEY] = json.dumps(acceptance_checks)
    if metadata:
        # bd create has its own --metadata flag (a JSON object string) --
        # checked live against bd v1.2.2's own --help rather than assumed
        # (a first draft of this guessed it didn't exist and planned a
        # wasteful second `bd update` round trip instead). Distinct from
        # `--set-metadata key=value` (bd update's flag, used by
        # assign_to_seat/set_acceptance_criteria below) -- this one takes
        # the whole metadata object as JSON, for create specifically.
        args += ["--metadata", json.dumps(metadata)]
    return json.loads(_run(args))


def children_of(issue_id: str) -> list[dict]:
    """Direct children of an issue (`--parent`, checked live against bd
    v1.2.2's own --help) -- what the board UI walks to render a
    project's epics, and each epic's stories, without needing a second
    data store to track the tree shape (Beads' own hierarchy already is
    the tree)."""
    return json.loads(
        _run(["list", "--all", "--parent", issue_id, "--sort", "priority", "--limit", "0"])
    )


def list_top_level(issue_type: str | None = None) -> list[dict]:
    """Root issues only (`--no-parent`, checked live against bd v1.2.2's
    own --help) -- the projects concept (2026-08-29) deliberately reuses
    Beads' native hierarchy rather than a parallel table: a project is a
    top-level issue, an epic is its child, a story/subtask is the
    grandchild (create_subtask/add_subtask_to_epic already produce this
    shape). Sorted by priority (0=highest) so the highest-priority
    project/epic naturally comes first -- what the product-owner's
    time-slicing logic reads to decide what to work next."""
    args = ["list", "--all", "--no-parent", "--sort", "priority", "--limit", "0"]
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
    return json.loads(_run(["list", "--all", "--limit", "0"]))


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


# Set when a person -- or the product-owner answering for one -- has JUDGED
# a ticket, as opposed to merely parking it. The verifier defers to it.
#
# Without this, an acceptance was not durable: the verifier judged the
# ticket again on its next pass, failed it on the ticket's own
# contaminated commit record, and reopened it. Found live 2026-09-17 on
# workspace-9jg.1.1.1, whose notes read, in order:
#     human response: Accepted as already delivered / no-op by construction.
#     Ruling: src/TickLoop.ts is the single canonical fixed-step loop...
#     verifier fail #1: The acceptance criteria claim src/FixedStepLoop.ts
#     does not exist at HEAD, but the diff's deletion of that file is not
#     reflected in the live working tree...
# So the one path that could end this class of ticket was being undone
# within minutes, and the ticket went round again -- another agent run, the
# same refusal, the same fail. A decision by a person is not evidence the
# model gets to re-weigh.
DECISION_KEY = "human_decision"


def human_decision(issue: dict) -> str | None:
    """`answered` or `dismissed` when a person has judged this ticket, else
    None. See DECISION_KEY."""
    return (issue.get("metadata") or {}).get(DECISION_KEY)


def parked_for_human(include_closed: bool = False) -> list[dict]:
    """Every human-flagged issue, WITH notes/metadata/labels.

    `list_all` returns the lean shape (`bd list --all` carries no notes,
    metadata or labels -- see its docstring), so recovering *why* something
    was parked needs this `--long` variant. Used by the maintenance script
    that requeues failed verifications.

    Closed tickets are excluded unless `include_closed` -- plain `bd list`
    never returns them. The escalation queue needs them: a
    verifier-exhausted ticket is closed AND human-labelled, and it is the
    one case no agent can answer for itself."""
    args = ["list", "--label", "human", "--long", "--limit", "0"]
    if include_closed:
        args.append("--all")
    return json.loads(_run(args))


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
    verified working, rather than depending on the broken subcommand.

    Also drops the `human` label, because the label means "parked for a
    person" and a person has now answered. Leaving it on made an ANSWERED
    ticket indistinguishable from a parked one: the dashboard's Blocked
    column is nothing but "carries the human label" (public/index.html,
    isBlocked -- deliberately status-independent, so that a closed ticket
    is not credited as done), so work that had been decided and closed went
    on being reported as blocked. Found live 2026-09-17: workspace-9jg.2.4
    was answered by the product-owner as already delivered and still sat in
    Blocked. Verifier-exhausted tickets never come through here -- they are
    parked by flag_for_human and only ever answered by this call -- so they
    are unaffected and still read as blocked, which is the whole point."""
    append_note(issue_id, f"human response: {response}", actor=actor)
    remove_human_flag(issue_id, actor=actor)
    # Record that a PERSON judged this ticket, so the verifier does not
    # re-litigate it. See DECISION_KEY.
    set_metadata(issue_id, DECISION_KEY, "answered")
    return close(issue_id, reason="Responded")


def dismiss_human(issue_id: str, reason: str | None = None, actor: str = DEFAULT_ACTOR) -> dict:
    """See respond_to_human's docstring -- same "bd human dismiss is
    broken on embedded Dolt" workaround, composed from close() alone, and
    the same rule about the label: a dismissed ticket has been dealt with,
    so it must stop reading as blocked."""
    if reason:
        append_note(issue_id, f"dismissed: {reason}", actor=actor)
    remove_human_flag(issue_id, actor=actor)
    set_metadata(issue_id, DECISION_KEY, "dismissed")
    return close(issue_id, reason="Dismissed")


def prime() -> str:
    result = subprocess.run(
        ["bd", "prime"], cwd=WORKSPACE_ROOT, capture_output=True, text=True, timeout=30
    )
    return result.stdout
