"""
External agent seats: sessions running OUTSIDE Custos (a Claude Code or
opencode session on someone's machine) that take tickets from the same board,
work them in their own checkout, and submit them for the same verifier-gated
merge an internal agent's work goes through.

Shape of one ticket's life:

1. `request_ticket` -- the seat's own in-flight or reworked ticket first, else
   the earliest ready ticket in roadmap order that dispatch itself would
   consider (dispatchable, no open children, no open blocker, not parked for a
   human, project not on hold, not harness-targeted). The ticket is re-seated
   to the external seat BEFORE it is claimed, so the dispatcher, which skips
   external seats, stops considering it; the claim is strict (no --force), so
   an internal agent that got there first keeps it.
2. The session clones the project over git, commits on a branch of its own,
   and pushes it to `external/<ticket-id>`. That is the only ref a push may
   touch (an `update` hook enforces it): `ticket/<id>` is usually checked out
   in the harness's worktree, which git refuses to push into, and the
   integration branch is only ever advanced by the gated merge.
3. `submit_ticket` points the ticket's worktree at the pushed tip, records the
   completion summary and `work_commit = base..tip`, and closes the ticket --
   exactly the state worker.work_one_ticket leaves an internal ticket in. The
   dispatcher then verifies it (and the verifier lands it on a pass), so the
   judgement is the same one an internal agent's work gets.

A lease stops an abandoned session from holding a ticket forever: every call
extends it, and the dispatcher hands expired tickets back to the pool.

External seats live in the `seats` table with status 'external', which keeps
them out of every roster the product-owner and the dispatcher's fast path read
(both list status 'active').
"""

import hashlib
import hmac
import logging
import os
import secrets
import subprocess
import time

import psycopg

from . import beads, seats, toolchain, workspaces

log = logging.getLogger("external")

EXTERNAL_STATUS = "external"
LEASE_SECONDS = int(os.environ.get("EXTERNAL_LEASE_SECONDS", "1800"))
LEASE_KEY = "external_lease_until"
PREVIOUS_SEAT_KEY = "external_previous_seat"

# How a session reaches a project's repository. `{project_id}` is filled in;
# the default is the box's own path over SSH, which is what an operator's
# machine with root SSH to the box can use as-is.
GIT_URL_TEMPLATE = os.environ.get(
    "EXTERNAL_GIT_URL_TEMPLATE",
    "ssh://root@192.168.100.231/mnt/dockermain/appdata/custos-v2/projects/{project_id}",
)

PUSH_PREFIX = "external/"

_UPDATE_HOOK = """#!/bin/sh
# Installed by custos (harness/external.py). External agents may only push
# their own ticket branches; the integration branch is advanced only by the
# verifier-gated merge.
case "$1" in
  refs/heads/external/*) exit 0 ;;
esac
echo "custos: pushes are only accepted to refs/heads/external/<ticket-id> (got $1)" >&2
exit 1
"""


class ExternalError(Exception):
    """A request the caller got wrong -- surfaced as a 4xx, not a crash."""


# -- seats and tokens ---------------------------------------------------------


def init_table(conn) -> None:
    seats.init_table(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_seat_tokens (
            seat_id TEXT PRIMARY KEY,
            token_sha256 TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_submissions (
            ticket_id TEXT PRIMARY KEY,
            seat_id TEXT NOT NULL,
            submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            verified_at TIMESTAMPTZ
        )
        """
    )


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def register_seat(conn, seat_id: str, description: str) -> str:
    """Create (or re-key) an external seat and return its token. The token is
    shown once; only its hash is stored."""
    if not seat_id or any(c.isspace() for c in seat_id):
        raise ExternalError("seat_id must be a non-empty identifier without spaces")
    existing = seats.get(conn, seat_id)
    if existing and existing["status"] != EXTERNAL_STATUS:
        raise ExternalError(f"{seat_id!r} is already an internal seat")
    if not existing:
        seats.create(conn, seat_id, f"external agent: {description}", created_by="operator")
    conn.execute("UPDATE seats SET status = %s WHERE seat_id = %s", (EXTERNAL_STATUS, seat_id))
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO external_seat_tokens (seat_id, token_sha256) VALUES (%s, %s) "
        "ON CONFLICT (seat_id) DO UPDATE SET token_sha256 = EXCLUDED.token_sha256, created_at = now()",
        (seat_id, _hash(token)),
    )
    return token


def seat_for_token(conn, token: str | None) -> str | None:
    if not token:
        return None
    digest = _hash(token)
    for seat_id, stored in conn.execute(
        "SELECT t.seat_id, t.token_sha256 FROM external_seat_tokens t "
        "JOIN seats s ON s.seat_id = t.seat_id WHERE s.status = %s",
        (EXTERNAL_STATUS,),
    ).fetchall():
        if hmac.compare_digest(stored, digest):
            return seat_id
    return None


_ids_cache: dict = {"at": -1e9, "ids": frozenset()}
_IDS_TTL = 60.0


def seat_ids(conn_string: str | None = None) -> frozenset[str]:
    """Every external seat id, cached briefly -- dispatch asks every poll.
    Fails open to the last known set (initially empty): a database hiccup must
    not stall dispatch."""
    now = time.monotonic()
    if now - _ids_cache["at"] < _IDS_TTL:
        return _ids_cache["ids"]
    try:
        with psycopg.connect(conn_string or os.environ["DATABASE_URL"], autocommit=True) as conn:
            seats.init_table(conn)
            rows = conn.execute(
                "SELECT seat_id FROM seats WHERE status = %s", (EXTERNAL_STATUS,)
            ).fetchall()
        _ids_cache.update(at=now, ids=frozenset(r[0] for r in rows))
    except Exception:
        log.exception("could not read external seats; keeping the last known set")
        _ids_cache["at"] = now
    return _ids_cache["ids"]


def invalidate_cache() -> None:
    _ids_cache["at"] = -1e9


# -- git ----------------------------------------------------------------------


def push_branch(ticket_id: str) -> str:
    return f"{PUSH_PREFIX}{ticket_id}"


def _git(args: list[str], cwd: str, timeout: int = 120):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


def ensure_push_guard(project_id: str) -> None:
    """Install the update hook that confines pushes to external/* branches."""
    repo = workspaces.ensure(project_id)
    hooks = os.path.join(repo, ".git", "hooks")
    os.makedirs(hooks, exist_ok=True)
    path = os.path.join(hooks, "update")
    try:
        with open(path) as fh:
            if fh.read() == _UPDATE_HOOK:
                return
    except OSError:
        pass
    with open(path, "w") as fh:
        fh.write(_UPDATE_HOOK)
    os.chmod(path, 0o755)


def _branch_tip(repo: str, branch: str) -> str | None:
    out = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo)
    return out.stdout.strip() if out.returncode == 0 else None


# -- leases -------------------------------------------------------------------


def _extend_lease(ticket_id: str, seat_id: str) -> float:
    until = time.time() + LEASE_SECONDS
    beads.set_metadata(ticket_id, LEASE_KEY, str(int(until)), actor=seat_id)
    return until


def lease_until(issue: dict) -> float | None:
    raw = (issue.get("metadata") or {}).get(LEASE_KEY)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _release(issue: dict, seat_id: str, reason: str) -> None:
    """Hand a ticket back: its previous seat (or the unassigned pool), open,
    unclaimed. The pushed branch is deleted so its commits cannot leak into a
    later attempt's diff (commits_for_ticket reads `git log --all`); its tip is
    recorded in the note so the work stays recoverable until git gc."""
    ticket_id = issue["id"]
    repo = workspaces.path_for(workspaces.project_id_for(ticket_id))
    tip = None
    if os.path.isdir(repo):
        tip = _branch_tip(repo, push_branch(ticket_id))
        if tip:
            _git(["branch", "-D", push_branch(ticket_id)], repo)
    previous = (issue.get("metadata") or {}).get(PREVIOUS_SEAT_KEY)
    note = f"released by external seat {seat_id}: {reason}"
    if tip:
        note += f" (its pushed work was at {tip[:12]})"
    beads.append_note(ticket_id, note, actor=seat_id)
    # --if-assignee: a no-op, not an overwrite, if the claim changed hands.
    args = ["update", ticket_id, "--status", "open", "--assignee", "", "--if-assignee", seat_id,
            "--unset-metadata", LEASE_KEY, "--unset-metadata", PREVIOUS_SEAT_KEY]
    args += (["--set-metadata", f"assigned_seat={previous}"] if previous
             else ["--unset-metadata", "assigned_seat"])
    beads._run(args, actor=seat_id)


def sweep_expired(conn_string: str | None = None, now: float | None = None) -> list[str]:
    """Release external claims whose lease has lapsed. Returns their ids."""
    ids = seat_ids(conn_string)
    if not ids:
        return []
    now = time.time() if now is None else now
    released = []
    for issue in beads.in_progress():
        seat_id = beads.assigned_seat(issue)
        if seat_id not in ids or beads.is_flagged_for_human(issue):
            continue
        until = lease_until(issue)
        if until is not None and until < now:
            try:
                _release(issue, seat_id, "lease expired without a heartbeat or submit")
                released.append(issue["id"])
            except Exception:
                log.exception("could not release expired external claim on %s", issue["id"])
    return released


# -- the ticket lifecycle -----------------------------------------------------


def _own(issue: dict, seat_id: str) -> None:
    if beads.assigned_seat(issue) != seat_id:
        raise ExternalError(f"{issue['id']} is not held by {seat_id}")


def _brief(issue: dict, seat_id: str) -> dict:
    record = beads.show(issue["id"])
    project_id = workspaces.project_id_for(issue["id"])
    meta = record.get("metadata") or {}
    return {
        "ticket": {
            "id": record["id"],
            "title": record.get("title"),
            "description": record.get("description"),
            "notes": record.get("notes"),
            "priority": record.get("priority"),
            "acceptance_criteria": beads.acceptance_criteria(record),
            "acceptance_checks": beads.acceptance_checks(record),
            "rework_reason": meta.get("rework_reason"),
            "dependencies": [
                {"id": d.get("id"), "title": d.get("title"), "status": d.get("status")}
                for d in record.get("dependencies") or []
            ],
        },
        "seat_id": seat_id,
        "lease_expires_at": lease_until(record),
        "git": {
            "clone_url": GIT_URL_TEMPLATE.format(project_id=project_id),
            "base_branch": workspaces.integration_ref(project_id),
            "push_branch": push_branch(record["id"]),
            "how": (
                f"Clone (or fetch) the repository, branch from "
                f"{workspaces.integration_ref(project_id)}, commit your work, then "
                f"`git push origin HEAD:{push_branch(record['id'])}`. That branch is the only "
                "ref a push is accepted to. Call submit_ticket once it is pushed."
            ),
        },
    }


def _eligible(issue: dict, held: dict, parents: set, blocked: set, external_ids) -> bool:
    from . import dispatcher

    if not dispatcher.dispatchable(issue):
        return False
    if issue["id"] in parents or issue["id"] in blocked or beads.is_flagged_for_human(issue):
        return False
    project_id = toolchain.project_id_for(issue["id"])
    if project_id in held or dispatcher.targets_harness(issue["id"]):
        return False
    seat = beads.assigned_seat(issue)
    return seat is None or seat not in external_ids


def _claim(issue: dict, seat_id: str) -> None:
    """Re-seat, then claim strictly. Re-seating first is what makes the
    dispatcher skip the ticket; claiming without --force is what stops an
    external request from stealing a ticket an internal agent already has."""
    ticket_id = issue["id"]
    previous = beads.assigned_seat(issue)
    args = ["update", ticket_id, "--set-metadata", f"assigned_seat={seat_id}"]
    if previous and previous != seat_id:
        args += ["--set-metadata", f"{PREVIOUS_SEAT_KEY}={previous}"]
    beads._run(args, actor=seat_id)
    try:
        beads._run(["update", ticket_id, "--claim"], actor=seat_id)
    except beads.BeadsError:
        # Someone else claimed it first. Put the seat back so nothing is
        # stranded on a seat that does not hold it.
        undo = ["update", ticket_id, "--unset-metadata", PREVIOUS_SEAT_KEY]
        undo += (["--set-metadata", f"assigned_seat={previous}"] if previous
                 else ["--unset-metadata", "assigned_seat"])
        beads._run(undo, actor=seat_id)
        raise


def request_ticket(seat_id: str, project_id: str | None = None,
                   conn_string: str | None = None) -> dict | None:
    """The next ticket for this external seat, claimed and leased, or None."""
    from . import dispatcher

    external_ids = seat_ids(conn_string) | {seat_id}

    # Already holding one: hand it back rather than taking a second.
    for issue in beads.in_progress():
        if beads.assigned_seat(issue) == seat_id and not beads.is_flagged_for_human(issue):
            _extend_lease(issue["id"], seat_id)
            return _brief(issue, seat_id)

    ready = beads.ready()
    held = dispatcher.held_projects()
    parents = dispatcher._parents_with_open_children()
    blocked = beads.blocked_ids()

    def _in_project(issue):
        return project_id is None or toolchain.project_id_for(issue["id"]) == project_id

    # Rework sent back to this seat by the verifier comes before new work.
    mine = [
        i for i in ready
        if beads.assigned_seat(i) == seat_id and _in_project(i)
        and i["id"] not in parents and i["id"] not in blocked
        and not beads.is_flagged_for_human(i)
    ]
    fresh = [
        i for i in ready
        if _in_project(i) and _eligible(i, held, parents, blocked, external_ids)
    ]
    for pool in (mine, fresh):
        for issue in sorted(pool, key=dispatcher._order):
            try:
                _claim(issue, seat_id)
            except beads.BeadsError:
                log.info("external %s lost %s to another claimant; trying the next", seat_id, issue["id"])
                continue
            ensure_push_guard(toolchain.project_id_for(issue["id"]))
            _extend_lease(issue["id"], seat_id)
            log.info("external seat %s claimed %s", seat_id, issue["id"])
            return _brief(issue, seat_id)
    return None


def heartbeat(seat_id: str, ticket_id: str) -> float:
    issue = beads.show(ticket_id)
    _own(issue, seat_id)
    if issue.get("status") != "in_progress":
        raise ExternalError(f"{ticket_id} is {issue.get('status')}, not in progress")
    return _extend_lease(ticket_id, seat_id)


def get_ticket(seat_id: str, ticket_id: str) -> dict:
    issue = beads.show(ticket_id)
    _own(issue, seat_id)
    return _brief(issue, seat_id)


def release_ticket(seat_id: str, ticket_id: str, reason: str) -> None:
    issue = beads.show(ticket_id)
    _own(issue, seat_id)
    _release(issue, seat_id, reason or "released by the session")


def submit_ticket(conn, seat_id: str, ticket_id: str, summary: str) -> dict:
    """Put a pushed external branch through the same completion an internal
    agent's work gets. Returns what was recorded."""
    from . import verifier

    if not summary or not summary.strip():
        raise ExternalError("a completion summary is required: say what was done and how it was checked")
    issue = beads.show(ticket_id)
    _own(issue, seat_id)
    if issue.get("status") != "in_progress":
        raise ExternalError(f"{ticket_id} is {issue.get('status')}, not in progress")

    project_id = workspaces.project_id_for(ticket_id)
    repo = workspaces.path_for(project_id)
    tip = _branch_tip(repo, push_branch(ticket_id))
    if not tip:
        raise ExternalError(
            f"nothing pushed: push your work to {push_branch(ticket_id)} before submitting"
        )
    base_ref = workspaces.integration_ref(project_id)
    base = _git(["merge-base", base_ref, tip], repo).stdout.strip() if base_ref else ""
    if not base:
        raise ExternalError(f"{push_branch(ticket_id)} does not share history with {base_ref}")
    if base == tip:
        raise ExternalError(f"{push_branch(ticket_id)} has no commits beyond {base_ref}")

    # The ticket's own worktree is the tree the verifier judges and runs the
    # suite in, and ticket/<id> is the branch land() merges -- so both move to
    # the pushed work. Resetting (not merging) is right: the external branch IS
    # this ticket's attempt, based on the integration tip it was cut from.
    worktree = workspaces.for_ticket(ticket_id)
    out = _git(["reset", "--hard", tip], worktree)
    if out.returncode != 0:
        raise RuntimeError(f"could not move {ticket_id}'s worktree to {tip[:12]}: {out.stderr.strip()}")
    _git(["clean", "-qfd", "-e", "node_modules"], worktree)

    span = f"{base}..{tip}"
    beads.set_metadata(ticket_id, "completion_summary", summary[:4000], actor=seat_id)
    beads.set_metadata(ticket_id, "work_commit", span, actor=seat_id)
    beads._run(["update", ticket_id, "--unset-metadata", LEASE_KEY], actor=seat_id)

    # A ticket with no criteria is never judged, so nothing else would land
    # it -- same rule as worker.work_one_ticket.
    current = beads.show(ticket_id)
    if not beads.acceptance_criteria(current) and not beads.acceptance_checks(current):
        if not verifier.land(ticket_id, conn):
            return {"ticket_id": ticket_id, "work_commit": span, "state": "flagged: could not merge"}

    beads.close(ticket_id, reason=summary[:500], actor=seat_id)
    conn.execute(
        "INSERT INTO external_submissions (ticket_id, seat_id) VALUES (%s, %s) "
        "ON CONFLICT (ticket_id) DO UPDATE SET seat_id = EXCLUDED.seat_id, "
        "submitted_at = now(), verified_at = NULL",
        (ticket_id, seat_id),
    )
    log.info("external seat %s submitted %s (%s)", seat_id, ticket_id, span)
    return {"ticket_id": ticket_id, "work_commit": span, "state": "submitted: awaiting verification"}


def pending_verification(conn) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT ticket_id FROM external_submissions WHERE verified_at IS NULL ORDER BY submitted_at"
        ).fetchall()
    ]


def mark_verified(conn, ticket_id: str) -> None:
    conn.execute(
        "UPDATE external_submissions SET verified_at = now() WHERE ticket_id = %s", (ticket_id,)
    )
