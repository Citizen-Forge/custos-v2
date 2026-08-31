"""
What a project needs installed, and whether it's actually there.

Exists because of a real failure (2026-08-31): Silent Run is a TypeScript
project, the harness image had no Node, and nothing anywhere noticed.
Agents were dispatched onto TypeScript tickets they could not build or
test, wrote files, and produced work that was unverified by construction.
The ticket looked perfectly workable the whole time.

The point is to make that class of failure loud. A project declares the
commands its work requires, as `toolchain` metadata on the project issue
(comma-separated, e.g. "node,npm"). Before dispatch starts an agent, the
declared commands are checked; if any are missing the ticket is not
started, and the reason is recorded rather than discovered hours later in
a transcript.

Deliberately just "is this command on PATH". Not a version solver, not a
package manager, not a container spec. The failure being prevented is
"the toolchain is absent entirely", which is the one that actually
happened and the one that silently wastes inference. Version pinning
lives in the Dockerfile, where it can be reproduced.

Projects with no declared toolchain are unaffected -- absence means "no
requirements", not "requires nothing available", so this cannot retro-
actively block existing work.
"""

import shutil

from . import beads

METADATA_KEY = "toolchain"


def declared_for(issue: dict) -> list[str]:
    """Commands this project declares it needs. Empty means unconstrained."""
    raw = (issue.get("metadata") or {}).get(METADATA_KEY) or ""
    return [c.strip() for c in raw.split(",") if c.strip()]


def missing(commands: list[str]) -> list[str]:
    """Which of these aren't on PATH, in the order declared."""
    return [c for c in commands if shutil.which(c) is None]


def project_id_for(ticket_id: str) -> str:
    """The project a ticket belongs to.

    Beads ids encode the hierarchy -- `workspace-9jg.1.5` is a story under
    epic `workspace-9jg.1` under project `workspace-9jg` -- so the root is
    the id up to the first dot. Same convention api._parent_id relies on,
    and guarded by the same test."""
    return ticket_id.split(".", 1)[0]


def check_ticket(ticket_id: str) -> list[str]:
    """Commands missing for this ticket's project, or [] if it can proceed.

    Fails OPEN: if the project can't be read, this returns [] rather than
    blocking. A preflight check that itself breaks must not become a
    reason no work ever runs -- the failure it prevents is bad output,
    which is worse than a missed check but not worse than total paralysis."""
    try:
        project = beads.show(project_id_for(ticket_id))
    except Exception:
        return []
    return missing(declared_for(project))


def set_for_project(project_id: str, commands: list[str]) -> dict:
    """Declare the commands a project's work requires."""
    return beads.set_metadata(project_id, METADATA_KEY, ",".join(commands))


def report() -> list[dict]:
    """Every project with a declared toolchain and whether it is satisfied
    -- so an operator can see a gap before agents run into it."""
    out = []
    for project in beads.list_all():
        if "." in project["id"]:
            continue
        declared = declared_for(project)
        if not declared:
            continue
        gaps = missing(declared)
        out.append(
            {
                "project_id": project["id"],
                "title": project.get("title"),
                "declared": declared,
                "missing": gaps,
                "satisfied": not gaps,
            }
        )
    return out
