"""
File-based project wiki -- human-facing documentation, distinct from
Beads' own per-issue notes/comments (checked live against `bd --help`,
2026-08-29: no wiki/doc/page concept exists there -- same answer as the
Slack "does bd already do this" question earlier). Lives under
WORKSPACE_ROOT/wiki/ as plain markdown files, reusing the existing
workspace-boundary safety check (permissions.check_within_workspace)
rather than a new storage layer -- also means a human can browse/edit
these directly on disk, and the files are naturally git-friendly if the
workspace is ever put under version control. An external-wiki
integration (the user's own "maybe") is a real future option once this
proves useful -- files can always be synced/exported later without
redesigning the storage layer.

Agent profile pages live at a fixed convention, agents/<seat_id> -- see
meta_agent.create_specialist_seat, which writes a new seat's own initial
profile page as part of creation (user's own framing: "a sort of profile
of who they are," reinforcing the seat's chosen identity --
seats.display_name/pronouns -- with something a human can actually read,
and something the seat itself can always read back later).
"""

import os

from .config import WORKSPACE_ROOT
from .permissions import check_within_workspace

WIKI_DIR = "wiki"


def _wiki_path(slug: str) -> str:
    slug = slug.strip("/")
    if not slug.endswith(".md"):
        slug = f"{slug}.md"
    return f"{WIKI_DIR}/{slug}"


def read_page(slug: str) -> str | None:
    """None if the page doesn't exist -- not an error, a wiki naturally
    has gaps (e.g. an agent profile that hasn't been written yet)."""
    path = _wiki_path(slug)
    check_within_workspace(path, WORKSPACE_ROOT)
    resolved = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if not os.path.exists(resolved):
        return None
    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


def write_page(slug: str, content: str) -> str:
    """Returns the page's workspace-relative path. Always overwrites --
    callers that want to preserve history should read-then-append
    themselves; this module doesn't version pages."""
    path = _wiki_path(slug)
    check_within_workspace(path, WORKSPACE_ROOT)
    resolved = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def list_pages() -> list[str]:
    """Slugs (relative to wiki/, no .md extension) of every page that
    exists -- empty list if the wiki directory doesn't exist yet, not an
    error (a fresh workspace has no pages, a normal state)."""
    wiki_root = os.path.join(WORKSPACE_ROOT, WIKI_DIR)
    if not os.path.isdir(wiki_root):
        return []
    slugs = []
    for dirpath, _, filenames in os.walk(wiki_root):
        for name in filenames:
            if name.endswith(".md"):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, wiki_root)
                slugs.append(rel[:-3].replace(os.sep, "/"))
    return sorted(slugs)


def agent_profile_slug(seat_id: str) -> str:
    return f"agents/{seat_id}"
