"""
Minimal HTTP API over the harness's existing state -- Beads tickets,
pending prompt proposals, per-role outcomes. Mostly read/narrow-write
(approve a pending prompt, respond to/dismiss a human-flagged ticket),
matching v1's "autonomy off by default" posture -- this is an admin
surface for a human to inspect and resolve things, not a general write
API.

Exception: POST /projects, /projects/{id}/epics, /epics/{id}/stories
(added 2026-08-30, for the MCP-based chat interface agent -- see
PLAN.md). These mirror product_owner.py's create_project/create_epic/
add_subtask_to_epic tools exactly (same beads.create() calls, same
issue_type/parent shape) -- until now, creating new project structure
was only reachable from inside the product-owner's own agent session,
never from the outside. A human (or an agent acting on a human's
behalf, e.g. a chat interface) deciding what new work should exist is
a fundamentally different kind of write than the narrow
"resolve/approve something already in flight" ones above -- creating
new backlog is closer to what a human already does by talking to the
product-owner conversationally than to a privileged action needing a
tighter gate.

Auth: a shared bearer token via API_AUTH_TOKEN (see auth.py) -- optional,
same "flagged, not hidden" posture the module docstring used to carry
here when this was unbuilt. /health stays open (docker healthcheck has
no way to carry a token) and the static dashboard files are served
unauthenticated (the dashboard JS itself carries the token on its own
API calls, see public/index.html) -- everything else requires it
whenever API_AUTH_TOKEN is set.
"""

import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import psycopg
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import avatar, beads, model_registry, outcomes, prompts, seats, self_mod, settings, tool_proposals, verifications, wiki
from .auth import require_auth
from .config import PROJECT_TREE_TTL


class RespondBody(BaseModel):
    response: str


class DismissBody(BaseModel):
    reason: str | None = None


class CostSliderBody(BaseModel):
    value: int


class AvatarStyleBody(BaseModel):
    value: str


class CreateProjectBody(BaseModel):
    name: str
    description: str
    priority: int


class CreateEpicBody(BaseModel):
    title: str
    description: str
    priority: int | None = None


class CreateStoryBody(BaseModel):
    title: str
    description: str
    priority: int | None = None


class PriorityBody(BaseModel):
    priority: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Found live: without this, a fresh workspace (no .beads yet) 500s on
    # the very first /tickets request instead of returning an empty list.
    # worker.py already did this on its own startup; api.py hadn't, since
    # it can be the very first thing to touch a workspace (e.g. checking
    # the dashboard before any ticket work has happened).
    beads.ensure_initialized()
    yield


app = FastAPI(title="Custos v2 harness API", lifespan=lifespan)

# Every route below except /health requires API_AUTH_TOKEN when it's
# set (see auth.py) -- grouped on a router rather than each endpoint
# individually so a new endpoint is protected by default, not by
# remembering to add the dependency each time.
router = APIRouter(dependencies=[Depends(require_auth)])

_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "public")


def _parent_id(issue_id: str) -> str | None:
    """Beads ids encode the hierarchy: `workspace-9jg` is a project,
    `workspace-9jg.1` an epic under it, `workspace-9jg.1.1` a story under
    that. `bd list` returns no parent field of its own (see
    beads.list_all), so this convention is how the tree gets rebuilt from
    one flat call instead of one call per node.

    This IS an assumption about bd's id format rather than a documented
    contract -- tests/test_api.py cross-checks a tree built this way
    against one built from bd's own --parent walk, so a format change
    fails loudly there instead of silently rendering an empty board."""
    return issue_id.rsplit(".", 1)[0] if "." in issue_id else None


def _sort_key(issue: dict) -> tuple:
    """Priority first (0=highest, matching `bd list --sort priority`),
    then the id read naturally so `.2` precedes `.10` -- a plain string
    sort puts `.10` first, which visibly mis-ordered the board once
    epic counts passed 9."""
    segments = []
    for segment in issue["id"].split("."):
        segments.append((0, int(segment)) if segment.isdigit() else (1, segment))
    return (issue.get("priority", 9), segments)


def _tree_from_flat(issues: list[dict]) -> list[dict]:
    """Build the project -> epic -> story tree from one flat issue list.

    Projects are top-level issues of type `epic` -- the same filter the
    per-node walk used, and it matters for the same reason documented in
    list_projects below: plain top-level `task` tickets are not projects
    and must not render as bare ones."""
    by_parent: dict[str | None, list[dict]] = defaultdict(list)
    for issue in issues:
        by_parent[_parent_id(issue["id"])].append(issue)

    tree = []
    for project in sorted(by_parent[None], key=_sort_key):
        if project.get("issue_type") != "epic":
            continue
        epics = sorted(by_parent.get(project["id"], []), key=_sort_key)
        for epic in epics:
            epic["stories"] = sorted(by_parent.get(epic["id"], []), key=_sort_key)
        project["epics"] = epics
        tree.append(project)
    return tree


# Serving the tree straight from `bd` put a ~5s call in the request path
# of a dashboard polling every 5s, so requests overlapped indefinitely
# and the board never settled. Stale-while-revalidate instead: a fresh
# snapshot is returned immediately, a stale one is returned immediately
# AND refreshed in the background, and only a cold start blocks.
_tree_load_lock = threading.Lock()
_tree_state_lock = threading.Lock()
_tree_state: dict = {"tree": None, "at": 0.0}


def _load_project_tree() -> list[dict]:
    tree = _tree_from_flat(beads.list_all())
    with _tree_state_lock:
        _tree_state["tree"] = tree
        _tree_state["at"] = time.monotonic()
    return tree


def _project_tree() -> tuple[list[dict], float]:
    """Returns (tree, age_seconds). Age is exposed so a caller can tell
    how stale the snapshot is rather than having to assume it is live."""
    with _tree_state_lock:
        tree, at = _tree_state["tree"], _tree_state["at"]
    age = time.monotonic() - at

    if tree is not None and age < PROJECT_TREE_TTL:
        return tree, age

    if tree is None:
        # Cold start has to block, but only one caller should pay for it.
        with _tree_load_lock:
            with _tree_state_lock:
                if _tree_state["tree"] is not None:
                    return _tree_state["tree"], 0.0
            return _load_project_tree(), 0.0

    # Warm but stale: refresh behind the response, never block on it.
    if _tree_load_lock.acquire(blocking=False):
        def refresh():
            try:
                _load_project_tree()
            except Exception as e:  # noqa: BLE001 -- a failed refresh must
                # never kill the thread silently or poison the cache; the
                # last good tree keeps being served until one succeeds.
                print(f"project tree refresh failed: {e}")
            finally:
                _tree_load_lock.release()

        threading.Thread(target=refresh, daemon=True).start()

    return tree, age


def _invalidate_project_tree() -> None:
    """Drop the cached tree so the next /projects reflects a write
    immediately. Without this, creating a project through the API and
    then reloading the board could show the pre-write tree for up to
    PROJECT_TREE_TTL seconds, which reads as the write having failed."""
    with _tree_state_lock:
        _tree_state["tree"] = None
        _tree_state["at"] = 0.0


def _prompt_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    return conn


def _seats_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    seats.init_table(conn)
    verifications.init_table(conn)
    return conn


def _tool_proposals_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    tool_proposals.init_table(conn)
    return conn


def _self_mod_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    self_mod.init_table(conn)
    return conn


@app.get("/health")
def health():
    return {"status": "ok"}


@router.get("/tickets")
def list_tickets(status: str = "ready"):
    if status == "ready":
        return beads.ready()
    if status == "in_progress":
        return [i for i in beads.in_progress() if not beads.is_flagged_for_human(i)]
    if status == "human":
        return [i for i in beads.in_progress() if beads.is_flagged_for_human(i)]
    raise HTTPException(400, f"unknown status {status!r}, expected ready|in_progress|human")


@router.get("/projects")
def list_projects(response: Response):
    """The full project -> epic -> story tree for the board UI. Walks
    Beads' own hierarchy live (no cached/parallel structure to drift out
    of sync) -- fine at today's scale (a handful of projects/epics), not
    optimized for hundreds.

    Real bug found live 2026-08-30: this used to call
    `beads.list_top_level()` with no filter, which returns EVERY
    top-level issue -- including plain one-off `task`-type tickets that
    were never meant to be projects (ad-hoc validation tickets predating
    the projects concept, or from enqueue_demo.py). `product_owner.
    create_project` always creates real projects with `issue_type="epic"`
    (see product_owner.py), so filtering on that excludes stray tasks
    without excluding any real project -- caught because the new board/
    roadmap UI (2026-08-29) made stray tickets rendering as bare
    "projects" with no epics/stories immediately obvious in a way the
    old nested-outline view didn't.

    Rewritten 2026-08-31 from an N+1 walk (one `bd` call per project and
    per epic) to a single `bd list --all` plus in-process tree building.
    Measured on the real deployment: one bd call is ~5s, and a 2-project
    /14-epic tree cost ~17 of them -- roughly 85s per response against a
    dashboard polling every 5s. The tree is rebuilt from the dotted id
    convention (see _parent_id) because `bd list` carries no parent
    field, and served through a short-lived cache (see _project_tree) so
    the bd call sits behind responses rather than inside them.

    The response body is unchanged -- still a plain list -- so existing
    callers (the dashboard, mcp-server) keep working. Snapshot age is
    reported in the X-Tree-Age-Seconds header instead of in the body."""
    tree, age = _project_tree()
    response.headers["X-Tree-Age-Seconds"] = f"{age:.1f}"
    return tree


@router.patch("/issues/{issue_id}/priority")
def set_issue_priority(issue_id: str, body: PriorityBody):
    """Change any issue's priority (0-4, 0=highest).

    Filling a real gap: POST /projects could set a priority at creation
    and nothing else could ever change one, so a backlog could be built
    through this API but never ordered. Ordering this project's own
    epics needed `docker exec ... bd update --priority` against the
    container, which is not something an API-driven caller should have
    to reach for."""
    try:
        updated = beads.update_priority(issue_id, body.priority)
    except beads.BeadsTimeout as e:
        raise HTTPException(503, str(e)) from e
    except beads.BeadsError as e:
        raise HTTPException(400, str(e)) from e
    _invalidate_project_tree()
    return updated


@router.post("/projects")
def create_project(body: CreateProjectBody):
    """Mirrors product_owner.py's create_project tool exactly (same
    beads.create() call) -- see this module's docstring for why this is
    a write endpoint despite the surface's otherwise-narrow posture."""
    project = beads.create(body.name, body.description, issue_type="epic", priority=body.priority)
    _invalidate_project_tree()
    return project


@router.post("/projects/{project_id}/epics")
def create_epic(project_id: str, body: CreateEpicBody):
    """Mirrors product_owner.py's create_epic tool. 404s if project_id
    doesn't exist -- beads.create's own --parent validation surfaces as
    a BeadsError, same pattern as get_ticket below."""
    try:
        epic = beads.create(
            body.title, body.description, issue_type="epic",
            parent=project_id, priority=body.priority,
        )
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e
    _invalidate_project_tree()
    return epic


@router.post("/epics/{epic_id}/stories")
def create_story(epic_id: str, body: CreateStoryBody):
    """Mirrors product_owner.py's add_subtask_to_epic tool."""
    try:
        story = beads.create(
            body.title, body.description, parent=epic_id, priority=body.priority,
        )
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e
    _invalidate_project_tree()
    return story


@router.get("/tickets/{issue_id}")
def get_ticket(issue_id: str):
    try:
        return beads.show(issue_id)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/tickets/{issue_id}/respond")
def respond_to_ticket(issue_id: str, body: RespondBody):
    """Resolve a human-flagged ticket with a response, closing it -- see
    beads.respond_to_human's docstring for why this composes append_note
    + close rather than calling the real (broken) `bd human respond`."""
    try:
        return beads.respond_to_human(issue_id, body.response)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/tickets/{issue_id}/dismiss")
def dismiss_ticket(issue_id: str, body: DismissBody = DismissBody()):
    try:
        return beads.dismiss_human(issue_id, body.reason)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/prompts/pending")
def list_pending_prompts(role: str | None = None):
    conn = _prompt_conn()
    try:
        return prompts.pending(conn, role)
    finally:
        conn.close()


@router.post("/prompts/{role}/{version}/approve")
def approve_prompt(role: str, version: int):
    conn = _prompt_conn()
    try:
        prompts.approve(conn, role, version)
        return {"role": role, "version": version, "status": "active"}
    finally:
        conn.close()


@router.get("/outcomes/{actor}")
def get_outcomes(actor: str):
    return outcomes.summary(actor)


@router.get("/seats")
def list_seats():
    conn = _seats_conn()
    try:
        roster = seats.list_all(conn)
        for s in roster:
            s["outcomes"] = outcomes.summary(s["seat_id"])
            s["verification"] = verifications.summary(conn, s["seat_id"])
            s["queue"] = outcomes.queue_stats(s["seat_id"])
        return roster
    finally:
        conn.close()


@router.get("/tool-proposals")
def list_tool_proposals(status: str = "reviewed"):
    """Default to `reviewed`: what a human should look at (sandboxed +
    a reviewer verdict attached, still short of active either way) --
    PLAN.md Phase 7's promotion gate."""
    conn = _tool_proposals_conn()
    try:
        return tool_proposals.list_by_status(conn, status)
    finally:
        conn.close()


@router.get("/wiki")
def list_wiki_pages():
    return {"pages": wiki.list_pages()}


@router.get("/wiki/{slug:path}")
def get_wiki_page(slug: str):
    content = wiki.read_page(slug)
    if content is None:
        raise HTTPException(status_code=404, detail=f"no wiki page at {slug!r}")
    return {"slug": slug, "content": content}


@router.get("/avatars/{seat_id}")
def get_avatar(seat_id: str):
    """A real Gemini-generated portrait for this seat, if one exists
    (avatar.py -- optional, needs GEMINI_API_KEY configured). 404 when
    none was generated -- the dashboard's <img onerror> falls back to a
    deterministic DiceBear avatar in that case, not an error state."""
    path = avatar.avatar_path(seat_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no generated avatar for {seat_id!r}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/settings/cost-slider")
def get_cost_slider():
    """0 (slow/free) to 100 (fast/costly) -- steers which configured
    provider tier the product-owner should reach for. See
    model_registry.py -- scaffolding today, not yet wired to automatic
    per-call routing (only a local provider is typically configured)."""
    conn = _seats_conn()  # any connection works; reuses the seats helper's init pattern
    try:
        settings.init_table(conn)
        return {"value": settings.get_cost_slider(conn)}
    finally:
        conn.close()


@router.put("/settings/cost-slider")
def set_cost_slider(body: CostSliderBody):
    conn = _seats_conn()
    try:
        settings.init_table(conn)
        try:
            settings.set_cost_slider(conn, body.value)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"value": settings.get_cost_slider(conn)}
    finally:
        conn.close()


@router.get("/settings/model-registry")
def get_model_registry():
    registry = model_registry.load_registry()
    return [{"name": p.name, "model": p.model, "cost_tier": p.cost_tier} for p in registry]


@router.get("/settings/avatar-style")
def get_avatar_style():
    """A DiceBear (api.dicebear.com) style name, applied to every seat's
    avatar in the dashboard via its seat_id as the seed."""
    conn = _seats_conn()
    try:
        settings.init_table(conn)
        return {"value": settings.get_avatar_style(conn)}
    finally:
        conn.close()


@router.put("/settings/avatar-style")
def set_avatar_style(body: AvatarStyleBody):
    conn = _seats_conn()
    try:
        settings.init_table(conn)
        try:
            settings.set_avatar_style(conn, body.value)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"value": settings.get_avatar_style(conn)}
    finally:
        conn.close()


@router.post("/tool-proposals/{proposal_id}/approve")
def approve_tool_proposal(proposal_id: int):
    """Manual override: `reviewer.review_proposal` already calls this
    itself for an "allow" verdict (2026-08-29), so a proposal is normally
    already approved by the time a human sees it here -- this endpoint
    exists for a human to override a "deny" verdict, not as the only path
    to approval anymore."""
    conn = _tool_proposals_conn()
    try:
        tool_proposals.approve(conn, proposal_id)
        return tool_proposals.get(conn, proposal_id)
    finally:
        conn.close()


@router.post("/tool-proposals/{proposal_id}/reject")
def reject_tool_proposal(proposal_id: int, body: DismissBody = DismissBody()):
    conn = _tool_proposals_conn()
    try:
        tool_proposals.reject(conn, proposal_id, body.reason)
        return tool_proposals.get(conn, proposal_id)
    finally:
        conn.close()


@router.get("/self-mod-proposals")
def list_self_mod_proposals(status: str = "approved"):
    """Default to `approved`: the reviewer's own verdict already
    approves/rejects a self-modification proposal (2026-08-30, no human
    review step -- see reviewer.review_self_modification's docstring),
    so this is what's about to be deployed (or already was), not a
    human decision queue. `reviewed` is still a valid filter for audit
    purposes -- every proposal passes through it on the way to
    approved/rejected."""
    conn = _self_mod_conn()
    try:
        return self_mod.list_by_status(conn, status)
    finally:
        conn.close()


@router.post("/self-mod-proposals/{proposal_id}/approve")
def approve_self_mod_proposal(proposal_id: int):
    """Manual override: reviewer.review_self_modification already calls
    this itself for an "allow" verdict, so a proposal is normally
    already approved by the time a human sees it here. This exists to
    override a "deny" verdict -- run_self_mod_deploy.py still won't
    touch anything for a proposal that isn't 'approved' AND doesn't have
    a clean sandboxed test run (see that script), so calling this alone
    still doesn't deploy anything by itself."""
    conn = _self_mod_conn()
    try:
        self_mod.approve(conn, proposal_id)
        return self_mod.get(conn, proposal_id)
    finally:
        conn.close()


@router.post("/self-mod-proposals/{proposal_id}/reject")
def reject_self_mod_proposal(proposal_id: int, body: DismissBody = DismissBody()):
    conn = _self_mod_conn()
    try:
        self_mod.reject(conn, proposal_id, body.reason)
        return self_mod.get(conn, proposal_id)
    finally:
        conn.close()


app.include_router(router)

# Mounted last, deliberately: a StaticFiles mount at "/" only catches
# paths not matched by the routes above it, since Starlette checks routes
# in registration order. Vanilla HTML/JS, no build step -- following v1's
# admin.html precedent rather than introducing a new frontend framework
# decision (see PLAN.md Phase 6).
if os.path.isdir(_PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")
