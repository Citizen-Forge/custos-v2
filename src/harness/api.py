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
from contextlib import asynccontextmanager

import psycopg
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import avatar, beads, model_registry, outcomes, prompts, seats, self_mod, settings, tool_proposals, verifications, wiki
from .auth import require_auth


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


class CreateStoryBody(BaseModel):
    title: str
    description: str


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
def list_projects():
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
    old nested-outline view didn't."""
    projects = beads.list_top_level(issue_type="epic")
    tree = []
    for project in projects:
        epics = beads.children_of(project["id"])
        for epic in epics:
            epic["stories"] = beads.children_of(epic["id"])
        project["epics"] = epics
        tree.append(project)
    return tree


@router.post("/projects")
def create_project(body: CreateProjectBody):
    """Mirrors product_owner.py's create_project tool exactly (same
    beads.create() call) -- see this module's docstring for why this is
    a write endpoint despite the surface's otherwise-narrow posture."""
    project = beads.create(body.name, body.description, issue_type="epic", priority=body.priority)
    return project


@router.post("/projects/{project_id}/epics")
def create_epic(project_id: str, body: CreateEpicBody):
    """Mirrors product_owner.py's create_epic tool. 404s if project_id
    doesn't exist -- beads.create's own --parent validation surfaces as
    a BeadsError, same pattern as get_ticket below."""
    try:
        epic = beads.create(body.title, body.description, issue_type="epic", parent=project_id)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e
    return epic


@router.post("/epics/{epic_id}/stories")
def create_story(epic_id: str, body: CreateStoryBody):
    """Mirrors product_owner.py's add_subtask_to_epic tool."""
    try:
        story = beads.create(body.title, body.description, parent=epic_id)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e
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
