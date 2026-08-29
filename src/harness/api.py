"""
Minimal HTTP API over the harness's existing state -- Beads tickets,
pending prompt proposals, per-role outcomes. Read-mostly; the write
endpoints (approve a pending prompt, respond to/dismiss a human-flagged
ticket) are deliberately narrow, matching v1's "autonomy off by default"
posture -- this is an admin surface for a human to inspect and resolve
things, not a general write API.

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

from . import avatar, beads, model_registry, outcomes, prompts, seats, settings, tool_proposals, verifications, wiki
from .auth import require_auth


class RespondBody(BaseModel):
    response: str


class DismissBody(BaseModel):
    reason: str | None = None


class CostSliderBody(BaseModel):
    value: int


class AvatarStyleBody(BaseModel):
    value: str


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
    optimized for hundreds."""
    projects = beads.list_top_level()
    tree = []
    for project in projects:
        epics = beads.children_of(project["id"])
        for epic in epics:
            epic["stories"] = beads.children_of(epic["id"])
        project["epics"] = epics
        tree.append(project)
    return tree


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
    return FileResponse(path, media_type="image/png")


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


app.include_router(router)

# Mounted last, deliberately: a StaticFiles mount at "/" only catches
# paths not matched by the routes above it, since Starlette checks routes
# in registration order. Vanilla HTML/JS, no build step -- following v1's
# admin.html precedent rather than introducing a new frontend framework
# decision (see PLAN.md Phase 6).
if os.path.isdir(_PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")
