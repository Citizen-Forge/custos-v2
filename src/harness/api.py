"""
Minimal HTTP API over the harness's existing state -- Beads tickets,
pending prompt proposals, per-role outcomes. Read-mostly; the write
endpoints (approve a pending prompt, respond to/dismiss a human-flagged
ticket) are deliberately narrow, matching v1's "autonomy off by default"
posture -- this is an admin surface for a human to inspect and resolve
things, not a general write API.

No auth yet -- matches where Phases 1-5 already are, nothing here is
exposed beyond the docker-compose network today. Needed before this is
reachable from anywhere but localhost; flagged, not hidden (same posture
v1 took explicitly, see project memory on v1's admin/remote auth).
"""

import os
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import beads, outcomes, prompts


class RespondBody(BaseModel):
    response: str


class DismissBody(BaseModel):
    reason: str | None = None


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

_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "public")


def _prompt_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    return conn


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickets")
def list_tickets(status: str = "ready"):
    if status == "ready":
        return beads.ready()
    if status == "in_progress":
        return [i for i in beads.in_progress() if not beads.is_flagged_for_human(i)]
    if status == "human":
        return [i for i in beads.in_progress() if beads.is_flagged_for_human(i)]
    raise HTTPException(400, f"unknown status {status!r}, expected ready|in_progress|human")


@app.get("/tickets/{issue_id}")
def get_ticket(issue_id: str):
    try:
        return beads.show(issue_id)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/tickets/{issue_id}/respond")
def respond_to_ticket(issue_id: str, body: RespondBody):
    """Resolve a human-flagged ticket with a response, closing it -- see
    beads.respond_to_human's docstring for why this composes append_note
    + close rather than calling the real (broken) `bd human respond`."""
    try:
        return beads.respond_to_human(issue_id, body.response)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/tickets/{issue_id}/dismiss")
def dismiss_ticket(issue_id: str, body: DismissBody = DismissBody()):
    try:
        return beads.dismiss_human(issue_id, body.reason)
    except beads.BeadsError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/prompts/pending")
def list_pending_prompts(role: str | None = None):
    conn = _prompt_conn()
    try:
        return prompts.pending(conn, role)
    finally:
        conn.close()


@app.post("/prompts/{role}/{version}/approve")
def approve_prompt(role: str, version: int):
    conn = _prompt_conn()
    try:
        prompts.approve(conn, role, version)
        return {"role": role, "version": version, "status": "active"}
    finally:
        conn.close()


@app.get("/outcomes/{actor}")
def get_outcomes(actor: str):
    return outcomes.summary(actor)


# Mounted last, deliberately: a StaticFiles mount at "/" only catches
# paths not matched by the routes above it, since Starlette checks routes
# in registration order. Vanilla HTML/JS, no build step -- following v1's
# admin.html precedent rather than introducing a new frontend framework
# decision (see PLAN.md Phase 6).
if os.path.isdir(_PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")
