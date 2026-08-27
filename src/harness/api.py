"""
Minimal HTTP API over the harness's existing state -- Beads tickets,
pending prompt proposals, per-role outcomes. Read-mostly; the one write
endpoint (approve a pending prompt) is deliberately narrow, matching v1's
"autonomy off by default" posture -- this is an admin surface for a human
to inspect and approve things, not a general write API. `bd human
respond/dismiss` (actually resolving a refused ticket) isn't wrapped yet
-- reading the state is the part Phase 6's UI needs first.

No auth yet -- matches where Phases 1-5 already are, nothing here is
exposed beyond the docker-compose network today. Needed before this is
reachable from anywhere but localhost; flagged, not hidden (same posture
v1 took explicitly, see project memory on v1's admin/remote auth).
"""

import os

import psycopg
from fastapi import FastAPI, HTTPException

from . import beads, outcomes, prompts

app = FastAPI(title="Custos v2 harness API")


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
