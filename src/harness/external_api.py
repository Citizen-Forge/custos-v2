"""
HTTP surface for external agent seats (see external.py).

Two layers of auth: the router sits behind api.py's shared API_AUTH_TOKEN like
every other route, and each ticket call also needs the seat's own token in
`X-Custos-Seat-Token`, which is what says WHICH external seat is calling.
Registering a seat needs only the shared token -- it is an operator action.
"""

import os

import psycopg
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from . import beads, external

router = APIRouter(prefix="/external")


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    external.init_table(conn)
    return conn


def _seat(token: str | None) -> str:
    with _conn() as conn:
        seat_id = external.seat_for_token(conn, token)
    if not seat_id:
        raise HTTPException(status_code=401, detail="missing or unknown X-Custos-Seat-Token")
    return seat_id


def _run(fn, *args):
    try:
        return fn(*args)
    except external.ExternalError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except beads.BeadsError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 409, detail=str(e))


class RegisterBody(BaseModel):
    seat_id: str
    description: str


class RequestBody(BaseModel):
    project_id: str | None = None


class SubmitBody(BaseModel):
    summary: str


class ReleaseBody(BaseModel):
    reason: str = ""


@router.post("/seats")
def register_seat(body: RegisterBody):
    with _conn() as conn:
        token = _run(external.register_seat, conn, body.seat_id, body.description)
    external.invalidate_cache()
    return {"seat_id": body.seat_id, "token": token,
            "note": "shown once -- configure it as CUSTOS_SEAT_TOKEN for this session"}


@router.post("/tickets/request")
def request_ticket(body: RequestBody = RequestBody(),
                   x_custos_seat_token: str | None = Header(default=None)):
    seat_id = _seat(x_custos_seat_token)
    brief = _run(external.request_ticket, seat_id, body.project_id)
    return brief or {"ticket": None, "detail": "no ticket is ready for this seat right now"}


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, x_custos_seat_token: str | None = Header(default=None)):
    return _run(external.get_ticket, _seat(x_custos_seat_token), ticket_id)


@router.post("/tickets/{ticket_id}/heartbeat")
def heartbeat(ticket_id: str, x_custos_seat_token: str | None = Header(default=None)):
    until = _run(external.heartbeat, _seat(x_custos_seat_token), ticket_id)
    return {"ticket_id": ticket_id, "lease_expires_at": until}


@router.post("/tickets/{ticket_id}/submit")
def submit(ticket_id: str, body: SubmitBody, x_custos_seat_token: str | None = Header(default=None)):
    seat_id = _seat(x_custos_seat_token)
    with _conn() as conn:
        return _run(external.submit_ticket, conn, seat_id, ticket_id, body.summary)


@router.post("/tickets/{ticket_id}/release")
def release(ticket_id: str, body: ReleaseBody = ReleaseBody(),
            x_custos_seat_token: str | None = Header(default=None)):
    _run(external.release_ticket, _seat(x_custos_seat_token), ticket_id, body.reason)
    return {"ticket_id": ticket_id, "released": True}
