"""
Shared-token auth for api.py, gating this from "docker-compose network
only" to "safe to expose beyond localhost" (see api.py's module
docstring -- flagged there as the one thing missing before Phase 6).

Deliberately a single shared bearer token, not per-user accounts: this
is a one-operator admin surface, the same posture v1 took for its own
admin auth (see project memory). Optional, same pattern as
SLACK_BOT_TOKEN/GEMINI_API_KEY elsewhere in this project -- unset means
auth is off, matching where the rest of the project already stood, not
a silently-broken deployment.
"""

import hmac
import os

from fastapi import Header, HTTPException


def require_auth(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("API_AUTH_TOKEN")
    if not token:
        return  # auth disabled -- see module docstring

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    supplied = authorization.removeprefix("Bearer ")
    # constant-time compare -- this is a credential check, not a data
    # equality check, so no early-exit-on-first-mismatch timing leak.
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="invalid bearer token")
