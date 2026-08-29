"""
Slack notifier -- gives a human visibility into agent activity as a
team-visible feed (user's own framing, 2026-08-29): seat creation posts a
welcome message, a worker starting a ticket announces it, and agents can
scan recent channel history for context. Deliberately NOT routed through
Beads' own `bd comment`/`bd comments` -- checked live against `bd
--help` rather than assumed: those are per-ISSUE comments (like a GitHub
issue thread), not a team-wide activity feed, the wrong shape for what
this is.

Needs a real Slack App with a BOT token (not an incoming webhook --
webhooks are post-only, and reading recent history for agent context
needs the real Web API), scopes `chat:write` + `channels:history` (or
`groups:history` for a private channel), invited into the target
channel. NOT live-verified against a real Slack workspace in this
session -- no credentials were available to test against. Every
function here fails soft (logs a warning, returns a safe empty/False
value, never raises) so a harness with Slack unconfigured behaves
exactly as it did before this module existed -- this is optional
infrastructure, not a hard dependency of seat creation or ticket
claiming.
"""

import logging
import os

import httpx

log = logging.getLogger("slack")

API_BASE = "https://slack.com/api"


def _configured() -> tuple[str, str] | None:
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        return None
    return token, channel


def post_message(text: str) -> bool:
    """Post a message to the configured channel. Returns True on success,
    False on any failure including "not configured" -- callers (seat
    creation, ticket claiming) treat this as fire-and-forget and never
    let a Slack failure block real work."""
    cfg = _configured()
    if cfg is None:
        log.debug("Slack not configured (SLACK_BOT_TOKEN/SLACK_CHANNEL_ID unset) -- skipping post")
        return False
    token, channel = cfg
    try:
        response = httpx.post(
            f"{API_BASE}/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": text},
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            log.warning("Slack post failed: %s", data.get("error"))
            return False
        return True
    except httpx.HTTPError as e:
        log.warning("Slack post failed: %s", e)
        return False


def recent_messages(limit: int = 20) -> list[str]:
    """Recent message texts from the configured channel (Slack's own
    newest-first order) -- empty list if unconfigured or on any failure,
    never raises. What an agent scanning for context at session start
    would read."""
    cfg = _configured()
    if cfg is None:
        return []
    token, channel = cfg
    try:
        response = httpx.get(
            f"{API_BASE}/conversations.history",
            headers={"Authorization": f"Bearer {token}"},
            params={"channel": channel, "limit": limit},
            timeout=10,
        )
        data = response.json()
        if not data.get("ok"):
            log.warning("Slack history fetch failed: %s", data.get("error"))
            return []
        return [m.get("text", "") for m in data.get("messages", [])]
    except httpx.HTTPError as e:
        log.warning("Slack history fetch failed: %s", e)
        return []
