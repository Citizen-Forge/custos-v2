"""
Realistic agent avatars via Gemini's image generation API ("Nano Banana",
model `gemini-3.1-flash-image` as of Google's own docs, checked live
2026-08-29 -- https://ai.google.dev/gemini-api/docs/image-generation).
User's own call: prefers this over DiceBear's illustrated avatars for
realism, now that a free tier makes it practical (reportedly ~500
free images/day per third-party reporting -- Google's own docs page
didn't show the exact number; worth confirming in the AI Studio
console's own quota display).

Entirely optional, same posture as slack.py: every function fails soft
(logs a warning, returns None) when GEMINI_API_KEY is unset or a call
fails, so a harness without it configured behaves exactly as before --
the dashboard falls back to the DiceBear avatar (see public/index.html)
whenever this returns nothing for a seat.

Generated images are written to WORKSPACE_ROOT/avatars/<seat_id>.png,
served via GET /avatars/{seat_id} (api.py) rather than embedded as
base64 anywhere -- keeps the seats API response small and lets the
browser cache images normally.

NOT live-verified against a real Gemini API call in this session -- no
API key was available to test against. The exact request/response shape
below is checked against Google's own docs page directly (not guessed),
but hasn't been proven against a real response.
"""

import base64
import logging
import os

import httpx

from .config import WORKSPACE_ROOT

log = logging.getLogger("avatar")

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-image"
AVATAR_DIR = "avatars"


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY")


def generate_avatar(seat_id: str, description: str) -> str | None:
    """Generates a portrait image from a text description (the seat's own
    written personality/appearance, e.g. from its wiki profile) and saves
    it to WORKSPACE_ROOT/avatars/<seat_id>.png. Returns the saved path on
    success, None on any failure (including "not configured") -- callers
    treat this as fire-and-forget, same as Slack, and should fall back to
    a deterministic avatar (DiceBear) rather than block on this."""
    api_key = _api_key()
    if not api_key:
        log.debug("Gemini not configured (GEMINI_API_KEY unset) -- skipping avatar generation")
        return None

    model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_MODEL)
    prompt = (
        "A realistic portrait photo of a person, based on this description: "
        f"{description}. Professional headshot style, plain background, single "
        "person, face clearly visible."
    )

    try:
        response = httpx.post(
            f"{API_BASE}/interactions",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": model,
                "input": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            },
            timeout=60,
        )
        data = response.json()
        image_b64 = _extract_image_b64(data)
        if image_b64 is None:
            log.warning("Gemini response had no image data: %s", str(data)[:500])
            return None
        image_bytes = base64.b64decode(image_b64)
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("Gemini avatar generation failed: %s", e)
        return None

    avatar_root = os.path.join(WORKSPACE_ROOT, AVATAR_DIR)
    os.makedirs(avatar_root, exist_ok=True)
    path = os.path.join(avatar_root, f"{seat_id}.png")
    with open(path, "wb") as f:
        f.write(image_bytes)
    return f"{AVATAR_DIR}/{seat_id}.png"


def _extract_image_b64(data: dict) -> str | None:
    """Response shape checked against Google's own docs page directly,
    not guessed -- but NOT proven against a real response (no API key
    available this session). If Gemini's actual response shape differs,
    this is the one place to fix -- everything else in this module is
    shape-agnostic."""
    try:
        for item in data.get("output", []):
            for block in item.get("content", []):
                if block.get("type") in ("image", "output_image") and "data" in block:
                    return block["data"]
    except (AttributeError, TypeError):
        pass
    return None


def avatar_path(seat_id: str) -> str | None:
    """The saved avatar file's absolute path, or None if one was never
    generated (not configured, generation failed, or hasn't run yet for
    this seat) -- what GET /avatars/{seat_id} checks before serving."""
    path = os.path.join(WORKSPACE_ROOT, AVATAR_DIR, f"{seat_id}.png")
    return path if os.path.exists(path) else None
