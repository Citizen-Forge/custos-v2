"""
Have existing seats rewrite their own wiki profile pages.

The seat-creation prompt was reframed (2026-09-03): it used to ask for a
bio covering "your work AND real personality" with a suggested menu of
attributes, which produced a roster of pages sharing one shape -- how I
work / what I like and dislike / a few true things, with a favourite band
and a favourite city apiece. It now asks the agent to imagine writing an
intranet introduction for new colleagues and decide for itself what to
say.

Seats created under the old prompt keep the old page until something
rewrites it, so this gives each of them the new brief and lets them write
their own again. Their identity is NOT regenerated: name, pronouns,
seat_id and specialty are theirs already and are passed back in. Only the
page changes.

    docker compose run --rm harness python scripts/rewrite_seat_profiles.py
    docker compose run --rm harness python scripts/rewrite_seat_profiles.py --seat <seat_id>
    docker compose run --rm harness python scripts/rewrite_seat_profiles.py --dry-run
"""

import argparse
import json
import logging
import os

import psycopg

from harness import seats, wiki
from harness.meta_agent import PROFILE_BRIEF
from harness.providers import ProviderConfig, build_chat_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rewrite-profiles")

PROMPT = """You already exist on this team. This is not an introduction to a stranger writing
you into being -- you are {display_name}, you use {pronouns}, and you are the seat known as
{seat_id}. That is settled and is not up for revision here.

What you are doing is rewriting your own intranet profile page, because the last one was
written to a brief that pushed everyone into the same shape and you can do better.

{profile_brief}

Your current page, for reference. Keep whatever still feels true and yours; discard whatever
reads like it was written to a form:
---
{current_page}
---

Other people on the team, so you can go somewhere they did not:
{others}

Respond with strict JSON and nothing else: {{"profile_page": "<your new page, markdown>"}}
"""


def model_reachable(timeout: int = 15) -> tuple[bool, str]:
    """Is a model server actually there? Liveness only -- NOT a test
    generation.

    Learned twice on 2026-09-03. First: with nothing obviously wrong,
    this script sat for eighty minutes producing no output, because the
    generation call simply never returned. So a preflight was added.
    Then the preflight itself was wrong -- it asked for a one-token
    completion, which queues behind whatever the server is already doing
    and timed out even though the server was healthy. Measured at the
    time: 3.66 tok/s prompt, 0.62 tok/s generation, one task taking 532
    seconds. A busy server is not a down server, and refusing to run
    against a slow one would be the timeout mistake this project
    deliberately avoids everywhere else.

    /health answers in milliseconds regardless of load, which is exactly
    the distinction wanted: nothing listening, versus listening and
    swamped."""
    import urllib.request

    base = os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1")
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    for url in (f"{root}/health", f"{root}/v1/models"):
        try:
            with urllib.request.urlopen(url, timeout=timeout):
                return True, url
        except Exception as e:
            last = f"{url}: {type(e).__name__} {e}"
    return False, last


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="profile-rewrite",
        base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        model=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"),
        # A profile page is prose, not a document. Kept modest because
        # generation on this deployment was measured at 0.62 tok/s -- an
        # unnecessarily large budget is hours of wall clock.
        max_tokens=1200,
    )


def rewrite(model, seat: dict, others: list[dict], dry_run: bool = False) -> str | None:
    slug = wiki.agent_profile_slug(seat["seat_id"])
    current = wiki.read_page(slug) or "(no page yet)"
    others_summary = "\n".join(
        f"- {o.get('display_name') or o['seat_id']}: {(o.get('specialty') or '')[:120]}"
        for o in others
    ) or "(nobody else yet)"

    response = model.invoke(
        PROMPT.format(
            display_name=seat.get("display_name") or seat["seat_id"],
            pronouns=seat.get("pronouns") or "they/them",
            seat_id=seat["seat_id"],
            profile_brief=PROFILE_BRIEF,
            current_page=current[:4000],
            others=others_summary,
        )
    )
    content = getattr(response, "content", response)
    try:
        page = json.loads(content)["profile_page"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.error("%s: could not parse response (%s)", seat["seat_id"], e)
        return None

    if not page or not page.strip():
        log.error("%s: empty page returned", seat["seat_id"])
        return None

    if dry_run:
        log.info("%s: would write %s chars to %s", seat["seat_id"], len(page), slug)
        return page

    wiki.write_page(slug, page)
    log.info("%s: rewrote %s (%s chars)", seat["seat_id"], slug, len(page))
    return page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seat", help="only this seat_id")
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args()

    ok, detail = model_reachable()
    if not ok:
        log.error("no model server at that address -- %s", detail)
        log.error("nothing was changed. Start the model server and run this again.")
        raise SystemExit(1)

    model = build_chat_model(_provider())
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        seats.init_table(conn)
        roster = seats.list_all(conn)

    targets = [s for s in roster if not args.seat or s["seat_id"] == args.seat]
    if not targets:
        log.error("no matching seat")
        return

    log.info(
        "rewriting %s profile(s)%s -- local inference is slow, expect minutes per seat",
        len(targets), " (dry run)" if args.dry_run else "",
    )
    for seat in targets:
        others = [o for o in roster if o["seat_id"] != seat["seat_id"]]
        try:
            rewrite(model, seat, others, dry_run=args.dry_run)
        except Exception:
            log.exception("%s: rewrite failed, leaving the existing page alone", seat["seat_id"])


if __name__ == "__main__":
    main()
