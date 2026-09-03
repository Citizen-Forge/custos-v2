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


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="profile-rewrite",
        base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
        model=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"),
        max_tokens=4000,
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

    model = build_chat_model(_provider())
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        seats.init_table(conn)
        roster = seats.list_all(conn)

    targets = [s for s in roster if not args.seat or s["seat_id"] == args.seat]
    if not targets:
        log.error("no matching seat")
        return

    log.info("rewriting %s profile(s)%s", len(targets), " (dry run)" if args.dry_run else "")
    for seat in targets:
        others = [o for o in roster if o["seat_id"] != seat["seat_id"]]
        try:
            rewrite(model, seat, others, dry_run=args.dry_run)
        except Exception:
            log.exception("%s: rewrite failed, leaving the existing page alone", seat["seat_id"])


if __name__ == "__main__":
    main()
