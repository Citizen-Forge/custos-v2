# Custos v2 interface agent -- sidecar setup

A Claude Code session, run wherever you normally run Claude Code (not
inside custos-v2's own docker-compose stack), talking to a live
deployment through `server.py`'s MCP tools. Gives you Sonnet-tier
conversational planning from your phone or laptop via Claude Code's own
Remote Control, at subscription pricing rather than per-token API
billing, while reusing custos-v2's real API instead of a parallel
system.

## Setup

1. Install this server's dependencies once, wherever you'll run it:
   `pip install -r mcp-server/requirements.txt` (or run it in a
   container the way `smoke_test.py` does -- see that file's docstring).
2. Register it with Claude Code:
   ```
   claude mcp add --transport stdio custos -- \
     env CUSTOS_API_URL=http://192.168.250.238:8000 \
     python /path/to/mcp-server/server.py
   ```
   **On the Unraid deployment, `api` is macvlan-attached to `br0` at its
   own static LAN IP (`192.168.250.238`, see `docker-compose.prod.yml`)
   with port publishing explicitly reset to none** -- the Unraid host's
   own IP (`192.168.100.231`) cannot reach it at all, standard macvlan
   behavior (a macvlan child is only reachable from other real devices on
   the physical LAN, never from the Docker host itself). Verified live,
   not assumed: `curl 192.168.100.231:8000` fails to connect,
   `curl 192.168.250.238:8000` returns 200. Use the macvlan IP, not the
   host IP, for any deployment using `docker-compose.prod.yml`'s overlay
   -- this changed 2026-08-30 (`git log -- docker-compose.prod.yml`) and
   silently broke any sidecar still pointed at the old host-IP URL.
   Swap in `CUSTOS_API_TOKEN=...` too if the target deployment has
   `API_AUTH_TOKEN` set (see the main project's `.env.example`).
3. Enable Remote Control on this Claude Code instance so it's reachable
   from claude.ai/code and the mobile app -- see Claude Code's own docs
   for that part, nothing custos-v2-specific about it.

## Role

Paste something like this into the session (as a system prompt / the
first message / a `CLAUDE.md` in whatever directory you run it from):

> You are the conversational interface to a running Custos v2 harness --
> an autonomous multi-agent software delivery system. Your job mirrors
> `product_owner.py`'s role, but live and conversational instead of a
> scheduled batch pass: help the user think through a rough idea out
> loud, and turn it into real project structure as you go, not as a
> summary at the end.
>
> Before proposing anything, check `list_projects` (does this idea
> belong in something that already exists?) and `list_seats` (what
> specialists already exist, so you don't suggest work with no one to
> do it). For a genuinely new idea: `create_project` once for the
> overall goal, then `create_epic` for each coherent slice, then
> `create_story` for concrete, individually-workable tasks under each
> epic -- match the depth of breakdown to the idea's actual size, a
> small idea might be one epic with two stories, not three empty tiers.
>
> When the user asks for a status update, use `list_projects`,
> `list_tickets`, and `list_seats` to answer from real current state,
> not from what you remember being true earlier in the conversation --
> the harness works asynchronously in the background and its state
> changes between your turns.
>
> If `list_tickets(status="human")` shows something flagged, that's a
> real decision the harness is blocked on -- surface it plainly (what's
> being asked, why) rather than deciding it yourself, and use
> `respond_to_ticket`/`dismiss_ticket` once the user actually tells you
> what to do.

## Bubble-up: proactive notifications

Claude Code has no webhook that wakes a dormant session on an external
event (checked against the docs before assuming otherwise) -- so this
has to be the session *staying alive and periodically checking*, not a
push arriving from outside. Concretely:

1. Keep this session running (or use whatever recurring/scheduled
   mechanism your Claude Code setup provides to re-enter it
   periodically -- e.g. a `/schedule`-style loop).
2. On each check-in, call `list_tickets(status="human")`.
3. Track which ticket ids you've already surfaced -- easiest is a
   plain local file (e.g. `.claude/notified-tickets.txt` next to
   wherever you run this) or, if you want it durable across a fresh
   session, a note recorded through `create_project`'s same connection
   -- either way, the point is: don't re-notify about the same flagged
   ticket every cycle.
4. For anything new, send the user a real message (this reaches their
   phone via Remote Control the same as any other turn) with the
   ticket id, what it's asking, and why it's blocked -- not just "you
   have 1 new item," enough to act on without opening the dashboard.

A few-minutes interval is a reasonable starting point -- responsive
without being noisy. Tune it once you see how often things actually get
flagged in practice.
