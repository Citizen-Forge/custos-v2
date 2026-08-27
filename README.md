# Custos v2

Full rewrite of [Custos](https://github.com/Citizen-Forge/custos): a
durable, queue-friendly agent harness for slow/low-concurrency local
models, with frontier models used narrowly. See [PLAN.md](PLAN.md) for the
phased roadmap and the reasoning behind every major decision — read that
first, this README is just "how to run what exists so far."

## Status

Phases 1–5 built and live-tested against real Postgres + real Beads
(scripted fake models standing in for the still-unreachable Ollama).
Phase 6 has a working API layer; the actual frontend isn't started. See
PLAN.md for the detailed status of each phase.

Work items live in [Beads](https://github.com/gastownhall/beads) (`bd`),
not a bespoke queue table — see PLAN.md's "Decisions locked in" for why.
`bd`'s own `.beads/` data directory lives in the mounted `workspace/`
folder alongside whatever files an agent's tool calls touch. Test runs
use their own isolated temp workspace (`tests/conftest.py`), not this
one — found and fixed a real bug where they didn't, early on.

## Running it (Docker only — no local Python install, see
[[feedback_docker-for-runtimes]])

```bash
# all tests, including the real end-to-end resume proof (needs Postgres +
# the real bd CLI, no LLM needed — see tests/test_worker_resume.py)
docker compose up -d postgres
docker compose run --rm harness pytest -v

# create a real ticket and let the worker pick it up
docker compose run --rm harness python scripts/enqueue_demo.py \
    "demo ticket" "list the files in the workspace"
docker compose up harness
```

`LOCAL_MODEL_BASE_URL` defaults to `http://host.docker.internal:11434/v1`
(a host-machine Ollama). Override in `.env` (copy from `.env.example`) to
point at a different OpenAI-compatible endpoint. No local model is
reachable in this environment yet, so end-to-end LLM behavior is still
unverified against a real model — `tests/test_worker_resume.py` proves the
durability mechanism itself using a scripted fake model instead.

## API + dashboard

```bash
docker compose up -d api   # http://localhost:8000, brings up postgres too
```

Open `http://localhost:8000/` for the dashboard (vanilla HTML/JS, no
build step — `public/index.html`): ready/in-progress/needs-a-human ticket
lists, pending prompt proposals with an Approve button, and an outcomes
lookup. Polls every 5s.

```bash
curl localhost:8000/tickets?status=ready
curl localhost:8000/prompts/pending
curl -X POST localhost:8000/prompts/worker/1/approve
```

Read-mostly by design (PLAN.md Phase 6) — no auth yet, not exposed beyond
the docker-compose network today.

## Meta-agent

`scripts/run_meta_agent.py` reviews a role's recent outcomes (sourced from
Beads' own audit trail) and proposes a system-prompt revision — queued as
*pending*, never applied automatically:

```bash
docker compose run --rm harness python scripts/run_meta_agent.py
# then review + apply via the API:
curl localhost:8000/prompts/pending
curl -X POST localhost:8000/prompts/worker/<version>/approve
```

## Proving the durability guarantee manually (with a real model)

1. `docker compose run --rm harness python scripts/enqueue_demo.py "<title>" "<prompt requiring a couple tool calls>"`
2. `docker compose up harness` and let it start working.
3. `docker compose kill harness` partway through.
4. `docker compose up harness` again — it should pick the same ticket back
   up from its last checkpoint (via `bd list --status=in_progress`, since
   `bd ready` won't show it anymore), not restart from scratch.
