# Custos v2

Full rewrite of [Custos](https://github.com/Citizen-Forge/custos): a
durable, queue-friendly agent harness for slow/low-concurrency local
models, with frontier models used narrowly. See [PLAN.md](PLAN.md) for the
phased roadmap and the reasoning behind every major decision — read that
first, this README is just "how to run what exists so far."

## Status

Phase 1 in progress: durable harness core (LangGraph + Postgres
checkpointer + a minimal work queue), no routing/personality/meta-agent
yet.

Work items live in [Beads](https://github.com/gastownhall/beads) (`bd`),
not a bespoke queue table — see PLAN.md's "Decisions locked in" for why.
`bd`'s own `.beads/` data directory lives in the mounted `workspace/`
folder alongside whatever files an agent's tool calls touch.

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

## Proving the durability guarantee manually (with a real model)

1. `docker compose run --rm harness python scripts/enqueue_demo.py "<title>" "<prompt requiring a couple tool calls>"`
2. `docker compose up harness` and let it start working.
3. `docker compose kill harness` partway through.
4. `docker compose up harness` again — it should pick the same ticket back
   up from its last checkpoint (via `bd list --status=in_progress`, since
   `bd ready` won't show it anymore), not restart from scratch.
