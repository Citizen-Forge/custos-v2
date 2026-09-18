# Custos v2

A durable, self-hosted agentic software-development lifecycle engine. Work is
a real product roadmap in [Beads](https://github.com/gastownhall/beads) (`bd`);
a bounded pool of named agent **seats** works it; a separate **verifier**
judges each finished ticket against its own acceptance criteria; and the
harness can improve its own source under a sandbox-and-review boundary. See
[PLAN.md](PLAN.md) for the phase-by-phase design and the reasoning behind each
decision — this README is "how it runs today."

## Status (2026-09)

All seven planned phases have working substrate, live-tested against real
Postgres, real Beads, and real Docker. The pieces, all live-proven:

- **Dispatcher** — capacity-bounded, roadmap-ordered, one ticket at a time per
  project; verifies on close and only then merges the work.
- **Emergent seats** — a product-owner agent assigns work to named specialists
  and creates new ones when nothing existing fits.
- **Projects & workspaces** — a project is a Beads epic; each gets its own
  workspace and product repo, and each ticket gets its own worktree.
- **Acceptance-criteria verifier** with a bounded rework loop, plus an
  **escalation** handler for tickets that get parked.
- **Scheduler** running triage, verification, progress checks and more on a
  loop.
- **Overwatch / reviewer** and **self-modification** of `src/harness/*` under
  containment.

The first product built on it is **Subspatial**
(`workspace-o0n`): a Godot 4 / GDScript isometric starship game, the native
successor to the suspended TypeScript project **Silent Run** (`workspace-9jg`,
kept on `dispatch_hold`).

## How it works

- **Work graph is Beads.** `bd`'s dotted hierarchy (`project → epic → story`)
  *is* the data model — a project is a top-level `epic`, its epics are
  children, stories are grandchildren. There is no parallel queue table.
- **Dispatcher** (`src/harness/dispatcher.py`) picks assigned work in roadmap
  order (project priority → epic priority → story priority → natural id),
  starting nothing else in a project until its earliest unfinished ticket is
  done. Orphans (`in_progress`) resume first. Human-labelled and
  dispatch-held projects are skipped. Capacity is `MAX_RUNNING_AGENTS`.
- **Worker** (`src/harness/worker.py`) runs one ticket's LangGraph thread. In
  a fresh start it lays out the opening prompt (`bd prime` context + ticket
  text + acceptance criteria + any verifier finding), then the agent works and
  commits. On success the harness commits the ticket's own worktree and closes
  it.
- **Verifier** (`src/harness/verifier.py`) judges a closed ticket against its
  acceptance criteria. Machine-checkable criteria (`acceptance_checks`, e.g.
  "file X exists", "the suite runs at least N tests") are evaluated in code
  first: a failed check fails the ticket with no model call, and a ticket whose
  criteria are all mechanical can pass without one. Free-text criteria go to
  the model, which also sees the mechanical results. A pass merges into the
  project's integration branch; a fail reopens it with the finding (bounded by
  `MAX_VERIFIER_REWORKS`) and then parks it for a human.
- **Escalations** (`src/harness/escalations.py`) let the product-owner resolve
  parked tickets (fix the spec and requeue, reassign, or dismiss), bounded by
  `MAX_ESCALATION_ATTEMPTS`.
- **Postgres** holds LangGraph checkpoints and the harness's own tables
  (prompts, seats, verifications, proposals).

## Acceptance checks (structured criteria)

Beyond free-text criteria, a ticket can carry `acceptance_checks`: a JSON list
of machine-checkable assertions the verifier evaluates in code before any model
call. Supported forms (`harness/acceptance.py`); paths are relative to the
project workspace and confined there:

```
{"type": "file_exists",     "path": "project.godot"}
{"type": "file_absent",     "path": "old/TODO"}
{"type": "contains",        "path": "README.md", "text": "## Test"}
{"type": "matches",         "path": "src/x.gd", "pattern": "func \\w+"}
{"type": "tests_at_least",  "count": 1}
```

A failed check is a fail with no model call; a ticket whose criteria are all
mechanical and pass can pass with no model call too. The model is asked only
when the ticket also has free-text criteria, and it sees the mechanical results
as measured evidence. The product-owner sets checks when it creates a story (or
on an escalation), and the API accepts `acceptance_checks` on the create-story
body.

## Running it (Docker only)

```bash
cp .env.example .env            # then edit: model endpoints/keys live here

docker compose up -d            # postgres + harness (dispatcher) + api + scheduler
```

- **harness** — `python -m harness.dispatcher`: the dispatcher/agent pool.
- **api** — FastAPI + dashboard on `:8000`.
- **scheduler** — periodic jobs (see below), on by default.
- **postgres** — checkpoints + harness tables.
- **sandbox-runner** — only under `--profile sandbox`; the one service that
  mounts the Docker socket (used by the sandbox/self-mod pipeline).

Full test suite (uses a throwaway Postgres DB and temp workspace —
`tests/conftest.py`):

```bash
docker compose run --rm harness pytest -v
```

Create a ticket:

```bash
docker compose run --rm harness python scripts/enqueue_demo.py \
    "demo ticket" "list the files in the workspace"
```

A fresh ticket has **no seat** until it is assigned. Either let the
product-owner triage it, or assign it directly:

```bash
docker compose run --rm harness python scripts/run_product_owner.py

# or bootstrap: assign to the default "worker" seat
docker compose run --rm harness python -c "
from harness import beads
beads.ensure_initialized()
beads.assign_to_seat('<ticket-id>', 'worker')
"
```

### Model configuration

`LOCAL_MODEL_*` is the shared chain every role falls back to (this deployment
points at a DeepSeek OpenAI-compatible endpoint; a local llama.cpp/vLLM/Ollama
server works the same way). `CLASSIFIER_*` configures the permission-gate model
separately — a small fast local model is a good fit there. The scheduler
resolves each job's model as `<ROLE>_MODEL_*` → `SCHEDULER_MODEL_*` →
`LOCAL_MODEL_*`, and the verifier/reflection use `VERIFIER_MODEL_*` /
`REFLECTION_MODEL_*`, so advisory roles (`progress`, `reflection`) can run on a
local model while the seats, product-owner and verifier stay on a frontier one.
`LOCAL_FALLBACK_*` adds a second chain entry, engaged automatically on a
provider error (e.g. a 402 once credits run out). All variables in `.env` reach
the containers (`env_file`), not just a hand-picked few.

## Dispatch, seats, and the product-owner

Seats are **emergent, not a fixed roster**: the product-owner looks at
unassigned ready tickets and the current seat roster (with each seat's
outcomes), assigns tickets to existing seats, and creates new specialist seats
(active immediately) when nothing fits. That is a model judgment call — but
only the *seat* is.

The ticket itself is deterministic: the next one is the project's roadmap
front, highest priority, `bd`-ready (no open dependencies). `dispatcher.
assign_front()` assigns it to the project's seat with **no model call** when
the project already has exactly one seat working it; the product-owner is only
woken when a project has no seat yet or several (a real choice of specialist).
A seat that already declined a ticket is never auto-picked for it.

## Projects and workspaces

- A project maps to a Beads epic (`issue_type="epic"`) and gets
  `PROJECTS_ROOT/<project_id>` (mounted at `/projects`) with its own git repo —
  product code never lands beside the harness's issue database, and projects
  can't reach each other's files.
- Each **ticket** gets its own `git worktree` on a branch off the project's
  integration branch (default `master`), so no two agents write the same tree
  and each ticket's commit is attributable to it. Work is merged only after the
  verifier passes.
- **Toolchain**: a project declares the commands its work needs as `toolchain`
  metadata (e.g. `node,npm` or `godot`). Dispatch refuses a ticket whose
  toolchain is missing, loudly, instead of producing unbuildable work. The
  shared image ships Node and Godot so agents can actually build and test.
- **Test command**: a project declares `test_command` (e.g.
  `godot --headless --path . --script tests/run_tests.gd`); the verifier runs
  it and reads `# tests/# pass/# fail` plus the exit code. A project with no
  declared command falls back to a `package.json` + `npm test` script.

## Permissions and containment

Three deliberately separate layers (`src/harness/permissions.py`):

1. **Allow-list fast path** — ordinary workspace-scoped build/test/local-git/
   `bd`-read commands skip the classifier.
2. **LLM classifier** — everything else is judged per call (tool name + args).
3. **Hard denials** — a small set is *not* the classifier's call, because they
   are integrity boundaries, not task judgments: agents may not move the
   integration branch (`git update-ref`, `branch -f`, `reset`, `checkout`, …)
   or mutate the board outside the harness's pathways (`bd close`, `update`,
   `delete`, …). A mechanical backstop (`workspaces.
   revert_unexpected_integration_move`) reverts an integration-branch move the
   gated merge did not make, since a blacklist is bypassable by construction.

File tools are additionally confined to the ticket's workspace by
`permissions.check_within_workspace` (a hard invariant, not classifier-
overridable). `shell_exec` is not path-jailed — the classifier is its gate,
same as this project's stated posture.

## API and dashboard

```bash
docker compose up -d api        # http://localhost:8000
```

The dashboard is a no-build vanilla HTML/JS page (`public/index.html`, polls
every 5s) with four tabs:

- **Home** — a project-wide progress bar, the **Roadmap**, and the **Board**
  (Open / In Progress / Blocked / Done). An in-progress ticket shows its
  assigned seat and **"last move 12s ago · N steps"**, read from the
  checkpointer, so you can see it's moving without opening another tab.
  `human`-labelled tickets render as Blocked even when closed (closed + human
  is failed work, not done).
- **Agents** — Working Now (running agents) and the seat roster with outcomes.
- **Wiki** — agent profile pages.
- **Settings** — cost slider, avatar style, model registry.

Key endpoints (`API_AUTH_TOKEN` in `.env` gates everything except `/health`
and the static dashboard; send `Authorization: Bearer <token>`):

```
GET    /tickets?status=ready|in_progress|human
GET    /projects
PATCH  /issues/{id}/priority
POST   /projects | /projects/{id}/epics | /epics/{id}/stories
GET    /tickets/{id}          POST /tickets/{id}/respond | /dismiss
GET    /seats                 GET  /agents/running
GET    /outcomes/{actor}      GET  /toolchain
GET    /prompts/pending       POST /prompts/{role}/{version}/approve
GET    /tool-proposals        POST /tool-proposals/{id}/approve|reject
GET    /self-mod-proposals    POST /self-mod-proposals/{id}/approve|reject
GET    /wiki | /wiki/{slug}   GET  /settings/{cost-slider,model-registry,avatar-style}
```

## Scheduler

`scripts/run_scheduler.py` runs, in order each cycle (default every 1800s),
first skipping model-backed jobs when a cycle has nothing to act on:

- **escalations** — resolve parked tickets first (the front of a project
  blocks everything behind it).
- **product_owner** — triage unassigned work.
- **overwatch** — capability-gap scanning / tool proposals.
- **meta_agent** — prompt-revision review per seat.
- **verifier** — judge closed tickets awaiting a verdict.
- **progress** — flag agents that look stalled/looping (cheap checkpointer
  signal, model only past a threshold; deliberately does not kill a run).

To opt a deployment back out of automatic scheduling:
`docker compose up postgres harness api`.

## Meta-agent, overwatch, and tool promotion

`src/harness/meta_agent.py` reviews a role's recent outcomes and proposes a
system-prompt revision; it **activates on its own verdict** (the model's
judgment *is* the review — no human step), and stays reachable via
`POST /prompts/{role}/{version}/approve` as a manual override.

`src/harness/overwatch.py` proposes tools from real capability-gap evidence,
`sandbox.run_sandboxed` runs candidate code in a maximally-restricted ephemeral
container (`--network none`, `--read-only`, `--cap-drop=ALL`, no socket, no
secrets), and `src/harness/reviewer.py` forms an allow/deny verdict on the
source **and** the sandbox evidence. A clean sandbox run alone is not enough to
allow — the reviewer reads the source.

```bash
docker compose run --rm -e OVERWATCH_BRIEF="..." harness python scripts/run_overwatch.py
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_sandbox_for_proposals.py
docker compose run --rm harness python scripts/run_reviewer.py
```

Sandbox containment is live-proven (`tests/test_sandbox.py`, run via the
`sandbox-runner` profile): no env vars visible, read-only mount, no network,
`--pids-limit` caps a fork bomb, and a timed-out sandbox is actually killed.

## Self-modification (the harness improving its own source)

Same shape for `src/harness/*` — the "control plane" no agent may write
directly. A self-modifier agent only writes to an isolated checkout; the only
code with write access to the real tree is `run_self_mod_deploy.py`'s trusted
orchestration, and deploy enforces a hard non-LLM precondition (the sandboxed
run had zero real test failures). An `allow` verdict approves and deploys
without a human gate; git history is the rollback.

```bash
docker compose run --rm -e SELF_MOD_BRIEF="..." harness python scripts/run_self_modifier.py
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_self_mod_sandbox.py
docker compose run --rm harness python scripts/run_self_mod_reviewer.py
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_self_mod_deploy.py
```

## Sidecar interface agent (conversational, from your phone)

`mcp-server/` is a separate MCP server exposing this harness's API
(projects/tickets/seats) as tools to an ordinary Claude Code session, run
wherever you normally run Claude Code with Remote Control enabled. See
[`mcp-server/SIDECAR.md`](mcp-server/SIDECAR.md) for setup, the role prompt,
and which API URL to use (it depends on where the sidecar runs relative to the
deployment).

## Production deployment (macvlan overlay)

`docker-compose.prod.yml` deploys alongside other apps on a LAN box: internal
services (postgres, harness, scheduler) stay on the private compose network,
while `api` gets a static LAN IP via Docker's macvlan driver (`br0`) plus a
pinned static IP on the internal bridge — so a host-level process (like the
sidecar) can reach it, since a Docker host cannot reach its own macvlan
children.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d harness api scheduler
```

`.env` lives only on the deployment target. `REPO_HOST_PATH` /
`SANDBOX_SCRATCH_HOST_PATH` resolve from `${PWD}` at compose time. A common
operator workflow is to commit locally, push, then on the box
`git fetch origin -q && git reset --hard origin/main`, and restart only the
service a change affects (`docker restart custos-v2-harness-1` for
dispatcher/agent code, `custos-v2-scheduler-1` for scheduled jobs; the
dashboard is served live).

## Proving the durability guarantee manually

1. Enqueue a ticket that needs a couple of tool calls, and assign it to a seat.
2. `docker compose up -d harness`, let it start — watch for
   `starting thread <id>`.
3. `docker compose kill harness` (once a real tool-call round trip is in the
   logs).
4. `docker compose up -d harness` — look for `resuming thread <id>` (distinct
   from `starting thread`), confirming checkpoint resume rather than restart.

## Repository layout

```
src/harness/     the harness (dispatcher, worker, verifier, product_owner,
                 seats, routing, permissions, api, self_mod, sandbox, …)
public/          dashboard (index.html, no build step)
scripts/         operator entrypoints (scheduler, product-owner, verifier,
                 escalations, self-mod, requeue helpers, probes)
tests/           pytest suite (real Postgres + real bd; isolated per session)
projects/        per-project workspaces (created at runtime, not tracked)
workspace/       the harness's own store (.beads, wiki, avatars)
mcp-server/      sidecar interface agent
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free to use, modify and share for any
noncommercial purpose.
