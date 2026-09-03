# Custos v2

Full rewrite of [Custos](https://github.com/Citizen-Forge/custos): a
durable, queue-friendly agent harness for slow/low-concurrency local
models, with frontier models used narrowly. See [PLAN.md](PLAN.md) for the
phased roadmap and the reasoning behind every major decision — read that
first, this README is just "how to run what exists so far."

## Status

Phases 1–7 have working substrate, live-tested against real Postgres +
real Beads (+ real Docker, for the Phase 7 sandbox). As of 2026-08-29 a
real local model is reachable too, and the core mechanisms (product-owner
judgment, kill/resume durability, the permission gate) are proven live
against it, not just scripted fakes — see PLAN.md's "Open questions" for
the full writeup. Work is now assigned
to specific named agent **seats** by a product-owner agent, not claimed
generically by one worker — see PLAN.md's Phase 4 "Emergent seat system"
and the "Seats" section below. As of 2026-08-30 the harness can also
modify its own source under the same containment substrate — see
"Self-modification" below. See PLAN.md for the detailed status of each
phase.

Work items live in [Beads](https://github.com/gastownhall/beads) (`bd`),
not a bespoke queue table — see PLAN.md's "Decisions locked in" for why.
`bd`'s own `.beads/` data directory lives in the mounted `workspace/`
folder alongside whatever files an agent's tool calls touch. Test runs
use their own isolated temp workspace *and* a fresh throwaway Postgres
database (`tests/conftest.py`) — both were found live to be missing at
different points this session (real tickets/seats were leaking into the
same store the actual services use) and fixed.

## Running it (Docker only — no local Python install, see
[[feedback_docker-for-runtimes]])

```bash
# all tests, including the real end-to-end resume proof (needs Postgres +
# the real bd CLI, no LLM needed — see tests/test_worker_resume.py)
docker compose up -d postgres
docker compose run --rm harness pytest -v

# create a real ticket
docker compose run --rm harness python scripts/enqueue_demo.py \
    "demo ticket" "list the files in the workspace"
```

**A freshly enqueued ticket has no seat assigned yet** — since the Phase 4
seat system landed, a worker only claims tickets explicitly assigned to
its own seat (`ready_for_seat`), never just "any ready ticket." Assign it
one of two ways before `docker compose up harness` will pick it up:

```bash
# realistic path: let the product-owner triage it (creates a specialist
# seat if nothing existing fits, its own judgment call)
docker compose run --rm harness python scripts/run_product_owner.py

# OR bootstrap path: assign it directly to the default "worker" seat
docker compose run --rm harness python -c "
from harness import beads
beads.ensure_initialized()
beads.assign_to_seat('<ticket-id-from-enqueue_demo-output>', 'worker')
"

docker compose up harness
```

`docker compose up harness` runs the `worker` seat (`DEFAULT_SEAT_ID`) by
default — set `SEAT_ID=<seat_id>` (now correctly wired through
docker-compose.yml as of 2026-08-29; previously silently ignored — a real
bug, see PLAN.md) to run a worker process for a specialist seat the
product-owner has created instead. `LOCAL_MODEL_*` env vars are the
shared model chain every seat falls back to unless it's explicitly
registered its own (`routing.py`'s `default_role`) — defaults to
`http://host.docker.internal:11434/v1`, a host-machine Ollama. Override in
`.env` (copy from `.env.example`) to point at a different OpenAI-compatible
endpoint — e.g. a llama.cpp/vLLM server on another machine on the LAN.

## Seats and the product-owner

```bash
docker compose run --rm harness python scripts/run_product_owner.py
```

One triage session: the product-owner looks at unassigned ready tickets
and the current seat roster (each with its outcomes), assigns tickets to
existing seats, and creates new specialist seats (via the meta-agent,
active immediately, no approval needed) when nothing existing fits — its
own judgment call, not a rule table. See the dashboard's Seats section or
`GET /seats` to see the roster it's built. Not scheduled yet, run
manually for now — same as the meta-agent below.

## API + dashboard

```bash
docker compose up -d api   # http://localhost:8000, brings up postgres too
```

Open `http://localhost:8000/` for the dashboard (vanilla HTML/JS, no
build step — `public/index.html`): the seat roster (with each seat's
outcomes), ready/in-progress/needs-a-human ticket lists, pending prompt
proposals with an Approve button, and an outcomes lookup. Polls every 5s.

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

## Sandbox (Phase 7 — overwatch's containment boundary)

```bash
# needs the Docker socket -- deliberately never mounted in harness/api,
# see PLAN.md Phase 7 for why. A separate profile so it isn't started by
# `docker compose up` by accident.
docker compose --profile sandbox run --rm sandbox-runner pytest tests/test_sandbox.py -v
```

Proves the actual containment properties live against real Docker: no
secrets visible by default, the mount is read-only, `--network none`
blocks network access, `--pids-limit` caps a fork-bomb attempt, and a
timed-out sandbox container is actually killed, not left running. The
regular test suite (`docker compose run --rm harness pytest`) skips these
automatically — no Docker socket there, which is itself the boundary
this phase is about, not an oversight.

`src/harness/tool_proposals.py` + `GET/POST /tool-proposals` + the
dashboard's Tool Proposals section carry a candidate tool from
`propose` → `sandboxed` → `reviewed` → `approved`/`rejected`. Nothing
auto-activates at any stage, unlike a new seat's first prompt — `approve`
is always a distinct, human-triggered call.

```bash
# overwatch proposes a tool from real capability-gap evidence (or an
# explicit brief while that evidence is still thin):
docker compose run --rm -e OVERWATCH_BRIEF="..." harness python scripts/run_overwatch.py
# sandboxing needs the Docker socket, so it's a separate step, sandbox-runner only:
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_sandbox_for_proposals.py
# reviewer forms a real allow/deny verdict on the source + sandbox evidence:
docker compose run --rm harness python scripts/run_reviewer.py
# then a human approves/rejects via the API/dashboard, same as any tool proposal.
```

## Self-modification (the harness improving its own source)

Same containment shape as tool proposals above, but for `src/harness/*`
itself — the "control plane" PLAN.md's Phase 7 design walls off ("no
agent, including overwatch, ever writes here directly"). The only code
with real write access to the actual tree is `run_self_mod_deploy.py`'s
own trusted orchestration; a self-modifier agent only ever writes to a
separate, isolated checkout.

```bash
# propose a change to the harness's own source, from real capability-gap
# evidence or an explicit brief:
docker compose run --rm -e SELF_MOD_BRIEF="..." harness python scripts/run_self_modifier.py
# sandboxed test run against the diff -- a real isolated image build +
# throwaway Postgres, needs the Docker socket, sandbox-runner only:
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_self_mod_sandbox.py
# reviewer forms a real allow/deny verdict on the diff + sandbox evidence:
docker compose run --rm harness python scripts/run_self_mod_reviewer.py
# approved proposals: applies the diff to the REAL tree, re-tests it
# there as a final hard gate, commits, and rebuilds/restarts the
# affected services -- also sandbox-runner only (needs both the Docker
# socket and write access to the real repo):
docker compose --profile sandbox run --rm sandbox-runner python scripts/run_self_mod_deploy.py
```

**No human approval gate here, deliberately** (the user's own explicit
call) — an `allow` verdict immediately approves, and deploy immediately
applies it once its own hard mechanical precondition is met (the
sandboxed run had zero real test failures, not just an "allow" from the
model). Git history is the actual rollback mechanism: a bad "allow" is a
`git revert` away, not something a pre-deployment approval step was ever
going to catch better than a real, passing test suite already does.
`approve`/`reject` stay reachable directly via the API as a manual
override path, same as every other proposal system in this project, but
nothing waits on it. Live-proven end to end 2026-08-30 against a real
(intentionally trivial) proposal — see PLAN.md's dated section for the
full trace, including two real bugs this surfaced and fixed
(Docker-out-of-Docker forcing raw `docker build`/`run`/`network` calls
instead of `docker compose` for the sandboxed/final test runs, and a
missing Compose CLI plugin in `sandbox-runner`'s own image).

## Sidecar interface agent (conversational, from your phone)

`mcp-server/` is a separate MCP server exposing this harness's real API
(projects/tickets/seats) as tools to an ordinary Claude Code session —
run wherever you normally run Claude Code, not inside this project's own
docker-compose stack, with Claude Code's own Remote Control enabled so
it's reachable from claude.ai/code and the mobile app. Gives
conversational, Sonnet-tier planning against a live deployment at
subscription pricing instead of a parallel per-token system. See
[`mcp-server/SIDECAR.md`](mcp-server/SIDECAR.md) for setup, the
role prompt, and — importantly — which API URL to use, which depends on
*where* the sidecar itself runs relative to the deployment (a host-level
process on the same box needs a different address than one on another
LAN device, if the deployment uses `docker-compose.prod.yml`'s macvlan
overlay below).

## Production deployment (macvlan overlay)

`docker-compose.prod.yml` is an overlay for deploying alongside other
apps on a LAN box (this project's convention, shared with
[irl](https://github.com/Citizen-Forge/irl) and others): internal
services (postgres, harness, scheduler) stay on the private
docker-compose network only, while `api` gets a real static LAN IP via
Docker's macvlan driver (`br0`) so it's reachable from other real
devices on the network — plus a *second*, pinned static IP on the
internal bridge network specifically so a host-level process (like a
sidecar above, if it runs on the same box) can still reach it, since a
Docker host can never reach its own macvlan children.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d harness api scheduler
```

No git-based deploy today — sync the working tree over (`tar`-over-SSH,
excluding `.git`/`.env`/`workspace`/`sandbox-scratch`), rebuild, restart.
`.env` lives only on the deployment target, never synced from a dev
checkout; `REPO_HOST_PATH`/`SANDBOX_SCRATCH_HOST_PATH` (needed for
self-modification's Docker-out-of-Docker calls above) resolve from
`${PWD}` at compose-invocation time, not from `.env`, so they need no
manual per-deployment configuration as long as commands run from the
deployment's own working directory.

## Verifier (acceptance-criteria pass/fail)

A real, automated positive-feedback signal (replaces the earlier idea of
a human-feedback "Laurels" surface): give a ticket explicit acceptance
criteria at creation time, and once a seat closes it, a separate verifier
agent judges the real evidence against those criteria and records a
pass/fail — not self-graded by the seat that did the work.

```bash
docker compose run --rm harness python scripts/run_verifier.py
```

## Scheduler (on by default)

Runs product-owner triage, meta-agent revision proposals, overwatch
capability scanning, and the verifier on a loop instead of manual
`docker compose run` invocations for each — starts automatically with
`docker compose up`, same as harness/api. Deliberately a reversal of v1's
"autonomy off by default" posture, per the user's own call: this project
is built around a local, unmetered model, so the cost/risk calculus that
justified gating recurring work behind manual activation elsewhere
doesn't really apply here. Narrowly scoped, though — the things that
protect against a *bad change silently taking effect* (prompts.py's
revision-approval step, tool_proposals.py's approve/reject gate) are
untouched; this only flips whether recurring work gets kicked off on its
own, not whether generated tool code or prompt revisions auto-activate.

```bash
# to opt back OUT of automatic scheduling for a given deployment:
docker compose up postgres harness api
```

## Proving the durability guarantee manually (with a real model)

Verified live 2026-08-29 against a real model, not just described here —
see PLAN.md's "Open questions" for the actual log evidence.

1. `docker compose run --rm harness python scripts/enqueue_demo.py "<title>" "<prompt requiring a couple tool calls>"`
2. Assign it to a seat (see "Running it" above — a fresh ticket has no
   seat by default).
3. `docker compose up harness` and let it start working — watch for
   `starting thread <id>` in the logs.
4. `docker compose kill harness` partway through (once you've seen at
   least one real tool-call round trip in the logs).
5. `docker compose up harness` again — look for `resuming thread <id>` in
   the logs (a distinct message from the fresh-claim `starting thread`,
   confirming it picked the same ticket back up from its last checkpoint
   via `bd list --status=in_progress`, not a restart from scratch).

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free to use, modify and share for any noncommercial purpose.
