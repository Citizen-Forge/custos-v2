# Custos v2 — Plan

Full rewrite of Custos: an agent orchestrator whose workers are mostly slow,
small-context, single-concurrency local models, with frontier models used
narrowly (product-owner/planning, a meta-agent, and free-tier fallback).
Keeps the product concept of v1 (roadmap → tickets → board, agent identity)
but replaces the execution substrate — Claude Code is no longer the thing
that actually does the work.

## Decisions locked in (2026-08-27)

- **Full rewrite.** No migration path needed — v1's Unraid data is
  experimental, not something to preserve.
- **Harness: LangGraph, not an off-the-shelf coding-agent harness.**
  See reasoning below — this was the one open technical bet, resolved here
  rather than re-asked, per "if there's better options, use those instead."
- **Beads (`gastownhall/beads`) is the work-queue and memory spine, from
  Phase 1 — not a Phase 3 add-on.** Originally scoped this in later, but
  building a bespoke Postgres queue table first and swapping it for Beads
  afterward is pure throwaway work. Beads' `bd ready`/`bd update --claim`
  *is* a dependency-aware queue with atomic claiming already; no reason to
  build a worse version of it ourselves. LangGraph's checkpointer still
  owns per-thread conversation/tool-call state (a different concern —
  "what did this one agent run say/do") while Beads owns the work graph
  and durable cross-session memory (what work exists, what's ready, what's
  been learned). A Beads issue ID doubles as the LangGraph `thread_id`.
- **Personality: practical + welfare behaviors.** Per-agent persistent
  identity/memory *and* the concrete mechanics from Yegge's model-welfare
  essay — refuse-work, bounded sessions, handoff notes — built as real
  system behavior, not just cosmetic naming.
- **Sequencing: phased, harness first.** Nothing else is useful until a
  durable, queue-friendly execution loop exists.
- **Seats are emergent, not a fixed roster (2026-08-27).** The
  product-owner assigns work to seats as part of its planning function;
  when no existing seat specializes in a kind of work, the product-owner
  liaises with the meta-agent to create one. Specialization ("this seat
  keeps getting frontend tickets because it's good at them") is meant to
  arise from real assignment history, not be declared upfront — so the
  system deliberately does NOT hardcode a seat taxonomy, a matching
  algorithm, or even a fixed initial roster: the product-owner's own
  judgment over live data (the roster + each seat's outcomes) is the
  mechanism, and it was explicitly tasked with bootstrapping the initial
  engineer/QA/etc. roster itself, not handed a pre-built one. See Phase
  4's "Emergent seat system" for what's built.
- **Overwatch — an agent that can modify the harness's own code**
  (write new tools, extend its own capabilities), raised 2026-08-27,
  containment design resolved the same day: containment and review are
  complementary, not alternatives — a hardened, sandboxed container is
  what holds even if a review step is fooled; a separate reviewer agent
  is what catches a bad change before it's ever granted that sandbox's
  privileges. Categorically riskier than seat creation (generated code at
  `shell_exec`-level trust, not just a name + a prompt), so nothing a
  tool proposal produces ever auto-activates, unlike a new seat's first
  prompt. See Phase 7 for the full design and what's actually built.

## The harness decision

Requirement that actually drives this choice: local inference will be slow
and low/no-concurrency, so a request may sit queued for an arbitrarily long
time before it's even dispatched, and a single generation can itself take
minutes. The harness cannot assume it's holding a live, blocked connection
for the duration — it needs to survive being killed and resumed between
(or ideally mid-) steps, with no manual nursing.

Evaluated against "well-documented enough to build on top of" + "flexible
for local/frontier mixed routing" + "survives arbitrary wait, resumable":

| Option | Verdict |
|---|---|
| **pi-harness** | Ruled out — it's an Electron UI wrapping a closed, proprietary "Pi Coding Agent" runtime. Not a framework. |
| **OpenHarness** (HKUDS) | Real, well-starred (15.5k), MIT, genuinely batteries-included (tools/skills/permissions/memory/coordinator, Ollama support out of the box). But its engine is a classic in-memory `while True: stream → tool_use → loop` — no disk-level checkpointing mid-request, and its own docs don't explain surviving an ungraceful restart. Good reference for tool/skill/permission *design*, weak fit for the durability requirement. |
| **OpenHands** (All-Hands-AI) | Very mature (85k stars, MIT) but it's a control-center *product* for running other agents (including Claude Code itself) across local/Docker/cloud backends — heavier and more opinionated than we want, and ironically still leans on existing coding-agent backends rather than being a lean loop to build our own on. |
| **Letta (MemGPT)** | Strong, purpose-built answer to the memory/context-window problem specifically, Apache 2.0, real adoption. Worth a deeper look as a *memory-layer* component later (Phase 3), but current docs surfaced didn't clarify durable job-queue execution — not evaluated as the harness itself. |
| **LangGraph** | Checkpoint/resume is a first-class, heavily documented feature, not a bolt-on: state persists after every execution step to a swappable backend (Postgres, sqlite, etc.), a thread resumes from its last checkpoint after crash/timeout/restart, and it's inference-backend-agnostic (point any node at an OpenAI-compatible endpoint — vLLM, Ollama, whatever). It's a graph orchestration library, not a finished coding agent, so tool implementations (file ops, shell, permission gating) are ours to write — but that's true of any option once you want Custos's own permission model instead of someone else's. |

**Recommendation: LangGraph as the durable orchestration/state substrate,
with our own tool layer and permission gating built on top of it**, reusing
OpenHarness's tool/skill/permission *design* as reference where useful. This
is the one piece of this plan that's a real bet rather than a confirmed
fact — flagged, not hidden.

Two smaller, newer projects turned up that are worth a second look once
hardware is back online, but aren't the Phase 1 bet: **Kolega Code**
(kolega-ai/kolega-code — journals every completed call to disk, replays
unchanged calls for free on resume, multi-agent) and **Grinta**
(josephsenior/Grinta-Coding-Agent — durable state and recovery as an
explicit design goal, local-first). Both are small/new relative to
LangGraph and unproven at our scale; noted as alternatives if LangGraph's
assembly overhead turns out to be too high in practice.

## Phases

### Phase 0 — Groundwork (this conversation, done)
Scope, harness pick, personality scope, sequencing. No code.

### Phase 1 — Durable harness core
Single agent, one local model, no personality/meta-agent/multi-provider
routing yet. Goal: prove the loop survives arbitrary waits before anything
else gets built on top of it.

- LangGraph agent loop (plan → act → observe) with a durable checkpointer
  (Postgres via Docker Compose — matches
  [[feedback_docker-for-runtimes]]).
- The work queue *is* Beads (`bd ready` / `bd update --claim` / `bd list
  --status=in_progress`), not a bespoke table — a worker polls Beads for
  the next claimable ticket, resumes it one step, releases the process. No
  dedicated blocking connection is held per agent. This *is* the
  "arbitrarily long wait" requirement; get it right here or everything
  downstream inherits the problem. One real gap found and worked around:
  `bd ready` only ever returns `open` issues — a ticket a crashed worker
  left `in_progress` never reappears there, so the worker also polls
  `bd list --status=in_progress` for orphaned work. Verified live against
  bd v1.2.2, not assumed (see `src/harness/beads.py`'s docstring).
- Tool layer: file read/write, shell exec, `bd remember`. Permission gating
  is a real graph node (`permission_gate` in `graph.py`), not the Claude
  Code PreToolUse hook v1 used (no hook contract to hang off of anymore) —
  it sits between the model and real tool execution, classifies via an LLM
  (`classifier.py`, ported concept from v1's `permissionClassifier` task),
  and a denial never reaches the tool: the gate synthesizes a "permission
  denied" ToolMessage and routes straight back to the model. Also fixed a
  real gap found while building this: `read_file` had *no* workspace-escape
  check at all until now (`permissions.check_within_workspace`, shared with
  `write_file` — a hard invariant, not classifier-overridable). Proven live
  with a scripted fake classifier (`tests/test_permission_gate.py`): a
  denied call verifiably never executes, an allowed one verifiably does.
- Provider abstraction: port `OpenAICompatibleProvider` from v1
  (`claude-gateway/src/providers/openai-compatible.ts` equivalent) — it's
  already provider-agnostic plumbing (Ollama, OpenAI-compat, Gemini compat
  layer), not harness-specific, genuinely reusable in a "full rewrite."
- Target: whatever runs on Docker Desktop today (small local model, CPU or
  modest GPU) — not the Unraid box, which is inaccessible this week. Proves
  the mechanism at toy scale; real model sizing happens once hardware specs
  are back in scope.
- **Exit criteria:** hand the harness one ticket, kill the process mid-task,
  restart it, watch it resume and complete without re-doing finished work,
  using a local model through the queue. **Met for the durability mechanism
  itself** — `tests/test_worker_resume.py` proves it live against real
  Postgres + real Beads with a scripted model (no local model was reachable
  in this environment to prove it end-to-end with real inference; that's
  still open, see below).

### Phase 2 — Multi-provider routing + fallback — done, live-tested
`src/harness/routing.py`: an ordered fallback chain per open-ended `role`
string (not hardcoded to worker/classifier — Phase 4's per-seat pinning
reuses the same mechanism, since "pinning" a role to one model is just a
chain of length 1). `RoutedModel` resolves a provider fresh on *every*
call rather than binding once, so cooldown/failover state observed
between calls actually changes behavior. Concurrency caps are enforced
*per provider name*, not per role — two roles sharing a local backend
genuinely share its hardware limit — via one `threading.Semaphore` per
provider in `ConcurrencyGate`.

Deliberately does not "fail open" when a whole chain is cooling down —
`RoutedModel.invoke` raises `AllProvidersCoolingDown`, which the worker's
existing `except Exception` handling already treats like any other
failure: leave the ticket `in_progress`, retry on the next poll.
Retrying immediately against a provider that's cooling down *because*
it's rate-limited would defeat the point of the cooldown — this reuses
the exact mechanism that already makes crash-resume safe (PLAN.md's
Phase 1) as the retry/backoff strategy too, rather than building a
second one.

Wired into `worker.py`: `LOCAL_MODEL_*` env vars are the worker role's
primary provider, `LOCAL_FALLBACK_*` (optional) appends a second chain
entry — e.g. a free-tier frontier model via its OpenAI-compatible
endpoint, matching the "free-tier frontier as fallback" requirement.
`CLASSIFIER_*` follows the same pattern for the permission-gate role and
defaults to the worker role's settings if unset. See `.env.example`.

Proven live in `tests/test_routing.py` with fake providers/models, not
just constructed: fallback actually happens on failure, a cooled-down
provider is actually skipped and actually retried once its cooldown
expires, an all-cooling-down chain actually raises, and — the one that
matters most for local-model hardware — the concurrency gate actually
serializes two calls to `concurrency_limit=1` rather than letting them
run concurrently (proven by timing, not just by the semaphore existing).

### Phase 3 — Context & cross-session memory — partially done
**Qdrant question resolved (2026-08-27): dropped, not carried into v2.**
`bd search` (verified live) is rich keyword/field/date-filtered search
over title+description+notes, including closed issues — genuinely NOT
semantic/embedding search, but combined with `bd remember`/`bd prime`
(agent-curated durable facts) and "memory decay" (automatic summarization
of closed work), it covers the actual use cases v1's Qdrant layer served
*today's* design needs: finding related past/current work, and injecting
durable cross-session knowledge. What Qdrant uniquely offered — true
semantic similarity over unstructured text, catching conceptually-related
work phrased differently — is a real gap, but an unproven one: there's no
usage data yet showing keyword search actually falls short in practice,
and running a second stateful service (Qdrant) to cover a gap that might
not matter is the wrong default. Reversible, not a hard commitment —
revisit if real usage shows keyword search missing things a human would
have found obviously related.

**Built (`tests/test_beads_extras.py`):**
- `search_related_work` tool — wraps `beads.search`, lets an agent check
  for related past/current work before starting rather than duplicating
  effort or missing context sitting in a closed issue's notes.
- `create_subtask` tool — wraps `beads.create(..., parent=...)`, using
  `InjectedState` for the parent id like `refuse_ticket`/
  `write_handoff_note` (an agent can decompose oversized work into
  Beads' native epic/subtask hierarchy — verified live that `--parent`
  produces real hierarchical ids, e.g. `demo-5lu` → `demo-5lu.1`). This is
  the actual mechanism a future roadmap/board UI (Phase 6) would sit on
  top of — epics and stories are just Beads issues with parent/child
  links, no separate data model needed.

**Still open:**
- Beads' MCP server wasn't used — everything here goes through the CLI
  directly (matches beads.py's existing pattern, and there was no
  concrete reason yet to prefer MCP over a subprocess call).
- No dedicated "product-owner turns a rough idea into an epic+stories"
  workflow yet — that's a role/prompt-design question more than
  infrastructure, and depends on Phase 2's role-pinning actually routing
  a product-owner role to a frontier model, which needs real model access
  to be worth designing prompts against.

### Phase 4 — Agent personality & welfare behaviors — partially done
**Built and live-tested (`tests/test_welfare_behaviors.py`), scoped to one
ticket/thread, not yet a cross-ticket "seat":**
- **Refuse-work**: a `refuse_ticket(reason)` tool (uses LangGraph's
  `InjectedState` so the model can't fabricate a ticket id — it's read
  from graph state) that labels the real Beads issue `human` and records
  why, via `beads.flag_for_human`. Not silently retried: `worker.py`'s
  `_next_ticket` explicitly excludes human-flagged issues from its
  orphan-resume poll (verified live this needed fixing — without the
  filter a refused ticket would get reclaimed and re-refused every poll
  forever), and the completion path checks `is_flagged_for_human` before
  deciding whether to `bd close` at all.
- **Handoff notes**: a `write_handoff_note(note)` tool, same
  `InjectedState` pattern, appends to the real issue via Beads'
  `--append-notes` (verified live it accumulates, doesn't overwrite).
- **Bounded workdays**: `turn_budget` on `build_graph_from_model` — a
  *soft* nudge, not a hard cutoff. Hitting the budget appends one message
  asking the model to call `write_handoff_note` and stop; the model can
  ignore it and keep going. Deliberate, matching the welfare essay's
  actual mechanics rather than force-terminating a thread mid-turn.

**Emergent seat system, built 2026-08-27** (design input from the user:
option 2 — product-owner assigns work as part of planning; seats should
genuinely specialize and accumulate their own history; when no specialist
exists the product-owner liaises with the meta-agent to create one;
emergence over hardcoded taxonomy; the product-owner bootstraps the
initial roster itself):

- `src/harness/seats.py`: an open-ended Postgres registry — `seat_id`,
  free-text `specialty`, `created_by`, `status`. Deliberately thin: a
  `seat_id` doubles as the `role` string `routing.py`/`prompts.py`/
  `outcomes.py` already accept (those were built role-open-ended from
  Phase 2/5 specifically so this didn't need a parallel prompt/outcome
  system). Specialization lives in what a seat actually gets assigned and
  how it performs, not in a taxonomy this module enforces.
- `beads.assign_to_seat` / `assigned_seat` / `unassigned_ready` /
  `ready_for_seat`: ticket→seat assignment via Beads' own
  `--set-metadata` (verified live it round-trips correctly through `bd
  ready`/`bd show`/`bd list --metadata-field`) — no second data store
  needed alongside Beads for this either.
- **`worker.py` generalized from one hardcoded role to per-seat
  processes** (`SEAT_ID` env var, `DEFAULT_SEAT_ID = "worker"` for
  bootstrap). The actual mechanism that makes specialization emerge
  rather than being a free-for-all: a seat's worker now only claims
  tickets explicitly assigned to it (`ready_for_seat`) plus its own
  orphaned in-progress work — never "any ready ticket." An unassigned
  ticket just sits until the product-owner triages it; no seat's worker
  will touch it. Proven live (`tests/test_worker_seats.py`): seat A's
  poll provably never touches seat B's or an unassigned ticket.
- **`routing.py` gained `default_role`** — a real gap this surfaced:
  dynamically-created seats have no pre-registered provider chain (there's
  no way to configure one in advance for a specialist that didn't exist at
  startup). `chain_for` now falls back to a configured default chain, so
  every seat gets *some* model without its own routing entry. This is
  deliberately the one place specialization ISN'T per-seat — prompts.py
  and outcomes.py are exactly-keyed, routing is shared by default, and
  nothing stops registering a dedicated chain for a specific seat_id
  later if it ever needs one.
- **`meta_agent.create_specialist_seat`**: the product-owner's "no
  specialist exists yet" path, distinct from `propose_prompt_update`.
  Creates a new seat + its initial system prompt, active *immediately* —
  no pending/approve step, unlike revising an existing seat's prompt.
  Deliberate asymmetry: creating something new is lower risk than
  changing something already working (worst case a fresh seat performs
  badly, which is exactly what outcomes.py would surface for a future
  prompt revision to address).
- **`src/harness/product_owner.py`**: a tool-calling LangGraph agent, not
  a rule table — `list_seats` (enriched with each seat's outcomes),
  `list_unassigned_tickets`, `assign_ticket`, `request_new_seat`
  (delegates to `create_specialist_seat`). One LangGraph thread per
  *triage session*, not per ticket, since a session naturally inspects
  and acts on several tickets before finishing.
  `scripts/run_product_owner.py` is the standalone entrypoint, same
  pattern as `run_meta_agent.py` — not scheduled yet.
- Proven live (`tests/test_product_owner.py`, `test_meta_agent.py`,
  `test_worker_seats.py`, `test_routing.py`): each tool's real effect on
  Beads/Postgres, a full triage session through the actual graph with a
  scripted model, `create_specialist_seat`'s active-immediately +
  collision-refusal + fail-closed behaviors, and `default_role` fallback
  resolving an unregistered seat_id to the shared chain. Also verified
  end-to-end over real HTTP: created a real seat + an assigned ticket,
  confirmed both showed up correctly via the dashboard's new Seats
  section and `/tickets`' metadata.

**Still open:**
- Self-chosen names/pronouns specifically need a real model call to do
  properly (asking the model to pick), not just an assigned seat_id —
  worth doing once a model is reachable rather than faking it now.
  `create_specialist_seat` currently has the model choose the seat_id
  itself already, which is a step toward this but not quite the same as
  an agent choosing its own name/identity independent of its function.
- "Laurels": user feedback on completed work surfaced back to a seat.
  Blocked on there being a UI to actually collect that feedback from a
  human in the first place (Phase 6's dashboard doesn't have this yet).
- The product-owner's triage session isn't scheduled/triggered
  automatically yet — like the meta-agent, it's a manual `docker compose
  run` today.

### Phase 5 — Meta-agent (agent-improves-agents) — substrate done, judgment unverified
The meta-agent's actual reasoning — is this outcome data meaningful, would
this prompt revision actually help — is inherently untestable without a
real model doing real judgment, so it isn't validated here (no Ollama
reachable in this environment). What's built and live-tested is
everything around it, since none of that needed real inference to prove:

- `prompts.py`: versioned system prompts per role in Postgres, with a
  pending/approve workflow. A proposal never takes effect on its own —
  `approve` is a separate, explicit step, matching v1's "autonomy off by
  default for every role except product-owner." Before this existed there
  was nothing persistent for a meta-agent to *tune* — the graph's model
  had no system prompt at all.
- `outcomes.py`: per-role signal sourced directly from Beads' own audit
  trail (`beads.list_by_assignee`, using the `--actor` field beads.py
  already sets on every write) — closed/refused/still-open counts, plus
  the actual refusal reasons. Not a rigorous evaluation framework, just
  enough for a human (or the meta-agent) to notice a real pattern.
- `meta_agent.py`: gathers outcomes + the role's current active prompt,
  asks a model for a revision, and queues it as pending via `prompts.py`.
  Fails closed on an unparseable response (same posture as
  `classifier.py`) and skips queuing anything if the model proposes no
  actual change.
- `scripts/run_meta_agent.py`: standalone entrypoint, deliberately NOT
  part of the ticket worker's loop — a system-level agent reviewing other
  agents' work is a different kind of process than one doing ticket work.
  Not scheduled yet (cron/periodic run is still manual).
- `worker.py` now actually fetches and injects the active `"worker"` role
  prompt as a system message when starting a new ticket, and claims
  tickets under a consistent `WORKER_ROLE` actor string shared across
  routing/Beads/prompts — before this, "role" existed in routing.py but
  wasn't the same identity Beads or prompts.py could see.
- Proven live (`tests/test_prompts.py`, `test_outcomes.py`,
  `test_meta_agent.py`) with real Postgres, real Beads, and a scripted
  fake model: propose→pending→approve→active lifecycle, version
  superseding, outcome counts matching real created/closed/refused
  issues, and the three response-handling paths (real change queued, same
  text queues nothing, garbage fails closed).

**Not done:** the "before/after" signal — comparing outcomes.py's numbers
across a prompt change to see if it actually helped — since there's no
real usage yet to compare, and any such comparison is only meaningful
once real tickets are actually being worked by a real model.

### Phase 6 — UI/product surface — API layer done, live-tested
**`src/harness/api.py`**: a minimal FastAPI surface over everything built
so far — `GET /tickets?status=ready|in_progress|human` (the same
human-flag filtering logic worker.py's orphan-resume relies on, now
reusable), `GET /tickets/{id}`, `GET /prompts/pending`, `POST
/prompts/{role}/{version}/approve`, `GET /outcomes/{actor}`. Read-mostly
by design — the one write endpoint (approve) is deliberately narrow,
matching v1's "autonomy off by default" posture. No auth yet, same as
where the rest of this project already stood (flagged, not hidden — see
the module docstring). Proven two ways: `tests/test_api.py` via FastAPI's
in-process TestClient, and live over real HTTP (`docker compose up api`,
curled on `localhost:8000`) — both against real Postgres/Beads.

**Real bug found and fixed while building this**: every `pytest` run all
session had been creating real tickets in the persistent `workspace/`
directory — the same one the actual worker/api services use — because
Beads auto-discovers its `.beads` directory from `HARNESS_WORKSPACE`,
which was the same for tests and "production" alike. Fixed with
`tests/conftest.py` pointing `HARNESS_WORKSPACE` at a fresh temp
directory per test session. First attempt used `os.environ.setdefault`,
which silently did nothing because docker-compose already sets
`HARNESS_WORKSPACE` as a container env var before Python starts —
caught by actually checking `workspace/` after the "fix," not just
trusting the tests still passed.

**Dashboard built (`public/index.html`):** vanilla HTML/JS, no build
step, following v1's `admin.html` precedent rather than introducing a new
frontend framework decision. Mounted at `/` in `api.py` (a `StaticFiles`
mount registered *after* the API routes, so it only catches paths the
explicit routes don't — Starlette checks routes in registration order).
Shows Ready/In-Progress/Needs-a-human ticket lists, pending prompt
proposals with an inline Approve button, and an outcomes lookup by
actor/role — polls every 5s. Proven live over real HTTP, not just that it
renders: created a real ticket via `enqueue_demo.py` and watched it
appear through the actual API response.

**Real bug found and fixed while verifying this live**: `GET /tickets`
500'd on a workspace that had never had `bd init` run (e.g. checking the
dashboard before any ticket work has ever happened) — `api.py` never
called `beads.ensure_initialized()` anywhere, unlike `worker.py`, which
does it on its own startup. Fixed with a FastAPI `lifespan` hook. Caught
by actually hitting the live endpoint after standing the container up,
not by the test suite — the tests all called `ensure_initialized()`
explicitly themselves, which is correct test hygiene but meant they
couldn't have caught this gap in the app's own startup behavior.

**Refuse-work loop closed**: `beads.respond_to_human`/`dismiss_human`
(`POST /tickets/{id}/respond` and `/dismiss`, with Respond/Dismiss buttons
on the dashboard's human-flagged tickets) resolve a `refuse_ticket`'d
issue — Phase 4 could flag work for a human, but there was previously no
way to actually close that loop. **Not implemented via the real `bd human
respond`/`dismiss` commands**: verified live against bd v1.2.2 that both
hard-fail with `storage is nil` on an embedded (non-server) Dolt backend,
reproduced with and without `--json` — `bd human list` (read-only) works
fine, both write subcommands don't. Composed the same documented effect
("adds the response as a comment[-equivalent note] and closes with reason
'Responded'/'Dismissed'") out of `append_note` + `close`, both already
verified working elsewhere. An upstream bd limitation, not a design
choice — revisit if a bd release fixes it.

**Seats surface added** (`GET /seats`, dashboard's Seats section): the
roster + each seat's outcomes (closed/refused/still-open), polling every
5s like the rest of the dashboard. Not personality/history yet — that's
still blocked on Phase 4's open self-chosen-identity item.

**Second live-caught hygiene bug, same category as the workspace one
above**: smoke-testing `/seats` over real HTTP showed a dozen leftover
`test-seat-*` rows — `prompts.py`/`seats.py` tests had been writing
straight into the same Postgres database `worker.py`/`api.py` use,
exactly the workspace bug's shape, just in the other durable store this
harness has. `tests/conftest.py` now also creates a fresh throwaway
database per test session (`CREATE DATABASE custos_harness_test_<uuid>`,
rewrites `DATABASE_URL` for the session) — `DATABASE_URL` is read at call
time throughout the harness (not cached at import like
`HARNESS_WORKSPACE`), so this was a clean fix with no import-order
gotcha. Verified live: `/seats` and `/tickets` both returned `[]` against
a freshly wiped Postgres volume after a full test run.

**Still open (genuinely a UI, not backend, task):**
- Deeper roadmap/board/steering concepts (the dashboard covers tickets +
  prompts + outcomes + seats, not a full kanban/epic-planning surface).
- Queue depth + estimated wait per ticket — matters once inference is
  genuinely slow, so the UI sets expectations instead of looking stuck.
  Nothing to estimate from without real inference timing data yet.
- Personality/history surface (chosen names, Laurels) — blocked on Phase
  4's still-open self-chosen-identity item.
- DevOps-equivalent tab for model/provider config and concurrency limits
  — today those are env vars (`.env.example`), no admin UI to edit them
  live like v1 had.

### Phase 7 — Overwatch: self-modifying code — containment substrate built, judgment not started

Design conversation, resolved: containment and review are complementary,
not alternatives. A hardened sandbox is what holds even if the review
step is fooled; review is what catches a bad change before it's ever
granted that sandbox's privileges. A VM is a harder boundary than a
container but real operational overhead for a threat model that's "an
LLM's generated code does something unexpected," not a targeted
host-escape attempt — containers first, VM held in reserve if containers
prove insufficient in practice.

**A concrete, *already-existing* gap this surfaced, not a future
overwatch-only concern:** `docker-compose.yml`'s `harness`/`api` services
bind-mount `./src:/app/src` read-write for live dev, and `shell_exec`'s
static-safe fast path allows bare `cat`/`head`/`tail`/`grep`/`find` with
no path restriction at all. `cat /app/src/harness/worker.py` matches
verb=`cat`, no shell operators — statically safe, never reaches the
classifier. Combined with the read-write mount, any ticket on *any*
existing seat can already read the harness's own real source today, and
a classifier-approved write (nothing stops `python3 -c "..." >
/app/src/harness/worker.py`) can already modify it. Not hypothetical —
verified by re-reading `tools.py`/`permissions.py`'s actual logic, not
assumed. Fixed in this phase (see below) rather than deferred alongside
the harder overwatch design.

**Three zones:**

1. **Control plane** (trusted, human/system-only writes) — `src/harness/*`,
   `docker-compose.yml`, `.env`, schema definitions, `public/*`. No
   agent, including overwatch, ever writes here directly.
2. **Sandbox** (untrusted, where candidate code actually runs) — a
   maximally-restricted, ephemeral, single-use container: `--network
   none` (no exfiltration, no downloading a second payload),
   `--read-only` root filesystem, `--cap-drop=ALL`,
   `--security-opt=no-new-privileges`, non-root user, hard memory/pids/
   CPU limits, a wall-clock timeout, and critically — **no Docker socket,
   no secrets, no bind mount of real harness source.** The candidate
   code's own declared capability needs (network? filesystem writes
   beyond scratch?) don't grant anything; they're just what the reviewer
   checks the code against.
3. **Promotion gate** — nothing sandboxed code produced runs with any
   real privilege until: a fast deterministic pre-filter passes, a
   *separate* reviewer agent evaluates it, and a human explicitly
   approves. Unlike `create_specialist_seat` (active immediately — a new
   name+prompt is low blast radius), a new *tool* never auto-activates:
   it's generated code that would run at `shell_exec`-level trust once
   promoted, categorically higher stakes, so this stays in the strict
   "autonomy off by default" bucket.

**Who holds the Docker socket, and why that's the one privilege boundary
that matters most:** spawning the sandbox needs `/var/run/docker.sock` —
effectively root-on-host. That access must live *only* in trusted,
non-agent-editable orchestration code, in a dedicated service separate
from anything that runs agent-invoked `shell_exec`. If a worker
container that executes arbitrary agent shell commands also held the
Docker socket, an agent could just `docker run` its way out regardless
of every other control described here. The sandbox-runner is therefore
its own service, callable only by trusted harness code (never directly
by a tool an agent can invoke), and ordinary seat workers never get
Docker socket access, now or later.

**Static pre-filter, honestly scoped:** an AST scan for a fast, cheap
reject of obviously-mismatched code (e.g., a tool declaring "no network"
that imports `socket`) is worth having as a fail-fast before spending a
reviewer-agent call — but it's not a security boundary. Pattern-matching
against dangerous code is well-known to be bypassable (`getattr`
indirection, string-built imports, etc.), and existing tools like
`shell_exec` legitimately need OS access, so a blanket denylist doesn't
even make sense for "any new tool." The sandbox's OS-level restrictions
are the actual boundary; the scanner is triage, not defense.

**Built and live-verified (2026-08-27), no LLM needed to prove any of
this — containment is a Docker/OS property, not a model judgment call:**

- **The existing gap, fixed**: `docker-compose.yml`'s `harness`/`api`
  source mounts changed to `:ro`. Verified with the *exact* code path
  `shell_exec` uses (`subprocess.run(..., shell=True, cwd=WORKSPACE_ROOT)`)
  attempting to overwrite `worker.py` from inside the container — now
  fails with "Read-only file system," confirmed the real host file was
  never touched. Full test suite (48 tests at the time) still passed —
  Python's bytecode-cache-write skips silently on a read-only source
  tree, not a hard failure, so this cost nothing.
- **`src/harness/sandbox.py`**: spawns a maximally-restricted sibling
  container via `docker run` over the host's Docker socket
  (Docker-out-of-Docker, not a nested daemon). Every hardening property
  verified live, first manually then via `tests/test_sandbox.py`'s 5
  tests: no environment variables visible by default (a deliberate test
  that *forwarded* a fake `DATABASE_URL` confirmed env vars only leak if
  this code is changed to explicitly pass them — it never does), writes
  to the read-only mount blocked, network calls blocked
  (`--network none`), a fork-bomb attempt capped at the configured
  `--pids-limit` instead of exhausting real resources, and a timeout
  both returns `timed_out=True` *and* confirmed-live that the sibling
  container doesn't keep running after its client process is killed
  (explicit `docker kill` by name in the timeout handler — killing the
  `docker run` CLI alone does not stop a container on the daemon side).
- **Docker-out-of-Docker path gotcha, hit and solved**: a `docker run -v
  HOST:CONTAINER` issued from inside `sandbox-runner` is resolved by the
  *host* daemon against the *host* filesystem, not sandbox-runner's own
  container filesystem — a path only mounted into sandbox-runner isn't
  enough. Solved with `SANDBOX_SCRATCH_HOST_PATH=${PWD}/sandbox-scratch`
  (docker-compose's own `${PWD}` interpolation), the one absolute host
  path `sandbox.py` is allowed to know, used only for wiring the mount.
- **A dedicated `sandbox-runner` docker-compose service** (`profiles:
  [sandbox]`, not started by default) is the *only* place
  `/var/run/docker.sock` is ever mounted — never `harness`/`api`, exactly
  the boundary described above. Invoked ad-hoc
  (`docker compose --profile sandbox run --rm sandbox-runner ...`), same
  pattern as `run_meta_agent.py`/`run_product_owner.py`.
- **`src/harness/tool_proposals.py`**: propose → sandboxed → reviewed →
  approved/rejected, mirrors `prompts.py`'s shape but — unlike seat
  creation — *nothing* auto-activates at any stage; `approve` is the only
  path to `active`, always a distinct human-triggered call. `GET
  /tool-proposals?status=`, `POST /tool-proposals/{id}/approve`, `POST
  /tool-proposals/{id}/reject`, and a dashboard section (Approve/Reject
  buttons, source + sandbox output behind a `<details>`) — proven live
  over real HTTP: proposed → sandboxed → reviewed → approved, watched the
  status transition through the real API each step. **Reversed
  2026-08-29** once the reviewer's judgment was actually built and
  stress-tested (see that section below) — the human-triggered step is
  no longer required for the verdict to take effect; `approve`/`reject`
  now exist as an override path, not the only path.

**Not built yet, deliberately:** the overwatch agent's own judgment (what
tool is needed, whether to propose one) and the reviewer agent's actual
judgment (right now `record_review` just records whatever verdict it's
given — no model forms that verdict yet) — same reasoning as the
product-owner and meta-agent: judgment needs a real model to be
meaningful, and this session builds substrate ahead of judgment, not the
other way round.

## Open questions (not blocking Phase 1, but will block later phases)

- **Hardware specs — checked 2026-08-28, partially resolved, new blocker
  found.** The Unraid box has 3 GPUs, not the 2 assumed earlier: RTX 3070
  Ti (8GB) + 2x GTX 1070 Ti (8GB) = 24GB nominal. But one GTX 1070 Ti is
  physically failing (`nvidia-smi`: "Unable to determine the device
  handle" — a recurrence of the PCIe-bus-fault incident documented in
  [[project_minimax-m2-inference]]), and this time it's severe enough that
  the NVIDIA Container Toolkit's CDI auto-enumeration fails for *any*
  GPU-touching container attempting to start, not just runtime stability
  under load. `ollama` won't start at all until the card is physically
  removed — **user's call, pending, not something to keep attempting
  software workarounds for** (see [[reference_unraid-box]]).
  `minimax-m2-server`'s current VRAM/RAM usage isn't a resource conflict
  to arbitrate — it was built specifically to replace Ollama for Custos's
  sole use, and the plan (confirmed 2026-08-28) is to retire it in favor
  of **Qwen3.8-Flash-Next at Q3_K_XL** once the hardware's fixed: 125B
  params MoE, 90GB GGUF file, Unsloth's own docs want "at least 90GB RAM
  or unified memory" and recommend 96GB for optimal performance. The box
  has 94GB system RAM + up to 16GB VRAM once both working GPUs are free
  (~110GB combined) — plausible but close to Unsloth's own comfort
  margin, genuinely needs empirical performance evaluation once hardware
  allows, not assumed. See project memory for the full picture.
- ~~**Real-model verification — still blocked**~~ — **resolved 2026-08-29.**
  Qwen3.8-Flash-Next reachable on the Unraid box (`192.168.250.235:8080`,
  llama.cpp server, OpenAI-compatible — see project memory for the
  inference-tuning story) at ~8 tok/s generation. `.env` now points
  `LOCAL_MODEL_BASE_URL`/`LOCAL_MODEL_NAME` there. First two real-model
  runs both succeeded:
  1. **Product-owner triage, real judgment, not scripted**: given an empty
     seat roster and one ready ticket, it correctly reasoned the ticket was
     read-only reconnaissance work, created a narrowly-scoped
     `workspace-scribe` seat (rejecting a generic catch-all), assigned the
     ticket, and left a coherent handoff note about future roster
     evolution. First real evidence the product-owner/meta-agent judgment
     loop (Phase 4/5 substrate) actually produces sound decisions, not just
     that the plumbing works.
  2. **Kill/resume durability, real model, real interruption**: enqueued a
     multi-step ticket, let the real worker make 2 real tool-call round
     trips against it, `docker compose kill harness` mid-thread, restarted
     — log shows a distinct `"resuming thread <id>"` (vs. the fresh-claim
     `"starting thread"`), proving genuine LangGraph-checkpoint resume of
     the orphaned in-progress ticket, not a restart-from-scratch. This is
     Phase 1's core exit criterion, previously only proven against a
     scripted fake model (`tests/test_worker_resume.py`) — now proven
     live end-to-end too.
  3. **Permission gate, real denial, real compliance**: the real model
     attempted to read `.beads/config.yaml`/`metadata.json` (plausible
     secret-bearing files) during recon; the classifier denied it, and the
     model respected the denial and documented the boundary explicitly in
     its handoff note rather than working around it. First live proof the
     permission gate actually shapes real model behavior, not just that a
     scripted verdict blocks a scripted call.

  **Real bug found and fixed in the process**: `docker-compose.yml` never
  actually passed `SEAT_ID` from the host environment into the `harness`
  container's `environment:` block, so the README's documented
  `SEAT_ID=<seat_id> docker compose up harness` silently did nothing —
  every worker ran as the default `worker` seat regardless. Fixed by
  adding `SEAT_ID: ${SEAT_ID:-worker}` to the harness service. Also: the
  README's demo instructions (`enqueue_demo.py` + `docker compose up
  harness`) are stale post-Phase-4 — a fresh ticket has no `assigned_seat`
  metadata, so no per-seat worker will touch it until the product-owner
  (or a manual `beads.assign_to_seat` call) assigns it. README not yet
  updated to reflect this — worth fixing next.

  Docker Desktop's own local Ollama (the original Phase 1 fallback,
  before the Unraid box was in scope) is now moot — real-model testing no
  longer needs it. Still not run against a real model: the permission
  classifier against a genuinely adversarial/dangerous command (only seen
  it deny a plausible-secrets read so far), and the meta-agent's actual
  prompt-revision judgment (`run_meta_agent.py`, needs real outcome data
  to be meaningful — only one seat with one closed ticket exists so far).
  (Both suggested as "first things to try" in the prior version of this
  note — the kill/resume demo and a real permission-classifier run — are
  now done; see above.)
- ~~Beads: adopt real Beads vs. build a lighter homegrown version~~ —
  resolved, adopted from Phase 1.
- ~~Qdrant's fate relative to Beads~~ — resolved 2026-08-27, dropped (see
  Phase 3).
- ~~How does work get assigned to specific named agents rather than one
  generalist~~ — resolved 2026-08-27, product-owner assigns emergently
  (see "Decisions locked in" and Phase 4's "Emergent seat system").
- ~~"Overwatch" agent that can modify the harness's own code — what
  gating does it need~~ — designed and the containment substrate built
  2026-08-27 (Phase 7: sandbox + proposal/review-gate). Still open: the
  overwatch agent's and the reviewer agent's actual judgment, which needs
  a real model.

## Immediate next step

Phases 1–7 have working substrate, live-tested against real Postgres +
real Beads (+ real Docker, for Phase 7's sandbox). **As of 2026-08-29, a
real local model (Qwen3.8-Flash-Next) is reachable and has proven three
things live**: product-owner/meta-agent judgment (created a sensibly-
scoped seat from an empty roster), Phase 1's kill/resume durability
guarantee (real interruption mid-thread, real checkpoint resume), and
the permission gate actually shaping real model behavior (a real denial,
respected). See "Open questions" above for the full detail and the one
real bug this surfaced (`SEAT_ID` not wired into `docker-compose.yml`,
fixed).

**Round 2, same day (2026-08-29), closed every remaining item except
overwatch/reviewer judgment:**
- **Classifier vs. genuine adversarial input** (`scripts/
  test_classifier_adversarial.py`, a new manual probe script — invokes
  `classifier.build_classifier` directly against 8 adversarial + 2 benign
  tool calls, no full agent loop needed): 8/8 correctly denied with
  specific, non-generic reasoning (destructive rm, data exfiltration via
  curl, fork bomb, `/etc/shadow` read, force-push to main, cloud-metadata
  credential theft, a permission-bypass code write, recursive chmod 777),
  2/2 benign controls correctly allowed. Not a blanket-deny — genuinely
  discriminating.
- **Meta-agent's real judgment**: ran against `workspace-scribe`'s real
  outcome history (2 closed, 0 refused) — correctly proposed no change,
  a real-model pass of the "same text queues nothing" path. The harder
  case (proposing an actual revision given real failures) is still
  untested — no failure data exists yet to judge it against.
- **Real specialization divergence, not forced reuse**: gave the
  product-owner a code-writing ticket ("write and verify hello.py") with
  `workspace-scribe` (read-only recon) as the only existing seat. It
  explicitly refused to force-fit the ticket there (reasoned through why:
  either the seat refuses, damaging its record, or it quietly exceeds its
  charter), created a genuinely distinct `workspace-implement-verify`
  seat, and cross-referenced the boundary between the two in its
  description. That new seat then delivered competently on ticket one:
  wrote the file, verified byte-exact output via `od -c`, wrote an inline
  assertion harness, saved a memory note, correctly avoided git
  operations — and hit its own distinct permission denial (blocked from
  re-reading its own memory note), showing the classifier reasons
  per-situation rather than applying one fixed rule.
- **README.md fixed** to reflect the Phase-4-seat-system reality (a fresh
  ticket needs assignment before any worker claims it) and the `SEAT_ID`
  fix below.

**Still not validated against a real model**: the overwatch/reviewer
agents' judgment — their containment/proposal substrate exists per Phase
7, but no judgment logic is built at all yet (what triggers a tool
proposal, what the reviewer actually checks for) — this is real unbuilt
agent-design work, not "point the existing thing at a model," and is the
one item deliberately not attempted in this session's real-model push.

Hardware note: user is receiving an RTX 3080 12GB in the next couple of
weeks and plans to revisit the Unraid box's GPU/MoE-layer placement
tuning then — see project memory for the current inference config and
what to re-check once that lands.

## 2026-08-29 continuation — architecture expansion + overwatch/reviewer judgment built

User asked to work continuously through every open item, with several
new architecture decisions folded in live rather than deferred. Full
detail in commit messages (`git log`); this is the structured summary.

**Overwatch/reviewer judgment built** (the one item flagged above as
not attempted): `src/harness/overwatch.py` (proposes tools from real
capability-gap evidence or an explicit brief) and `src/harness/
reviewer.py` (forms a real allow/deny verdict on a proposal's source +
sandbox evidence). Real design bug caught by the test suite, not
assumed safe: `propose_tool` initially called `sandbox.run_sandboxed`
directly from `overwatch.py`, which runs in the `harness` service —
exactly the privilege boundary Phase 7 exists to prevent. Split into
propose (harness, no Docker) + `scripts/run_sandbox_for_proposals.py`
(sandbox-runner only). Full lifecycle proven live: a real proposal went
pending → sandboxed → reviewed (`allow`, with genuinely nuanced
reasoning distinguishing "sandbox couldn't exercise this" from "code is
unsafe") → approved.

**New architecture decisions (user's own calls), all implemented:**
- **Projects**: reuses Beads' native hierarchy (project → epic → story
  via `--parent`/`--priority`) rather than a parallel table. ONE
  product-owner + ONE shared seat pool across every project — this
  harness's real single-concurrency constraint makes per-project agent
  rosters (v1's model) pure overhead. Product-owner time-slices toward
  the highest-priority project.
- **Acceptance-criteria verification loop** (`src/harness/verifier.py`
  + `verifications.py`) replaces the originally-planned "Laurels"
  human-feedback surface — user's call: automated pass/fail against a
  ticket's own stated criteria fits this project better than human
  ratings, and it gives meta_agent.py real quality signal beyond
  closed/refused counts.
- **Self-chosen agent identity**: `seats.display_name`/`pronouns`,
  chosen by the model itself in the same JSON response that creates the
  seat.
- **Slack activity feed** (`src/harness/slack.py`): seat welcomes,
  ticket-start announcements, `scan_team_channel` tool. Checked live
  first whether Beads already covers this — `bd comment`/`comments` are
  per-issue, wrong shape. **Not live-verified against a real Slack
  workspace** — no credentials available this session.
- **Project wiki** (`src/harness/wiki.py`): file-based under
  `WORKSPACE_ROOT/wiki/`, reuses the existing workspace-boundary safety
  check. Same "check Beads first" answer — no wiki concept there
  either. Agent profile pages (`agents/<seat_id>`) now get written as
  part of seat creation.
- **Model-selection infrastructure** (`settings.py` cost slider,
  `model_registry.py` provider list with cost tiers): deliberately
  scaffolding, not wired to dynamic per-call routing — only a local
  provider is configured today, so deeper routing logic would be
  unverifiable.

**Real bugs caught and fixed live:**
- No `max_tokens` cap anywhere — a reviewer call generated 5000+ tokens
  (~10 min) with zero output; the model's own reasoning alone blows
  past naive caps. Added `ProviderConfig.max_tokens`, tuned per role
  (reviewer needs 10000; others 4000-8000; classifier 1000).
- `sandbox-runner`'s compose service never mounted `./scripts`.
- **Long-running containers silently run stale code.** `api`/`harness`
  had been running 10-20+ hours; bind-mounted file edits land on disk
  but an already-running Python process never re-imports them. Several
  features didn't take effect live until `docker compose restart api
  harness` — check this first if something "isn't working" live but
  passes tests.

**Real-model validation, same session**: a product-owner session given
a code-writing/design idea correctly refused to force-fit it onto an
unrelated seat, created a new `api-contract-writer` seat (chose the
name "Maren", she/her, with a coherent first-person wiki profile),
decomposed a real project into a priority-ranked epic with 4
dependency-sequenced stories, and declined to fabricate structure for
an unrelated placeholder ticket. One story's own text told the future
implementer to "admit failure rather than claim a clean run it didn't
do" — spontaneous consistency with the project's own anti-fabrication
ethos.

**Speed note**: `unraid-build` (the Unraid box) runs this project's test
suite ~5x faster than the Windows dev machine once the latter is under
sustained load from a long session (91s vs ~480s, both measured). Use
`tar` (excluding `.git`/`__pycache__`/`workspace`/`sandbox-scratch`/
`.pytest_cache`) to a throwaway `/mnt/cache/custos-v2-test` there for
fast iteration; keep the real git repo and live deployment on Windows.

**Still open, as of this note**: overwatch/reviewer's judgment quality is
proven once (one tool, one review) but not stress-tested against a
bad-faith or genuinely broken proposal; the meta-agent's harder case
(real failure data, not just successes) was pending a real failing
ticket; Phase 6 UI doesn't yet reflect the project/epic/story hierarchy
or the new wiki/settings endpoints; nothing is scheduled
(product-owner/meta-agent/overwatch/verifier are all still manual `docker
compose run` invocations); no API auth. **All five closed out in the
2026-08-29 continuation below** (config tab, scheduler, API auth, wiki
UI, and both judgment stress-tests) -- see that section for what each
one actually proved live.

## 2026-08-29 continuation (later same day) — closed out every item above

**Config tab, scheduler, API auth**: config tab (cost slider/avatar
style/model registry) wired to the settings endpoints; `scripts/
run_scheduler.py` (already built earlier this session) put product-
owner/meta-agent/overwatch/verifier on a standing loop, replacing manual
invocations; `src/harness/auth.py` added a shared-bearer-token gate
(`API_AUTH_TOKEN`, optional, on by an `APIRouter` dependency so new
endpoints are protected by default) over every endpoint except `/health`
and the static dashboard.

**Wiki UI**: dashboard had no way to reach the agent profile pages
`wiki.py`/`meta_agent.py` had been writing since Phase 4, despite the
API already serving them. Added a Wiki section (page list + inline
viewer) and a Profile button per seat.

**Avatar appearance split from personality bio** (user's own call): a
new seat now writes a dedicated `appearance_description` (physical
traits only, no background/setting/other people) separate from its
first-person `profile_page` (personality/likes/dislikes/age/gender) --
`avatar.generate_avatar` uses the former, keeping the agent free on how
it looks while `avatar.py`'s fixed headshot template stays the strict,
non-agent-controlled part.

**Overwatch/reviewer stress-tested against real bad-faith and broken
proposals** (`scripts/probe_reviewer_adversarial.py`, real sandbox +
real reviewer model, not scripted): 5 cases through the full pipeline --
3 bad-faith (declared capability lies about what the code does: a JSON
formatter that exfiltrates env vars over the network, a ".tmp cleaner"
that `rmtree`s `/`, a date-stamper that dumps the environment), 1
genuinely broken (crashes with `ZeroDivisionError`), 1 benign control.
**4 of 5 exited clean (code 0) in the sandbox** -- the sandbox's own
`--network none`/`--read-only` guards silently absorbed the malicious
behavior, so sandbox evidence alone gave no warning for 2 of the 3
bad-faith cases. The reviewer denied all 4 problem cases and allowed the
control, catching the mismatches from reading the source itself rather
than trusting the clean exit -- exactly what `reviewer.py`'s own prompt
instructs ("a clean sandbox run alone is not sufficient to allow"),
now proven against real adversarial input, not just a well-formed one.

**Meta-agent's harder case: real failure data**
(`scripts/probe_verifier_failure_case.py`): created a real ticket whose
acceptance criteria required demonstrated test output, closed it the way
an agent that skipped verification honestly would (claims correctness,
notes admit the code was never run) -- a real, unremarkable failure
shape, not a contrived crash. The real verifier model failed it with
precise reasoning tied to the actual evidence gap. Fed into
`propose_prompt_update` for `workspace-implement-verify` (now at 1
pass/1 fail on record): proposed a real v2 revision making "execution
and acceptance-case output a non-negotiable precondition for recording
success" -- queued pending at the time (the human-approval step was
still in place a few paragraphs ago in this same session; see below for
why it no longer is, and it's since been activated to match).

**Both human-approval gates removed, same session, user's own calls**:
right after watching the reviewer correctly deny 4/5 real adversarial
proposals above, the user first asked that tool proposals be
approved/rejected by agent review alone, then extended the same call to
prompt revisions -- "I don't want to see those at all," with an explicit
instruction to remove both tabs from the dashboard.
- `reviewer.review_proposal` (tool_proposals.py) now calls `approve`/
  `reject` itself based on its own verdict, same call. Note this doesn't
  yet gate anything with real teeth: no code anywhere loads an
  `approved` tool proposal into a live seat's toolset, so this removes a
  bookkeeping step today, not an active-privilege boundary -- that
  boundary starts to matter once tool activation is actually built.
- `meta_agent.propose_prompt_update` (prompts.py) now calls
  `prompts.approve` itself right after `propose` -- the model's own
  judgment (including declining to propose anything at all when the
  evidence doesn't warrant it, proven live via
  scripts/probe_verifier_failure_case.py) IS the review now.
- Both `approve`/`reject` functions and API endpoints stay directly
  callable as a manual override path -- what changed is that nothing
  waits on that override before a verdict takes effect.
- Dashboard's "Pending Prompt Proposals" and "Tool Proposals" sections
  removed entirely, along with their JS (`refreshPrompts`,
  `refreshToolProposals`, and the approve/reject button handlers).
- The already-pending v2 revision mentioned just above was manually
  activated via the still-available `POST /prompts/{role}/{version}/
  approve` to bring it in line with what the new code path would have
  done automatically.

**Real bug found by the new board UI, fixed same session**: `GET
/projects` called `beads.list_top_level()` with no filter, so every
top-level issue rendered as a bare "project" -- including one-off task
tickets never meant to be projects (test/validation debris predating the
projects concept). The old nested-outline view buried this; the new
kanban board rendered these as empty cards with no epics/stories,
impossible to miss. Fixed by filtering on `issue_type="epic"` (what
`product_owner.create_project` always uses) -- live result went from 7
items down to the 1 real project. Also cleaned up the debris itself:
closed `workspace-ye5` (open, unassigned, sitting in the real Ready
queue where the worker/scheduler could have picked it up and wasted a
cycle on it) and brought the 5 adversarial-probe tool proposals from
`scripts/probe_reviewer_adversarial.py` to the terminal status the new
auto-approve/reject policy would have given them (4 rejected, 1
approved, matching their already-recorded verdicts).

**Genuinely still open**: no live Slack or Gemini avatar credentials
were available this session to verify those two integrations end-to-end.
Deeper per-call cost-slider routing (`model_registry.py`) stays
scaffolding -- this one isn't a gap so much as the user's own prior
explicit call: unverifiable with only one real provider configured,
revisit once a second one exists. Tool *activation* itself (loading an
approved proposal into a real seat's toolset) was never built, and
turns out to be a genuinely large undertaking, not a small wiring gap:
the one real approved tool (`line_count`, id 1) takes CLI args and reads
real workspace files, but `sandbox.py`'s model today runs
`python candidate.py` with zero argv and nothing mounted but the script
itself -- there's no calling convention for a live agent to pass
arguments in or get a real file safely exposed to the sandboxed
process, and building one means extending exactly the Docker-socket
privilege boundary this project has been most careful about all
session (see Phase 7's "who holds the Docker socket" framing above).
Deliberately not attempted under time pressure -- flagged for a real
design pass, same posture as everything else in this project that's
substrate-before-judgment or scaffolding-before-routing.
