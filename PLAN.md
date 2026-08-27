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

**Still open, needs a real cross-ticket concept before it makes sense:**
- Persistent identity ("seat") that spans *multiple* tickets — chosen
  name/pronouns, accumulated history. Today's handoff notes live on the
  Beads issue itself (one ticket = one thread = one place for notes to
  live), which is a reasonable v1 but isn't yet "the same agent worked
  this seat across 40 different tickets and remembers all of them."
  Self-chosen names specifically need a real model call to do properly
  (asking the model to pick), not just an assigned default — worth doing
  once a model is reachable rather than faking it now.
- "Laurels": user feedback on completed work surfaced back to a seat.
  Blocked on there being a UI (Phase 6) to actually collect that feedback
  from a human in the first place.

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

**Still open (genuinely a UI, not backend, task):**
- Rebuild roadmap/board/steering concepts as an actual frontend. v1's
  precedent (`public/admin.html` — vanilla HTML/JS, no build step) is the
  natural default to follow rather than introducing a new frontend
  framework decision.
- Queue depth + estimated wait per ticket in the UI — matters once
  inference is genuinely slow, so the UI sets expectations instead of
  looking stuck. The API doesn't expose an ETA yet (nothing to estimate
  from without real inference timing data).
- Agent seats/personalities/history surface — blocked on Phase 4's
  still-open cross-ticket seat identity.
- DevOps-equivalent tab for model/provider config and concurrency limits
  — today those are env vars (`.env.example`), no admin UI to edit them
  live like v1 had.

## Open questions (not blocking Phase 1, but will block later phases)

- **Hardware specs** — VRAM/GPU on the Unraid box, once you're back. Drives
  actual model choice (Qwen3.8-27B class vs. smaller), quantization, and
  real context length. Phase 1 doesn't need this — it targets whatever runs
  on Docker Desktop today.
- **Real-model verification** — no Ollama/local model was reachable in this
  dev environment, so everything above (durability, permission gate) is
  proven against real Postgres/Beads but a *scripted fake* model, not real
  inference. First thing to run once a model is reachable again: the manual
  kill/resume demo in README.md with an actual model, plus a real run of
  the permission classifier to see how it behaves against genuine tool-call
  arguments rather than a scripted verdict.
- ~~Beads: adopt real Beads vs. build a lighter homegrown version~~ —
  resolved, adopted from Phase 1.
- ~~Qdrant's fate relative to Beads~~ — resolved 2026-08-27, dropped (see
  Phase 3).

## Immediate next step

Start Phase 1: scaffold the LangGraph-based harness in this repo
(`custos-v2`), Docker Compose for the checkpointer store, port the
provider abstraction, build the minimal tool layer + permission gate.
