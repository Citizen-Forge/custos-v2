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
- Tool layer: file read/write, shell exec — minimal set. Port the
  classifier-based permission gating concept from v1
  (`claude-gateway/src/server` PreToolUse hook) into a direct tool-call
  boundary inside the graph, since there's no more Claude Code hook
  contract to hang off of.
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
  using a local model through the queue.

### Phase 2 — Multi-provider routing + fallback
- Priority/fallback chains on top of the Phase 1 provider abstraction
  (local model → free-tier frontier → paid frontier), reusing v1's
  cooldown-on-rate-limit logic.
- Role-to-model pinning (product-owner → frontier, workers → local),
  reusing v1's `custos:<provider>/<model>` alias pattern.
- Concurrency caps enforced *per backend*, not globally — local endpoints
  default to 1 concurrent call, frontier calls aren't artificially limited
  by the same constraint.

### Phase 3 — Context & cross-session memory
- Small context windows mean blind truncation is out. Evaluate **adopting
  real Beads** (Go + Dolt, has an MCP server) rather than reinventing it —
  it already does dependency-graph issue tracking *and* "memory decay"
  semantic compaction of closed work *and* `bd prime`-style context
  injection, which is exactly this problem. A LangGraph tool node can call
  Beads' MCP server directly.
- Decide what Beads replaces vs. complements: it could directly back v1's
  roadmap/board data model (nice fit — you said you like that part of the
  UI) rather than being a separate system next to it.
- Decide the fate of v1's Qdrant semantic-memory layer: fold into Beads,
  keep as a third loose-recall layer, or drop. Open question, not resolved
  here.

### Phase 4 — Agent personality & welfare behaviors
- Per-agent persistent identity ("seat"): chosen name/pronouns, accumulated
  history distinct from a single session.
- Handoff notes: an agent writes a closure note before a session ends
  (voluntarily, or forced by a bounded workday); the next session for that
  seat receives it as context.
- Refuse-work as a real, first-class response type — surfaced in the UI,
  not silently retried by the orchestrator.
- Bounded workdays: configurable turn/session budget per seat before a
  forced handoff.
- "Laurels": user feedback on completed work gets surfaced back to the
  specific seat that did it and stored as part of that seat's history.

### Phase 5 — Meta-agent (agent-improves-agents)
- Frontier-backed agent reads completed work + outcomes (including Laurels)
  across seats, proposes system-prompt diffs per seat/role.
- Human-approval gate before a prompt change takes effect — matches v1's
  existing "autonomy off by default for every role except product-owner"
  pattern.
- Track a simple before/after signal per seat (completion rate, rework
  rate, refusal rate) to judge whether a proposed change actually helped.

### Phase 6 — UI/product surface
- Rebuild roadmap/board/steering concepts on the new backend.
- New surfaces the old UI didn't need: queue depth + estimated wait per
  ticket (important once inference is genuinely slow — the UI has to set
  expectations instead of looking stuck), agent seats/personalities/history,
  DevOps-equivalent tab for model/provider config and concurrency limits.

## Open questions (not blocking Phase 1, but will block later phases)

- **Hardware specs** — VRAM/GPU on the Unraid box, once you're back. Drives
  actual model choice (Qwen3.8-27B class vs. smaller), quantization, and
  real context length. Phase 1 doesn't need this — it targets whatever runs
  on Docker Desktop today.
- **Beads: adopt real Beads vs. build a lighter homegrown version** —
  leaning adopt (Phase 3), not yet confirmed.
- **Qdrant's fate** relative to Beads (Phase 3).

## Immediate next step

Start Phase 1: scaffold the LangGraph-based harness in this repo
(`custos-v2`), Docker Compose for the checkpointer store, port the
provider abstraction, build the minimal tool layer + permission gate.
