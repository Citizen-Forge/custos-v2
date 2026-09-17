# Custos v2 — session handoff

You are picking up an in-progress engineering/operations session on the **Custos v2**
project: a self-hosted, agentic software-development lifecycle engine running on an
Unraid box. Read this whole file before acting. It is written to be tool-agnostic; if
your agent has its own project-instructions file (`AGENTS.md`, a system prompt, a
rules file), copy the "Operating rules" section into it.

---

## Prime prompt (paste this first)

> You are the engineer/operator for the Custos v2 harness on the Unraid box
> `unraid-build`. Read `HANDOFF.md` in full before doing anything. Work from
> `/mnt/user/appdata/custos-v2` on the box. Verify claims with the test suite
> (`docker exec custos-v2-harness-1 python -m pytest -q`) and with `bd`/the API;
> never assume a ticket's state from memory. Make small, tested changes, commit in
> the repo's explanatory style, push, sync the box, and restart only the services a
> change affects. Never print or commit secrets. Ask before destructive actions.

---

## Access & environment

- **SSH:** `ssh unraid-build` (host `192.168.100.231`, user `root`).
- **Deployment (source of truth):** `/mnt/user/appdata/custos-v2` — a git checkout, bind-mounted into
  the containers. This is the real, actively-developed tree; the GitHub repo can be behind it.
- **Host OS quirk:** Unraid's `/` is tmpfs. Persistent things live on `/mnt/user/...` or
  `/boot/config/...`.
- **Be careful with shell quoting over SSH from Windows/PowerShell:** `$(...)`, parentheses, and
  nested quotes get mangled. Prefer writing a small `.sh` locally, `scp` it, run it, delete it.
  (`scp` and `ssh` work; PowerShell is the problem, not the box.)

### Containers (Docker names)

| Name | Role |
|---|---|
| `custos-v2-harness-1` | dispatcher + agent runs (`python -m harness.dispatcher`) |
| `custos-v2-scheduler-1` | scheduled jobs: product_owner, overwatch, meta_agent, verifier, escalations, progress |
| `custos-v2-api-1` | FastAPI JSON API + dashboard (`public/index.html`) |
| `custos-v2-postgres-1` | Postgres (LangGraph checkpoints, prompts/seats/verifications/self-mod tables) |
| `custos-v2-test-postgres-1` | separate test DB |
| `custos-reasonix-sidecar` | the conversational sidecar: Reasonix web UI + `custos` MCP on :4096 (replaced opencode 2026-09-15) |
| `custos-bridge` | the mobile app's backend: HTTP+SSE over ACP to `reasonix acp`, on :8788 |
| `custos-classifier` | Ollama serving the tool-permission classifier |
| `qwen3-coder-server` | **stopped** (was the old local model; no longer needed) |

### Addresses

- API from the **host**: `http://172.31.0.100:8000` (the bridge IP; the host cannot reach its own
  macvlan child).
- API from **other LAN devices**: `http://192.168.250.238:8000` (macvlan `br0`).
- Classifier from the harness: `http://192.168.250.239:11434` (Ollama, `qwen2.5:7b-instruct`).

---

## Repo & deploy workflow

- Local clone: `C:\Users\tall_\Documents\custos` (origin `git@github-custos:Citizen-Forge/custos-v2.git`
  / `https://github.com/Citizen-Forge/custos-v2.git`).
- **Windows clone uses `core.autocrlf=true`.** Commit locally (git normalises to LF), push, then on
  the box:
  ```
  cd /mnt/user/appdata/custos-v2 && git fetch origin -q && git reset --hard origin/main
  ```
  **Do not** edit files on Windows and `scp` them onto the box and commit there — the CRLF makes git
  see every line changed.
- **Tests:** `docker exec custos-v2-harness-1 python -m pytest -q` (uses a throwaway DB + temp
  workspace; safe). Targeted: append paths, e.g. `tests/test_dispatcher.py`.
- **Restart only what a change affects:** `docker restart custos-v2-harness-1` (dispatcher/agent code)
  and/or `docker restart custos-v2-scheduler-1` (scheduled jobs). The dashboard
  (`public/index.html`) is served live; no restart needed.
  **Restart BOTH when you change anything `reopen()`/`verifier.py` touches.** The verifier and the
  escalation handler run in the *scheduler*, so a fix landed only in the harness leaves the scheduler
  calling the old code — that is exactly how a stale `reopen()` kept re-creating the claim wedge
  described below after the harness had already been fixed (54 more dispatch failures, 2026-09-15).
- **Commit style:** imperative title, then a body explaining the live failure found and why the fix is
  shaped that way. Match the existing log.

---

## Architecture mental model

- **Work items live in Beads (`bd`)**, a dotted hierarchy: project (`workspace-9jg`) → epic
  (`workspace-9jg.3`) → story (`workspace-9jg.3.5`). The DB is `/workspace/.beads` inside the
  containers. `status` is `open` / `in_progress` / `closed`; **`human` is a label**, not a status.
- **Dispatcher** (`src/harness/dispatcher.py`): capacity-bounded (currently `MAX_RUNNING_AGENTS=3`).
  Picks assigned work in **roadmap order** — `(project priority → epic priority → story priority →
  natural id)` via a cached `bd list --all`. Orphans (`in_progress`) resume first. Skips human-labelled
  and held projects.
- **Worker** (`src/harness/worker.py`): runs one ticket's LangGraph thread. On success calls
  `commit_all` (commit subject `"<ticket-id>: <summary>"`), records `work_commit`, closes.
- **Agents** call `complete_ticket` / `refuse_ticket` / `decline_ticket`. The graph has a one-turn
  **completion nudge** (`completion_gate=True` in `worker.build_seat_runtime`) — a run that stops
  without a terminal tool call gets one extra turn before the worker treats it as unclaimed.
- **Verifier** (`src/harness/verifier.py`): judges closed tickets against acceptance criteria. A fail
  **requeues with the finding** (bounded by `MAX_VERIFIER_REWORKS`, default 2); after that it flags
  `human`. It finds the ticket's diff via `workspaces.diff_for_ticket` — preferring commits whose
  subject is `<ticket-id>: …`, falling back to `work_commit`.
- **Product-owner** (`src/harness/product_owner.py`): triage/dispatch, idea decomposition, and — added
  this session — an **escalation handler** (`src/harness/escalations.py`) over the `human` queue.
- **Escalations:** `human`-labelled, non-closed tickets. Scheduled job `escalations` (and
  `scripts/run_escalations.py`) has the product-owner fix the specification and requeue, reassign, or
  resolve. Bounded by `MAX_ESCALATION_ATTEMPTS` (default 2), after which it's left for a person.
- **Dashboard** (`public/index.html`): Home tab has project tabs that scope Roadmap + Board. Blocked
  column counts `human` **and not closed** tickets only.

---

## Current state (snapshot at handoff)

- Tickets: **50 open, 7 in_progress, 125 closed.** The `human` label is independent of status; ~25
  carry it and most of those are `closed` (verifier-exhausted), which the dashboard deliberately does
  not count as Blocked.
- Tickets flow: dispatch is healthy and draining epic 7 as of this snapshot.
- Classifier: GPU (RTX 3070 Ti), `100% GPU`, ~0.7s/call, `CLASSIFIER_MODEL_CONCURRENCY=3`.
- Sidecar: **Reasonix** on :4096 (`http://192.168.100.231:4096`) — opencode retired. See "Sidecar".
- Load average ~3 (was ~15 before the Qwen server was shut down and the classifier moved to GPU).
- Full test suite green at last run: **432 passed, 5 skipped**.

---

## What was done this session (2026-09-15 — dispatcher deadlock, then the Reasonix sidecar)

1. **Fixed the dispatcher deadlock.** `beads.reopen()` set a ticket back to `open` but never released
   bd's `assignee`, and `bd --claim` is idempotent only for the actor already holding the ticket — so a
   reopened ticket whose seat had since changed could *never* be claimed by its new seat. The
   dispatcher's selector returns the same highest-priority ticket every cycle, so instead of falling
   through to the rest of the queue it retried that one ticket forever: dispatch was fully wedged for
   ~16h with 83 open tickets and three free agent slots. Repaired the two stranded tickets
   (`1.3`, `4.2`) and restarted the harness; **both containers** need the restart (see the warning in
   "Repo & deploy workflow"). Commit `7beec0c`, with a regression test that fails against the old
   `reopen()`. Full suite: 432 passed, 5 skipped.
2. **Replaced the opencode sidecar with Reasonix** (`custos-reasonix-sidecar`, :4096) — see "Sidecar".
3. **Built the mobile app's backend** (`custos-bridge`, :8788) — a FastAPI service that speaks the
   documented Agent Client Protocol to `reasonix acp` and re-exposes it as HTTP+SSE, plus thin mirrors
   of the Custos API. Verified end to end (session → message → SSE → the agent answering
   `BRIDGE-OK 2` from a live `mcp__custos__list_tickets` call). See "Sidecar" and
   `custos-sidecar/bridge/README.md`. The Android client itself is not written yet.
4. **Found (but did not fix) a second trigger for the same wedge:** an `in_progress` orphan whose
   `assigned_seat` differs from its `assignee` makes `start_agent` fail the claim every cycle too. The
   dispatcher still has no fall-through for a ticket it cannot claim, so *any* one such ticket stalls
   the whole queue. See "Open issues".
5. **Worked on "blocked tickets building up again" — NOT solved. Be sceptical of the two fixes.**
   Two real defects were found and fixed (`ef37168`, then `bf9d6b8`, both on `main`):
   - **Dangling tool calls.** A run that dies between the model returning `tool_calls` and the tool
     results being recorded checkpoints an assistant message whose calls nothing answers, and the
     provider rejects the whole request for that. Ten threads had the shape. `graph.py` now answers
     the missing calls in place in `call_model`. Verified directly against a real thread.
   - **Unbounded history.** Long threads had grown to 600–830 KB of checkpointed messages (4.2 832 KB,
     1.3 800 KB, 3.5 715 KB, 6.1 661 KB) while a 195 KB one ran fine. `graph.py` now drops the oldest
     messages until it fits, caps individual tool results, and forces the retained window to start on
     a group boundary. Cuts 4.2 from 642 KB to 183 KB. Cannot regress: a history that fits is
     returned unchanged.

   **But the live 400 was never reproduced, so neither fix is known to address it.** The live error is
   `400 ... insufficient tool messages following tool_calls message`, and against it:
   - the failing threads' histories pass a positional provider-rule walk in the wire form
     `convert_to_openai_messages` produces — raw *and* after the bound;
   - 4.2 still 400s at 183 KB, and the same list both passed and failed across attempts, so it is not
     simply size;
   - isolated `ChatOpenAI` reproductions of these threads fail with a **different** error —
     `reasoning_content in the thinking mode must be passed back` — so they are not faithful to the
     worker's model configuration.

   **Neither fix is proven to address the live 400, but the queue has since drained.** By the end of
   this session: 400s down to ~2 in 40 minutes, and closed tickets up from 103 to 125. Three of the
   stranded tickets (`4.2`, `5.2`, `6.1`) went on to run clean once the verifier's rework path gave
   them a **fresh thread** — so a thread reset is the practical cure when this shape appears. (Note
   the probes above ran against threads that were subsequently reset; re-check that a thread still
   exists before trusting an earlier measurement.)

6. **Fixed why the human queue was really filling: the verifier was judging the wrong diff.** Two
   independent attribution faults, both found by asking why so many parked tickets had fail reasons
   like *"the diff under review is not the probe ticket's work at all"*. All 17 parked tickets were
   verifier-exhausted, and nearly all of their reasons were this, not real shortcomings.
   - **Range diff.** `diff_for_ticket` diffed the range from a ticket's FIRST commit to its LAST. The
     workspace is shared by every seat, so other tickets commit in between: `9jg.7.3`'s own work is
     2 files, but the range spanned **137 other commits and 206 files / 66,418 insertions**. Now each
     of the ticket's commits is diffed separately (`4071a17`).
   - **The harness's own store was in every commit.** `.beads/` lives inside the project workspace,
     `bd init` does not ignore it and it ends up tracked, so `git add -A` swept `interactions.jsonl`
     into ticket diffs — `9jg.2.3` failed for exactly that reason. Now ignored AND unstaged
     (`1ff3670`). Verified live: the newest ticket commits carry 0 `.beads` files.
   - **Re-verified 3 of the 17 on the corrected diff (`7.3`, `7.4`, `2.3`): 0 flipped, and that is the
     fix working, not failing.** `7.4`'s new reason is a substantive review of the right code
     (*"`findBestTargetInCone` never reads `seekerConeAngle`"*) and `2.3`'s is *"touches only
     `.beads/interactions.jsonl` and two one-line scaffolding"* — both true. Previously they were
     failed on other tickets' work. So the remaining parked tickets mostly need **specification or
     rework decisions**, which is the escalation role's job, not an attribution fix.
   - Rows for those 3 were backed up to a `verifications_backup_reverify` table before clearing.

---

## What was done in the previous session

1. **Corrected the local repo** to `custos-v2`; pushed the deployment's unpushed commits to GitHub and
   pulled them locally.
2. **`shell_exec` timeout fix:** was a hardcoded 120s that killed whole runs; now `SHELL_TIMEOUT`
   (default 600), the process group is killed on expiry, and the timeout comes back as a tool result.
3. **Concurrency raised to 3**, plus two real bugs found: compose's `environment:` shadowed `env_file`
   (it was running 1), and `next_assigned_ticket` re-selected the ticket it had just claimed
   (stalled at one agent).
4. **Dashboard:** project tabs now scope the whole Home view (Roadmap + Board); closed tickets are
   never shown as Blocked.
5. **Verifier rework loop:** a fail now reopens the ticket with the verifier's finding, clears the old
   completion claim, resets the LangGraph thread, and puts the finding in the next attempt's prompt —
   bounded by `MAX_VERIFIER_REWORKS`. Verification idempotency is per-`work_commit`.
6. **Completion nudge:** one extra turn when a run stops without `complete_ticket`.
7. **Roadmap-order dispatch** (above), with the product-owner prompt updated to drain one epic at a
   time.
8. **Verifier diff fallback** (`workspaces.diff_for_ticket`): fixes a class of false fails where
   `work_commit` was empty or pointed at another seat's commit (`1.6`, `13.1`, `13.4`, `1.5`).
9. **Escalation handler** (product-owner role + scheduler job + `run_escalations.py`).
10. **Created epic `workspace-9jg.14` "Human interface and rendering"** with stories `14.1` Rendering
    technology and app shell, `14.2` Shared HUD/plot style, `14.3` Ship exterior renderer, `14.4`
    Module visual assembly. Wired existing UI stories (`2.5, 3.4, 3.6, 5.7, 6.6, 7.5, 9.5, 11.3`)
    to depend on `14.1`. A `render-presentation-ts` seat was auto-created for it.
11. **Classifier/GPU + ops:** stopped `qwen3-coder-server`; recreated `custos-classifier` on the GPU
    (`--runtime=nvidia`, `NVIDIA_VISIBLE_DEVICES=<3070Ti UUID>`, `OLLAMA_NUM_PARALLEL=3`); set
    `CLASSIFIER_MODEL_CONCURRENCY=3` in `.env`. Latency 2–13s → ~0.7s; CPU ~1200% → ~0%.
12. **open-code sidecar** (see below).

---

## Open issues / next steps

1. **The intermittent 400 — much rarer now, but not root-caused.**
   `400 ... insufficient tool messages following tool_calls message`. At its worst this fired every
   few seconds and parked tickets within minutes; after the fixes above and the recreations, it is
   down to roughly **2 in 40 minutes** and the tickets it had stranded have closed (`9jg.6.1` closed
   on a fresh thread; the last 400 was 11:57). So it is no longer blocking, but the trigger is still
   not isolated and it can come back.

   What is now established, and the wrong turns, so they are not repeated:
   - **Use `build_chat_model`/`ProviderConfig` to reproduce, never a hand-built `ChatOpenAI`.** The
     hand-built client omits the provider's `extra_body={"thinking":{"type":"disabled"}}`, so it
     fails with a *different* error (`reasoning_content in the thinking mode must be passed back`)
     and sends you after the wrong fault. That mistake cost two false conclusions here.
   - **It is deterministic and shape-related, not size.** With the faithful client, every history
     budget failed — 400 KB down to 60 KB (71 messages) — so trimming does not cure it.
   - **A valid-looking group can still tip a passing prefix into failure.** Bisecting the 642 KB
     `4.2` history with the faithful client, the prefix of 113 messages passed and 115 failed, and
     the group added was an ordinary `assistant(shell_exec) -> tool` pair that matches in the wire
     form. So the provider is enforcing something stricter than "every tool_call id is answered by a
     following tool message" — the positional walk the harness assumes is not the whole rule.
   - **Thread resets cure the affected tickets.** `4.2`, `5.2` and `6.1` all resumed failing on their
     old checkpoint and then ran clean once the verifier's rework path (`_reset_thread`) gave them a
     fresh thread. That is the practical escape hatch if it returns.
   - Note the probes above ran against threads that were *subsequently reset*, so re-check the thread
     still exists before trusting any earlier measurement.

2. **`LOCAL_FALLBACK_*` was removed from `.env`** (2026-09-15, operator's decision). The host it
   pointed at, `192.168.250.235`, accepted no connections on any port — it is the stopped qwen
   server — so every DeepSeek hiccup fell back to a dead endpoint and died with
   `OpenAIConnectionError`. Removed rather than repointed: failures now fail fast and cleanly.
   Backup at `/mnt/user/appdata/custos-v2/.env.bak-pre-fallback-removal` (0600). Remember that
   `docker restart` does **not** re-read `env_file` — the containers had to be recreated with
   `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate harness
   scheduler`.
3. **One unclaimable ticket still stalls the whole queue.** The dispatcher's selector returns the same
   highest-priority ticket every cycle, and `start_agent` just returns `False` when the claim throws —
   so it never falls through to the rest of the board. `reopen()` no longer *creates* the bad state
   (fixed 2026-09-15), and the scheduler restart removed the last source of it, but a second trigger
   exists in the same shape: an `in_progress` orphan whose `assigned_seat` differs from its `assignee`
   (e.g. the product-owner reassigns a ticket while a seat still holds it). The real fix is
   dispatcher-side: skip (and log) a ticket it cannot claim, or release a stale claim on an `open`
   ticket, so no single record can starve the other 80. Deliberately left as a separate change.
4. **"Ended with no completion claim" is a recurring park.** Several tickets (e.g. `3.5`) are flagged
   for stopping without calling `complete_ticket`, even after the completion nudge. Worth watching
   whether that is model flakiness or a prompt that under-sells the terminal call.
5. **The verifier-exhausted `human`+`closed` tickets now need a real decision.** With attribution
   fixed (see "what was done", item 6), the remaining parked tickets are failing on their own work —
   `7.4` on a genuine algorithm gap, `2.3` on commits that contain no product code at all. So the
   lever is now specification and rework, not attribution: either let the escalation role reconsider
   closed+human tickets whose last verdict is a fail (it currently skips closed by design), or rework
   them by hand. Re-verifying alone will not flip them — measured 0 of 3.
6. **Ticket attribution is still imperfect.** The harness commits `<ticket-id>: …`, but agents also
   self-commit and work lands in ranges; several tickets were judged on the wrong commit. The fallback
   helps but the worker could record `work_commit` more reliably (e.g. always record the range since
   `work_base`).
7. **Model reliability:** DeepSeek (`deepseek-flash`, a reasoning model) occasionally rejects large
   tool-call payloads (a ~35KB single `write_file`/`shell_exec` argument → HTTP 400 → local fallback →
   malformed tool JSON → 500 → run crash). Mitigation so far is prompt-level (write files in small
   pieces); a code guard (catch model/tool-call errors, return them as tool results, split large
   writes) would be better.
8. **`custos-classifier` has no Unraid template or compose file** — it was created by hand. Recreating
   it from the UI would lose GPU + `OLLAMA_NUM_PARALLEL`. A small compose file would make it
   reproducible. (The new Reasonix sidecar and its bridge both have one, so they are the model to copy.)
9. **Roadmap-order nuance:** in-flight orphans resume before ordering applies, so a later-epic ticket
   already in `in_progress` will still run. That's deliberate (don't discard partial work).

---

## Sidecar (the conversational interface)

**Reasonix** — a container, `custos-reasonix-sidecar`, defined in
`/mnt/user/appdata/custos-sidecar/reasonix/` (Dockerfile + compose + `config/reasonix.toml` + a
README that explains the two non-obvious container settings). It replaced opencode on 2026-09-15.

- **Web UI:** `http://192.168.100.231:4096`. Password auth; the password is unchanged from opencode and
  still lives in `/mnt/user/appdata/custos-sidecar/opencode/server-password` (0600 — read it there,
  don't paste it around). Only a bcrypt hash is in the config.
- **Role prompt:** `system_prompt` in `config/reasonix.toml` (interface-agent + harness-engineer),
  adapted from the old opencode `AGENTS.md`. Its MCP tool names are `mcp__custos__<tool>`.
- **Autostart:** a `docker compose up -d` retry loop in `/boot/config/go`; the container's own
  `restart: unless-stopped` does the real work. The opencode watchdog block was removed from that file.
- **Gotchas that cost real time here** (all in the sidecar README, all verified live):
  - Reasonix reads provider credentials from *its own* `$HOME/.reasonix/.env`, not the process
    environment, so `entrypoint.sh` copies the key there for it.
  - It jails every shell command with bubblewrap; inside a container that needs `SYS_ADMIN` **and**
    `seccomp:unconfined`, or it refuses to run *any* shell command at all.
  - The config template keeps a `REPLACE_WITH_BCRYPT_HASH` placeholder; re-run
    `set-password.sh` after copying a fresh config over the live one, then restart — otherwise
    every login 401s.
- **opencode** is retired but its directory is still on disk (`/mnt/user/appdata/custos-sidecar/opencode/`);
  nothing starts it. The older **Claude Code** sidecar + watchdog also still live under
  `/mnt/user/appdata/custos-sidecar/`.
- **`custos-bridge` (:8788)** is the *app's* backend, not the operator's console: a FastAPI service in
  `/mnt/user/appdata/custos-sidecar/bridge/` that drives `reasonix acp` over the documented Agent
  Client Protocol and exposes HTTP+SSE, plus thin mirrors of the Custos API. Auth is a bearer token in
  `bridge/state/bridge-token` (0600, `./set-token.sh`). Built FROM the sidecar image. Its README has
  the endpoint table and the two traps (credentials come from Reasonix's own `.env`; the ACP client
  must not re-enter its start lock). `e2e_check.py` is the acceptance test — it should print
  `BRIDGE-OK <n>`. **The Android client is not written yet**; the bridge is the finished half.

---

## Operational runbook

```bash
# tests
docker exec custos-v2-harness-1 python -m pytest -q

# restart after a code change
docker restart custos-v2-harness-1          # dispatcher/agent code
docker restart custos-v2-scheduler-1        # scheduled jobs

# tickets (read)
docker exec -w /workspace custos-v2-api-1 bd list --label human --all --long --json --limit 0
docker exec -w /workspace custos-v2-api-1 bd show <id> --json
docker exec -w /workspace custos-v2-api-1 bd list --status=in_progress --json

# API (from the host)
curl -s http://172.31.0.100:8000/projects
curl -s "http://172.31.0.100:8000/tickets?status=human"

# requeue a parked ticket with a directive
docker exec custos-v2-harness-1 python scripts/requeue_ticket.py <id> --reason "..."

# run the escalation handler on demand
docker exec custos-v2-scheduler-1 python scripts/run_escalations.py

# DB access
docker exec custos-v2-postgres-1 psql -U custos -d custos_harness -c "select ..."
```

---

## Operating rules

- **Verify, don't assume.** Read `bd`/the API for ticket state; run the tests for code changes.
- **Small, reversible, tested changes.** Match the repo's conventional style and explanatory commit
  bodies.
- **Never print or commit secrets** — `.env` (DeepSeek key), `~/.ssh`, the sidecar password, Slack/
  Gemini tokens. Do not modify `.env` unless asked.
- **Ask before destructive actions:** bulk `bd` changes, `docker compose down`, force pushes, anything
  outside `custos-v2` / `custos-sidecar`.
- Restarting `custos-v2-harness-1` kills in-flight agent runs; that's safe (threads resume from
  checkpoints) but say so first.
- **Don't over-block the board.** A `human` label means "parked for a person"; the escalation role
  exists precisely to keep that queue small. Prefer fixing a specification and requeueing over
  parking.
