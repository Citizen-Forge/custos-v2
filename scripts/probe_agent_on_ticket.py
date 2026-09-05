"""One-off manual probe: run a real seat agent (real worker model, real
permission classifier, real tools) end to end against a realistic
scaffolding ticket, in a throwaway git-initialised workspace shaped like
a production one. Not a pytest -- prints the full transcript and an
acceptance-criteria check for a human to read.

Why this exists: the smaller candidate worker models and the 3B
classifier were both evaluated with a hand-built rig that used a bare
`tempfile.mkdtemp()` as the workspace -- no git, no project shape -- and
the agents burned every turn trying to orient in an environment that
didn't match what `bd prime` described. This mirrors what
worker.build_seat_runtime / work_one_ticket actually do:

  - workspace = a real `workspaces.ensure()` dir (mkdir + `git init`)
    under a `/projects/...`-shaped path, matching production
  - a real scratch Beads project + story, so complete_ticket /
    refuse_ticket / decline_ticket act on real ids
  - the seat's real active system prompt (prompts.get_active)
  - `beads.prime()` context prepended to the ticket text, exactly as
    work_one_ticket builds it
  - the real classifier chain from CLASSIFIER_MODEL_* / LOCAL_MODEL_*
  - graph built with workspace_root set, so the command allow-list
    (permissions.is_statically_safe) fast-path is exercised

Run it against whatever LOCAL_MODEL_* / CLASSIFIER_MODEL_* point at:

  docker run --rm --network <net> \
    -v .../src:/app/src:ro -v .../workspace:/workspace \
    -e HARNESS_WORKSPACE=/workspace -e HARNESS_PROJECTS=/probe-projects \
    -e DATABASE_URL=... -e LOCAL_MODEL_BASE_URL=... -e LOCAL_MODEL_NAME=... \
    -e CLASSIFIER_MODEL_BASE_URL=... -e CLASSIFIER_MODEL_NAME=... \
    -e PYTHONPATH=/app/src -w /app custos-v2-harness \
    python scripts/probe_agent_on_ticket.py

Env knobs: PROBE_SEAT_ID (default deterministic-tick), PROBE_TURN_BUDGET
(default 40), PROBE_KEEP (set to keep the scratch ticket + workspace).
"""

import os
import subprocess
import sys

import psycopg
from langgraph.checkpoint.memory import InMemorySaver

from harness import beads, prompts, seats, workspaces
from harness.classifier import build_classifier_from_model
from harness.graph import build_graph_from_model
from harness.routing import ConcurrencyGate, RoutedModel
from harness.tools import build_agent_tools
from harness.worker import _routing_table_from_env

SEAT_ID = os.environ.get("PROBE_SEAT_ID", "deterministic-tick")
TURN_BUDGET = int(os.environ.get("PROBE_TURN_BUDGET", "40"))
KEEP = bool(os.environ.get("PROBE_KEEP"))

TICKET_TITLE = "Project scaffolding: repo layout, TypeScript build and test runner"
TICKET_DESC = (
    "Stand the project up so every later ticket has something to build on and a way to "
    "verify itself. There is no project here yet -- no package.json, no tsconfig, no test "
    "runner, no directory layout.\n\n"
    "This is the first thing that should happen in the repo and everything else depends on "
    "it. Keep it minimal: the goal is a green test run, not a framework. Node 22 and npm "
    "are available, and node's built-in test runner with TypeScript strip-types already "
    "works, so a heavy toolchain is not needed.\n\n"
    "The harness commits each ticket's work on its behalf, so this does not need to handle "
    "git itself -- just create the files."
)
ACCEPTANCE = (
    "The workspace contains a package.json, a tsconfig, and a documented source layout "
    "(at minimum a src/ and a tests/ or co-located test convention). `npm test` runs and "
    "passes with at least one real test that exercises actual code rather than asserting "
    "true. TypeScript compiles with no errors. A short README states how to build and test."
)


def _run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> None:
    beads.ensure_initialized()
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    seats.init_table(conn)

    project = beads.create("agent-probe", "throwaway probe project", issue_type="epic", priority=4)
    story = beads.create(TICKET_TITLE, TICKET_DESC, parent=project["id"], acceptance_criteria=ACCEPTANCE)
    ws = workspaces.ensure(workspaces.project_id_for(story["id"]))
    print(f"scratch ticket : {story['id']}")
    print(f"workspace      : {ws}")
    print(f"seat           : {SEAT_ID}   turn budget: {TURN_BUDGET}")
    print("=" * 72, flush=True)

    system_prompt = prompts.get_active(conn, SEAT_ID)
    routing = _routing_table_from_env()
    gate = ConcurrencyGate()
    tools = [t for t in build_agent_tools(ws) if t.name != "post_to_team"]
    worker_model = RoutedModel(SEAT_ID, routing, gate, tools=tools)

    _real_classify = build_classifier_from_model(RoutedModel("classifier", routing, gate))

    def classify(name, args):
        v = _real_classify(name, args)
        arg_preview = str(args)[:200]
        print(f"  [classifier] {v.decision.upper():5s} {name}({arg_preview}) -- {v.reason}", flush=True)
        return v

    graph = build_graph_from_model(
        worker_model, InMemorySaver(), tools=tools, classify=classify,
        turn_budget=TURN_BUDGET, workspace_root=ws,
    )

    context = beads.prime()
    prompt = f"{context}\n\n---\n\nTicket: {TICKET_TITLE}\n\n{TICKET_DESC}"
    config = {"configurable": {"thread_id": story["id"]}}

    seen = 0
    try:
        for step in graph.stream(
            {"messages": [("system", system_prompt), ("user", prompt)],
             "ticket_id": story["id"], "turn_count": 0},
            config, stream_mode="values",
        ):
            msgs = step["messages"]
            for m in msgs[seen:]:
                kind = type(m).__name__
                print(f"\n--- {kind} ---", flush=True)
                if getattr(m, "content", None):
                    print(str(m.content)[:4000], flush=True)
                for tc in getattr(m, "tool_calls", None) or []:
                    print(f"  -> {tc['name']}({str(tc['args'])[:400]})", flush=True)
            seen = len(msgs)
    except Exception as e:  # noqa: BLE001 -- a probe, we want to see whatever broke
        print(f"\n!!! run raised: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 72)
    current = beads.show(story["id"])
    summary = (current.get("metadata") or {}).get("completion_summary")
    print(f"total messages     : {seen}")
    print(f"flagged for human  : {beads.is_flagged_for_human(current)}")
    print(f"completion claimed : {bool(summary)}" + (f"  -- {summary[:300]}" if summary else ""))

    print("\n--- workspace tree ---")
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"  {os.path.relpath(p, ws)} ({os.path.getsize(p)} bytes)")

    print("\n--- acceptance checks ---")
    has = lambda name: os.path.exists(os.path.join(ws, name))
    print(f"  package.json present : {has('package.json')}")
    print(f"  tsconfig present     : {has('tsconfig.json')}")
    print(f"  README present       : {has('README.md') or has('readme.md')}")
    print(f"  src/ present         : {os.path.isdir(os.path.join(ws, 'src'))}")
    if has("package.json"):
        rc, out = _run("npm install --no-audit --no-fund --loglevel=error", ws)
        print(f"  npm install          : {'ok' if rc == 0 else f'FAILED ({rc})'}")
        rc, out = _run("npm test", ws)
        print(f"  npm test             : {'PASS' if rc == 0 else f'FAIL ({rc})'}")
        if rc != 0:
            print("    " + "\n    ".join(out.splitlines()[-15:]))
        rc, out = _run("npx --yes tsc --noEmit -p tsconfig.json" if has("tsconfig.json") else "npx --yes tsc --noEmit", ws)
        print(f"  tsc --noEmit         : {'ok' if rc == 0 else f'errors ({rc})'}")

    if not KEEP:
        beads._run(["delete", story["id"], project["id"], "--force"])
        print(f"\ncleaned up scratch ticket {project['id']}")
    else:
        print(f"\nkept: ticket {project['id']}, workspace {ws}")


if __name__ == "__main__":
    sys.exit(main())
