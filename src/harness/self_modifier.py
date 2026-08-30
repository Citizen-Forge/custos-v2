"""
Phase 7's self-modification agent: proposes changes to the harness's
OWN source. Works in a real, writable checkout at `/self-mod-checkout`
-- a separate Docker volume, deliberately decoupled from the live
system's read-only `/app/src` mount (see docker-compose.yml's comment
on that volume for why a second mount of the *same* host path would
have silently reopened the exact read-write gap Phase 7 already found
and fixed once). Editing this checkout can never affect anything
actually running.

Same tool-calling LangGraph shape as overwatch.py, narrower toolset
again: no shell_exec, no /workspace access (that's ticket work, a
different concern), and -- critically -- nothing here can apply its own
diff to the real tree or trigger a deploy, even though nothing later in
the pipeline waits on a human either (see self_mod.py's module
docstring). Proposing is the ceiling of this agent's own privileges.
Sandboxed testing (run_self_mod_sandbox.py, sandbox-runner only, needs
Docker access this role never gets), review (reviewer.py, a separate
model call that also decides approve/reject), and deployment
(run_self_mod_deploy.py, sandbox-runner only, the one script with
write access to the real tree) are all deliberately separate modules
this one has no path to invoke.

Scoped to `src/`, `tests/`, `scripts/` -- what harness's own container
already has read access to (see docker-compose.yml). Doesn't see
PLAN.md/README.md at the repo root; a real limitation, not an oversight,
kept this way to avoid mounting more of the host repo into a container
than this phase already decided harness should see.
"""

import os
import shutil
import subprocess
import uuid

from langchain_core.tools import tool

from . import self_mod
from .graph import build_graph_from_model
from .permissions import PermissionDenied, check_within_workspace

ROLE = "self_modifier"

CHECKOUT_ROOT = "/self-mod-checkout"
LIVE_SOURCE_ROOT = "/app"
LIVE_SOURCE_DIRS = ("src", "tests", "scripts")

SYSTEM_PROMPT = """You are the self-modification agent for an autonomous software delivery \
system -- you propose real improvements to the harness's OWN source code, not to a client's \
workspace. This is categorically higher-stakes than any other work in this system: a bad \
change here can break the whole harness, not just one ticket or one seat.

Work in your own writable checkout (sync_checkout_from_live to seed or refresh it, \
read_checkout_file/write_checkout_file/list_checkout_files to make real edits). Read the \
existing code carefully before changing it -- match its existing style and conventions, make \
the smallest change that actually addresses a real, specific problem you can point to, and \
never touch anything outside the scope of what you're actually fixing. You cannot run tests \
yourself; a separate, fully isolated process will run the real test suite against your exact \
diff before anyone (human or reviewer agent) ever sees it, so reason carefully rather than \
guessing and hoping the sandbox catches it.

When you have a real, complete, well-considered change, call propose_self_modification with a \
clear description of what it does and why. This queues it for sandboxed testing, then a \
separate reviewer agent's judgment, then human approval -- nothing you do here ever takes \
effect on its own. If you don't have a genuine, specific improvement to make, say so and stop \
rather than inventing a speculative change."""


def _checkout_path(relpath: str) -> str:
    check_within_workspace(relpath, CHECKOUT_ROOT)
    return os.path.abspath(os.path.join(CHECKOUT_ROOT, relpath))


def build_tools(conn):
    @tool
    def sync_checkout_from_live() -> str:
        """Seeds or refreshes your checkout from the harness's real
        current source (src/, tests/, scripts/). Refuses to overwrite
        uncommitted work -- propose or discard your current changes
        first if you want a fresh baseline."""
        git_dir = os.path.join(CHECKOUT_ROOT, ".git")
        if os.path.isdir(git_dir):
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT, capture_output=True, text=True
            )
            if status.stdout.strip():
                return "checkout has uncommitted changes -- propose_self_modification or discard them before re-syncing"

        try:
            os.makedirs(CHECKOUT_ROOT, exist_ok=True)
            for name in LIVE_SOURCE_DIRS:
                src = os.path.join(LIVE_SOURCE_ROOT, name)
                dst = os.path.join(CHECKOUT_ROOT, name)
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

            if not os.path.isdir(git_dir):
                subprocess.run(["git", "init", "-q"], cwd=CHECKOUT_ROOT, check=True)
            subprocess.run(["git", "add", "-A"], cwd=CHECKOUT_ROOT, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "baseline sync from live source", "--allow-empty"],
                cwd=CHECKOUT_ROOT,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            return f"error syncing checkout: {e}"
        return "checkout synced to current live source (src/, tests/, scripts/)"

    @tool
    def list_checkout_files(subdir: str = "") -> str:
        """List files under a directory in your checkout (e.g. "src/harness")."""
        try:
            path = _checkout_path(subdir)
        except PermissionDenied as e:
            return f"error: {e}"
        if not os.path.isdir(path):
            return f"no such directory: {subdir!r}"
        entries = []
        for root, _, files in os.walk(path):
            for f in files:
                entries.append(os.path.relpath(os.path.join(root, f), CHECKOUT_ROOT))
        return "\n".join(sorted(entries)) if entries else "(empty)"

    @tool
    def read_checkout_file(path: str) -> str:
        """Read a file from your checkout, relative to its root."""
        try:
            resolved = _checkout_path(path)
            with open(resolved, "r", encoding="utf-8") as f:
                return f.read()
        except PermissionDenied as e:
            return f"error: {e}"
        except OSError as e:
            return f"error reading {path!r}: {e}"

    @tool
    def write_checkout_file(path: str, content: str) -> str:
        """Write a file in your checkout, relative to its root -- creates
        new files or overwrites existing ones. Never touches the real
        running source; this is your own isolated copy."""
        try:
            resolved = _checkout_path(path)
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            return f"wrote {len(content)} bytes to {path}"
        except PermissionDenied as e:
            return f"error: {e}"
        except OSError as e:
            return f"error writing {path!r}: {e}"

    @tool
    def propose_self_modification(description: str) -> str:
        """Computes the real diff between your checkout's synced baseline
        and its current edited state, and queues it for sandboxed
        testing and review. Never applied automatically -- this is the
        ceiling of what this role can do."""
        diff_result = subprocess.run(
            ["git", "diff", "--", *LIVE_SOURCE_DIRS], cwd=CHECKOUT_ROOT, capture_output=True, text=True
        )
        diff = diff_result.stdout
        if not diff.strip():
            return "no changes to propose -- your checkout matches its synced baseline"

        proposal_id = self_mod.propose(conn, description, diff, proposed_by=ROLE)
        return (
            f"proposal #{proposal_id} recorded (status: pending). "
            f"Sandboxed testing, review, and human approval all happen in later, separate steps."
        )

    return [
        sync_checkout_from_live,
        list_checkout_files,
        read_checkout_file,
        write_checkout_file,
        propose_self_modification,
    ]


def run_self_mod_session(agent_model, tools, checkpointer, brief: str | None = None, thread_id: str | None = None) -> dict:
    """One self-modification pass -- same shape as overwatch.run_overwatch_session."""
    thread_id = thread_id or f"self-mod-{uuid.uuid4().hex[:8]}"
    graph = build_graph_from_model(agent_model, checkpointer, tools=tools)
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    if state.values:
        result = graph.invoke(None, config)
    else:
        user_message = brief or (
            "Sync your checkout, look for a real, specific improvement to the harness's own "
            "source, and propose it if you find one worth making."
        )
        result = graph.invoke(
            {
                "messages": [("system", SYSTEM_PROMPT), ("user", user_message)],
                "ticket_id": thread_id,
                "turn_count": 0,
            },
            config,
        )

    return {"thread_id": thread_id, "final_message": result["messages"][-1].content}
