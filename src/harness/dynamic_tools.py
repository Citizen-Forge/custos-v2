"""
Tool activation: turns an `approved` tool_proposals row into a real,
callable langchain tool a live agent session can invoke -- the gap
PLAN.md flagged repeatedly ("approve/reject is still the end of the
modeled lifecycle, not a trigger for anything").

Scoped 2026-08-30 per the user's own reminder of this project's
two-layer security model: MECHANICAL isolation (sandbox.py's Docker
container, no network/no persistent FS) is what proves a *candidate*
proposal safe enough to trust in the first place. Once trusted
(approved), execution is real and direct, gated per-call by the
LLM-driven classifier -- exactly how the existing built-in tools
already work (see tools.py's shell_exec: "gating happens one layer up,
in graph.py's permission_gate node ... no redundant check here"). A
dynamically-approved tool is NOT a new trust tier or a new privilege
boundary; it inherits the same real-time classifier gate every other
tool already goes through, unchanged -- classifier.py's PROMPT is
already generic over (tool_name, tool_args), so no classifier change
was needed to cover this. This is why tool activation turned out to be
smaller than first scoped: the judgment half (the "security agent")
already existed and already applies for free once a tool is wired into
a session's toolset; what was missing was purely the mechanical wiring.

Calling convention: every proposal reviewed so far (see
scripts/probe_reviewer_adversarial.py, and the one real approved tool,
`line_count`) is written as a standalone CLI script, invoked as
`python3 script.py [args...]` -- sandbox.py already proves this exact
invocation shape safe pre-approval, so activation keeps that same
shape rather than inventing a new one. Approved source is materialized
to disk once and run as a real subprocess with the agent's own `args`
string shlex-split into argv -- no exec()/import of untrusted code into
the harness process itself, same posture as shell_exec never importing
arbitrary code either. Deliberately NOT re-sandboxed in Docker at call
time: that would mean extending Docker-socket access into the `harness`
service, exactly the privilege boundary Phase 7 was built to keep
separate (see sandbox.py's own module docstring) -- mechanical isolation
stays a pre-trust, one-time proposal-review concern, not a recurring
per-call one.
"""

import logging
import os
import shlex
import subprocess
import sys

from langchain_core.tools import StructuredTool

from . import tool_proposals
from .config import WORKSPACE_ROOT

log = logging.getLogger("dynamic_tools")

TOOLS_DIR = ".approved_tools"
# Matches tools.py's shell_exec timeout exactly -- an activated tool is
# the same trust tier as any other in-process tool, not a stricter or
# looser one, so there's no principled reason for a different bound.
TIMEOUT_SECONDS = 120


def _materialize(proposal: dict) -> str:
    """Writes an approved proposal's source to a stable on-disk path
    outside the ticket-visible workspace tree, keyed by proposal id (not
    just tool_name) so two different proposals can never collide on the
    same file. Kept out of the ticket workspace root specifically so
    nothing a ticket does through its own real tools (shell_exec,
    write_file) can silently overwrite the running definition of an
    approved tool. Re-writes on every call -- idempotent and cheap given
    how rarely a tool gets approved, and it means a session always runs
    against the proposal's current row contents."""
    tools_root = os.path.join(WORKSPACE_ROOT, TOOLS_DIR)
    os.makedirs(tools_root, exist_ok=True)
    path = os.path.join(tools_root, f"{proposal['id']}_{proposal['tool_name']}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(proposal["source_code"])
    return path


def _make_runner(script_path: str, tool_name: str):
    """Returns a plain `(cli_args: str) -> str` function -- the shape
    StructuredTool.from_function introspects into a single-string-argument
    tool schema, matching every approved proposal's own CLI-args design
    (argparse, per line_count's docstring: `line_count PATH --json`).
    Named `cli_args`, not `args` -- pydantic's field-name handling mangles
    a literal `args` parameter (collides with its own `*args` convention,
    verified live: langchain passed it through as `v__args` and the call
    failed), caught by this module's own tests actually invoking the
    tool rather than just building it."""

    def run(cli_args: str = "") -> str:
        try:
            argv = shlex.split(cli_args) if cli_args else []
        except ValueError as e:
            return f"{tool_name}: could not parse args: {e}"

        try:
            result = subprocess.run(
                [sys.executable, script_path, *argv],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"{tool_name}: timed out after {TIMEOUT_SECONDS}s"

        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            return f"{tool_name}: exit {result.returncode}\n{output}"
        return output or f"{tool_name}: completed with no output"

    return run


def build_dynamic_tools(conn) -> list:
    """One real, callable tool per `approved` tool_proposals row --
    fetched fresh on every call, not cached at import time, so a newly
    approved proposal is available to the next session/worker process
    that builds its toolset without a code change. (A worker process
    already running when a proposal gets approved keeps its existing
    toolset until its next restart -- same "long-running containers
    don't pick up changes automatically" tradeoff this project has
    documented elsewhere, not a new one.)"""
    tool_proposals.init_table(conn)
    approved = tool_proposals.list_by_status(conn, "approved")

    tools = []
    for proposal in approved:
        script_path = _materialize(proposal)
        tools.append(
            StructuredTool.from_function(
                func=_make_runner(script_path, proposal["tool_name"]),
                name=proposal["tool_name"],
                description=proposal["declared_capabilities"],
            )
        )
    return tools
