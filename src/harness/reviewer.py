"""
Phase 7's other missing judgment call: `record_review` currently just
records whatever verdict it's given (tool_proposals.py) -- nothing forms
that verdict. This is the reviewer agent PLAN.md flagged as substrate
without judgment: given a candidate tool's source code, its declared
capabilities, and its real sandbox run (stdout/stderr/exit code), decide
allow/deny and why.

Single-shot, same shape as meta_agent.py's propose_prompt_update -- no
tools needed, just source + sandbox evidence in, a verdict out. Never
applies anything on its own: `record_review` only ever moves a proposal
to `reviewed` status, and `approve`/`reject` stay separate, human-only
calls (tool_proposals.py's docstring, PLAN.md Phase 7's promotion gate).
"""

import json

from . import tool_proposals

PROMPT = """You are reviewing a candidate tool proposed for an autonomous coding agent \
harness. The code will only ever run with real privileges if a human separately approves \
it after your review -- your job is to give that human an honest, specific verdict, not to \
be the final gate yourself.

Tool name: {tool_name}
Declared capabilities (what the proposer says this tool needs/does): {declared_capabilities}

Source code:
{source_code}

Sandbox run result (isolated container, no network, no secrets, read-only filesystem):
exit code: {exit_code}
stdout: {stdout}
stderr: {stderr}

Deny if: the code does something beyond its declared capabilities, the sandbox result shows \
a crash or clearly wrong behavior, it touches secrets/credentials/network in a way not \
justified by its declared purpose, or it's destructive/irreversible without being obviously \
that kind of tool. A clean sandbox run alone is not sufficient to allow -- the sandbox has no \
network and no real filesystem, so code can look fine there and still be dangerous once \
promoted with real privileges; read the code itself, not just its sandboxed output.

Respond with strict JSON and nothing else: \
{{"verdict": "allow"|"deny", "reasoning": "<one or two sentences>"}}
"""


def review_proposal(conn, proposal_id: int, model) -> dict | None:
    """Returns the recorded verdict dict, or None if the proposal doesn't
    exist. An unparseable model response is recorded as a "deny" verdict
    with that fact as the reasoning -- fails closed like classifier.py,
    but (unlike meta_agent.py's propose_prompt_update) still moves the
    proposal to `reviewed` status rather than leaving it stuck in
    `sandboxed` forever, since a human still makes the real approve/reject
    call regardless of this verdict."""
    proposal = tool_proposals.get(conn, proposal_id)
    if proposal is None:
        return None

    response = model.invoke(
        PROMPT.format(
            tool_name=proposal["tool_name"],
            declared_capabilities=proposal["declared_capabilities"],
            source_code=proposal["source_code"],
            exit_code=proposal["sandbox_exit_code"],
            stdout=proposal["sandbox_stdout"] or "(empty)",
            stderr=proposal["sandbox_stderr"] or "(empty)",
        )
    )
    content = getattr(response, "content", response)

    try:
        data = json.loads(content)
        verdict = data["verdict"]
        reasoning = data.get("reasoning", "")
        if verdict not in ("allow", "deny"):
            raise ValueError(f"unexpected verdict: {verdict!r}")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        verdict = "deny"
        reasoning = f"reviewer response unparseable: {e}"

    tool_proposals.record_review(conn, proposal_id, verdict, reasoning)
    return {"proposal_id": proposal_id, "verdict": verdict, "reasoning": reasoning}
