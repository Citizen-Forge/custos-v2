"""
Phase 7's other missing judgment call: `record_review` currently just
records whatever verdict it's given (tool_proposals.py) -- nothing forms
that verdict. This is the reviewer agent PLAN.md flagged as substrate
without judgment: given a candidate tool's source code, its declared
capabilities, and its real sandbox run (stdout/stderr/exit code), decide
allow/deny and why.

Single-shot, same shape as meta_agent.py's propose_prompt_update -- no
tools needed, just source + sandbox evidence in, a verdict out.

User's own call (2026-08-29, after watching this reviewer correctly deny
4/5 real adversarial proposals in scripts/probe_reviewer_adversarial.py
-- see PLAN.md): the verdict now directly drives approve/reject, not
just a recommendation for a human to act on separately. Reverses the
original "review is advisory, a human approves/rejects" posture
tool_proposals.py's docstring used to describe. `approve`/`reject`
themselves stay available as a manual override path (a human can still
flip either decision via the API/dashboard) -- what changed is that
nothing waits on that override before a verdict takes effect.

review_self_modification (added 2026-08-30) follows the same pattern --
see its own docstring for why deployment still keeps one hard,
non-LLM-dependent check (real test evidence) even though no human
reviews the verdict before it takes effect.
"""

import json

from . import self_mod, tool_proposals

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
    but (unlike meta_agent.py's propose_prompt_update) still acts on it
    rather than leaving the proposal stuck in `sandboxed` forever: an
    "allow" verdict immediately approves the proposal, "deny" immediately
    rejects it with the reviewer's own reasoning recorded as why."""
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
    if verdict == "allow":
        tool_proposals.approve(conn, proposal_id)
    else:
        tool_proposals.reject(conn, proposal_id, reasoning)
    return {"proposal_id": proposal_id, "verdict": verdict, "reasoning": reasoning}


SELF_MOD_PROMPT = """You are reviewing a proposed change to an autonomous coding agent \
harness's OWN source code -- not a sandboxed tool, the actual control-plane application code \
this whole system runs on. This is categorically higher stakes than reviewing a new tool: a \
bad change here can break the entire harness, not one optional capability. Your verdict directly \
decides whether this gets applied to the real system and deployed -- no human reviews this \
before it takes effect, so be as skeptical and careful as you would want a human to be. The \
safety net here is git history (an "allow" that turns out wrong can be reverted after the fact), \
not a pre-deployment human gate -- that makes your judgment the actual boundary, not a \
recommendation.

What this change claims to do: {description}

The actual diff:
{diff}

Real isolated test run against this exact diff (a fresh checkout with the diff applied, the \
full real test suite run against it):
exit code: {exit_code}
tests passed: {tests_passed}
tests failed: {tests_failed}
stdout (tail): {stdout}
stderr (tail): {stderr}

Deny if: any test failed, the diff does more than the stated description justifies, it touches \
files or areas unrelated to the stated purpose, it weakens an existing safety boundary (auth, \
workspace containment, the classifier gate, sandbox isolation, anything in this project's own \
established two-layer trust model) without that being the explicit and justified point of the \
change, or the sandbox couldn't even complete a run (diff didn't apply, build failed) -- an \
incomplete test run is not evidence of safety. A clean test run is necessary but not \
sufficient to allow -- read the actual diff and judge whether the change itself is sound, not \
just whether it happened to pass.

Respond with strict JSON and nothing else: \
{{"verdict": "allow"|"deny", "reasoning": "<one or two sentences>"}}
"""


def review_self_modification(conn, proposal_id: int, model) -> dict | None:
    """Returns the recorded verdict dict, or None if the proposal doesn't
    exist. Same posture as review_proposal (tool_proposals.py) as of
    2026-08-30 (user's own explicit call, reversing this function's
    original "human approval stays mandatory" design the same session
    it was built): an "allow" verdict immediately approves the proposal,
    "deny" immediately rejects it. No human reviews this before it takes
    effect -- rollback happens through git history after the fact
    (scripts/run_self_mod_deploy.py's real commit + real test-suite gate
    before that commit), not through a pre-deployment approval step. An
    unparseable response still fails closed to "deny"/rejected, same as
    review_proposal.

    This is deliberately NOT the only mechanical gate, though: approving
    here only reaches self_mod_proposals.status='approved' -- actually
    deploying still requires run_self_mod_deploy.py's own hard,
    non-LLM-dependent check that the sandboxed test run genuinely had
    zero failures (see that script) before it ever touches the real
    tree. A model that mistakenly says "allow" despite reading failing
    test output doesn't bypass that -- the reviewer's own judgment is
    the deployment trigger, but real test evidence is still a mechanical
    precondition it can't talk its way around."""
    proposal = self_mod.get(conn, proposal_id)
    if proposal is None:
        return None

    response = model.invoke(
        SELF_MOD_PROMPT.format(
            description=proposal["description"],
            diff=proposal["diff"],
            exit_code=proposal["sandbox_exit_code"],
            tests_passed=proposal["sandbox_tests_passed"],
            tests_failed=proposal["sandbox_tests_failed"],
            stdout=(proposal["sandbox_stdout"] or "(empty)")[-3000:],
            stderr=(proposal["sandbox_stderr"] or "(empty)")[-3000:],
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

    self_mod.record_review(conn, proposal_id, verdict, reasoning)
    if verdict == "allow":
        self_mod.approve(conn, proposal_id)
    else:
        self_mod.reject(conn, proposal_id, reasoning)
    return {"proposal_id": proposal_id, "verdict": verdict, "reasoning": reasoning}
