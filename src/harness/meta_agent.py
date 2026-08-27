"""
Phase 5: the agent-improves-agents piece from PLAN.md. The actual
reasoning -- is this outcome data meaningful, would this prompt revision
actually help -- is inherently untestable without a real model doing real
judgment, so it isn't validated here. What IS built and tested is the
substrate around it: gathering an outcomes summary from Beads' own audit
trail (outcomes.py), formatting it alongside the role's current active
prompt (prompts.py), and turning a model's response into a *pending*
proposal that a human must explicitly approve before it ever takes
effect. Matches v1's "autonomy off by default for every role except
product-owner" pattern -- the meta-agent proposes, it never applies.
"""

import json

from . import outcomes, prompts

PROMPT_TEMPLATE = """You are reviewing an AI agent's system prompt based on its recent \
track record, and proposing an improved version if one is warranted.

Current system prompt for role "{role}":
{current_prompt}

Recent outcomes for this role (from its work-tracking history):
{outcomes_summary}

If the current prompt already looks reasonable given these outcomes, respond with the \
SAME prompt text unchanged and explain why no change is needed. Otherwise propose a \
revised prompt that addresses a real pattern in the outcomes above -- not a speculative \
rewrite.

Respond with strict JSON and nothing else: \
{{"revised_prompt": "<full prompt text>", "reasoning": "<one or two sentences>"}}
"""


def propose_prompt_update(conn, role: str, model) -> dict | None:
    """Returns the proposal dict if one was queued, None if the model
    proposed no change or its response couldn't be parsed (fail closed,
    same posture as classifier.py -- an unparseable response never
    silently becomes a proposal)."""
    current = prompts.get_active(conn, role) or "(no system prompt set yet)"
    summary = outcomes.summary(role)

    response = model.invoke(
        PROMPT_TEMPLATE.format(
            role=role, current_prompt=current, outcomes_summary=json.dumps(summary, default=str)
        )
    )
    content = getattr(response, "content", response)

    try:
        data = json.loads(content)
        revised = data["revised_prompt"]
        reasoning = data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if revised.strip() == current.strip():
        return None

    version = prompts.propose(conn, role, revised, reasoning)
    return {"role": role, "version": version, "reasoning": reasoning}
