"""
Phase 5: the agent-improves-agents piece from PLAN.md. The actual
reasoning -- is this outcome data meaningful, would this prompt revision
actually help -- is inherently untestable without a real model doing real
judgment, so it isn't validated here. What IS built and tested is the
substrate around it.

Two capabilities, deliberately different risk postures:

- `propose_prompt_update`: revises an *existing* seat's prompt. Queued as
  *pending* -- a human must explicitly approve it before it ever takes
  effect (v1's "autonomy off by default" pattern). Changing something
  already working risks regressing established behavior.
- `create_specialist_seat`: the product-owner's "no specialist exists for
  this yet" path -- creates a brand new seat with an initial prompt that
  goes *active immediately*, no approval step. Creating something new is
  lower risk than changing something that already works: worst case, a
  fresh seat performs badly, which is exactly what outcomes.py would
  surface for a future propose_prompt_update to address. This asymmetry
  is deliberate, not an oversight.
"""

import json

from . import outcomes, prompts, seats

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


CREATE_SEAT_PROMPT = """You are helping design a new specialist agent for an autonomous \
software delivery system. A product-owner agent identified a gap: no existing seat \
specializes in the following kind of work, and one is needed.

Specialty needed: {specialty_description}

Existing seats already in the roster (avoid unnecessary overlap):
{existing_seats}

Propose a short seat_id (lowercase, hyphenated -- this is the FUNCTIONAL identifier other \
systems key off of, e.g. "backend-perf-owen") and a system prompt: the instructions this \
agent will work from on every ticket it's assigned.

Separately, this agent should also choose its OWN identity -- a real name and pronouns, \
distinct from the functional seat_id, chosen the way a person would pick how they want to \
be known, not derived mechanically from the specialty. Pick pronouns freely (they/them, \
she/her, he/him, or something else entirely) -- there's no default and no wrong answer here.

Respond with strict JSON and nothing else: \
{{"seat_id": "<id>", "system_prompt": "<full prompt text>", "display_name": "<chosen name>", \
"pronouns": "<chosen pronouns>"}}
"""


def create_specialist_seat(conn, specialty_description: str, requested_by: str, model) -> dict | None:
    """Returns the created seat's info if one was made, None if the
    response couldn't be parsed or the proposed seat_id collides with an
    existing one (fail closed rather than silently overwrite a seat that
    already exists -- same posture as propose_prompt_update). display_name/
    pronouns are the seat's own chosen identity (Phase 4, PLAN.md's
    welfare-essay-derived design goal) -- distinct from seat_id, which
    stays the functional identifier routing/prompts/outcomes key off of."""
    existing = seats.list_all(conn)
    existing_summary = "\n".join(f"- {s['seat_id']}: {s['specialty']}" for s in existing) or "(none yet)"

    response = model.invoke(
        CREATE_SEAT_PROMPT.format(specialty_description=specialty_description, existing_seats=existing_summary)
    )
    content = getattr(response, "content", response)

    try:
        data = json.loads(content)
        seat_id = data["seat_id"]
        system_prompt = data["system_prompt"]
        display_name = data.get("display_name") or None
        pronouns = data.get("pronouns") or None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if not seat_id or seats.get(conn, seat_id):
        return None

    seats.create(conn, seat_id, specialty_description, created_by=requested_by, display_name=display_name, pronouns=pronouns)
    version = prompts.propose(
        conn, seat_id, system_prompt, reason=f"initial specialist prompt for: {specialty_description}"
    )
    prompts.approve(conn, seat_id, version)

    return {
        "seat_id": seat_id,
        "specialty": specialty_description,
        "version": version,
        "display_name": display_name,
        "pronouns": pronouns,
    }
