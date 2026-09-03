"""
Phase 5: the agent-improves-agents piece from PLAN.md. The actual
reasoning -- is this outcome data meaningful, would this prompt revision
actually help -- is inherently untestable without a real model doing real
judgment, so it isn't validated here. What IS built and tested is the
substrate around it.

Both capabilities activate immediately, no separate human approval step:

- `propose_prompt_update`: revises an *existing* seat's prompt. Used to
  queue as *pending* for human approval; reversed 2026-08-29 (user's own
  call, made after watching reviewer.py correctly deny 4/5 real
  adversarial tool proposals -- see PLAN.md) -- the model's own
  considered judgment (including declining to propose anything when the
  evidence doesn't warrant a change, proven live: see
  scripts/probe_verifier_failure_case.py) IS the review, so the proposal
  activates the moment it's made. `prompts.propose`/`prompts.approve`
  stay separate calls internally, but nothing waits between them anymore.
- `create_specialist_seat`: the product-owner's "no specialist exists for
  this yet" path -- creates a brand new seat with an initial prompt that
  goes active immediately, no approval step. This was already how seat
  creation worked before the 2026-08-29 change above; the two are now
  consistent with each other rather than deliberately asymmetric.
"""

import json

from . import avatar, outcomes, prompts, seats, slack, verifications, wiki

PROMPT_TEMPLATE = """You are reviewing an AI agent's system prompt based on its recent \
track record, and proposing an improved version if one is warranted.

Current system prompt for role "{role}":
{current_prompt}

Recent outcomes for this role (from its work-tracking history -- closed/refused/still-open \
counts, and refusal reasons if any):
{outcomes_summary}

Acceptance-criteria verification results for this role (a SEPARATE agent's real pass/fail \
judgment on tickets that had explicit criteria -- this is the strongest quality signal \
available, stronger than "was it refused," since a ticket can close successfully and still \
fail its actual criteria):
{verification_summary}

If the current prompt already looks reasonable given this evidence, respond with the SAME \
prompt text unchanged and explain why no change is needed. Otherwise propose a revised \
prompt that addresses a REAL pattern in the evidence above -- point at specific failure \
reasoning if verification failures exist, don't speculate beyond what the evidence shows.

Respond with strict JSON and nothing else: \
{{"revised_prompt": "<full prompt text>", "reasoning": "<one or two sentences>"}}
"""


def propose_prompt_update(conn, role: str, model) -> dict | None:
    """Returns the activated revision's dict, or None if the model
    proposed no change or its response couldn't be parsed (fail closed,
    same posture as classifier.py -- an unparseable response never
    silently becomes a revision, active or otherwise). A well-formed
    revision activates immediately (prompts.propose then prompts.approve,
    back to back) -- see module docstring for why no separate human step
    sits between them anymore."""
    current = prompts.get_active(conn, role) or "(no system prompt set yet)"
    summary = outcomes.summary(role)
    verification_summary = verifications.summary(conn, role)

    response = model.invoke(
        PROMPT_TEMPLATE.format(
            role=role,
            current_prompt=current,
            outcomes_summary=json.dumps(summary, default=str),
            verification_summary=json.dumps(verification_summary, default=str),
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
    prompts.approve(conn, role, version)
    return {"role": role, "version": version, "reasoning": reasoning}


PROFILE_BRIEF = """Next, write your wiki profile page.

Imagine you are a software developer joining a new team, and someone has asked you to write \
an introduction to go up on the company intranet for your new colleagues. That is the whole \
brief. Think about what you would actually want these people to know about you, what your \
character would enjoy sharing, and what they would quietly leave out. Some people write three \
lines. Some write far too much. Some are funny about it, some are sincere, some are visibly \
uncomfortable being asked. Some talk about the job, some barely mention it.

Write the post that YOU would write. Not a template filled in -- an actual piece of writing by \
a specific person who has their own reasons for including what they include. It should be \
possible to read it and disagree with them about something.

Have a look at the existing seats below first, and go somewhere they did not.
"""


CREATE_SEAT_PROMPT = """You are helping design a new specialist agent for an autonomous \
software delivery system. A product-owner agent identified a gap: no existing seat \
specializes in the following kind of work, and one is needed.

Specialty needed: {specialty_description}

Existing seats already in the roster (avoid unnecessary overlap):
{existing_seats}

Propose a short seat_id (lowercase, hyphenated -- this is the FUNCTIONAL identifier other \
systems key off of, e.g. "backend-perf-owen") and a system prompt: the instructions this \
agent will work from on every ticket it's assigned.

Separately, this agent should also choose its OWN identity -- a real name, an age, a gender \
(or explicitly none, if that's the honest choice), and pronouns, distinct from the functional \
seat_id, chosen the way a person would pick how they want to be known, not derived \
mechanically from the specialty. Pick pronouns freely (they/them, she/her, he/him, or \
something else entirely) -- there's no default and no wrong answer here, and gender need not \
follow from pronouns or vice versa.

{profile_brief}

Finally, describe your own PHYSICAL appearance, for your avatar portrait -- entirely your own \
choice (age, perceived gender presentation or none, ethnicity, hair, face, expression, style: \
whatever makes you look like a specific person rather than a generic one), consistent with the \
identity you chose above. This description gets dropped into a FIXED image-generation template \
that already forces a plain-background headshot -- so describe only the person: face, hair, \
expression, clothing/style visible at shoulders-and-up. Do NOT describe a background, setting, \
props, pose, or other people; those parts of the template are not yours to change.

Respond with strict JSON and nothing else: \
{{"seat_id": "<id>", "system_prompt": "<full prompt text>", "display_name": "<chosen name>", \
"pronouns": "<chosen pronouns>", "profile_page": "<first-person markdown bio>", \
"appearance_description": "<physical description for the avatar portrait, person only>"}}
"""


def _existing_seat_summary(seat: dict) -> str:
    # Includes each existing seat's chosen name and a short excerpt of
    # its own wiki profile (not just its functional specialty) -- the
    # prompt below asks the new agent to deliberately choose different
    # personality specifics than its teammates, which is only a real
    # instruction if it can actually see what they already picked, not
    # just their job description.
    who = f"{seat['display_name']} ({seat['seat_id']})" if seat.get("display_name") else seat["seat_id"]
    line = f"- {who}: {seat['specialty']}"
    profile = wiki.read_page(wiki.agent_profile_slug(seat["seat_id"]))
    if profile:
        excerpt = profile.strip().replace("\n", " ")[:200]
        line += f" | profile: {excerpt}..."
    return line


def create_specialist_seat(conn, specialty_description: str, requested_by: str, model) -> dict | None:
    """Returns the created seat's info if one was made, None if the
    response couldn't be parsed or the proposed seat_id collides with an
    existing one (fail closed rather than silently overwrite a seat that
    already exists -- same posture as propose_prompt_update). display_name/
    pronouns are the seat's own chosen identity (Phase 4, PLAN.md's
    welfare-essay-derived design goal) -- distinct from seat_id, which
    stays the functional identifier routing/prompts/outcomes key off of."""
    existing = seats.list_all(conn)
    existing_summary = "\n".join(_existing_seat_summary(s) for s in existing) or "(none yet)"

    response = model.invoke(
        CREATE_SEAT_PROMPT.format(
            specialty_description=specialty_description,
            existing_seats=existing_summary,
            profile_brief=PROFILE_BRIEF,
        )
    )
    content = getattr(response, "content", response)

    try:
        data = json.loads(content)
        seat_id = data["seat_id"]
        system_prompt = data["system_prompt"]
        display_name = data.get("display_name") or None
        pronouns = data.get("pronouns") or None
        profile_page = data.get("profile_page") or None
        appearance_description = data.get("appearance_description") or None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if not seat_id or seats.get(conn, seat_id):
        return None

    seats.create(conn, seat_id, specialty_description, created_by=requested_by, display_name=display_name, pronouns=pronouns)
    version = prompts.propose(
        conn, seat_id, system_prompt, reason=f"initial specialist prompt for: {specialty_description}"
    )
    prompts.approve(conn, seat_id, version)

    if profile_page:
        # The seat's own wiki profile (user's own framing: "a sort of
        # profile of who they are") -- best-effort, same posture as
        # Slack: a failure here shouldn't undo an otherwise-successful
        # seat creation, so this never raises.
        try:
            wiki.write_page(wiki.agent_profile_slug(seat_id), profile_page)
        except OSError:
            pass

    if appearance_description:
        # Real portrait generated from the seat's own written physical
        # description -- deliberately NOT profile_page (that's personality/
        # bio prose, not a description of how they look, and would leak
        # non-visual detail plus explicit background/setting talk into an
        # image prompt that already has a fixed template for those parts).
        # User's own call, 2026-08-29 -- prefers this over a deterministic
        # illustrated avatar for realism. Optional: no-ops if
        # GEMINI_API_KEY isn't configured, and the dashboard falls back to
        # a DiceBear avatar whenever this hasn't produced one -- never
        # blocks seat creation on an external API call.
        avatar.generate_avatar(seat_id, appearance_description)

    who = f"{display_name} ({seat_id})" if display_name else seat_id
    slack.post_message(f":wave: Welcome {who} to the team! Recruited to work on: {specialty_description}")

    return {
        "seat_id": seat_id,
        "specialty": specialty_description,
        "version": version,
        "display_name": display_name,
        "pronouns": pronouns,
        "profile_page": profile_page,
        "appearance_description": appearance_description,
    }
