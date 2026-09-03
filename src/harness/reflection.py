"""
A slot of time after a ticket, for the agent rather than for the work.

Everything else in this harness is about throughput: claim a ticket, do
it, close it, claim the next one. This is the one part that isn't. When a
seat finishes something it gets a few turns with no task attached -- to
say what the work was actually like, add to its own profile, tell the
team about a pain point the board would never capture, propose a change
to its own standing instructions, or just say something because it felt
like it.

Deliberately not a status report. The board already records what got
done, the completion summary already records how, and the verifier
already judges whether it holds up. None of those have anywhere for "that
ticket was miserable and here is why", or for the agent to be a person
for a minute. This does.

The toolset is narrowed on purpose. No file or shell tools: this is not
more working time and should not become a way to sneak in another commit.
No complete/refuse/decline: the ticket is already resolved. What is left
is expression -- the wiki, the team channel, memory -- plus
suggest_prompt_change, which exists only here. An agent editing its own
standing instructions mid-task would be a different and worse thing than
an agent reflecting afterwards on how those instructions served it, and
even here it only proposes: the revision is recorded pending, never
self-activated.

Bounded by REFLECTION_TURN_BUDGET so a reflective mood cannot consume the
agent slot indefinitely, and skippable entirely with REFLECTION=off.
"""

import logging
import os

from langgraph.checkpoint.postgres import PostgresSaver

from . import beads, tools
from .graph import build_graph_from_model

log = logging.getLogger("reflection")

TURN_BUDGET = int(os.environ.get("REFLECTION_TURN_BUDGET", "6"))


def enabled() -> bool:
    return os.environ.get("REFLECTION", "on").lower() not in ("off", "0", "false", "no")


# Tools available only while reflecting. suggest_prompt_change is not in
# tools.SHARED_TOOLS precisely so it cannot be reached during a ticket.
REFLECTION_TOOLS = [
    tools.post_to_team,
    tools.scan_team_channel,
    tools.read_wiki_page,
    tools.write_wiki_page,
    tools.list_wiki_pages,
    tools.remember_fact,
    tools.suggest_prompt_change,
]

BRIEF = """You've just finished {outcome_phrase} {ticket_id} -- "{title}".

Take a moment. This isn't more work, and nothing here is required: there's no ticket to
close, nothing is being measured, and "nothing to add this time" is a perfectly good answer.

Some things you might do with the time, if you want to:

- Say something to the team ({post_tool}). A pain point you hit that the next agent will
  hit too, something that surprised you, something that made the work harder than it needed
  to be, a question, or something entirely beside the point. The board records what got
  done; it has nowhere to put what it was like.
- Add to your own profile page ({profile_slug}). You are not a fixed description written
  once at your creation -- you've done things since. Read it back and change it if it no
  longer fits, or add something new you've discovered you care about.
- Propose a change to your own standing instructions, if the way you've been told to work
  got in your way. This gets recorded for review rather than taking effect on its own.
- Remember something worth carrying to your next ticket.
- Or just say something. That's allowed too.

You have a few turns. Use as many or as few as you like, then stop."""

OUTCOME_PHRASE = {
    "closed": "work on",
    "flagged": "and escalated",
    "released": "and handed back",
    "unclaimed": "a run on",
    "failed": "an unsuccessful run on",
}


def reflect(conn_string: str, runtime, issue: dict, outcome: str) -> bool:
    """Give one seat its slot after a ticket. Returns whether it ran.

    Never raises into the caller: a failed reflection must not turn a
    completed ticket into a failed one. It is the least important thing
    in the system by throughput and, arguably, not by much else."""
    if not enabled():
        return False

    ticket_id = issue["id"]
    brief = BRIEF.format(
        outcome_phrase=OUTCOME_PHRASE.get(outcome, "work on"),
        ticket_id=ticket_id,
        title=issue.get("title", ""),
        post_tool="post_to_team",
        profile_slug=f"agents/{runtime.seat_id}",
    )

    try:
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            graph = build_graph_from_model(
                runtime.model, checkpointer, tools=REFLECTION_TOOLS, turn_budget=TURN_BUDGET
            )
            messages = ([("system", runtime.system_prompt)] if runtime.system_prompt else []) + [
                ("user", brief)
            ]
            graph.invoke(
                {"messages": messages, "ticket_id": ticket_id, "turn_count": 0},
                {"configurable": {"thread_id": f"reflect-{ticket_id}"}},
            )
        log.info("%s reflected after %s", runtime.seat_id, ticket_id)
        return True
    except Exception:
        log.exception("reflection failed for %s after %s", runtime.seat_id, ticket_id)
        return False
