"""
Is a running agent actually getting anywhere?

Deliberately not a timeout. This harness runs against slow local
inference, so elapsed wall-clock time says nothing on its own -- an agent
holding a ticket for hours is expected (see Dispatcher._agent_thread).
Progress has to be judged from what the agent has actually done.

Two signals, with very different costs, and the cheap one runs first:

1. Free, no model: the newest checkpoint timestamp for a ticket. LangGraph's
   PostgresSaver writes a checkpoint row per graph step, each carrying
   `checkpoint->>'ts'`, and thread_id IS the ticket id (worker.py makes
   them the same deliberately). If that has not moved in hours, the agent
   has taken no graph step at all -- that is not slow thinking, it is
   stopped. Plain SQL, no inference, no new storage.

2. Costly, needs judgment: the checkpoints ARE advancing but the agent is
   going in circles -- re-reading the same file, retrying the same failing
   command. Only a model reading the recent transcript can tell that from
   real work.

Because (1) is free it runs over every agent every cycle, and (2) is only
ever asked about agents (1) could not clear. On a healthy roster this
costs nothing.

Nothing here kills an agent. A stall is reported, never terminated --
terminating on a clock is exactly the timeout this design rejects. What
should happen instead is the caller's decision; see check_running_agents.
"""

import json
import os

from langgraph.checkpoint.postgres import PostgresSaver

from . import beads

# How long with no graph step at all before an agent is worth a second
# look. Generous on purpose: a single turn against a slow local model can
# take many minutes, so this wants to be well clear of "thinking".
STALL_AFTER_SECONDS = int(os.environ.get("AGENT_STALL_SECONDS", "3600"))

# Transcripts get long and the tail is what matters -- an agent looping is
# visible in its last few exchanges, not its first.
TRANSCRIPT_MESSAGES = int(os.environ.get("PROGRESS_TRANSCRIPT_MESSAGES", "8"))
MESSAGE_CHARS = 800

PROMPT = """You are judging whether a working agent is making progress or is stuck in a loop.

Ticket: {title}
Description: {description}

Its most recent activity, oldest first:
{transcript}

The agent runs on a slow local model, so taking a long time is NORMAL and is not by itself \
evidence of a problem. Judge only what the messages show it doing.

Say "looping" only if the recent activity actually repeats itself -- the same command retried \
with no change, the same file read over and over, the same failed approach restated. Varied \
work, or a long gap with real steps either side, is progress.

If you cannot tell from the evidence, say "progressing" -- a false alarm interrupts real work, \
which is worse than noticing a loop one cycle later.

Respond with strict JSON and nothing else: \
{{"verdict": "progressing"|"looping", "reasoning": "<one or two sentences>"}}
"""


def last_activity(conn, thread_ids: list[str]) -> dict[str, str]:
    """Newest checkpoint timestamp per thread. The free signal -- one
    indexed query, no model, and it reads the checkpointer's own tables
    rather than duplicating any state."""
    if not thread_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT thread_id, max(checkpoint->>'ts') "
            "FROM checkpoints WHERE thread_id = ANY(%s) GROUP BY thread_id",
            (list(thread_ids),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def idle_seconds(last_ts: str | None, now=None) -> float | None:
    """Seconds since a thread's last graph step, or None if it has never
    checkpointed (a just-started agent, not a stalled one)."""
    if not last_ts:
        return None
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    try:
        then = datetime.fromisoformat(last_ts)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds()


def transcript(conn_string: str, thread_id: str, last_n: int = TRANSCRIPT_MESSAGES) -> list[dict]:
    """The tail of a ticket's message history, read straight from the
    checkpointer.

    Goes through PostgresSaver rather than querying the tables directly:
    messages are serialized into checkpoint_blobs, and get_tuple is what
    knows how to deserialize them. The judge has no compiled graph to call
    get_state on, so this is its only route in."""
    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        tuple_ = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
    if not tuple_:
        return []
    messages = (tuple_.checkpoint.get("channel_values") or {}).get("messages") or []
    out = []
    for message in messages[-last_n:]:
        content = str(getattr(message, "content", "") or "")
        out.append({"type": type(message).__name__, "content": content[:MESSAGE_CHARS]})
    return out


def _format(messages: list[dict]) -> str:
    if not messages:
        return "(no messages recorded yet)"
    return "\n".join(f"[{m['type']}] {m['content']}" for m in messages)


def judge_progress(model, issue: dict, messages: list[dict]) -> dict:
    """Single-shot verdict, same evidence-in/verdict-out shape as
    verifier.py and reviewer.py.

    Fails OPEN to "progressing", unlike verifier.py which fails closed:
    the cost of being wrong is asymmetric here. A wrong "looping" verdict
    raises a false alarm about an agent doing real work; a wrong
    "progressing" verdict just means the next cycle catches it."""
    response = model.invoke(
        PROMPT.format(
            title=issue.get("title", ""),
            description=(issue.get("description") or "")[:1500],
            transcript=_format(messages),
        )
    )
    content = getattr(response, "content", response)
    try:
        data = json.loads(content)
        verdict = data["verdict"]
        if verdict not in ("progressing", "looping"):
            raise ValueError(f"unexpected verdict: {verdict!r}")
        return {"verdict": verdict, "reasoning": data.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {
            "verdict": "progressing",
            "reasoning": f"could not parse judge response ({e}) -- assuming progress",
        }


def check_running_agents(conn, conn_string: str, model=None, stall_after: int | None = None) -> list[dict]:
    """Assess every currently-running agent.

    Returns one report per agent: seat, ticket, seconds idle, a verdict
    and why. Verdicts are 'progressing', 'idle' (no graph step for
    stall_after seconds, established without a model) or 'looping' (the
    model read the transcript and saw repetition).

    Reports only. Nothing here flags, closes, or kills anything -- what to
    do about a stalled agent is a policy decision with real side effects
    (flagging for human parks the ticket AND frees the seat, because both
    worker._next_ticket and dispatcher.next_assigned_ticket skip flagged
    issues), so it belongs to the caller, not here."""
    from .dispatcher import running_agents

    stall_after = STALL_AFTER_SECONDS if stall_after is None else stall_after
    agents = running_agents()
    if not agents:
        return []

    ticket_ids = [a["ticket_id"] for a in agents]
    activity = last_activity(conn, ticket_ids)

    reports = []
    for agent in agents:
        ticket_id = agent["ticket_id"]
        last_ts = activity.get(ticket_id)
        idle = idle_seconds(last_ts)
        report = {
            "seat_id": agent.get("seat_id"),
            "ticket_id": ticket_id,
            "title": agent.get("title"),
            "last_activity": last_ts,
            "idle_seconds": idle,
        }

        if idle is None:
            report.update(verdict="progressing", reasoning="just started; no checkpoint yet")
            reports.append(report)
            continue

        if idle < stall_after:
            report.update(
                verdict="progressing",
                reasoning=f"took a graph step {int(idle)}s ago",
            )
            reports.append(report)
            continue

        # Idle past the threshold. Without a model that is all we can say,
        # and it is still worth saying -- no graph step in this long is a
        # real signal on its own.
        if model is None:
            report.update(
                verdict="idle",
                reasoning=f"no graph step for {int(idle)}s and no judge model available",
            )
            reports.append(report)
            continue

        try:
            issue = beads.show(ticket_id)
            messages = transcript(conn_string, ticket_id)
            judged = judge_progress(model, issue, messages)
        except Exception as e:  # noqa: BLE001 -- a judge failure must not
            # take down the scheduled check for every other agent.
            report.update(verdict="idle", reasoning=f"idle {int(idle)}s; judge failed: {e}")
            reports.append(report)
            continue

        report.update(
            verdict="looping" if judged["verdict"] == "looping" else "idle",
            reasoning=judged["reasoning"],
        )
        reports.append(report)

    return reports
