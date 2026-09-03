"""
Working a harness-improvement ticket through self-modification.

A ticket whose project targets the harness's own source cannot be done by
an ordinary agent: agents are rooted in their project's workspace and the
harness source is deliberately not in it. Rather than handing such a
ticket to a seat that will fail -- which is exactly what happened on
2026-09-01, costing about twelve hours of a seat searching for a file it
could never open -- dispatch routes it here.

The agent's job becomes proposing a diff against an isolated checkout
(self_modifier.py). Everything after that is the trusted loop's:
sandboxing and deployment run in sandbox-runner, the only service holding
the Docker socket. So routing a ticket here grants the agent no new
capability at all; it just points it at the one route that can actually
work.

The ticket is left open and claimed while the proposal moves through the
pipeline, and closed by the loop once its change is deployed -- so the
board reflects real state rather than "an agent talked about it".
"""

import logging

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

from . import beads, self_mod, self_modifier
from .routing import ConcurrencyGate, RoutedModel, RoutingTable

log = logging.getLogger("self-mod-ticket")

BRIEF = """You are improving the harness's own source code, working from this ticket.

Ticket {ticket_id}: {title}

{description}

Acceptance criteria: {acceptance_criteria}

You are editing an isolated checkout of the harness, not the running system. Make the change
there and propose it. It will then be tested in a sandbox and reviewed by a separate agent
before anything is deployed -- an unclean test run blocks deployment no matter how good the
change looks, so there is nothing to gain from overstating it.

Keep the change as small as it can be while genuinely satisfying the acceptance criteria, and
describe honestly what you changed and what you did not."""


def work_ticket(conn_string: str, routing: RoutingTable, gate: ConcurrencyGate, issue: dict) -> int | None:
    """Run one self-modification session for a ticket. Returns the
    proposal id if one was raised, else None."""
    ticket_id = issue["id"]

    with psycopg.connect(conn_string, autocommit=True) as conn:
        self_mod.init_table(conn)
        before = {p["id"] for p in self_mod.list_by_status(conn, "pending")}
        tools = self_modifier.build_tools(conn)
        agent_model = RoutedModel(self_modifier.ROLE, routing, gate, tools=tools)

        brief = BRIEF.format(
            ticket_id=ticket_id,
            title=issue.get("title", ""),
            description=issue.get("description", ""),
            acceptance_criteria=beads.acceptance_criteria(issue) or "(none set)",
        )

        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()
            self_modifier.run_self_mod_session(
                agent_model, tools, checkpointer, brief=brief, thread_id=ticket_id
            )

        # Attribute whatever proposal the session raised to this ticket,
        # so the pipeline can report back to the board when it deploys.
        raised = [p for p in self_mod.list_by_status(conn, "pending") if p["id"] not in before]
        if not raised:
            log.warning("%s: self-modification session raised no proposal", ticket_id)
            beads.append_note(
                ticket_id,
                "self-modification session ended without proposing a change",
            )
            return None

        proposal_id = raised[0]["id"]
        conn.execute(
            "UPDATE self_mod_proposals SET ticket_id=%s WHERE id=%s", (ticket_id, proposal_id)
        )
        beads.append_note(
            ticket_id, f"self-modification proposal #{proposal_id} raised; awaiting sandbox and review"
        )
        log.info("%s: raised proposal #%s", ticket_id, proposal_id)
        return proposal_id


def report_deployment(conn, proposal: dict) -> None:
    """Close the ticket a deployed proposal came from.

    This is what stops self-modification happening somewhere the board
    cannot see. The close reason carries the proposal id and the review
    verdict, so the ticket satisfies the completion gate with real
    evidence rather than an agent's say-so."""
    ticket_id = proposal.get("ticket_id")
    if not ticket_id:
        return
    summary = (
        f"deployed via self-modification proposal #{proposal['id']}: "
        f"{(proposal.get('description') or '')[:300]}"
    )
    try:
        beads.set_metadata(ticket_id, "completion_summary", summary[:2000])
        beads.append_note(
            ticket_id,
            f"{summary}\nreview: {proposal.get('review_verdict')} -- "
            f"{(proposal.get('review_reasoning') or '')[:300]}",
        )
        beads.close(ticket_id, reason=summary[:500])
        log.info("%s: closed after deploying proposal #%s", ticket_id, proposal["id"])
    except Exception:
        log.exception("could not report deployment onto %s", ticket_id)
