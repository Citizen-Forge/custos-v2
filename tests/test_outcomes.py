"""
Phase 5 groundwork: outcome signals sourced from Beads' own audit trail.
Runs against the real bd CLI.
"""

import uuid

from harness import beads, outcomes


def test_summary_counts_closed_and_refused_correctly():
    beads.ensure_initialized()
    actor = f"test-actor-{uuid.uuid4().hex[:8]}"

    done = beads.create("will be closed", "x")
    beads.claim(done["id"], actor=actor)
    beads.close(done["id"], reason="finished")

    refused = beads.create("will be refused", "x")
    beads.claim(refused["id"], actor=actor)
    beads.flag_for_human(refused["id"], "needs a human call", actor=actor)

    still_open = beads.create("still going", "x")
    beads.claim(still_open["id"], actor=actor)

    result = outcomes.summary(actor)

    assert result["total"] == 3
    assert result["closed"] == 1
    assert result["refused"] == 1
    assert result["still_open"] == 1
    assert "needs a human call" in result["refused_reasons"]
