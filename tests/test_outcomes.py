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


def test_queue_stats_computes_real_average_completion_time():
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    done = beads.create("will be closed", "x")
    beads.claim(done["id"], actor=seat_id)
    beads.close(done["id"], reason="finished")

    result = outcomes.queue_stats(seat_id)

    assert result["sample_size"] == 1
    assert result["avg_completion_seconds"] is not None
    assert result["avg_completion_seconds"] >= 0


def test_queue_stats_reports_none_not_zero_with_no_history():
    beads.ensure_initialized()
    seat_id = f"test-seat-never-completed-{uuid.uuid4().hex[:8]}"

    result = outcomes.queue_stats(seat_id)

    assert result["sample_size"] == 0
    assert result["avg_completion_seconds"] is None
    assert result["estimated_wait_seconds"] is None  # not a misleading 0


def test_queue_stats_counts_ready_and_in_progress_for_the_seat():
    beads.ensure_initialized()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    ready_ticket = beads.create("ready work", "x")
    beads.assign_to_seat(ready_ticket["id"], seat_id)

    in_progress_ticket = beads.create("in progress work", "x")
    beads.assign_to_seat(in_progress_ticket["id"], seat_id)
    beads.claim(in_progress_ticket["id"], actor=seat_id)

    result = outcomes.queue_stats(seat_id)

    assert result["ready"] == 1
    assert result["in_progress"] == 1
