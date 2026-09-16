"""
scripts/run_scheduler.py's orchestration logic -- run_one_cycle is split
out from main()'s infinite loop specifically so this is testable without
running forever. Real job functions (product-owner/overwatch/meta-agent/
verifier sessions) aren't invoked here -- those already have their own
tests against real Postgres/Beads; what's new and worth testing here is
the scheduler's own resilience: one job failing must not stop the rest
of the cycle, or a real transient error (a network blip hitting the
model server) would silently wedge every job after it forever.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_scheduler  # noqa: E402
from run_scheduler import run_one_cycle  # noqa: E402


def test_all_jobs_run_in_order():
    calls = []
    jobs = [
        ("first", lambda conn_string: calls.append("first")),
        ("second", lambda conn_string: calls.append("second")),
        ("third", lambda conn_string: calls.append("third")),
    ]

    run_one_cycle("fake-conn-string", jobs=jobs)

    assert calls == ["first", "second", "third"]


def test_a_failing_job_does_not_stop_the_rest_of_the_cycle():
    calls = []

    def failing_job(conn_string):
        raise RuntimeError("simulated real-world failure (e.g. model server unreachable)")

    jobs = [
        ("before", lambda conn_string: calls.append("before")),
        ("boom", failing_job),
        ("after", lambda conn_string: calls.append("after")),
    ]

    run_one_cycle("fake-conn-string", jobs=jobs)  # must not raise

    assert calls == ["before", "after"]


def test_the_escalation_queue_is_worked_before_the_other_sessions():
    """A ticket parked for a human is its project's front, and the front gate
    holds every other ticket in that project until it is dealt with -- so the
    job that can clear it must not sit behind four others. Found live
    2026-09-16: workspace-9jg.1.1 held 46 dispatchable tickets for 3.5h while
    the cycle was still several jobs away from ever reaching escalations."""
    names = [name for name, _ in run_scheduler.JOBS]

    assert names[0] == "escalations"


def test_an_idle_cycle_skips_the_jobs_that_would_cost_a_session(monkeypatch):
    """Each model-backed job is a full agent session even when it finds
    nothing, which is how the account drained while the board was wedged."""
    monkeypatch.setattr(run_scheduler, "_nothing_to_do", lambda conn_string: True)
    calls = []
    jobs = [
        ("escalations", lambda cs: calls.append("escalations")),
        ("product_owner", lambda cs: calls.append("product_owner")),
        ("overwatch", lambda cs: calls.append("overwatch")),
        ("meta_agent", lambda cs: calls.append("meta_agent")),
        ("verifier", lambda cs: calls.append("verifier")),
        ("progress", lambda cs: calls.append("progress")),
    ]

    run_one_cycle("fake-conn-string", jobs=jobs)

    # The jobs that guard themselves still run: asking escalations about an
    # empty queue, or the verifier about candidates, costs no model call.
    assert calls == ["escalations", "verifier", "progress"]


def test_a_cycle_with_work_runs_every_job(monkeypatch):
    monkeypatch.setattr(run_scheduler, "_nothing_to_do", lambda conn_string: False)
    calls = []
    jobs = [
        ("product_owner", lambda cs: calls.append("product_owner")),
        ("overwatch", lambda cs: calls.append("overwatch")),
    ]

    run_one_cycle("fake-conn-string", jobs=jobs)

    assert calls == ["product_owner", "overwatch"]


def test_a_cycle_that_cannot_gate_anything_never_pays_for_the_lookup(monkeypatch):
    """The lookup reads the board, so a caller passing jobs the scheduler
    cannot gate must not trigger it at all."""

    def boom(conn_string):
        raise AssertionError("must not be consulted")

    monkeypatch.setattr(run_scheduler, "_nothing_to_do", boom)
    calls = []

    run_one_cycle("fake-conn-string", jobs=[("first", lambda cs: calls.append("first"))])

    assert calls == ["first"]


def test_all_jobs_receive_the_connection_string():
    received = []
    jobs = [("job", lambda conn_string: received.append(conn_string))]

    run_one_cycle("the-real-conn-string", jobs=jobs)

    assert received == ["the-real-conn-string"]
