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


def test_all_jobs_receive_the_connection_string():
    received = []
    jobs = [("job", lambda conn_string: received.append(conn_string))]

    run_one_cycle("the-real-conn-string", jobs=jobs)

    assert received == ["the-real-conn-string"]
