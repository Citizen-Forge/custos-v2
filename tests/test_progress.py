"""Stall detection that isn't a timeout.

The rule this has to respect: elapsed time alone never condemns an agent.
Local inference is slow and a long gap is normal, so the cheap signal is
"has it taken any graph step at all", and only past that does a model get
asked whether the steps are actually going anywhere.
"""

from datetime import datetime, timedelta, timezone

import pytest

from harness import beads, progress


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()


class FakeModel:
    """Stands in for the judge. Records what it was asked so a test can
    assert the model is NOT consulted on the cheap path."""

    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)

        class R:
            pass

        r = R()
        r.content = self.content
        return r


def _ago(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


# -- the free signal --------------------------------------------------


def test_idle_seconds_measures_from_last_checkpoint():
    assert progress.idle_seconds(_ago(120)) == pytest.approx(120, abs=5)


def test_idle_seconds_is_none_before_any_checkpoint():
    """A just-started agent has no checkpoint yet. That is not a stall."""
    assert progress.idle_seconds(None) is None


def test_idle_seconds_tolerates_a_malformed_timestamp():
    assert progress.idle_seconds("not-a-timestamp") is None


def test_last_activity_is_empty_for_no_threads():
    assert progress.last_activity(None, []) == {}


# -- judging ----------------------------------------------------------


def test_judge_reports_looping_when_the_model_says_so():
    model = FakeModel('{"verdict": "looping", "reasoning": "same command five times"}')
    result = progress.judge_progress(model, {"title": "t", "description": "d"}, [])
    assert result["verdict"] == "looping"


def test_judge_fails_open_to_progressing_on_unparseable_output():
    """Asymmetric on purpose: a false 'looping' interrupts real work,
    a missed one is caught next cycle. Note this is the opposite of
    verifier.py, which fails closed."""
    model = FakeModel("not json at all")
    result = progress.judge_progress(model, {"title": "t", "description": "d"}, [])
    assert result["verdict"] == "progressing"


def test_judge_rejects_an_unexpected_verdict_value():
    model = FakeModel('{"verdict": "exploded", "reasoning": "x"}')
    result = progress.judge_progress(model, {"title": "t", "description": "d"}, [])
    assert result["verdict"] == "progressing"


# -- the check as a whole ---------------------------------------------


def test_no_running_agents_is_not_an_error(monkeypatch):
    monkeypatch.setattr("harness.dispatcher.running_agents", lambda: [])
    assert progress.check_running_agents(None, "postgresql://unused") == []


def test_recent_activity_never_calls_the_model(monkeypatch):
    """The whole point of the cheap signal: a healthy roster costs no
    inference at all."""
    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "s", "ticket_id": "t1", "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {"t1": _ago(60)})
    model = FakeModel('{"verdict": "looping", "reasoning": "should never be asked"}')

    reports = progress.check_running_agents(None, "postgresql://unused", model=model)

    assert reports[0]["verdict"] == "progressing"
    assert model.calls == [], "model must not be consulted for a recently-active agent"


def test_a_slow_agent_is_not_reported_as_stalled(monkeypatch):
    """25 minutes with no step is well within normal for slow local
    inference, and must not raise anything."""
    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "s", "ticket_id": "t1", "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {"t1": _ago(1500)})

    reports = progress.check_running_agents(None, "postgresql://unused", model=None)

    assert reports[0]["verdict"] == "progressing"


def test_long_idle_without_a_model_is_reported_as_idle(monkeypatch):
    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "s", "ticket_id": "t1", "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {"t1": _ago(7200)})

    reports = progress.check_running_agents(None, "postgresql://unused", model=None)

    assert reports[0]["verdict"] == "idle"
    assert reports[0]["idle_seconds"] == pytest.approx(7200, abs=30)


def test_never_checkpointed_agent_is_not_stalled(monkeypatch):
    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "s", "ticket_id": "t1", "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {})

    reports = progress.check_running_agents(None, "postgresql://unused", model=None)

    assert reports[0]["verdict"] == "progressing"


def test_a_failing_judge_does_not_break_the_check(monkeypatch):
    """One bad transcript read must not take down the scheduled check for
    every other agent."""
    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "s", "ticket_id": "t1", "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {"t1": _ago(7200)})

    def boom(*a, **k):
        raise RuntimeError("transcript unavailable")

    monkeypatch.setattr(progress, "transcript", boom)
    model = FakeModel('{"verdict": "progressing", "reasoning": "x"}')

    reports = progress.check_running_agents(None, "postgresql://unused", model=model)

    assert reports[0]["verdict"] == "idle"
    assert "judge failed" in reports[0]["reasoning"]


def test_check_reports_never_mutate_the_ticket(monkeypatch):
    """check_running_agents is reporting only -- flagging parks a ticket
    AND frees the seat, which would be a timeout by another name."""
    project = beads.create("progress proj", "d", issue_type="epic", priority=1)
    story = beads.create("progress story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "seat-p")
    beads.claim(story["id"], actor="seat-p")

    monkeypatch.setattr(
        "harness.dispatcher.running_agents",
        lambda: [{"seat_id": "seat-p", "ticket_id": story["id"], "title": "T"}],
    )
    monkeypatch.setattr(progress, "last_activity", lambda conn, ids: {story["id"]: _ago(9999)})

    progress.check_running_agents(None, "postgresql://unused", model=None)

    current = beads.show(story["id"])
    assert beads.is_flagged_for_human(current) is False
    assert current["status"] == "in_progress"
