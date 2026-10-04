"""
The acceptance-criteria verification loop's judgment substrate -- runs
against a real Beads ticket (like test_welfare_behaviors.py), since the
point is proving verify_ticket reads and reacts to real ticket state, not
a fake. Whether a real model's pass/fail judgment is actually *correct*
needs real inference to validate (see PLAN.md's real-model session log).
"""

import json
import os

import psycopg

from harness import beads, verifications, verifier
from harness.verifier import MAX_VERIFIER_REWORKS, verify_ticket


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Response", (), {"content": self.content})()


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    verifications.init_table(conn)
    return conn


def test_ticket_with_no_acceptance_criteria_is_not_a_candidate():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("no criteria set", "x")
    beads.claim(issue["id"])
    beads.close(issue["id"])

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None
    assert verifications.get_for_issue(conn, issue["id"]) is None


def test_an_answered_ticket_is_never_re_judged():
    """A person's decision ends the question. Without this the verifier judged
    accepted tickets again on their own contaminated commit record, failed
    them, and reopened them -- so the one path that could end this class of
    ticket was undone within minutes. Found live 2026-09-17 on
    workspace-9jg.1.1.1, whose notes read, in order: "human response:
    Accepted as already delivered / no-op by construction. Ruling:
    src/TickLoop.ts is the single canonical fixed-step loop...", then
    "verifier fail #1: The acceptance criteria claim src/FixedStepLoop.ts
    does not exist at HEAD...". The ticket had two more agent runs in front
    of it, for the same refusal."""
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create(
        "already accepted", "x", acceptance_criteria="must do the thing"
    )
    beads.claim(issue["id"])
    beads.flag_for_human(issue["id"], "was this ever delivered?")
    beads.respond_to_human(issue["id"], "accepted as already delivered")

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None
    assert verifications.get_for_issue(conn, issue["id"]) is None


def test_ticket_not_yet_closed_is_not_a_candidate():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("still open", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    # deliberately not closed

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None


def test_well_formed_pass_verdict_gets_recorded():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create(
        "print hello world", "write a script that prints hello world", acceptance_criteria="output is exactly 'hello world'"
    )
    beads.claim(issue["id"])
    beads.append_note(issue["id"], "wrote the script, ran it, output was 'hello world'")
    beads.close(issue["id"], reason="done, verified output matches")

    model = FakeModel(json.dumps({"verdict": "pass", "reasoning": "close reason confirms exact output match"}))
    result = verify_ticket(conn, issue["id"], model)

    # no Custos seat assignment in this test (assigned_seat is unset), so
    # this falls back to Beads' own `assignee` field, which is always
    # populated with the acting actor (DEFAULT_ACTOR here) -- "unknown"
    # is the harder-to-hit fallback for the case where BOTH are somehow
    # empty, not the common case. beads.create()'s own response is a
    # leaner shape than beads.show()'s (no `assignee` key at all) -- look
    # the real value up via show(), the same call verify_ticket itself
    # makes internally, rather than trusting create()'s response shape.
    real_assignee = beads.show(issue["id"])["assignee"]
    assert result == {
        "issue_id": issue["id"],
        "seat_id": real_assignee,
        "verdict": "pass",
        "reasoning": "close reason confirms exact output match",
    }
    stored = verifications.get_for_issue(conn, issue["id"])
    assert stored["verdict"] == "pass"


def test_already_verified_ticket_is_skipped_idempotent():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("already checked", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "pass", "already judged once")

    result = verify_ticket(conn, issue["id"], FakeModel("should never be called"))

    assert result is None
    # untouched -- the original verdict, not overwritten by this no-op call
    assert verifications.get_for_issue(conn, issue["id"])["reasoning"] == "already judged once"


def test_unparseable_response_fails_closed_to_fail():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("bad response test", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])

    result = verify_ticket(conn, issue["id"], FakeModel("not json at all"))

    assert result["verdict"] == "fail"
    assert "unparseable" in result["reasoning"]
    assert verifications.get_for_issue(conn, issue["id"])["verdict"] == "fail"


def test_prompt_includes_real_ticket_evidence():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create(
        "evidence check", "the description text", acceptance_criteria="the specific criteria text"
    )
    beads.claim(issue["id"])
    beads.append_note(issue["id"], "the specific notes text")
    beads.close(issue["id"], reason="the specific close reason text")

    captured = {}

    class CapturingModel:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type("Response", (), {"content": json.dumps({"verdict": "pass", "reasoning": "n/a"})})()

    verify_ticket(conn, issue["id"], CapturingModel())

    assert "the specific criteria text" in captured["prompt"]
    assert "the specific notes text" in captured["prompt"]
    assert "the specific close reason text" in captured["prompt"]


# -- the rework loop --------------------------------------------------
#
# A fail used to be recorded and then parked (reopened only when the test
# suite was objectively broken, otherwise just flagged), so the verifier's
# finding sat unactioned on the board. It now goes back to an agent with
# the finding attached, bounded by MAX_VERIFIER_REWORKS.


def test_failed_verification_requeues_with_the_finding():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("rework me", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "completion_summary", "I did the thing")
    beads.set_metadata(issue["id"], "work_commit", "abc123")
    beads.close(issue["id"], reason="done")

    model = FakeModel(json.dumps({"verdict": "fail", "reasoning": "the thing was not done"}))
    result = verify_ticket(conn, issue["id"], model)

    assert result["verdict"] == "fail"
    current = beads.show(issue["id"])
    assert current["status"] == "open", "a failed ticket must go back in the queue"
    assert beads.is_flagged_for_human(current) is False, "requeued work must be dispatchable"
    meta = current.get("metadata") or {}
    assert int(meta.get("rework_count")) == 1
    assert "the thing was not done" in (meta.get("rework_reason") or "")
    assert not meta.get("completion_summary"), "the old claim must not re-close the ticket"
    assert verifications.get_for_issue(conn, issue["id"])["verdict"] == "fail"


def test_a_pass_lands_the_work_and_a_fail_does_not(monkeypatch):
    """The merge into the integration branch is gated on this verdict. An
    approved ticket's work lands; a rejected one stays on its own branch,
    so what the next ticket starts from is only ever accepted work."""
    landed = []
    monkeypatch.setattr(verifier, "land", lambda ticket_id, conn=None: landed.append(ticket_id) or True)

    conn = _conn()
    beads.ensure_initialized()

    passed = beads.create("lands proj", "x", acceptance_criteria="must do the thing")
    beads.close(passed["id"], reason="done")
    verify_ticket(conn, passed["id"], FakeModel(json.dumps({"verdict": "pass", "reasoning": "ok"})))

    failed = beads.create("does not land", "x", acceptance_criteria="must do the thing")
    beads.close(failed["id"], reason="done")
    verify_ticket(
        conn, failed["id"], FakeModel(json.dumps({"verdict": "fail", "reasoning": "not done"}))
    )

    assert landed == [passed["id"]], "exactly the approved ticket lands"


def test_requeue_is_bounded_then_flags_for_a_human():
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("exhausted", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "rework_count", str(MAX_VERIFIER_REWORKS))
    beads.close(issue["id"], reason="done")

    result = verify_ticket(
        conn, issue["id"], FakeModel(json.dumps({"verdict": "fail", "reasoning": "still wrong"}))
    )

    assert result["verdict"] == "fail"
    assert beads.is_flagged_for_human(beads.show(issue["id"])) is True


def test_legacy_verdict_is_trusted_not_rejudged():
    """A verdict recorded before work_commit existed (NULL) must be left
    alone -- re-judging every settled ticket would reopen it and re-block
    its dependents."""
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("legacy pass", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "pass", "judged before work_commit existed")

    assert verify_ticket(conn, issue["id"], FakeModel("should never be called")) is None


def test_legacy_verdict_is_rejudged_when_requeued():
    """A requeued ticket's old verdict must not block judging its new work,
    even though that verdict predates the work_commit column."""
    conn = _conn()
    beads.ensure_initialized()
    issue = beads.create("legacy rework", "x", acceptance_criteria="must do the thing")
    beads.claim(issue["id"])
    beads.set_metadata(issue["id"], "rework_count", "1")
    beads.set_metadata(issue["id"], "work_commit", "newsha")
    beads.close(issue["id"])
    verifications.record(conn, issue["id"], "some-seat", "fail", "judged before work_commit existed")

    result = verify_ticket(
        conn, issue["id"], FakeModel(json.dumps({"verdict": "pass", "reasoning": "now it does"}))
    )

    assert result["verdict"] == "pass"


# -- the ticket's own verifier command, and earlier verdicts ------------
#
# Found live 2026-10-04 on workspace-o0n.18.14: the fallback model failed it
# twice -- the second time quoting the first word for word -- for "the test
# suite result does not verify the specific acceptance criteria", while the
# suite its criteria named passed 47/0. The verifier now runs that command
# itself, and the model no longer sees earlier verdicts.

CRITERIA = (
    "1. VERIFIER COMMAND: godot --headless --path . --script tests/run_one.gd -- --suite=res://tests/x_test.gd\n"
    "2. ASSERTIONS: the thing works.\n3. TEST-COUNT: '# fail 0'."
)


class CapturingModel:
    def __init__(self, verdict="pass"):
        self.prompt = None
        self.verdict = verdict

    def invoke(self, prompt):
        self.prompt = prompt
        return type("Response", (), {"content": json.dumps({"verdict": self.verdict, "reasoning": "n/a"})})()


def _closed_ticket(title, criteria=CRITERIA, notes=()):
    beads.ensure_initialized()
    issue = beads.create(title, "x", acceptance_criteria=criteria)
    beads.claim(issue["id"])
    for n in notes:
        beads.append_note(issue["id"], n)
    beads.close(issue["id"], reason="done")
    return issue["id"]


def _godot_project(monkeypatch, run_result=None, calls=None):
    monkeypatch.setattr(
        verifier.toolchain, "test_command_for",
        lambda project_id: "godot --headless --path . --script tests/run_tests.gd",
    )
    monkeypatch.setattr(verifier, "_tests_for", lambda issue: None)

    def run(ticket_id, command, timeout=600):
        if calls is not None:
            calls.append(command)
        return dict(run_result)

    monkeypatch.setattr(verifier.workspaces, "run_command_for_ticket", run)


def test_verifier_command_is_read_from_the_criteria():
    assert verifier.verifier_command_in(CRITERIA) == (
        "godot --headless --path . --script tests/run_one.gd -- --suite=res://tests/x_test.gd"
    )
    assert verifier.verifier_command_in("1. VERIFIER COMMAND: `npm test`") == "npm test"
    assert verifier.verifier_command_in("no command here") is None
    assert verifier.verifier_command_in(None) is None


def test_the_tickets_own_suite_result_reaches_the_model(monkeypatch):
    conn = _conn()
    calls = []
    _godot_project(monkeypatch, {"ran": 47, "passed": 47, "failed": 0, "exit": 0, "tail": "# pass 47\n# fail 0"}, calls)
    ticket = _closed_ticket("own suite passes")
    model = CapturingModel()

    result = verify_ticket(conn, ticket, model)

    assert calls == [verifier.verifier_command_in(CRITERIA)]
    assert "THE TICKET'S SUITE PASSED: 47 tests ran and passed" in model.prompt
    assert result["verdict"] == "pass"


def test_a_failing_own_suite_is_a_mechanical_fail(monkeypatch):
    conn = _conn()
    _godot_project(monkeypatch, {"ran": 47, "passed": 45, "failed": 2, "exit": 1, "tail": "# fail 2"})
    ticket = _closed_ticket("own suite fails")

    result = verify_ticket(conn, ticket, FakeModel("should never be called"))

    assert result["verdict"] == "fail"
    assert "verifier command failed" in result["reasoning"]


def test_an_own_suite_that_ran_nothing_is_a_mechanical_fail(monkeypatch):
    conn = _conn()
    _godot_project(monkeypatch, {"ran": 0, "passed": 0, "failed": 0, "exit": 0, "tail": ""})
    ticket = _closed_ticket("own suite empty")

    result = verify_ticket(conn, ticket, FakeModel("should never be called"))

    assert result["verdict"] == "fail"
    assert "ran zero tests" in result["reasoning"]


def test_a_command_for_another_program_is_not_run(monkeypatch):
    conn = _conn()
    calls = []
    _godot_project(monkeypatch, {"ran": 1, "passed": 1, "failed": 0, "exit": 0, "tail": ""}, calls)
    ticket = _closed_ticket("smuggled command", criteria="1. VERIFIER COMMAND: rm -rf /\n2. ASSERTIONS: x")
    model = CapturingModel()

    verify_ticket(conn, ticket, model)

    assert calls == [], "only the project's own test program may be run"
    assert "NOT RUN: rm -rf /" in model.prompt


def test_earlier_verdicts_are_withheld_from_the_model(monkeypatch):
    conn = _conn()
    _godot_project(monkeypatch, {"ran": 3, "passed": 3, "failed": 0, "exit": 0, "tail": ""})
    ticket = _closed_ticket("rejudged afresh", notes=[
        "the agent's own account of the work",
        "verifier fail #1: the file still contains a parsing error",
        "verifier fail #2: the file still contains a parsing error",
    ])
    model = CapturingModel()

    verify_ticket(conn, ticket, model)

    assert "the agent's own account of the work" in model.prompt
    assert "still contains a parsing error" not in model.prompt
    notes = beads.show(ticket).get("notes") or ""
    assert "verifier fail #1" in notes, "the notes themselves are untouched"
