"""The classifier response parser tolerates the shapes a chatty model
actually emits, and still fails closed on anything it cannot read. The
retry-once path exists because the 2026-09-19 denial audit found the real
over-strictness was EMPTY responses, not a strict model."""

from harness.classifier import build_classifier_from_model, parse_verdict


def test_plain_json_object():
    verdict = parse_verdict('{"decision": "allow", "reason": "ok"}')
    assert verdict.decision == "allow"
    assert verdict.reason == "ok"


def test_json_in_code_fence():
    verdict = parse_verdict('```json\n{"decision": "deny", "reason": "outside"}\n```')
    assert verdict.decision == "deny"
    assert verdict.reason == "outside"


def test_json_buried_in_prose():
    verdict = parse_verdict('Sure! Verdict: {"decision": "allow", "reason": "fine"}')
    assert verdict.decision == "allow"


def test_empty_response_fails_closed():
    verdict = parse_verdict("")
    assert verdict.decision == "deny"
    assert "unparseable" in verdict.reason


def test_bad_decision_fails_closed():
    verdict = parse_verdict('{"decision": "maybe", "reason": "hmm"}')
    assert verdict.decision == "deny"
    assert "unparseable" in verdict.reason


class _Fake:
    """Returns the scripted responses in order; counts calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        content = self.responses.pop(0)

        class _Response:
            pass

        response = _Response()
        response.content = content
        return response


def test_empty_then_valid_retries_once():
    model = _Fake(["", '{"decision": "allow", "reason": "ok"}'])
    classify = build_classifier_from_model(model)
    verdict = classify("shell_exec", {"command": "ls"})
    assert verdict.decision == "allow"
    assert model.calls == 2


def test_two_unparseable_responses_deny():
    model = _Fake(["", ""])
    classify = build_classifier_from_model(model)
    verdict = classify("write_file", {"path": "x", "content": "y"})
    assert verdict.decision == "deny"
    assert model.calls == 2


def test_valid_response_is_not_retried():
    model = _Fake(['{"decision": "allow", "reason": "ok"}'])
    classify = build_classifier_from_model(model)
    classify("shell_exec", {"command": "ls"})
    assert model.calls == 1


def test_escalates_to_primary_when_still_unparseable():
    local = _Fake(["", ""])
    frontier = _Fake(['{"decision": "allow", "reason": "escalated"}'])
    classify = build_classifier_from_model(local, escalation_model=frontier)
    verdict = classify("write_file", {"path": "x", "content": "y"})
    assert verdict.decision == "allow"
    assert verdict.reason == "escalated"
    assert local.calls == 2
    assert frontier.calls == 1


def test_no_escalation_when_local_answers():
    local = _Fake(['{"decision": "deny", "reason": "nope"}'])
    frontier = _Fake([])
    classify = build_classifier_from_model(local, escalation_model=frontier)
    verdict = classify("shell_exec", {"command": "x"})
    assert verdict.decision == "deny"
    assert frontier.calls == 0
