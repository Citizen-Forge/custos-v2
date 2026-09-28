"""The classifier prompt must not carry a whole file: large tool-args are
truncated before they reach the model.

Found 2026-09-28: a write_file's content made every classifier call ~2050
prompt tokens and ~500 output tokens, ~11-15s on the 3070 Ti, so a board full
of agents hammered the card."""

from harness.classifier import _ARGS_MAX_CHARS, _short_args


def test_large_tool_args_are_truncated():
    text = _short_args({"path": "src/a.gd", "content": "x" * 5000})

    assert "src/a.gd" in text, "the path (what the judgement turns on) is kept"
    assert "truncated" in text
    assert len(text) <= _ARGS_MAX_CHARS + 80


def test_small_tool_args_are_untouched():
    text = _short_args({"command": "bd ready"})
    assert text == str({"command": "bd ready"})


def test_decision_variants_are_normalised():
    from harness.classifier import parse_verdict

    assert parse_verdict('{"decision": "denied", "reason": "x"}').decision == "deny"
    assert parse_verdict('{"decision": "allowed", "reason": "x"}').decision == "allow"
    assert parse_verdict('{"decision": "blocked", "reason": "x"}').decision == "deny"
    assert parse_verdict('{"decision": "ALLOW", "reason": "x"}').decision == "allow"
