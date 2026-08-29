"""
Slack notifier: request construction and graceful no-op behavior, tested
against a mocked HTTP layer -- no real Slack workspace/credentials were
available in this environment to test against live (see slack.py's
module docstring). What's proven here: every function fails soft (no
exception, safe empty/False return) whether unconfigured or on a real
API-level failure, and the real request shape (auth header, channel,
payload) is correct -- not that Slack's real API actually accepts it.
"""

from unittest.mock import MagicMock, patch

from harness import slack


def test_post_message_noops_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)

    with patch("harness.slack.httpx.post") as mock_post:
        result = slack.post_message("hello")

    assert result is False
    mock_post.assert_not_called()  # never even attempts a real request


def test_recent_messages_returns_empty_list_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_CHANNEL_ID", raising=False)

    with patch("harness.slack.httpx.get") as mock_get:
        result = slack.recent_messages()

    assert result == []
    mock_get.assert_not_called()


def test_post_message_sends_correct_request_shape(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")

    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}

    with patch("harness.slack.httpx.post", return_value=mock_response) as mock_post:
        result = slack.post_message("welcome to the team")

    assert result is True
    call = mock_post.call_args
    assert call.args[0] == f"{slack.API_BASE}/chat.postMessage"
    assert call.kwargs["headers"]["Authorization"] == "Bearer xoxb-test-token"
    assert call.kwargs["json"] == {"channel": "C12345", "text": "welcome to the team"}


def test_post_message_returns_false_on_api_level_error(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")

    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

    with patch("harness.slack.httpx.post", return_value=mock_response):
        result = slack.post_message("hello")

    assert result is False  # Slack's own {"ok": false} shape, not an HTTP-level error


def test_post_message_returns_false_on_network_failure(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")

    import httpx

    with patch("harness.slack.httpx.post", side_effect=httpx.ConnectError("refused")):
        result = slack.post_message("hello")

    assert result is False  # never raises -- Slack is optional infrastructure


def test_recent_messages_extracts_text_from_real_response_shape(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C12345")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "messages": [{"text": "newest message"}, {"text": "older message"}],
    }

    with patch("harness.slack.httpx.get", return_value=mock_response) as mock_get:
        result = slack.recent_messages(limit=5)

    assert result == ["newest message", "older message"]
    assert mock_get.call_args.kwargs["params"]["limit"] == 5
