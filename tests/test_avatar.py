"""
Realistic agent avatars via Gemini's image generation API. Tested here
against a mocked HTTP layer (fast, no real API cost/latency per test
run) -- the mocked response shapes below match what a real call was
confirmed to return live (see avatar.py's module docstring, fixed
2026-08-30 after the first live call revealed the original guessed
shape was wrong). What's proven here: every function fails soft (no
exception, None return) whether unconfigured or on a real API-level
failure, and a well-formed response gets decoded and saved correctly.
"""

import base64
from unittest.mock import MagicMock, patch

from harness import avatar


def test_generate_avatar_noops_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))

    with patch("harness.avatar.httpx.post") as mock_post:
        result = avatar.generate_avatar("some-seat", "a friendly person")

    assert result is None
    mock_post.assert_not_called()


def test_generate_avatar_saves_a_well_formed_response(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))

    fake_image_bytes = b"not a real jpeg but bytes are bytes for this test"
    fake_b64 = base64.b64encode(fake_image_bytes).decode()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "steps": [{"type": "model_output", "content": [{"type": "output_image", "data": fake_b64}]}]
    }

    with patch("harness.avatar.httpx.post", return_value=mock_response) as mock_post:
        result = avatar.generate_avatar("some-seat", "a friendly person who loves jazz")

    assert result == "avatars/some-seat.jpg"
    call = mock_post.call_args
    assert call.kwargs["headers"]["x-goog-api-key"] == "fake-key"
    assert "jazz" in call.kwargs["json"]["input"][0]["text"]
    assert call.kwargs["json"]["response_format"]["type"] == "image"

    saved_path = tmp_path / "avatars" / "some-seat.jpg"
    assert saved_path.read_bytes() == fake_image_bytes


def test_generate_avatar_returns_none_when_response_has_no_image(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))

    mock_response = MagicMock()
    mock_response.json.return_value = {"steps": []}

    with patch("harness.avatar.httpx.post", return_value=mock_response):
        result = avatar.generate_avatar("some-seat", "a description")

    assert result is None
    assert not (tmp_path / "avatars" / "some-seat.jpg").exists()


def test_generate_avatar_returns_none_on_network_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))

    import httpx as real_httpx

    with patch("harness.avatar.httpx.post", side_effect=real_httpx.ConnectError("refused")):
        result = avatar.generate_avatar("some-seat", "a description")

    assert result is None  # never raises -- avatar generation is optional


def test_avatar_path_returns_none_when_never_generated(monkeypatch, tmp_path):
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))
    assert avatar.avatar_path("never-generated-seat") is None


def test_avatar_path_returns_real_path_after_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(avatar, "WORKSPACE_ROOT", str(tmp_path))

    fake_b64 = base64.b64encode(b"fake jpeg bytes").decode()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "steps": [{"type": "model_output", "content": [{"type": "image", "data": fake_b64}]}]
    }

    with patch("harness.avatar.httpx.post", return_value=mock_response):
        avatar.generate_avatar("some-seat", "a description")

    path = avatar.avatar_path("some-seat")
    assert path is not None
    assert path.endswith("some-seat.jpg")
