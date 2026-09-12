"""
Proves routing.py's actual behaviors, not just that it constructs:
fallback to the next provider on failure, cooldown causing a provider to
be skipped, cooldown expiring and the provider being tried again, a fully
cooled-down chain raising instead of silently retrying a rate-limited
provider, and the concurrency gate actually serializing calls rather than
just holding a number nobody checks.

Uses fake providers/models throughout -- no real endpoint needed, no
Ollama dependency.
"""

import threading
import time

import pytest

from harness.providers import ProviderConfig
from harness.routing import AllProvidersCoolingDown, ConcurrencyGate, RoutedModel, RoutingTable


class FakeModel:
    def __init__(self, name, fail=False, delay=0.0, calls=None):
        self.name = name
        self.fail = fail
        self.delay = delay
        self.calls = calls if calls is not None else []

    def invoke(self, messages):
        start = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        self.calls.append((self.name, start, time.monotonic()))
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        return f"response from {self.name}"


def _cfg(name, concurrency_limit=1):
    return ProviderConfig(name=name, base_url="http://fake", model="fake", concurrency_limit=concurrency_limit)


def test_dynamic_seat_falls_back_to_default_role_chain():
    worker_chain = [_cfg("shared-local")]
    routing = RoutingTable({"worker": worker_chain}, default_role="worker")

    # "aria-ux" was never registered -- a seat the product-owner created
    # at runtime -- but should still resolve to the shared default chain.
    assert routing.chain_for("aria-ux") == worker_chain
    assert routing.chain_for("worker") == worker_chain


def test_unknown_role_without_default_still_raises():
    routing = RoutingTable({"worker": [_cfg("shared-local")]})  # no default_role
    with pytest.raises(KeyError):
        routing.chain_for("aria-ux")


def test_falls_back_to_next_provider_on_failure():
    primary = _cfg("primary")
    backup = _cfg("backup")
    models = {"primary": FakeModel("primary", fail=True), "backup": FakeModel("backup")}

    routing = RoutingTable({"worker": [primary, backup]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: models[cfg.name])

    result = routed.invoke("hi")

    assert result == "response from backup"
    assert routing.is_cooling_down(primary)
    assert not routing.is_cooling_down(backup)


def test_cooldown_skips_provider_until_it_expires():
    primary = _cfg("primary")
    backup = _cfg("backup")
    models = {"primary": FakeModel("primary"), "backup": FakeModel("backup")}
    routing = RoutingTable({"worker": [primary, backup]})

    routing.report_failure(primary, cooldown_seconds=0.1)

    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: models[cfg.name])
    assert routed.invoke("hi") == "response from backup"  # primary skipped while cooling down

    time.sleep(0.15)
    assert routed.invoke("hi") == "response from primary"  # cooldown expired, primary tried again


def test_all_providers_cooling_down_raises():
    primary = _cfg("primary")
    routing = RoutingTable({"worker": [primary]})
    routing.report_failure(primary, cooldown_seconds=60)

    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: FakeModel(cfg.name))

    with pytest.raises(AllProvidersCoolingDown):
        routed.invoke("hi")


def test_concurrency_gate_serializes_calls_to_the_same_provider():
    cfg = _cfg("solo", concurrency_limit=1)
    calls = []
    routing = RoutingTable({"worker": [cfg]})
    routed = RoutedModel(
        "worker", routing, ConcurrencyGate(),
        model_factory=lambda c: FakeModel(c.name, delay=0.15, calls=calls),
    )

    threads = [threading.Thread(target=routed.invoke, args=("hi",)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 2
    (_, start_a, end_a), (_, start_b, end_b) = sorted(calls, key=lambda c: c[1])
    # with concurrency_limit=1, the second call cannot start until the
    # first one's finished -- if the gate weren't enforcing this, both
    # calls' 0.15s sleeps would overlap and start_b would be ~0, not >= end_a.
    assert start_b >= end_a


class _HTTPError(RuntimeError):
    """Stands in for openai's error types, which carry status_code."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class RejectingModel:
    """Fails every call with a given exception, and counts the attempts."""

    def __init__(self, exc):
        self.exc = exc
        self.attempts = 0

    def invoke(self, messages):
        self.attempts += 1
        raise self.exc


def test_content_rejection_does_not_cool_the_provider_down():
    # A 400 means the request was unacceptable, not that the backend is
    # unwell. Cooling down here is what flagged four tickets on 2026-09-11
    # as having produced nothing -- see the comment in routing.py.
    primary = _cfg("primary")
    overflow = _HTTPError(
        "Error code: 400 - request (66117 tokens) exceeds the available "
        "context size (65536 tokens)",
        status_code=400,
    )
    routing = RoutingTable({"worker": [primary]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: RejectingModel(overflow))

    with pytest.raises(_HTTPError):
        routed.invoke("hi")

    assert not routing.is_cooling_down(primary)


def test_a_retry_after_a_content_rejection_still_reaches_the_provider():
    # The point of not cooling down: the *next* attempt is a real attempt.
    # Before this, it died instantly with AllProvidersCoolingDown without a
    # single call leaving the harness.
    primary = _cfg("primary")
    model = RejectingModel(_HTTPError("Error code: 400 - bad request", status_code=400))
    routing = RoutingTable({"worker": [primary]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: model)

    for _ in range(3):
        with pytest.raises(_HTTPError):
            routed.invoke("hi")

    assert model.attempts == 3  # not 1 attempt then two instant cooldown failures


def test_malformed_tool_call_is_treated_as_content_despite_being_a_500():
    # llama.cpp returns 500 when the model emits a tool-call argument long
    # enough to be truncated mid-string. The server is fine; we sent junk.
    primary = _cfg("primary")
    parse_failure = _HTTPError(
        "Error code: 500 - Failed to parse tool call arguments as JSON: "
        "syntax error while parsing value - missing closing quote",
        status_code=500,
    )
    routing = RoutingTable({"worker": [primary]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: RejectingModel(parse_failure))

    with pytest.raises(_HTTPError):
        routed.invoke("hi")

    assert not routing.is_cooling_down(primary)


def test_ordinary_provider_failure_still_cools_down():
    # The rate-limit case this mechanism was built for must keep working.
    primary = _cfg("primary")
    backup = _cfg("backup")
    models = {"primary": FakeModel("primary", fail=True), "backup": FakeModel("backup")}
    routing = RoutingTable({"worker": [primary, backup]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: models[cfg.name])

    assert routed.invoke("hi") == "response from backup"
    assert routing.is_cooling_down(primary)


def test_a_500_without_a_known_marker_still_cools_down():
    primary = _cfg("primary")
    backup = _cfg("backup")
    models = {
        "primary": RejectingModel(_HTTPError("Error code: 500 - internal server error", status_code=500)),
        "backup": FakeModel("backup"),
    }
    routing = RoutingTable({"worker": [primary, backup]})
    routed = RoutedModel("worker", routing, ConcurrencyGate(), model_factory=lambda cfg: models[cfg.name])

    assert routed.invoke("hi") == "response from backup"
    assert routing.is_cooling_down(primary)
