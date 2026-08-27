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
