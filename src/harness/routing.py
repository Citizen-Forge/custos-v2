"""
Multi-provider routing: an ordered fallback chain per role (e.g.
"worker", "classifier", "product_owner"), with cooldown-on-failure and a
concurrency cap enforced *per provider*, not per role -- two roles
pointed at the same backend genuinely share its hardware concurrency
limit (a local model is the bottleneck regardless of which role is
calling it).

Ported concept from v1's per-task priority lists + rate-limit cooldown
(claude-gateway's router), generalized: v1 baked "task" (general/
classifier/curator) directly into its routing keys; this uses an open
`role` string so Phase 4's per-seat pinning can reuse the same mechanism
without another routing layer on top. "Pinning" a role to one model is
just a chain of length 1 -- no separate concept needed.

Deliberately does NOT implement a "fail open" fallback when an entire
chain is cooling down: `RoutedModel.invoke` raises instead, and relies on
the worker's existing crash-safe retry (a failed ticket stays
`in_progress` and gets retried on the next poll, same mechanism that
makes process-kill recovery safe -- see worker.py). Retrying immediately
against a provider that's cooling down specifically *because* it's rate-
limited would defeat the cooldown's purpose.
"""

import threading
import time
from dataclasses import dataclass

from .providers import ProviderConfig, build_chat_model

DEFAULT_COOLDOWN_SECONDS = 60.0


class AllProvidersCoolingDown(Exception):
    pass


@dataclass
class _Cooldown:
    until: float = 0.0

    def active(self) -> bool:
        return time.monotonic() < self.until

    def start(self, seconds: float) -> None:
        self.until = time.monotonic() + seconds


class ConcurrencyGate:
    """One semaphore per provider *name*, shared across every role that
    resolves to that provider."""

    def __init__(self):
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()

    def acquire(self, cfg: ProviderConfig) -> threading.Semaphore:
        with self._lock:
            if cfg.name not in self._semaphores:
                self._semaphores[cfg.name] = threading.Semaphore(cfg.concurrency_limit)
            return self._semaphores[cfg.name]


class RoutingTable:
    def __init__(self, chains: dict[str, list[ProviderConfig]]):
        self._chains = chains
        self._cooldowns: dict[str, _Cooldown] = {}

    def chain_for(self, role: str) -> list[ProviderConfig]:
        try:
            return self._chains[role]
        except KeyError:
            raise KeyError(f"no provider chain configured for role {role!r}") from None

    def is_cooling_down(self, cfg: ProviderConfig) -> bool:
        return self._cooldowns.get(cfg.name, _Cooldown()).active()

    def report_failure(self, cfg: ProviderConfig, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS) -> None:
        self._cooldowns.setdefault(cfg.name, _Cooldown()).start(cooldown_seconds)


class RoutedModel:
    """A `.invoke()`-compatible model that resolves a provider from a
    RoutingTable on *every* call, not once at construction -- so
    failover/cooldown state observed between calls actually changes which
    backend gets used. Duck-types as a plain chat model: pass it directly
    to `graph.build_graph_from_model` as the `model` argument."""

    def __init__(self, role: str, routing: RoutingTable, gate: ConcurrencyGate, tools=None, model_factory=build_chat_model):
        self._role = role
        self._routing = routing
        self._gate = gate
        self._tools = tools
        self._model_factory = model_factory
        self._model_cache: dict[str, object] = {}

    def _model_for(self, cfg: ProviderConfig):
        if cfg.name not in self._model_cache:
            model = self._model_factory(cfg)
            if self._tools:
                model = model.bind_tools(self._tools)
            self._model_cache[cfg.name] = model
        return self._model_cache[cfg.name]

    def invoke(self, messages):
        chain = self._routing.chain_for(self._role)
        last_error = None
        attempted_any = False

        for cfg in chain:
            if self._routing.is_cooling_down(cfg):
                continue
            attempted_any = True
            semaphore = self._gate.acquire(cfg)
            with semaphore:
                try:
                    return self._model_for(cfg).invoke(messages)
                except Exception as e:  # noqa: BLE001 -- any provider failure triggers fallback
                    last_error = e
                    self._routing.report_failure(cfg)

        if not attempted_any:
            raise AllProvidersCoolingDown(f"every provider for role {self._role!r} is cooling down")
        raise last_error
