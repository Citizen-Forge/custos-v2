"""
Flat registry of available model providers, each tagged with a cost tier
(0-100, matching settings.py's cost slider). User's own architecture
call, 2026-08-29: replaces the idea of a human pre-deciding one fixed
ordered fallback chain per role (routing.py's cooldown/retry-on-failure
logic stays as-is -- an actual provider outage is a different concern
from "which tier to use for cost reasons," not something this replaces).

Configured via MODEL_REGISTRY, a JSON array, one entry per provider:
    [{"name": "local-qwen", "base_url": "...", "model": "...", "cost_tier": 0},
     {"name": "frontier-x", "base_url": "...", "model": "...", "api_key": "...", "cost_tier": 80}]
Defaults to a single local entry (cost_tier 0) built from LOCAL_MODEL_*
if MODEL_REGISTRY is unset, so today's real setup -- only the local
model configured -- needs zero new configuration to keep working
exactly as before.

Scaffolding, not fully wired to dynamic per-call routing yet -- the
user's own explicit priority ("to start with I'll probably only have
the local provider configured so this won't be as important... but we
need to make sure the infrastructure is in place"). product_owner.py's
check_model_options tool gives an agent visibility into the registry and
the current slider; a session doesn't yet re-route worker calls to a
different-tier provider mid-ticket based on that. Building deeper
dynamic routing against a registry of one real provider would be
unverifiable today -- the natural next step once a second real provider
exists to actually choose between.
"""

import json
import os

from .providers import ProviderConfig


def load_registry() -> list[ProviderConfig]:
    raw = os.environ.get("MODEL_REGISTRY")
    if not raw:
        return [
            ProviderConfig(
                name="local",
                base_url=os.environ.get("LOCAL_MODEL_BASE_URL", "http://host.docker.internal:11434/v1"),
                model=os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct"),
                cost_tier=0,
            )
        ]
    entries = json.loads(raw)
    return [
        ProviderConfig(
            name=e["name"],
            base_url=e["base_url"],
            model=e["model"],
            api_key=e.get("api_key"),
            cost_tier=e.get("cost_tier", 0),
        )
        for e in entries
    ]


def providers_at_or_below(registry: list[ProviderConfig], slider: int) -> list[ProviderConfig]:
    """Providers whose cost_tier fits within the current slider setting,
    most-capable-that-still-fits first -- what a caller should choose
    from. Empty if nothing in the registry fits (e.g. slider at 0 but
    every configured provider has cost_tier > 0) -- a real, valid state
    to surface to a caller, not an error to hide."""
    eligible = [p for p in registry if p.cost_tier <= slider]
    return sorted(eligible, key=lambda p: -p.cost_tier)
