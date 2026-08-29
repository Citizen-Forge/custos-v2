"""
Flat provider registry with cost tiers. No real Postgres needed --
purely env-var-driven config parsing and in-memory filtering.
"""

from harness import model_registry


def test_default_registry_is_a_single_local_provider_at_cost_tier_zero(monkeypatch):
    monkeypatch.delenv("MODEL_REGISTRY", raising=False)
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "test-model")

    registry = model_registry.load_registry()

    assert len(registry) == 1
    assert registry[0].name == "local"
    assert registry[0].model == "test-model"
    assert registry[0].cost_tier == 0


def test_configured_registry_parses_multiple_providers(monkeypatch):
    monkeypatch.setenv(
        "MODEL_REGISTRY",
        '[{"name": "cheap", "base_url": "http://a", "model": "m1", "cost_tier": 0}, '
        '{"name": "pricey", "base_url": "http://b", "model": "m2", "cost_tier": 80, "api_key": "k"}]',
    )

    registry = model_registry.load_registry()

    assert len(registry) == 2
    assert registry[0].name == "cheap"
    assert registry[1].cost_tier == 80
    assert registry[1].api_key == "k"


def test_providers_at_or_below_filters_and_orders_most_capable_first():
    cheap = model_registry.ProviderConfig(name="cheap", base_url="a", model="m", cost_tier=0)
    mid = model_registry.ProviderConfig(name="mid", base_url="b", model="m", cost_tier=50)
    pricey = model_registry.ProviderConfig(name="pricey", base_url="c", model="m", cost_tier=90)
    registry = [cheap, mid, pricey]

    result = model_registry.providers_at_or_below(registry, slider=60)

    assert result == [mid, cheap]  # most-capable-that-still-fits first, pricey excluded


def test_providers_at_or_below_empty_when_nothing_fits():
    pricey = model_registry.ProviderConfig(name="pricey", base_url="c", model="m", cost_tier=90)

    result = model_registry.providers_at_or_below([pricey], slider=10)

    assert result == []
