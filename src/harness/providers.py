"""
Provider abstraction for OpenAI-compatible chat endpoints (Ollama, vLLM,
OpenAI itself, Gemini's /v1beta/openai compat layer, etc).

Ported concept from Custos v1's `OpenAICompatibleProvider`
(claude-gateway/src/providers/openai-compatible.ts) — that code was already
provider-agnostic plumbing, not tied to Claude Code, so it survives the
rewrite. Same convention kept: `base_url` includes the version prefix
(".../v1", or Gemini's ".../v1beta/openai") to match how the underlying
SDKs configure `base_url` directly.
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    # Local model backends default to 1 — this is the hard constraint that
    # drove the whole harness choice (Phase 1, PLAN.md). Frontier providers
    # can override this higher once Phase 2 adds routing.
    concurrency_limit: int = 1
    # Caught live 2026-08-29: nothing here capped response length, and this
    # project's reasoning-heavy local model generated 5000+ tokens (~10
    # minutes at ~9 tok/s) for what should have been a short JSON verdict
    # from reviewer.py, with no sign of stopping on its own. `None`
    # preserves the old "let the API/model decide" behavior for any caller
    # that doesn't set this — but every entrypoint script now does (see
    # scripts/run_*.py), since unbounded generation is a real
    # reliability/cost risk here, not a hypothetical one.
    max_tokens: int | None = None
    # Cost/capability tier, 0-100 (0=free/local, 100=most expensive) --
    # added 2026-08-29 per the user's own architecture call: rather than
    # a human pre-deciding one fixed ordered fallback chain, the
    # product-owner should be able to choose which provider to use per
    # piece of work, steered by a single system-wide slider (settings.py)
    # rather than needing separate policy per role. See model_registry.py.
    cost_tier: int = 0


def build_chat_model(cfg: ProviderConfig) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key or "not-needed",
        max_tokens=cfg.max_tokens,
    )
