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


def build_chat_model(cfg: ProviderConfig) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key or "not-needed",
    )
