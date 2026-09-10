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


# A model call that never returns takes the whole project down with it.
# The dispatcher runs one agent at a time, so a worker blocked on a socket
# holds that slot indefinitely: no exception, no outcome logged, nothing
# for the retry logic to act on, and every ready ticket simply waits.
# Observed twice on 2026-09-10 -- a run made a successful call at
# 10:35:26 and then sat at 0% CPU for 65 minutes with the model server
# idle, until the harness was restarted by hand.
#
# Generous rather than tight: local inference on a 30B MoE is slow, and a
# long generation is normal here (see the no-agent-timeout note -- stall
# timeouts on the AGENT are wrong). This bounds one HTTP request, not the
# agent's thinking. On expiry the call raises, work_one_ticket logs the
# failure, and the dispatcher retries and eventually flags -- all of which
# are recoverable states, unlike silence.
_REQUEST_TIMEOUT_S = 900
_MAX_RETRIES = 2


def build_chat_model(cfg: ProviderConfig) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key or "not-needed",
        max_tokens=cfg.max_tokens,
        timeout=_REQUEST_TIMEOUT_S,
        max_retries=_MAX_RETRIES,
    )
