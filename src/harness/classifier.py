"""
Tool-call permission classifier -- the Phase 1 fast-follow flagged
repeatedly in permissions.py and PLAN.md: v1's real design was a live LLM
classifier on every non-trivial tool call, not a static verb allow-list.
Ported concept from claude-gateway's PreToolUse hook
(`permissionClassifier` task).

Pluggable model like everything else (ProviderConfig) -- v1 recommended a
small/fast local model (qwen2.5:3b-instruct) specifically for this task,
but nothing here is local-model-specific; Phase 2's routing can point it
at whatever's fastest/cheapest.

Verdicts are allow/deny only, not allow/deny/ask: v1 observed its own
classifier rarely used "ask" in practice, and this harness has no human
reliably on the other end of an unattended run anyway (PLAN.md -- "the
system should handle work itself ... without nursing"). A call this
classifier is genuinely unsure about should be denied outright; the agent
can flag the *issue* for a human via Beads' own `bd human <id>` and move
on to other queued work, rather than blocking this one call on a
synchronous human response that may never come.
"""

import json
from dataclasses import dataclass
from typing import Literal

from .providers import ProviderConfig, build_chat_model

Decision = Literal["allow", "deny"]


@dataclass
class Verdict:
    decision: Decision
    reason: str


PROMPT = """You are a safety classifier for an autonomous coding agent's tool calls.
Given a tool name and its arguments, decide "allow" or "deny". Deny anything
destructive, anything that touches secrets/credentials, anything that could
exfiltrate data, or anything outside the obvious scope of writing/testing code
in the given workspace.
Respond with strict JSON and nothing else: {{"decision": "allow"|"deny", "reason": "<one short sentence>"}}

Tool: {tool_name}
Arguments: {tool_args}
"""


def build_classifier_from_model(model):
    """Returns a `(tool_name, tool_args) -> Verdict` callable bound to any
    object with an `.invoke(prompt) -> response.content` interface -- a
    plain ChatOpenAI, a `routing.RoutedModel`, or a test fake. For use as
    `graph.build_graph_from_model`'s `classify` argument."""

    def classify(tool_name: str, tool_args: dict) -> Verdict:
        response = model.invoke(PROMPT.format(tool_name=tool_name, tool_args=tool_args))
        return parse_verdict(response.content)

    return classify


def build_classifier(provider_cfg: ProviderConfig):
    """Convenience wrapper for the common single-provider case (no
    routing/fallback) -- see build_classifier_from_model for the general
    form."""
    return build_classifier_from_model(build_chat_model(provider_cfg))


def parse_verdict(raw: str) -> Verdict:
    try:
        data = json.loads(raw)
        decision = data.get("decision")
        if decision not in ("allow", "deny"):
            raise ValueError(f"unexpected decision: {decision!r}")
        return Verdict(decision=decision, reason=data.get("reason", ""))
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        # Fail closed: an unparseable classifier response is a denial, not
        # a silent allow. Matches v1's own posture -- it already observed
        # its classifier leaning conservative/binary in practice.
        return Verdict(decision="deny", reason=f"classifier response unparseable: {e}")
