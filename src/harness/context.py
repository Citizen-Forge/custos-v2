"""Request-size bounding.

A provider's context window is a hard limit, and in a fallback chain the
secondary is usually far smaller than the primary -- a hosted model's big
window versus a local 30B's 64k. Bounding therefore belongs to the provider,
applied in routing right before each call, not once globally in the graph.

`bound_history` is the one implementation; the graph applies it as an outer
cap (DEFAULT_HISTORY_MAX_CHARS) and routing tightens it per provider.

Found live 2026-09-21: DeepSeek ran out of balance (402), the 30B fallback
engaged, and it refused a 330k-token request against its 64k window -- 15
tickets were flagged. The bound had also been counting only message content,
missing tool-call arguments (a write_file carries the whole file), which is
what let the request grow that far.
"""

import os
import re

from langchain_core.messages import ToolMessage

# Outer / default bound (bytes of text) when a provider declares no bound of its
# own. Override per provider with `<PREFIX>_MODEL_CONTEXT_CHARS` /
# `<PREFIX>_FALLBACK_CONTEXT_CHARS` (see worker._chain_from_env).
DEFAULT_HISTORY_MAX_CHARS = int(os.environ.get("HISTORY_MAX_CHARS", "400000"))

# The default for a FALLBACK provider, tuned to fit a local 30B served at a
# 65536-token window: ~150k chars of code leaves room for the system prompt,
# tool schemas and the reply. Override with `<PREFIX>_FALLBACK_CONTEXT_CHARS`.
FALLBACK_HISTORY_MAX_CHARS = int(os.environ.get("FALLBACK_HISTORY_MAX_CHARS", "150000"))

# One tool result is capped too -- an unbounded single result is its own way to
# blow the window even when the total is otherwise trimmed.
TOOL_MESSAGE_MAX_CHARS = int(os.environ.get("TOOL_MESSAGE_MAX_CHARS", "20000"))


def cap_tool_message(message: ToolMessage) -> ToolMessage:
    """Shorten one tool result, marking the cut so the model knows it is partial."""
    content = getattr(message, "content", None)
    if not isinstance(content, str) or len(content) <= TOOL_MESSAGE_MAX_CHARS:
        return message
    return ToolMessage(
        content=(
            content[:TOOL_MESSAGE_MAX_CHARS]
            + f"\n... [truncated at {TOOL_MESSAGE_MAX_CHARS} characters]"
        ),
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
    )


def message_size(message) -> int:
    """Chars a message contributes to the request: its content PLUS its
    tool-call arguments.

    The args are the load-bearing part for a coding agent -- a `write_file`
    carries the whole file -- and counting only `content` let a thread grow far
    past the bound in real tokens."""
    size = len(str(getattr(message, "content", "") or ""))
    for call in getattr(message, "tool_calls", None) or []:
        size += len(str(call.get("args", "")))
    return size


# `bd prime` inlines EVERY persistent memory. Reflection adds memories after
# each run, and by 2026-09-30 prime was 1.69M chars (724 memories) -- every
# fresh brief was ~450k tokens, over both DeepSeek's window and the 30B's.
PRIME_MAX_CHARS = int(os.environ.get("PRIME_MAX_CHARS", "40000"))

# The lead-in (system prompt + brief) may take at most this share of a
# provider's budget; the rest is for the conversation itself.
LEAD_IN_MAX_SHARE = 0.5

BRIEF_TICKET_MARKER = "\n\n---\n\nTicket:"

_MEMORIES_HEADING = re.compile(r"(?m)^## Persistent Memories.*$")
_TOP_LEVEL_HEADING = re.compile(r"(?m)^# ")
_MEMORY_ENTRY = re.compile(r"(?m)^(?=### )")


def _cut_middle(text: str, max_chars: int) -> str:
    marker = f"\n\n... [{len(text) - max_chars} characters omitted to fit the context budget] ...\n\n"
    keep = max(max_chars - len(marker), 0)
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head):]


def _mentions_project(entry: str, project_id: str) -> bool:
    short = project_id.rsplit("-", 1)[-1]
    return re.search(rf"(?i)(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", entry) is not None


def cap_prime(text: str, max_chars: int = PRIME_MAX_CHARS, project_id: str | None = None) -> str:
    """Fit `bd prime` output into `max_chars` by dropping persistent memories.

    The workflow sections before and after the memories are kept whole. Of the
    memories, those mentioning `project_id` are kept first, then the rest in
    their original order, until the budget is spent; a note says how many were
    left out and how to search them.
    """
    if len(text) <= max_chars:
        return text
    heading = _MEMORIES_HEADING.search(text)
    if not heading:
        return _cut_middle(text, max_chars)
    start = heading.end()
    tail = _TOP_LEVEL_HEADING.search(text, start)
    end = tail.start() if tail else len(text)
    before, after = text[: heading.start()], text[end:]
    entries = [e for e in _MEMORY_ENTRY.split(text[start:end]) if e.startswith("### ")]

    budget = max_chars - len(before) - len(after) - 400
    if budget <= 0:
        return _cut_middle(text, max_chars)

    order = list(range(len(entries)))
    if project_id:
        order.sort(key=lambda i: not _mentions_project(entries[i], project_id))
    kept: set[int] = set()
    used = 0
    for i in order:
        if used + len(entries[i]) <= budget:
            kept.add(i)
            used += len(entries[i])

    note = (
        f"## Persistent Memories ({len(kept)} of {len(entries)} shown)\n\n"
        f"{len(entries) - len(kept)} more were left out to fit the context budget. "
        "Search them with `bd memories <keyword>` or the search_related_work tool.\n\n"
    )
    return before + note + "".join(entries[i] for i in sorted(kept)) + after


def cap_brief(content: str, max_chars: int, project_id: str | None = None) -> str:
    """Fit a ticket brief -- `bd prime`, then the ticket text -- into `max_chars`,
    shortening only the prime part. The ticket text is the one part an agent
    cannot work without."""
    if len(content) <= max_chars:
        return content
    split = content.find(BRIEF_TICKET_MARKER)
    if split > 0:
        ticket = content[split:]
        prime_budget = max_chars - len(ticket)
        if prime_budget > 1000:
            return cap_prime(content[:split], prime_budget, project_id) + ticket
    return _cut_middle(content, max_chars)


def bound_history(messages: list, max_chars: int) -> list:
    """Drop the oldest messages until the request fits `max_chars`, cutting only
    at boundaries.

    The retained window must never BEGIN on a tool message: a tool reply whose
    assistant message was dropped is invalid on its own, which is the exact
    failure this exists to prevent. Keeping a suffix (rather than a prefix plus
    a suffix) is what makes that a one-line guarantee -- every assistant
    tool_calls inside a suffix still has its replies directly after it.

    A history that already fits is returned unchanged.
    """
    capped = [cap_tool_message(m) if isinstance(m, ToolMessage) else m for m in messages]
    total = sum(message_size(m) for m in capped)
    if total <= max_chars:
        return capped

    # Keep the LEAD-IN, which for a ticket thread is [system prompt, brief].
    # Dropping the brief is not a tidy truncation: it removes the only copy of
    # the ticket text, and agents then refuse with "no ticket text reached this
    # turn". Found live 2026-09-15, after the first version of this bound
    # shipped -- 6.1, 6.3, 6.5 and 9.4 were all parked saying exactly that.
    # Only system/human messages are kept, never an assistant turn, so this can
    # never separate a tool_calls message from its replies.
    head: list = []
    for message in capped[:2]:
        if getattr(message, "type", None) in ("system", "human"):
            head.append(message)
        else:
            break
    body = capped[len(head):]

    # The brief itself can be too big: a checkpoint written before briefs were
    # capped carries the whole of `bd prime`, and resuming it replays that
    # verbatim (workspace-o0n.17.1.5, 2026-09-30: a 1.69M-char brief).
    lead_in_budget = int(max_chars * LEAD_IN_MAX_SHARE)
    head = [
        m.model_copy(update={"content": cap_brief(m.content, lead_in_budget)})
        if getattr(m, "type", None) == "human"
        and isinstance(m.content, str)
        and len(m.content) > lead_in_budget
        else m
        for m in head
    ]
    budget = max_chars - sum(message_size(m) for m in head)

    kept: list = []
    used = 0
    for message in reversed(body):
        size = message_size(message)
        if kept and used + size > budget:
            break
        kept.append(message)
        used += size
    kept.reverse()
    while kept and isinstance(kept[0], ToolMessage):
        kept.pop(0)
    return head + kept
