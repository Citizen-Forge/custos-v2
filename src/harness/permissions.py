"""
Tool-call permission gate — Phase 1 skeleton only.

This is a static per-verb allow-list, which is exactly the design Custos
v1 started with *and later flagged as a real safety bug*: a coarse verb
whitelist let `rm foo.txt` silently whitelist `rm -rf /` forever after.
v1's fix was a live LLM classifier on every call (see
claude-gateway/src/server's PreToolUse hook, `permissionClassifier` task).

Porting that classifier is a deliberate fast-follow, not done here — Phase
1's exit criteria is proving the durable queue/resume loop, not permission
correctness. Do not point real filesystem/shell access at this as-is.
"""


class PermissionDenied(Exception):
    pass


_SAFE_READONLY_VERBS = {"ls", "cat", "pwd", "head", "tail", "grep", "find"}
_SHELL_OPERATORS = ("|", ">", ">>", "&&", ";", "`", "$(")


def check_shell(command: str) -> None:
    stripped = command.strip()
    verb = stripped.split()[0] if stripped else ""
    has_operator = any(op in command for op in _SHELL_OPERATORS)
    if verb in _SAFE_READONLY_VERBS and not has_operator:
        return
    raise PermissionDenied(f"shell command requires explicit approval: {command!r}")


def check_write(path: str, workspace_root: str) -> None:
    import os

    resolved = os.path.abspath(os.path.join(workspace_root, path))
    if not resolved.startswith(os.path.abspath(workspace_root)):
        raise PermissionDenied(f"write path escapes workspace: {path!r}")
