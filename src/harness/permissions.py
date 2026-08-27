"""
Tool-call permission layer. Two deliberately separate concerns:

1. `is_statically_safe` -- a fast-path allow-list so obviously-safe calls
   skip the LLM classifier (classifier.py) entirely. Mirrors v1's design:
   always-allow read-only tools and a small static set of
   argument-invariant-safe verbs. This is a *speed* optimization, not the
   security boundary -- anything not statically safe falls through to the
   classifier, which is the real gate (wired in graph.py's permission_gate
   node).

2. `check_within_workspace` -- a hard invariant enforced *inside* the file
   tools themselves (tools.py), independent of whatever the static
   allow-list or classifier decided. This one is not classifier-
   overridable on purpose: workspace containment is a sandbox boundary,
   not a task-semantic judgment call, so it doesn't belong to the same
   layer that's reasoning about intent.
"""

import os


class PermissionDenied(Exception):
    pass


_SAFE_READONLY_VERBS = {"ls", "cat", "pwd", "head", "tail", "grep", "find"}
_SHELL_OPERATORS = ("|", ">", ">>", "&&", ";", "`", "$(")


def _is_safe_shell(command: str) -> bool:
    stripped = command.strip()
    verb = stripped.split()[0] if stripped else ""
    has_operator = any(op in command for op in _SHELL_OPERATORS)
    return verb in _SAFE_READONLY_VERBS and not has_operator


def _is_within_workspace(path: str, workspace_root: str) -> bool:
    resolved = os.path.abspath(os.path.join(workspace_root, path))
    return resolved.startswith(os.path.abspath(workspace_root))


def is_statically_safe(tool_name: str, tool_args: dict, workspace_root: str) -> bool:
    if tool_name == "remember_fact":
        return True  # additive, non-destructive by construction
    if tool_name == "read_file":
        return _is_within_workspace(tool_args.get("path", ""), workspace_root)
    if tool_name == "shell_exec":
        return _is_safe_shell(tool_args.get("command", ""))
    return False  # write_file and anything unrecognized always gets classified


def check_within_workspace(path: str, workspace_root: str) -> None:
    if not _is_within_workspace(path, workspace_root):
        raise PermissionDenied(f"path escapes workspace: {path!r}")
