"""
Tool-call permission layer. Three deliberately separate concerns:

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

3. `is_infrastructure` -- names injected into every project workspace by
   the harness or its tooling (.git, .beads, .claude, .codex, .agents),
   not the project's own content. Unlike (2) this is NOT a security
   boundary and never raises: an agent poking at one of these is
   ordinary orientation behavior, not an escape attempt, so callers
   return an explanatory string the agent can read and move on from --
   same "ordinary, not a security event" treatment read_file already
   gives a missing file. The actual security boundary for a sensitive
   file like .claude/settings.json is still the classifier (or workspace
   containment for an escape); this only keeps an agent from wasting a
   turn finding that out the hard way. Found live 2026-09-04: an agent
   listed its workspace, tried to read .claude/settings.json because it
   was sitting right there in the listing, got denied, and its very next
   turn came back completely empty -- no text, no tool call, no further
   progress for hours.
"""

import os


class PermissionDenied(Exception):
    pass


_SAFE_READONLY_VERBS = {"ls", "cat", "pwd", "head", "tail", "grep", "find"}
_SHELL_OPERATORS = ("|", ">", ">>", "&&", ";", "`", "$(")

HIDDEN_FROM_LISTING = {".git", ".beads", ".claude", ".codex", ".agents"}


def _is_safe_shell(command: str) -> bool:
    stripped = command.strip()
    verb = stripped.split()[0] if stripped else ""
    has_operator = any(op in command for op in _SHELL_OPERATORS)
    return verb in _SAFE_READONLY_VERBS and not has_operator


def _is_within_workspace(path: str, workspace_root: str) -> bool:
    """Containment by path components, NOT by string prefix.

    A plain `startswith` was wrong in a way that only bites once
    workspaces have siblings: from /projects/proj-a the path
    ../proj-abc/x resolves to /projects/proj-abc/x, which starts with
    /projects/proj-a and was therefore allowed. With one workspace that
    was a latent bug (/workspace-evil escaped /workspace); with a
    workspace per project it would have meant no isolation between
    projects at all, which is the whole point of having them."""
    root = os.path.abspath(workspace_root)
    resolved = os.path.abspath(os.path.join(root, path))
    if resolved == root:
        return True
    return resolved.startswith(root + os.sep)


def is_statically_safe(tool_name: str, tool_args: dict, workspace_root: str) -> bool:
    if tool_name == "remember_fact":
        return True  # additive, non-destructive by construction
    if tool_name in ("read_file", "list_directory"):
        return _is_within_workspace(tool_args.get("path", ""), workspace_root)
    if tool_name == "shell_exec":
        return _is_safe_shell(tool_args.get("command", ""))
    return False  # write_file and anything unrecognized always gets classified


def check_within_workspace(path: str, workspace_root: str) -> None:
    if not _is_within_workspace(path, workspace_root):
        raise PermissionDenied(f"path escapes workspace: {path!r}")


def is_infrastructure(path: str) -> bool:
    """True if any component of `path` names harness/tool infrastructure
    (see module docstring) rather than the project's own content. Checks
    every component, not just the first, so a nested reference like
    `src/../.claude/settings.json` or a deeper `sub/.beads/x` still
    matches."""
    parts = path.replace("\\", "/").strip("/").split("/")
    return any(p in HIDDEN_FROM_LISTING for p in parts)
