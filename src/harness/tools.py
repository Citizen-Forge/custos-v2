"""Minimal Phase 1 tool set: shell exec, file read/write, and Beads memory."""

import os
import subprocess

from langchain_core.tools import tool

from . import beads, permissions
from .config import WORKSPACE_ROOT


@tool
def shell_exec(command: str) -> str:
    """Run a shell command in the workspace and return its combined output."""
    permissions.check_shell(command)
    result = subprocess.run(
        command,
        shell=True,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.stdout or "") + (result.stderr or "")


@tool
def read_file(path: str) -> str:
    """Read a text file's contents, relative to the workspace root."""
    resolved = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file, relative to the workspace root."""
    permissions.check_write(path, WORKSPACE_ROOT)
    resolved = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} bytes to {path}"


@tool
def remember_fact(text: str) -> str:
    """Persist a durable insight via Beads (`bd remember`) so it survives
    across sessions and tickets, not just this conversation."""
    result = beads.remember(text)
    return f"remembered: {result.get('key', text)}"


ALL_TOOLS = [shell_exec, read_file, write_file, remember_fact]
