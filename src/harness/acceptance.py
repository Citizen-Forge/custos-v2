"""
Machine-checkable acceptance criteria.

Free-text `acceptance_criteria` needs a model to judge. But a large share of
real criteria are mechanical -- a file exists, the suite runs at least N
tests, the README contains a section -- and those do not need a model at
all. Checking them in code is both cheaper and stricter than asking one, so
the verifier evaluates them first and can decide a ticket without spending a
call. The frontier verifier is then left to judge only the genuinely
subjective half, which is the half that actually needs it.

Schema: a ticket carries a JSON list (`beads.CHECKS_KEY`); each entry is a
dict with a `type` and its parameters:

    {"type": "file_exists",     "path": "project.godot"}
    {"type": "file_absent",     "path": "old/TODO"}
    {"type": "contains",        "path": "README.md", "text": "## Test"}
    {"type": "matches",         "path": "src/x.gd", "pattern": "func \\w+"}
    {"type": "tests_at_least",  "count": 1}

Paths are relative to the ticket's own tree and confined there by
permissions.check_within_workspace -- the same boundary the file tools use,
so a check cannot read outside the workspace it is judging.
`tests_at_least` reads the project's mechanical test run (already executed
by the verifier) rather than running anything itself.
"""

import os
import re

from . import beads, permissions, workspaces

CHECK_TYPES = ("file_exists", "file_absent", "contains", "matches", "tests_at_least")


def evaluate(issue: dict, tests: dict | None) -> list[dict]:
    """One {"check", "ok", "detail"} per declared check, in declared order."""
    return [_one(issue, check, tests) for check in beads.acceptance_checks(issue)]


def failed(results: list[dict]) -> list[dict]:
    return [r for r in results if not r["ok"]]


def _result(check, ok, detail):
    return {"check": check, "ok": ok, "detail": detail}


def _resolve(issue: dict, raw_path: str) -> str:
    """Resolve a check's path inside the ticket's tree, refusing an escape."""
    tree = workspaces.tree_for_ticket(issue["id"])
    permissions.check_within_workspace(raw_path, tree)
    return os.path.join(tree, raw_path)


def _read(issue: dict, raw_path: str) -> str:
    with open(_resolve(issue, raw_path), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _one(issue: dict, check: dict, tests: dict | None) -> dict:
    kind = check.get("type")
    path = check.get("path", "")
    try:
        if kind == "file_exists":
            ok = os.path.isfile(_resolve(issue, path))
            return _result(check, ok, f"{path} {'exists' if ok else 'is missing'}")
        if kind == "file_absent":
            absent = not os.path.exists(_resolve(issue, path))
            return _result(check, absent, f"{path} is {'absent' if absent else 'present'}")
        if kind == "contains":
            needle = check.get("text", "")
            ok = needle in _read(issue, path)
            return _result(
                check, ok, f"{path} {'contains' if ok else 'does not contain'} {needle!r}"
            )
        if kind == "matches":
            pattern = check.get("pattern", "")
            ok = re.search(pattern, _read(issue, path)) is not None
            return _result(
                check, ok, f"{path} {'matches' if ok else 'does not match'} /{pattern}/"
            )
        if kind == "tests_at_least":
            want = int(check.get("count", 1))
            if tests is None:
                return _result(check, False, "no runnable test suite was found")
            ok = (
                tests.get("exit") == 0
                and tests.get("failed", 0) == 0
                and tests.get("ran", 0) >= want
            )
            return _result(
                check, ok,
                f"tests ran={tests.get('ran')} failed={tests.get('failed')} "
                f"exit={tests.get('exit')} (wanted >= {want} passing)",
            )
        return _result(check, False, f"unknown check type {kind!r}")
    except permissions.PermissionDenied as e:
        return _result(check, False, f"path escapes the ticket's workspace: {e}")
    except FileNotFoundError:
        return _result(check, False, f"file not found: {path}")
    except re.error as e:
        return _result(check, False, f"invalid pattern: {e}")
    except Exception as e:  # noqa: BLE001 -- a broken check is a fail, not a crash
        return _result(check, False, f"check could not be evaluated: {e}")


def describe(results: list[dict]) -> str:
    """A compact block for the verifier prompt (or the deterministic reason)."""
    if not results:
        return "(none declared)"
    lines = []
    for r in results:
        check = r["check"]
        label = check.get("type", "?")
        if check.get("path"):
            label += f" {check['path']}"
        elif check.get("type") == "tests_at_least":
            label += f" (>= {check.get('count', 1)} tests)"
        lines.append(f"- [{'OK' if r['ok'] else 'FAILED'}] {label}: {r['detail']}")
    return "\n".join(lines)
