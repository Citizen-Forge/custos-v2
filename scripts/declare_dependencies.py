"""Declare real dependency edges across a project's operator-authored roadmap.

The product-owner declares dependencies during ITS OWN breakdown, but a
roadmap authored directly (like Subspatial, 99/128 issues by
operator-subspatial) never runs that session, so `bd ready` sees no blocker
edges and dispatch has no ordering to enforce. This reads the open stories,
asks the model for the real prerequisites, and (with --apply) writes them as
`bd dep add` edges.

Dry run by default: it prints the proposed edges so a human can see them
before anything is blocked.

Usage (inside the harness container):
    python scripts/declare_dependencies.py <project_id> [--apply]
"""

import json
import os
import re
import sys

from harness import beads
from harness.providers import ProviderConfig, build_chat_model

PROMPT = """You are auditing the dependency graph of a software project's backlog.

Below is every open story in project {project} as:  <id> | <title> | <description snippet>

Declare the REAL prerequisites: for each story that cannot be correctly built or tested until \
another specific story is finished, emit one edge {{"blocked": "<story id>", "blocker": \
"<prerequisite id>"}}.

Rules:
- Only DIRECT, real dependencies -- a story needs another's artifact, data model, contract or \
toolchain. Do not emit transitive/implied edges (if A blocks B and B blocks C, do not also emit \
A blocks C).
- High confidence only. Prefer a smaller, correct set over a large speculative one.
- Never an edge from a story to itself, and no cycles.
- If a story has no real prerequisite, give it no edge.
- Output ONLY a JSON array of {{"blocked","blocker"}} objects, nothing else.

Stories:
{stories}
"""


def _extract_json_array(text: str) -> list | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, list) else None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    if not args:
        print("usage: declare_dependencies.py <project_id> [--apply]")
        return 2
    project = args[0]

    issues = beads.list_all()
    stories = [
        i
        for i in issues
        if i.get("id", "").startswith(project + ".")
        and i.get("issue_type") == "task"
        and i.get("status") != "closed"
    ]
    stories.sort(key=lambda i: i["id"])
    if not stories:
        print(f"no open stories under {project}")
        return 0

    known = {s["id"] for s in stories}
    lines = []
    for s in stories:
        desc = " ".join((s.get("description") or "").split())[:300]
        lines.append("%s | %s | %s" % (s["id"], s.get("title", ""), desc))

    cfg = ProviderConfig(
        name="dep-audit",
        base_url=os.environ.get(
            "SCHEDULER_MODEL_BASE_URL", os.environ.get("LOCAL_MODEL_BASE_URL")
        ),
        model=os.environ.get("SCHEDULER_MODEL_NAME", os.environ.get("LOCAL_MODEL_NAME")),
        api_key=os.environ.get("SCHEDULER_MODEL_API_KEY", os.environ.get("LOCAL_MODEL_API_KEY")),
        max_tokens=4000,
        extra_body=(
            {"thinking": {"type": "disabled"}}
            if os.environ.get("LOCAL_MODEL_DISABLE_THINKING")
            else None
        ),
    )
    model = build_chat_model(cfg)

    print(f"project {project}: {len(stories)} open stories -> asking the model for edges")
    response = model.invoke(PROMPT.format(project=project, stories="\n".join(lines)))
    edges = _extract_json_array(getattr(response, "content", "") or "")
    if edges is None:
        print("could not parse an edge list from the model response")
        return 1

    valid = []
    for edge in edges:
        blocked, blocker = edge.get("blocked"), edge.get("blocker")
        if blocked not in known or blocker not in known:
            print("  skip (unknown id):", edge)
            continue
        if blocked == blocker:
            print("  skip (self):", edge)
            continue
        valid.append((blocked, blocker))

    print(f"\nproposed edges ({len(valid)}):")
    for blocked, blocker in valid:
        print(f"  {blocked}  blocked by  {blocker}")

    if not apply:
        print("\n(dry run -- rerun with --apply to write these edges)")
        return 0

    print()
    for blocked, blocker in valid:
        try:
            beads.add_dependency(blocked, blocker, actor="operator")
            print(f"  added: {blocked} depends on {blocker}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {blocked} depends on {blocker}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
