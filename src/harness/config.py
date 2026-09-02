import os

# The harness's own store: the Beads issue database (.beads), the wiki
# and generated avatars. Agents are NOT rooted here any more -- see
# PROJECTS_ROOT and workspaces.py. Keeping the name and default means the
# live issue database does not have to move.
WORKSPACE_ROOT = os.environ.get("HARNESS_WORKSPACE", "/workspace")

# Parent directory of per-project workspaces. Each project gets
# PROJECTS_ROOT/<project_id>, which is the root an agent working that
# project's ticket sees -- so product code never lands beside the issue
# database, and one project cannot reach another's files.
PROJECTS_ROOT = os.environ.get("HARNESS_PROJECTS", "/projects")
DEFAULT_ACTOR = os.environ.get("BEADS_ACTOR", "custos-worker")

# Per-`bd`-invocation subprocess timeout. Was hardcoded at 60s in
# beads._run; raised and made configurable after a real failure
# (2026-08-31): a `bd create` exceeded 60s against a Dolt-backed
# workspace of only ~47 issues, which surfaced as an opaque HTTP 500 and
# aborted a bulk backlog load partway through. bd's cost grows with the
# workspace, so a fixed 60s is a backlog-size-dependent time bomb.
BD_TIMEOUT = int(os.environ.get("BD_TIMEOUT", "300"))

# How long /projects may serve a cached tree before refreshing it. The
# tree costs a full `bd list --all` (~5s on a real workspace), which is
# far too slow to sit in the request path of a dashboard that polls
# every 5s -- see api.list_projects.
PROJECT_TREE_TTL = float(os.environ.get("PROJECT_TREE_TTL", "15"))
