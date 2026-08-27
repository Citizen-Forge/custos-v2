import os

WORKSPACE_ROOT = os.environ.get("HARNESS_WORKSPACE", "/workspace")
DEFAULT_ACTOR = os.environ.get("BEADS_ACTOR", "custos-worker")
