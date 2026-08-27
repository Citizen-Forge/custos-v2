"""
Tests get their own isolated Beads workspace -- a fresh temp directory
per test session, never the persistent `workspace/` the real worker/api
services use -- so running the suite doesn't fill real ticket data with
test tickets (found live: every prior pytest run this session had been
creating real tickets in workspace/.beads).

Must run before any `harness.*` module is imported, since config.py reads
HARNESS_WORKSPACE once at import time. conftest.py is guaranteed by
pytest to load before test modules collect, which is what makes setting
the env var here (rather than in each test) actually take effect.

Unconditional assignment, not setdefault: docker-compose.yml's `harness`
service already sets HARNESS_WORKSPACE=/workspace as a container env var
before Python even starts, so setdefault (which only fills in a *missing*
key) silently did nothing -- caught live, this is what actually let the
first version of this fix ship without working.
"""

import os
import tempfile

os.environ["HARNESS_WORKSPACE"] = tempfile.mkdtemp(prefix="custos-test-workspace-")
