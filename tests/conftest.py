"""
Tests get their own isolated state -- a fresh Beads workspace AND a fresh
Postgres database per test session, never the persistent ones the real
worker/api/product-owner services use -- so running the suite doesn't
fill real data with test artifacts.

Both problems were found live, not anticipated up front: the workspace
one when repeated pytest runs left ~30 tickets in workspace/.beads; the
Postgres one afterward, when smoke-testing the seat system's /seats
endpoint over real HTTP showed a dozen leftover test-seat-* rows sitting
in what should have been the real roster -- prompts.py/seats.py tests
had been writing straight into the same DATABASE_URL worker.py/api.py
use, exactly the same category of bug the workspace fix addressed, just
in the other durable store this harness has.

Must run before any `harness.*` module is imported, since config.py reads
HARNESS_WORKSPACE once at import time. conftest.py is guaranteed by
pytest to load before test modules collect, which is what makes setting
env vars here (rather than in each test) actually take effect.

Unconditional assignment, not setdefault: docker-compose.yml's `harness`
service already sets these as container env vars before Python starts,
so setdefault (which only fills in a *missing* key) would silently do
nothing -- this exact mistake shipped once already for HARNESS_WORKSPACE
before being caught by actually checking the directory afterward.
"""

import os
import tempfile
import uuid

import psycopg

os.environ["HARNESS_WORKSPACE"] = tempfile.mkdtemp(prefix="custos-test-workspace-")

_base_url = os.environ.get("DATABASE_URL")
if _base_url:
    _test_db_name = f"custos_harness_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_base_url, autocommit=True) as _admin_conn:
        _admin_conn.execute(f'CREATE DATABASE "{_test_db_name}"')

    _base, _, _ = _base_url.rpartition("/")
    os.environ["DATABASE_URL"] = f"{_base}/{_test_db_name}"
