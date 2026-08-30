"""
self_modifier.py's own tools: real filesystem + real git operations
against an isolated checkout (never the actual harness source -- see
CHECKOUT_ROOT/LIVE_SOURCE_ROOT monkeypatching below, same pattern as
dynamic_tools.py's WORKSPACE_ROOT tests). What's proven here: syncing
seeds a real git-tracked checkout from a fake "live" source, re-syncing
refuses to clobber uncommitted work, writes stay contained to the
checkout, and proposing computes a real diff (not a guess at git's
output shape) and queues a real self_mod_proposals row.
"""

import os
import uuid

import psycopg

from harness import self_mod, self_modifier
from harness.self_modifier import build_tools


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    self_mod.init_table(conn)
    return conn


def _seed_live_source(live_root, contents: dict[str, str]):
    for relpath, text in contents.items():
        full = live_root / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text)


def test_sync_seeds_a_real_git_tracked_checkout(monkeypatch, tmp_path):
    live_root = tmp_path / "live"
    checkout_root = tmp_path / "checkout"
    _seed_live_source(live_root, {"src/harness/foo.py": "x = 1\n"})
    monkeypatch.setattr(self_modifier, "LIVE_SOURCE_ROOT", str(live_root))
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(checkout_root))

    tools = build_tools(_conn())
    sync = next(t for t in tools if t.name == "sync_checkout_from_live")
    result = sync.invoke({})

    assert "synced" in result
    assert (checkout_root / "src" / "harness" / "foo.py").read_text() == "x = 1\n"
    assert (checkout_root / ".git").is_dir()


def test_resync_refuses_to_clobber_uncommitted_edits(monkeypatch, tmp_path):
    live_root = tmp_path / "live"
    checkout_root = tmp_path / "checkout"
    _seed_live_source(live_root, {"src/harness/foo.py": "x = 1\n"})
    monkeypatch.setattr(self_modifier, "LIVE_SOURCE_ROOT", str(live_root))
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(checkout_root))

    tools = build_tools(_conn())
    sync = next(t for t in tools if t.name == "sync_checkout_from_live")
    write = next(t for t in tools if t.name == "write_checkout_file")

    sync.invoke({})
    write.invoke({"path": "src/harness/foo.py", "content": "x = 2  # uncommitted edit\n"})

    result = sync.invoke({})

    assert "uncommitted" in result
    # the edit must survive -- a refused sync must not have touched anything
    assert (checkout_root / "src" / "harness" / "foo.py").read_text() == "x = 2  # uncommitted edit\n"


def test_read_write_checkout_file_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(tmp_path))
    tools = build_tools(_conn())
    write = next(t for t in tools if t.name == "write_checkout_file")
    read = next(t for t in tools if t.name == "read_checkout_file")

    write.invoke({"path": "src/harness/new_file.py", "content": "real content\n"})
    result = read.invoke({"path": "src/harness/new_file.py"})

    assert result == "real content\n"


def test_write_checkout_file_rejects_path_escaping_the_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(tmp_path))
    tools = build_tools(_conn())
    write = next(t for t in tools if t.name == "write_checkout_file")

    # Real bug caught live 2026-08-30, not by this test originally: a
    # first version of this test accepted either a raised exception or
    # an error string as passing, which is exactly why it didn't catch
    # that the tool was letting PermissionDenied escape uncaught and
    # crash a real self-modifier session outright. Fixed to require the
    # fail-soft behavior every other tool in this codebase already
    # follows -- never raise, always return a readable error string.
    result = write.invoke({"path": "../../etc/passwd", "content": "malicious"})
    assert "escapes workspace" in result


def test_propose_self_modification_computes_a_real_diff(monkeypatch, tmp_path):
    live_root = tmp_path / "live"
    checkout_root = tmp_path / "checkout"
    _seed_live_source(live_root, {"src/harness/foo.py": "def f():\n    return 1\n"})
    monkeypatch.setattr(self_modifier, "LIVE_SOURCE_ROOT", str(live_root))
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(checkout_root))

    conn = _conn()
    tools = build_tools(conn)
    sync = next(t for t in tools if t.name == "sync_checkout_from_live")
    write = next(t for t in tools if t.name == "write_checkout_file")
    propose = next(t for t in tools if t.name == "propose_self_modification")

    sync.invoke({})
    write.invoke({"path": "src/harness/foo.py", "content": "def f():\n    return 2  # fixed the return value\n"})
    description = f"fix f() return value {uuid.uuid4().hex[:8]}"
    result = propose.invoke({"description": description})

    assert "proposal #" in result
    proposal_id = int(result.split("#")[1].split()[0])
    proposal = self_mod.get(conn, proposal_id)
    assert proposal["description"] == description
    assert proposal["proposed_by"] == "self_modifier"
    assert proposal["status"] == "pending"
    assert "-    return 1" in proposal["diff"]
    assert "+    return 2  # fixed the return value" in proposal["diff"]


def test_propose_self_modification_with_no_changes_does_not_queue_anything(monkeypatch, tmp_path):
    live_root = tmp_path / "live"
    checkout_root = tmp_path / "checkout"
    _seed_live_source(live_root, {"src/harness/foo.py": "x = 1\n"})
    monkeypatch.setattr(self_modifier, "LIVE_SOURCE_ROOT", str(live_root))
    monkeypatch.setattr(self_modifier, "CHECKOUT_ROOT", str(checkout_root))

    conn = _conn()
    tools = build_tools(conn)
    sync = next(t for t in tools if t.name == "sync_checkout_from_live")
    propose = next(t for t in tools if t.name == "propose_self_modification")

    sync.invoke({})
    result = propose.invoke({"description": "nothing actually changed"})

    assert "no changes" in result
