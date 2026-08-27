"""
Phase 5: the meta-agent's substrate, not its judgment -- whether a real
model's proposed revision is actually a *good* one can't be verified
without real inference (no Ollama reachable here). What's tested is that
a well-formed response gets queued as a pending proposal (never applied
automatically), a same-text response queues nothing, and an unparseable
response fails closed rather than silently becoming a proposal --
matching classifier.py's posture on the same kind of failure.
"""

import json
import os
import uuid

import psycopg

from harness import prompts
from harness.meta_agent import propose_prompt_update


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    return conn


class FakeModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return type("Response", (), {"content": self.content})()


def test_well_formed_revision_gets_queued_pending_not_applied():
    conn = _conn()
    role = f"test-role-{uuid.uuid4().hex[:8]}"
    v1 = prompts.propose(conn, role, "old prompt text")
    prompts.approve(conn, role, v1)

    model = FakeModel(json.dumps({"revised_prompt": "new prompt text", "reasoning": "refusing too often"}))
    result = propose_prompt_update(conn, role, model)

    assert result == {"role": role, "version": 2, "reasoning": "refusing too often"}
    assert prompts.get_active(conn, role) == "old prompt text"  # unchanged -- proposal isn't auto-applied
    pending = prompts.pending(conn, role)
    assert len(pending) == 1
    assert pending[0]["text"] == "new prompt text"


def test_same_text_response_queues_nothing():
    conn = _conn()
    role = f"test-role-{uuid.uuid4().hex[:8]}"
    v1 = prompts.propose(conn, role, "steady state prompt")
    prompts.approve(conn, role, v1)

    model = FakeModel(
        json.dumps({"revised_prompt": "steady state prompt", "reasoning": "outcomes look fine"})
    )
    result = propose_prompt_update(conn, role, model)

    assert result is None
    assert prompts.pending(conn, role) == []


def test_unparseable_response_fails_closed():
    conn = _conn()
    role = f"test-role-{uuid.uuid4().hex[:8]}"
    v1 = prompts.propose(conn, role, "some prompt")
    prompts.approve(conn, role, v1)

    model = FakeModel("not json at all")
    result = propose_prompt_update(conn, role, model)

    assert result is None
    assert prompts.pending(conn, role) == []
