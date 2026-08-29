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
from unittest.mock import patch

import psycopg

from harness import prompts, seats, verifications, wiki
from harness.meta_agent import create_specialist_seat, propose_prompt_update


def _conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    prompts.init_table(conn)
    seats.init_table(conn)
    verifications.init_table(conn)
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


def test_create_specialist_seat_goes_active_immediately_no_approval_needed():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    model = FakeModel(
        json.dumps(
            {
                "seat_id": seat_id,
                "system_prompt": "you specialize in X",
                "display_name": "Rowan",
                "pronouns": "they/them",
                "profile_page": "Hi, I'm Rowan! I specialize in X.",
                "appearance_description": "curly red hair, round glasses, warm smile",
            }
        )
    )
    with patch("harness.meta_agent.avatar.generate_avatar") as mock_generate:
        result = create_specialist_seat(conn, "specializes in X", requested_by="product_owner", model=model)

    assert result == {
        "seat_id": seat_id,
        "specialty": "specializes in X",
        "version": 1,
        "display_name": "Rowan",
        "pronouns": "they/them",
        "profile_page": "Hi, I'm Rowan! I specialize in X.",
        "appearance_description": "curly red hair, round glasses, warm smile",
    }
    created = seats.get(conn, seat_id)
    assert created is not None
    assert created["specialty"] == "specializes in X"
    assert created["created_by"] == "product_owner"
    # the seat's own chosen identity (Phase 4) -- distinct from seat_id,
    # which stays the functional identifier
    assert created["display_name"] == "Rowan"
    assert created["pronouns"] == "they/them"
    # the seat's own wiki profile page, written as part of creation (the
    # user's own framing: a real page a human -- or the seat itself,
    # later -- can read, not just a name in a table)
    assert wiki.read_page(wiki.agent_profile_slug(seat_id)) == "Hi, I'm Rowan! I specialize in X."
    # unlike propose_prompt_update, a brand-new seat's first prompt is
    # active immediately -- nothing existing to protect with a pending step
    assert prompts.get_active(conn, seat_id) == "you specialize in X"
    assert prompts.pending(conn, seat_id) == []
    # the avatar prompt gets the PHYSICAL description, not the personality
    # bio -- profile_page and appearance_description deliberately diverge
    # in this test so a mix-up would fail loudly
    mock_generate.assert_called_once_with(seat_id, "curly red hair, round glasses, warm smile")


def test_create_specialist_seat_skips_avatar_generation_without_appearance_description():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    model = FakeModel(
        json.dumps(
            {
                "seat_id": seat_id,
                "system_prompt": "you specialize in X",
                "display_name": "Rowan",
                "profile_page": "Hi, I'm Rowan!",
            }
        )
    )
    with patch("harness.meta_agent.avatar.generate_avatar") as mock_generate:
        result = create_specialist_seat(conn, "specializes in X", requested_by="product_owner", model=model)

    assert result["appearance_description"] is None
    mock_generate.assert_not_called()


def test_create_specialist_seat_prompt_includes_existing_profiles_for_differentiation():
    # The prompt asks a new agent to pick different personality specifics
    # than its teammates -- only a real instruction if it can actually
    # see what they picked, not just their job title.
    conn = _conn()
    existing_seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, existing_seat_id, "existing specialty", created_by="test", display_name="Juno")
    wiki.write_page(wiki.agent_profile_slug(existing_seat_id), "I love jazz and living in Lisbon.")

    captured = {}

    class CapturingModel:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "seat_id": f"new-seat-{uuid.uuid4().hex[:8]}",
                            "system_prompt": "x",
                            "display_name": "Someone",
                            "pronouns": "they/them",
                            "profile_page": "bio",
                        }
                    )
                },
            )()

    create_specialist_seat(conn, "new specialty", requested_by="product_owner", model=CapturingModel())

    assert "Juno" in captured["prompt"]
    assert "jazz" in captured["prompt"]
    assert "Lisbon" in captured["prompt"]


def test_create_specialist_seat_without_identity_fields_stores_null_not_a_placeholder():
    # A response that omits display_name/pronouns (e.g. an older prompt
    # version, or a model that just didn't include them) shouldn't fail
    # closed on that alone -- the seat still gets created, just without a
    # chosen identity yet, rather than a fabricated one.
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"

    model = FakeModel(json.dumps({"seat_id": seat_id, "system_prompt": "you specialize in Y"}))
    result = create_specialist_seat(conn, "specializes in Y", requested_by="product_owner", model=model)

    assert result["display_name"] is None
    assert result["pronouns"] is None
    assert result["profile_page"] is None
    assert wiki.read_page(wiki.agent_profile_slug(seat_id)) is None  # no fabricated page either
    assert seats.get(conn, seat_id)["display_name"] is None


def test_create_specialist_seat_refuses_to_collide_with_existing_seat():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"
    seats.create(conn, seat_id, "already exists", created_by="someone")

    model = FakeModel(json.dumps({"seat_id": seat_id, "system_prompt": "would overwrite"}))
    result = create_specialist_seat(conn, "new specialty", requested_by="product_owner", model=model)

    assert result is None
    assert seats.get(conn, seat_id)["specialty"] == "already exists"  # untouched


def test_create_specialist_seat_fails_closed_on_unparseable_response():
    conn = _conn()
    seat_id = f"test-seat-{uuid.uuid4().hex[:8]}"  # would-be id, never actually used
    model = FakeModel("not json")

    result = create_specialist_seat(conn, "some specialty", requested_by="product_owner", model=model)

    assert result is None
    assert seats.get(conn, seat_id) is None  # nothing got created
