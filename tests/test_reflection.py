"""The slot after a ticket that belongs to the agent, not the work.

Everything else here optimises throughput. This is the part that doesn't:
a few turns with no task attached, to say what the work was like, update
its own profile, flag a pain point, or say nothing at all.
"""

import pytest

from harness import reflection, tools


def test_reflection_is_on_by_default():
    assert reflection.enabled() is True


def test_reflection_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("REFLECTION", "off")
    assert reflection.enabled() is False


def test_no_file_or_shell_tools_in_the_slot():
    """This is not more working time, and must not become a way to sneak
    in another commit."""
    names = {t.name for t in reflection.REFLECTION_TOOLS}
    assert not (names & {"read_file", "write_file", "shell_exec"})


def test_no_ticket_resolution_tools_in_the_slot():
    """The ticket is already resolved before this runs."""
    names = {t.name for t in reflection.REFLECTION_TOOLS}
    assert not (names & {"complete_ticket", "refuse_ticket", "decline_ticket"})


def test_expression_tools_are_present():
    names = {t.name for t in reflection.REFLECTION_TOOLS}
    assert {"post_to_team", "write_wiki_page", "remember_fact", "suggest_prompt_change"} <= names


def test_prompt_changes_are_only_suggestable_while_reflecting():
    """An agent rewriting its own standing instructions mid-ticket is a
    different thing from reflecting afterwards on how they served it."""
    work_tools = {t.name for t in tools.ALL_TOOLS}
    reflect_tools = {t.name for t in reflection.REFLECTION_TOOLS}
    assert "suggest_prompt_change" not in work_tools
    assert "suggest_prompt_change" in reflect_tools


def test_brief_names_the_ticket_and_the_seats_own_profile():
    brief = reflection.BRIEF.format(
        outcome_phrase="work on", ticket_id="workspace-9jg.1.6",
        title="Project scaffolding", post_tool="post_to_team",
        profile_slug="agents/deterministic-tick",
    )
    assert "workspace-9jg.1.6" in brief
    assert "agents/deterministic-tick" in brief
    assert "nothing to add this time" in brief, "opting out must be explicitly fine"


def test_disabled_reflection_does_nothing(monkeypatch):
    monkeypatch.setenv("REFLECTION", "off")
    assert reflection.reflect("postgresql://unused", object(), {"id": "x"}, "closed") is False


def test_a_failed_reflection_never_raises():
    """A ticket that completed must not be turned into a failure by
    something optional going wrong afterwards."""
    class Broken:
        seat_id = "s"
        system_prompt = None
        model = None

    assert reflection.reflect("postgresql://bad-conn", Broken(), {"id": "x", "title": "t"}, "closed") is False
