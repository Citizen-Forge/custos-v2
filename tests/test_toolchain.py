"""Toolchain declaration and the dispatch preflight.

Guards the failure this was written for (2026-08-31): Silent Run is
TypeScript, the harness image had no Node, and nothing noticed. Agents
were dispatched onto tickets they could not build or test and produced
work that was unverified by construction, while the tickets looked
perfectly workable.
"""

import pytest

from harness import beads, dispatcher, toolchain
from harness.routing import RoutingTable


@pytest.fixture(autouse=True)
def _workspace():
    beads.ensure_initialized()


def test_missing_reports_absent_commands_only():
    gaps = toolchain.missing(["sh", "a-command-that-does-not-exist"])
    assert gaps == ["a-command-that-does-not-exist"]


def test_missing_is_empty_when_everything_is_present():
    assert toolchain.missing(["sh"]) == []


def test_project_id_is_the_root_of_a_ticket_id():
    assert toolchain.project_id_for("workspace-9jg.1.5") == "workspace-9jg"
    assert toolchain.project_id_for("workspace-9jg") == "workspace-9jg"


def test_declared_is_empty_when_nothing_is_set():
    project = beads.create("no toolchain proj", "d", issue_type="epic", priority=1)
    assert toolchain.declared_for(beads.show(project["id"])) == []


def test_declaration_round_trips():
    project = beads.create("tc proj", "d", issue_type="epic", priority=1)
    toolchain.set_for_project(project["id"], ["node", "npm"])
    assert toolchain.declared_for(beads.show(project["id"])) == ["node", "npm"]


def test_ticket_inherits_its_project_declaration():
    project = beads.create("tc2 proj", "d", issue_type="epic", priority=1)
    epic = beads.create("tc2 epic", "d", issue_type="epic", parent=project["id"])
    story = beads.create("tc2 story", "d", parent=epic["id"])
    toolchain.set_for_project(project["id"], ["definitely-not-installed"])

    assert toolchain.check_ticket(story["id"]) == ["definitely-not-installed"]


def test_undeclared_project_never_blocks():
    """Absence means 'no requirements', not 'requires nothing available'
    -- otherwise adding this check would retroactively block all existing
    work."""
    project = beads.create("tc3 proj", "d", issue_type="epic", priority=1)
    story = beads.create("tc3 story", "d", parent=project["id"])
    assert toolchain.check_ticket(story["id"]) == []


def test_check_fails_open_for_an_unreadable_project():
    """A broken preflight must not become the reason nothing ever runs."""
    assert toolchain.check_ticket("no-such-project.1.1") == []


def test_dispatch_refuses_a_ticket_whose_toolchain_is_missing(monkeypatch):
    project = beads.create("tc4 proj", "d", issue_type="epic", priority=1)
    story = beads.create("tc4 story", "d", parent=project["id"])
    beads.assign_to_seat(story["id"], "tc-seat")
    toolchain.set_for_project(project["id"], ["definitely-not-installed"])

    # Pin which ticket dispatch sees. Without this the test passes alone
    # and fails in the full suite: next_assigned_ticket scans the whole
    # workspace, so whichever assigned ticket other tests happened to
    # leave behind gets picked first.
    monkeypatch.setattr(
        dispatcher, "next_assigned_ticket", lambda: (beads.show(story["id"]), "tc-seat")
    )

    d = dispatcher.Dispatcher("postgresql://unused", RoutingTable({}), max_agents=1)
    assert d.tick() == "blocked on toolchain"

    # And crucially it did not claim or start anything.
    assert beads.show(story["id"])["status"] == "open"


def test_report_flags_an_unsatisfied_project():
    project = beads.create("tc5 proj", "d", issue_type="epic", priority=1)
    toolchain.set_for_project(project["id"], ["sh", "definitely-not-installed"])

    entry = next(r for r in toolchain.report() if r["project_id"] == project["id"])
    assert entry["satisfied"] is False
    assert entry["missing"] == ["definitely-not-installed"]


def test_report_ignores_projects_with_no_declaration():
    project = beads.create("tc6 proj", "d", issue_type="epic", priority=1)
    assert all(r["project_id"] != project["id"] for r in toolchain.report())


def test_node_and_npm_are_actually_installed():
    """The concrete regression: the harness image must carry the Node
    toolchain Silent Run needs. This is the check whose absence let
    agents be dispatched onto work they could not do."""
    assert toolchain.missing(["node", "npm"]) == []
