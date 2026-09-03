"""Guards the assumption that makes /projects fast.

api.list_projects used to walk Beads' hierarchy with one `bd` call per
project and per epic. That was replaced (2026-08-31) with a single
`bd list --all` whose result is reassembled in-process -- but `bd list`
returns no parent field, so the tree is rebuilt from the dotted id
convention (`workspace-x` -> `workspace-x.1` -> `workspace-x.1.1`).

That convention is bd's, not ours, and nothing in bd promises it. If it
ever changes, the prefix logic would not raise -- it would quietly
produce projects with no epics, i.e. an empty board that looks like
"no work yet" rather than a bug. These tests exist so that failure is
loud and lands here instead.
"""

import pytest

from harness import api, beads


@pytest.fixture(autouse=True)
def _workspace():
    """api.py does this in its lifespan hook; tests calling beads
    directly have to do it themselves or bd has no database to talk to."""
    beads.ensure_initialized()


def _walk_tree_via_parent_calls():
    """The original N+1 implementation, kept here purely as the oracle
    the fast path is checked against."""
    tree = []
    for project in beads.list_top_level(issue_type="epic"):
        epics = beads.children_of(project["id"])
        for epic in epics:
            epic["stories"] = beads.children_of(epic["id"])
        project["epics"] = epics
        tree.append(project)
    return tree


def _shape(tree):
    """Ids only -- the two builders sort independently, so compare the
    structure rather than whole issue dicts."""
    return {
        project["id"]: {
            epic["id"]: sorted(story["id"] for story in epic.get("stories", []))
            for epic in project.get("epics", [])
        }
        for project in tree
    }


def test_prefix_tree_matches_parent_walk_on_real_bd_data():
    project = beads.create("Tree probe project", "d", issue_type="epic", priority=1)
    epic_a = beads.create("Epic A", "d", issue_type="epic", parent=project["id"])
    epic_b = beads.create("Epic B", "d", issue_type="epic", parent=project["id"])
    beads.create("Story A1", "d", parent=epic_a["id"])
    beads.create("Story A2", "d", parent=epic_a["id"])
    beads.create("Story B1", "d", parent=epic_b["id"])

    fast = _shape(api._tree_from_flat(beads.list_all()))
    oracle = _shape(_walk_tree_via_parent_calls())

    assert fast == oracle, (
        "id-prefix tree diverged from bd's own --parent walk -- bd's id "
        "format may have changed; see this module's docstring"
    )
    assert len(fast[project["id"]]) == 2
    assert len(fast[project["id"]][epic_a["id"]]) == 2


def test_top_level_tasks_are_not_rendered_as_projects():
    """Regression guard carried over from the N+1 version: only
    top-level issues of type `epic` are projects. Plain top-level tasks
    (ad-hoc tickets) must not appear as bare projects on the board."""
    stray = beads.create("Stray top-level task", "d")
    tree = api._tree_from_flat(beads.list_all())
    assert stray["id"] not in {project["id"] for project in tree}


def test_parent_id_derivation():
    assert api._parent_id("workspace-9jg") is None
    assert api._parent_id("workspace-9jg.1") == "workspace-9jg"
    assert api._parent_id("workspace-9jg.1.1") == "workspace-9jg.1"


def test_children_sort_numerically_not_lexically():
    """A plain string sort puts `.10` before `.2`, which visibly
    mis-ordered epics once a project passed nine of them."""
    issues = [
        {"id": "p.10", "priority": 2},
        {"id": "p.2", "priority": 2},
        {"id": "p.1", "priority": 2},
    ]
    assert [i["id"] for i in sorted(issues, key=api._sort_key)] == ["p.1", "p.2", "p.10"]


def test_priority_beats_id_order():
    issues = [
        {"id": "p.1", "priority": 3},
        {"id": "p.2", "priority": 0},
    ]
    assert [i["id"] for i in sorted(issues, key=api._sort_key)] == ["p.2", "p.1"]
