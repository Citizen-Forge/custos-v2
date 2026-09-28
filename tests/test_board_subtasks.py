"""The board tree must carry a story's own subtasks (the 4th id level), so the
dashboard can show them as indented subcards. They were dropped before."""

from harness import api


def test_a_storys_subtasks_are_carried_in_the_tree():
    issues = [
        {"id": "proj", "issue_type": "epic", "priority": 1, "title": "p"},
        {"id": "proj.1", "issue_type": "epic", "priority": 1, "title": "e"},
        {"id": "proj.1.1", "issue_type": "task", "priority": 1, "title": "s"},
        {"id": "proj.1.1.1", "issue_type": "task", "priority": 1, "title": "sub"},
        {"id": "proj.1.1.2", "issue_type": "task", "priority": 2, "title": "sub2"},
    ]

    tree = api._tree_from_flat(issues)

    story = tree[0]["epics"][0]["stories"][0]
    assert [s["id"] for s in story["subtasks"]] == ["proj.1.1.1", "proj.1.1.2"]
