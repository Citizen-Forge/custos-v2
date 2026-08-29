"""
File-based project wiki -- human-facing documentation, distinct from
Beads' notes/comments (see wiki.py's module docstring). Runs against the
real filesystem under the test session's isolated HARNESS_WORKSPACE
(conftest.py), same as the file tools in test_graph.py/tools.py exercise
-- proves real file I/O and the real workspace-boundary check, not a
mock.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from harness import wiki
from harness.graph import build_graph_from_model
from harness.permissions import PermissionDenied


def test_read_missing_page_returns_none_not_an_error():
    assert wiki.read_page(f"never-written-{id(object())}") is None


def test_write_then_read_round_trips():
    slug = f"topic-{id(object())}"
    wiki.write_page(slug, "# Hello\n\nSome docs.")

    assert wiki.read_page(slug) == "# Hello\n\nSome docs."


def test_write_page_normalizes_slug_to_md_extension():
    slug = f"already-has-ext-{id(object())}"
    wiki.write_page(f"{slug}.md", "content")

    assert wiki.read_page(slug) == "content"  # same page, extension-agnostic lookup


def test_list_pages_includes_written_pages():
    slug = f"listed-page-{id(object())}"
    wiki.write_page(slug, "content")

    assert slug in wiki.list_pages()


def test_agent_profile_slug_convention():
    assert wiki.agent_profile_slug("some-seat") == "agents/some-seat"


def test_write_page_rejects_escaping_the_workspace():
    import pytest

    with pytest.raises(PermissionDenied):
        wiki.write_page("../../etc/passwd", "malicious")


class WritesWikiPageThenStops:
    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_wiki_page",
                    "args": {"slug": "tool-written-page", "content": "written via the tool"},
                    "id": "call-1",
                }
            ],
        )


def test_write_wiki_page_tool_writes_a_real_page():
    graph = build_graph_from_model(WritesWikiPageThenStops(), InMemorySaver())
    graph.invoke(
        {"messages": [("user", "document this")], "ticket_id": "t1", "turn_count": 0},
        {"configurable": {"thread_id": "t1"}},
    )

    assert wiki.read_page("tool-written-page") == "written via the tool"


class ReadsWikiPageThenStops:
    def invoke(self, messages):
        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"got: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[{"name": "read_wiki_page", "args": {"slug": "pre-existing-page"}, "id": "call-1"}],
        )


def test_read_wiki_page_tool_reads_a_real_page():
    wiki.write_page("pre-existing-page", "the real content")

    graph = build_graph_from_model(ReadsWikiPageThenStops(), InMemorySaver())
    result = graph.invoke(
        {"messages": [("user", "check the doc")], "ticket_id": "t2", "turn_count": 0},
        {"configurable": {"thread_id": "t2"}},
    )

    assert "the real content" in result["messages"][-1].content
