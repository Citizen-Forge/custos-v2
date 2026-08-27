from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class HarnessState(TypedDict):
    messages: Annotated[list, add_messages]
    ticket_id: str
    # Plain overwrite field (no reducer) -- last write wins, which is what
    # we want for a running count. Phase 4's bounded-workday nudge in
    # graph.py reads this. Scoped to one ticket/thread for now, not a
    # cross-ticket "seat" -- see PLAN.md Phase 4 for what's still open.
    turn_count: int
