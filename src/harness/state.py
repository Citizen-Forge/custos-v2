from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class HarnessState(TypedDict):
    messages: Annotated[list, add_messages]
    ticket_id: str
