"""
Shared state passed between every node in the LangGraph workflow.

TypedDict rather than a Pydantic model: LangGraph's StateGraph natively
expects a TypedDict (or dataclass) it can merge partial updates into after
each node — every node returns only the keys it changed, and LangGraph
merges them into this shape. Using a full Pydantic model here would fight
that update-merging mechanism rather than work with it.
"""

from typing import TypedDict


class EmailState(TypedDict, total=False):
    email_id: int
    sender: str
    subject: str
    body: str
    attachment_text: str

    retrieved_context: str
    valid_team_ids: list[int]

    department: str
    team: str
    team_id: int
    confidence: float
    reasoning: str

    summary: str

    # PROCESSING | ROUTED | REVIEW_REQUIRED | FAILED
    status: str
    error: str
