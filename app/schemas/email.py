"""
Pydantic schemas — the API's request/response contracts.

Kept separate from the SQLAlchemy models in app/database/models.py on
purpose: the API's shape and the database's shape are allowed to diverge
(e.g. we may want to expose a computed field or hide an internal column),
and coupling them makes both harder to change independently.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class EmailCreate(BaseModel):
    sender: str = Field(..., min_length=1, max_length=255)
    subject: str = Field(default="", max_length=500)
    body: str = Field(..., min_length=1)


class RoutingResultOut(BaseModel):
    department: str
    team: str
    team_id: int
    confidence: float
    reasoning: str
    retrieved_context: str
    summary: str
    routed_at: datetime

    model_config = {"from_attributes": True}


class EmailOut(BaseModel):
    id: int
    sender: str
    subject: str
    body: str
    status: str
    received_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class EmailDetailOut(EmailOut):
    routing_result: RoutingResultOut | None = None


class SummaryOut(BaseModel):
    email_id: int
    summary: str


class ReviewOut(BaseModel):
    id: int
    email_id: int
    predicted_team: str
    predicted_team_id: int
    predicted_confidence: float
    final_team: str | None
    final_team_id: int | None
    reviewer_decision: str
    created_at: datetime
    reviewed_at: datetime | None

    model_config = {"from_attributes": True}


class ReviewDecision(BaseModel):
    final_team: str = Field(..., min_length=1)
    final_team_id: int


class AnalyticsOut(BaseModel):
    total_emails: int
    auto_routed: int
    human_reviewed: int
    review_required: int
    average_confidence: float
    by_department: dict[str, int]
    by_team: dict[str, int]
