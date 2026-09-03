"""
Database models.

Three tables, matching the three real entities in this system:

- Email          — the raw inbound email as received.
- RoutingResult  — the AI's classification output for that email (1:1 with Email).
- Review         — a human's correction, only created for emails that
                    needed review (0:1 with Email).

Relationships are declared both directions (Email.routing_result,
RoutingResult.email) so repository/service code can navigate either way
without writing manual joins for the common cases.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sender: Mapped[str] = mapped_column(String(255), nullable=False)
    subject: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    attachment_text: Mapped[str] = mapped_column(Text, nullable=True, default="")
    attachment_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # PENDING -> PROCESSING -> ROUTED | REVIEW_REQUIRED -> RESOLVED
    # FAILED is used if the workflow could not complete (e.g. LLM failure).
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")

    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    routing_result: Mapped["RoutingResult"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )
    review: Mapped["Review"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )


class RoutingResult(Base):
    __tablename__ = "routing_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), unique=True, nullable=False)

    department: Mapped[str] = mapped_column(String(255), nullable=False)
    team: Mapped[str] = mapped_column(String(255), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False, default="")
    retrieved_context: Mapped[str] = mapped_column(Text, nullable=True, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=True, default="")

    routed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    email: Mapped["Email"] = relationship(back_populates="routing_result")


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), unique=True, nullable=False)

    predicted_team: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    final_team: Mapped[str | None] = mapped_column(String(255), nullable=True)
    final_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewer_decision: Mapped[str] = mapped_column(
        String(50), nullable=False, default="PENDING"
    )  # PENDING | APPROVED | CORRECTED

    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    email: Mapped["Email"] = relationship(back_populates="review")
