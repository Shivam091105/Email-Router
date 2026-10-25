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

    # Full status lifecycle with Kafka + security screening:
    #   PENDING_SECURITY -> (security screening) ->
    #     APPROVED -> PROCESSING -> ROUTED | REVIEW_REQUIRED -> RESOLVED
    #     SECURITY_REVIEW  (flagged, awaiting human review)
    #     QUARANTINED      (high-risk, blocked from processing)
    #   KAFKA_FAILED       (Kafka publish failed; eligible for retry sweep)
    #   FAILED             (workflow error: LLM failure, etc.)
    #
    # Legacy status PENDING is kept for backward compatibility with
    # existing emails created before Kafka integration.
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING_SECURITY")

    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    routing_result: Mapped["RoutingResult"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )
    review: Mapped["Review"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )
    security_screening: Mapped["SecurityScreening"] = relationship(
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


class SecurityScreening(Base):
    """
    Records the outcome of the security screening step for one email.

    Persists the decision (APPROVED/REVIEW_REQUIRED/QUARANTINED), a
    numeric risk score, a human-readable explanation of what was found,
    and the number of detections — but NEVER the actual detected secret
    values themselves.
    """
    __tablename__ = "security_screenings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), unique=True, nullable=False)

    decision: Mapped[str] = mapped_column(String(50), nullable=False)  # APPROVED | REVIEW_REQUIRED | QUARANTINED
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sensitive_content_found: Mapped[bool] = mapped_column(nullable=False, default=False)
    phishing_signals_found: Mapped[bool] = mapped_column(nullable=False, default=False)

    screened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    email: Mapped["Email"] = relationship(back_populates="security_screening")
