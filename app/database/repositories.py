"""
Repository layer.

Every direct SQLAlchemy query in the application lives here. Services
call these functions instead of touching `db.query(...)` themselves. This
is what "API layer -> service layer -> repository layer" means in
practice: if we ever changed ORMs or added caching, only this file would
need to change.
"""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import Email, Review, RoutingResult, SecurityScreening


# ---------- Email ----------

def create_email(
    db: Session,
    sender: str,
    subject: str,
    body: str,
    attachment_text: str = "",
    attachment_filename: str | None = None,
    status: str = "PENDING",

) -> Email:
    email = Email(
        sender=sender,
        subject=subject,
        body=body,
        attachment_text=attachment_text,
        attachment_filename=attachment_filename,
        status=status,
    )

    db.add(email)
    db.commit()
    db.refresh(email)
    return email


def get_email(db: Session, email_id: int) -> Email | None:
    return db.get(Email, email_id)


def list_emails(db: Session, limit: int = 100, offset: int = 0) -> list[Email]:
    stmt = select(Email).order_by(Email.created_at.desc()).limit(limit).offset(offset)
    return list(db.scalars(stmt))


def update_email_status(db: Session, email_id: int, status: str) -> Email | None:
    email = db.get(Email, email_id)
    if email is None:
        return None
    email.status = status
    db.commit()
    db.refresh(email)
    return email


def list_emails_by_status(db: Session, status: str, limit: int = 100) -> list[Email]:
    """Find emails with a given status — used for recovery sweeps (e.g. KAFKA_FAILED)."""
    stmt = select(Email).where(Email.status == status).limit(limit)
    return list(db.scalars(stmt))


# ---------- RoutingResult ----------

def create_routing_result(
    db: Session,
    email_id: int,
    department: str,
    team: str,
    team_id: int,
    confidence: float,
    reasoning: str,
    retrieved_context: str = "",
    summary: str = "",
) -> RoutingResult:
    result = RoutingResult(
        email_id=email_id,
        department=department,
        team=team,
        team_id=team_id,
        confidence=confidence,
        reasoning=reasoning,
        retrieved_context=retrieved_context,
        summary=summary,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_routing_result(db: Session, email_id: int) -> RoutingResult | None:
    stmt = select(RoutingResult).where(RoutingResult.email_id == email_id)
    return db.scalars(stmt).first()


# ---------- Review ----------

def create_review(
    db: Session,
    email_id: int,
    predicted_team: str,
    predicted_team_id: int,
    predicted_confidence: float,
) -> Review:
    review = Review(
        email_id=email_id,
        predicted_team=predicted_team,
        predicted_team_id=predicted_team_id,
        predicted_confidence=predicted_confidence,
        reviewer_decision="PENDING",
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def get_review(db: Session, email_id: int) -> Review | None:
    stmt = select(Review).where(Review.email_id == email_id)
    return db.scalars(stmt).first()


def list_pending_reviews(db: Session) -> list[Review]:
    stmt = select(Review).where(Review.reviewer_decision == "PENDING")
    return list(db.scalars(stmt))


def submit_review_decision(
    db: Session,
    email_id: int,
    final_team: str,
    final_team_id: int,
    decision: str,  # "APPROVED" or "CORRECTED"
) -> Review | None:
    review = get_review(db, email_id)
    if review is None:
        return None
    review.final_team = final_team
    review.final_team_id = final_team_id
    review.reviewer_decision = decision
    review.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(review)
    return review


# ---------- SecurityScreening ----------

def create_security_screening(
    db: Session,
    email_id: int,
    decision: str,
    risk_score: float,
    explanation: str,
    detection_count: int = 0,
    sensitive_content_found: bool = False,
    phishing_signals_found: bool = False,
) -> SecurityScreening:
    screening = SecurityScreening(
        email_id=email_id,
        decision=decision,
        risk_score=risk_score,
        explanation=explanation,
        detection_count=detection_count,
        sensitive_content_found=sensitive_content_found,
        phishing_signals_found=phishing_signals_found,
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return screening


def get_security_screening(db: Session, email_id: int) -> SecurityScreening | None:
    stmt = select(SecurityScreening).where(SecurityScreening.email_id == email_id)
    return db.scalars(stmt).first()


# ---------- Analytics ----------

def get_analytics_summary(db: Session) -> dict:
    total = db.scalar(select(func.count()).select_from(Email)) or 0
    routed = db.scalar(select(func.count()).select_from(Email).where(Email.status == "ROUTED")) or 0
    review_required = (
        db.scalar(select(func.count()).select_from(Email).where(Email.status == "REVIEW_REQUIRED")) or 0
    )
    reviewed = (
        db.scalar(
            select(func.count()).select_from(Review).where(Review.reviewer_decision != "PENDING")
        )
        or 0
    )
    avg_confidence = db.scalar(select(func.avg(RoutingResult.confidence))) or 0.0

    quarantined = db.scalar(
        select(func.count()).select_from(Email).where(Email.status == "QUARANTINED")
    ) or 0
    security_review = db.scalar(
        select(func.count()).select_from(Email).where(Email.status == "SECURITY_REVIEW")
    ) or 0

    by_department_rows = db.execute(
        select(RoutingResult.department, func.count()).group_by(RoutingResult.department)
    ).all()
    by_team_rows = db.execute(
        select(RoutingResult.team, func.count()).group_by(RoutingResult.team)
    ).all()

    return {
        "total_emails": total,
        "auto_routed": routed,
        "human_reviewed": reviewed,
        "review_required": review_required,
        "quarantined": quarantined,
        "security_review": security_review,
        "average_confidence": round(float(avg_confidence), 4),
        "by_department": {row[0]: row[1] for row in by_department_rows},
        "by_team": {row[0]: row[1] for row in by_team_rows},
    }

