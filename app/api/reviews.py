"""
Human-in-the-loop review endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import repositories
from app.database.database import get_db
from app.schemas.email import ReviewDecision, ReviewOut

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/pending", response_model=list[ReviewOut])
def list_pending_reviews(db: Session = Depends(get_db)):
    return repositories.list_pending_reviews(db)


@router.post("/{email_id}", response_model=ReviewOut)
def submit_review(email_id: int, decision: ReviewDecision, db: Session = Depends(get_db)):
    review = repositories.get_review(db, email_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"No pending review for email {email_id}")

    was_correction = decision.final_team_id != review.predicted_team_id
    updated = repositories.submit_review_decision(
        db,
        email_id=email_id,
        final_team=decision.final_team,
        final_team_id=decision.final_team_id,
        decision="CORRECTED" if was_correction else "APPROVED",
    )
    repositories.update_email_status(db, email_id, "RESOLVED")
    return updated
