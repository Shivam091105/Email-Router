from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import repositories
from app.database.database import get_db
from app.schemas.email import AnalyticsOut

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsOut)
def get_analytics(db: Session = Depends(get_db)):
    return repositories.get_analytics_summary(db)
