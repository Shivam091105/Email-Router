"""
Email endpoints.

Route handlers stay thin: validate input via Pydantic, call the service
layer, translate results into response schemas and HTTP status codes.
No direct database queries here — that belongs in the repository layer.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_llm_client, get_vectorstore
from app.database import repositories
from app.database.database import get_db
from app.llm.client import LLMClient
from app.schemas.email import EmailCreate, EmailDetailOut, EmailOut, SummaryOut
from app.services.email_service import create_email, process_email

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/emails", tags=["emails"])


@router.post("", response_model=EmailOut, status_code=201)
def submit_email(
    payload: EmailCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
    vectorstore=Depends(get_vectorstore),
):
    email = create_email(db, sender=payload.sender, subject=payload.subject, body=payload.body)

    # Returns immediately; the actual LLM/RAG work happens after the
    # response is sent. See app/services/email_service.py's docstring for
    # why this matters. llm_client/vectorstore are resolved via Depends
    # (not called directly) specifically so tests can override them with
    # fakes via app.dependency_overrides.
    background_tasks.add_task(process_email, email.id, llm_client, vectorstore)

    return email


@router.get("", response_model=list[EmailOut])
def list_all_emails(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return repositories.list_emails(db, limit=limit, offset=offset)


@router.get("/{email_id}", response_model=EmailDetailOut)
def get_email_detail(email_id: int, db: Session = Depends(get_db)):
    email = repositories.get_email(db, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")
    return email


@router.get("/{email_id}/classification")
def get_email_classification(email_id: int, db: Session = Depends(get_db)):
    email = repositories.get_email(db, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

    result = repositories.get_routing_result(db, email_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Email {email_id} has not been classified yet (status={email.status})",
        )
    return {
        "department": result.department,
        "team": result.team,
        "team_id": result.team_id,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "retrieved_context": result.retrieved_context,
    }


@router.get("/{email_id}/summary", response_model=SummaryOut)
def get_email_summary(email_id: int, db: Session = Depends(get_db)):
    email = repositories.get_email(db, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

    result = repositories.get_routing_result(db, email_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Email {email_id} has not been processed yet (status={email.status})",
        )
    return SummaryOut(email_id=email_id, summary=result.summary)
