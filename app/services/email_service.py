"""
Email service.

Owns the two operations the API needs around emails: creating one (and
persisting it immediately, in PENDING status) and processing one (running
the full LangGraph workflow, meant to be called from a FastAPI
BackgroundTask so the POST /emails request returns immediately rather
than blocking on LLM calls).

Why background processing here specifically: LLM inference and (later)
SMTP delivery are the slow, occasionally-flaky parts of this system. If
`process_email` ran synchronously inside the POST handler, a slow or
hung LLM call would hold the HTTP connection open and block the FastAPI
worker from serving other requests in the meantime.
"""

import logging

from sqlalchemy.orm import Session

from app.database import repositories
from app.database.database import SessionLocal
from app.graph.workflow import run_email_workflow

logger = logging.getLogger(__name__)


def create_email(
    db: Session,
    sender: str,
    subject: str,
    body: str,
    attachment_text: str = "",
    attachment_filename: str | None = None,
):
    return repositories.create_email(
        db,
        sender=sender,
        subject=subject,
        body=body,
        attachment_text=attachment_text,
        attachment_filename=attachment_filename,
    )


def process_email(email_id: int, llm_client, vectorstore) -> None:
    """
    Runs the LangGraph workflow for one email and persists the result.

    Designed to be handed to FastAPI's BackgroundTasks, so it opens its
    own DB session (via SessionLocal, the app's real session factory)
    rather than reusing the request-scoped session from `get_db()`, which
    will already be closed by the time this runs.
    """
    db = SessionLocal()
    try:
        email = repositories.get_email(db, email_id)
        if email is None:
            logger.error("process_email called for nonexistent email_id=%s", email_id)
            return
        sender, subject, body, attachment_text = email.sender, email.subject, email.body, email.attachment_text
    finally:
        db.close()

    try:
        run_email_workflow(
            llm_client=llm_client,
            vectorstore=vectorstore,
            session_factory=SessionLocal,
            email_id=email_id,
            sender=sender,
            subject=subject,
            body=body,
            attachment_text=attachment_text or "",
        )
    except Exception:
        # The graph's own save_result node already handles expected
        # failures (invalid LLM output, empty email) by writing FAILED to
        # the DB. This except is the last-resort catch for anything
        # unexpected (e.g. a DB connection drop mid-workflow), so a
        # background task crash is at least logged instead of vanishing
        # silently.
        logger.exception("Unhandled error processing email_id=%s", email_id)
        db2 = SessionLocal()
        try:
            repositories.update_email_status(db2, email_id, "FAILED")
        finally:
            db2.close()
