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


def process_parsed_email_sync(parsed: dict, llm_client, vectorstore) -> bool:
    """
    Entry point for IMAP-fetched emails (POST /emails/fetch). Creates the
    Email row from a parsed IMAP message dict (see email_parser.py for
    its shape) and runs it through process_email() — the exact same
    function POST /emails hands to BackgroundTasks.

    Called synchronously rather than via BackgroundTasks: /emails/fetch
    is an on-demand batch trigger, not a latency-sensitive per-request
    endpoint, and the IMAP layer needs to know per-message whether to
    mark it \\Seen before moving to the next one — that requires waiting
    for the result, which BackgroundTasks doesn't give us. This is the
    ONE place IMAP-sourced and API-sourced emails diverge; everything
    downstream of create_email() is identical.

    Returns True if the email was durably created and handed to the
    pipeline — regardless of the classification OUTCOME. ROUTED,
    REVIEW_REQUIRED, and FAILED are all legitimate terminal states that
    process_email() already persists correctly; none of them mean the
    IMAP message should be re-fetched. This returns False only if the
    email couldn't even be created (e.g. the database was unreachable),
    which IS worth retrying on the next fetch.
    """
    body = parsed.get("body_text") or parsed.get("body_html") or ""
    attachments = parsed.get("attachments") or []
    attachment_filename = attachments[0]["filename"] if attachments else None

    db = SessionLocal()
    try:
        email = create_email(
            db,
            sender=parsed.get("sender", ""),
            subject=parsed.get("subject", ""),
            body=body,
            attachment_filename=attachment_filename,
        )
        email_id = email.id
    except Exception:
        logger.exception("Failed to create Email row from IMAP message (sender=%s)", parsed.get("sender"))
        return False
    finally:
        db.close()

    # process_email() already catches its own exceptions internally and
    # writes a FAILED status rather than raising further — see its
    # docstring above. We don't need a try/except here for that reason.
    process_email(email_id, llm_client, vectorstore)
    return True


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