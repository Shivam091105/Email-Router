"""
LangGraph node implementations.

Every node is a plain function `state -> partial_state_dict`, which is
what LangGraph expects: nodes don't mutate state in place, they return
only the keys they changed and LangGraph merges the result in.

Nodes need access to shared resources (an LLM client, the vector store, a
DB session) that don't belong inside the graph's State itself — State
should hold data about *this one email*, not shared infrastructure. So
each node is produced by a small factory function that closes over those
dependencies. `app/graph/workflow.py` calls these factories once at
startup; `tests/test_graph.py` calls them with fake dependencies.
"""

import logging

from langchain_chroma import Chroma
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import repositories
from app.graph.state import EmailState
from app.llm.client import LLMClient
from app.rag.retriever import format_context, retrieve_teams
from app.schemas.classification import ClassificationError
from app.services.classification_service import classify_email as run_classification
from app.services.summary_service import generate_summary as run_summary_generation

logger = logging.getLogger(__name__)


def make_preprocess_email():
    def preprocess_email(state: EmailState) -> dict:
        subject = (state.get("subject") or "").strip()
        body = (state.get("body") or "").strip()
        attachment_text = (state.get("attachment_text") or "").strip()

        if not body and not attachment_text:
            logger.error("Email %s has no body or attachment text", state.get("email_id"))
            return {"status": "FAILED", "error": "Email has no content to classify"}

        return {"subject": subject, "body": body, "attachment_text": attachment_text, "status": "PROCESSING"}

    return preprocess_email


def make_retrieve_context(vectorstore: Chroma, top_k: int | None = None):
    k = top_k or settings.retrieval_top_k

    def retrieve_context(state: EmailState) -> dict:
        query_text = f"{state.get('subject', '')}\n{state.get('body', '')}\n{state.get('attachment_text', '')}".strip()
        try:
            results = retrieve_teams(vectorstore, query_text, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for email %s: %s", state.get("email_id"), exc)
            return {"status": "FAILED", "error": f"Retrieval failed: {exc}"}

        return {
            "retrieved_context": format_context(results),
            "valid_team_ids": [doc.metadata["team_id"] for doc, _score in results],
        }

    return retrieve_context


def make_classify_email(llm_client: LLMClient):
    def classify_email(state: EmailState) -> dict:
        try:
            result = run_classification(
                llm_client=llm_client,
                subject=state.get("subject", ""),
                body=state.get("body", ""),
                context=state.get("retrieved_context", ""),
                valid_team_ids=set(state.get("valid_team_ids", [])),
            )
        except ClassificationError as exc:
            logger.error("Classification failed for email %s: %s", state.get("email_id"), exc)
            return {"status": "FAILED", "error": str(exc)}

        return {
            "department": result.department,
            "team": result.team,
            "team_id": result.team_id,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    return classify_email


def decide_after_classification(state: EmailState) -> str:
    """
    Conditional edge function (not a regular node): reads state, returns
    the name of the next branch. Does not modify state itself.
    """
    if state.get("status") == "FAILED":
        return "failed"
    if state.get("confidence", 0.0) >= settings.confidence_threshold:
        return "high_confidence"
    return "low_confidence"


def make_route_email():
    def route_email(state: EmailState) -> dict:
        logger.info(
            "Auto-routing email %s to team_id=%s (%s), confidence=%.2f",
            state.get("email_id"), state.get("team_id"), state.get("team"), state.get("confidence", 0.0),
        )
        # A real SMTP notification to the destination team would happen here
        # (see app/integrations/smtp_client.py) — omitted from the graph
        # itself so the workflow's core logic can be tested without a mail
        # server, per app/integrations/smtp_client.py's own docstring.
        return {"status": "ROUTED"}

    return route_email


def make_human_review():
    def human_review(state: EmailState) -> dict:
        logger.info(
            "Email %s marked REVIEW_REQUIRED: predicted team_id=%s, confidence=%.2f below threshold %.2f",
            state.get("email_id"), state.get("team_id"), state.get("confidence", 0.0), settings.confidence_threshold,
        )
        return {"status": "REVIEW_REQUIRED"}

    return human_review


def make_generate_summary(llm_client: LLMClient):
    def generate_summary(state: EmailState) -> dict:
        summary = run_summary_generation(
            llm_client=llm_client,
            subject=state.get("subject", ""),
            body=state.get("body", ""),
        )
        return {"summary": summary}

    return generate_summary


def make_save_result(session_factory):
    """
    `session_factory` is a zero-arg callable returning a new SQLAlchemy
    Session (e.g. `SessionLocal`), so this node can open/close its own
    short-lived session rather than sharing one across the whole graph run.
    """

    def save_result(state: EmailState) -> dict:
        db: Session = session_factory()
        try:
            email_id = state["email_id"]
            status = state.get("status", "FAILED")

            repositories.update_email_status(db, email_id, status)

            if status == "FAILED":
                logger.error("Saving FAILED result for email %s: %s", email_id, state.get("error"))
                return {}

            repositories.create_routing_result(
                db,
                email_id=email_id,
                department=state.get("department", ""),
                team=state.get("team", ""),
                team_id=state.get("team_id", 0),
                confidence=state.get("confidence", 0.0),
                reasoning=state.get("reasoning", ""),
                retrieved_context=state.get("retrieved_context", ""),
                summary=state.get("summary", ""),
            )

            if status == "REVIEW_REQUIRED":
                repositories.create_review(
                    db,
                    email_id=email_id,
                    predicted_team=state.get("team", ""),
                    predicted_team_id=state.get("team_id", 0),
                    predicted_confidence=state.get("confidence", 0.0),
                )

            return {}
        finally:
            db.close()

    return save_result
