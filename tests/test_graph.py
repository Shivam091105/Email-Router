"""
LangGraph workflow tests.

Uses FakeLLMClient (no network) and a small in-memory Chroma store built
with DeterministicFakeEmbeddings (no network), plus an in-memory SQLite
database (no Postgres required) — the whole graph runs fully in-process.

We test the three branches the confidence_check conditional edge can take:
high confidence -> route_email, low confidence -> human_review, and the
failure path (invalid LLM output, or an empty email) -> straight to
save_result with status FAILED.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import models  # noqa: F401 - ensures models are registered on Base
from app.database import repositories
from app.database.database import Base
from app.graph.workflow import run_email_workflow
from app.llm.client import FakeLLMClient
from app.rag.embeddings import DeterministicFakeEmbeddings
from app.rag.vectorstore import build_vectorstore

SAMPLE_DOCS = [
    Document(
        page_content="Team: IT Support. Handles login problems and password resets.",
        metadata={"department": "IT", "team": "IT Support", "team_id": 101},
    ),
    Document(
        page_content="Team: Billing. Handles refunds and payment failures.",
        metadata={"department": "Finance", "team": "Billing", "team_id": 201},
    ),
]


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def vectorstore(tmp_path):
    embedding = DeterministicFakeEmbeddings()
    return build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))


def _make_email(session_factory, sender="a@b.com", subject="subj", body="body"):
    db = session_factory()
    email = repositories.create_email(db, sender=sender, subject=subject, body=body)
    email_id = email.id
    db.close()
    return email_id


def test_high_confidence_routes_automatically(session_factory, vectorstore):
    email_id = _make_email(session_factory, body="I can't log into my account")
    valid_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.95, "reasoning": "Login issue."}
    )
    llm = FakeLLMClient(responses=[valid_json, "Cannot log in."])

    state = run_email_workflow(
        llm, vectorstore, session_factory, email_id, "a@b.com", "subj", "I can't log into my account"
    )

    assert state["status"] == "ROUTED"
    assert state["team_id"] == 101

    db = session_factory()
    assert repositories.get_email(db, email_id).status == "ROUTED"
    assert repositories.get_routing_result(db, email_id).team_id == 101
    assert repositories.get_review(db, email_id) is None  # no review row for auto-routed emails
    db.close()


def test_routed_email_triggers_smtp_notification(session_factory, vectorstore):
    """
    Regression test for the IMAP/SMTP integration: once an email is
    auto-routed, notify_destination_team() must actually be called (it
    previously existed but was dead code, never wired into the graph).
    We patch send_routing_notification itself rather than requiring a
    real SMTP server.
    """
    email_id = _make_email(session_factory, sender="cust@gmail.com", subject="help", body="I can't log into my account")
    valid_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.95, "reasoning": "Login issue."}
    )
    llm = FakeLLMClient(responses=[valid_json, "Cannot log in."])

    with patch("app.services.routing_service.send_routing_notification") as mock_send:
        run_email_workflow(
            llm, vectorstore, session_factory, email_id, "cust@gmail.com", "help", "I can't log into my account"
        )

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["original_sender"] == "cust@gmail.com"
    assert call_kwargs["team_name"] == "IT Support"


def test_review_required_email_does_not_trigger_smtp_notification(session_factory, vectorstore):
    """Low-confidence emails must NOT be forwarded until a human confirms
    the team via POST /reviews/{id} — see save_result's comment."""
    email_id = _make_email(session_factory, body="something ambiguous")
    low_conf_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.3, "reasoning": "Weak match."}
    )
    llm = FakeLLMClient(responses=[low_conf_json, "Ambiguous request."])

    with patch("app.services.routing_service.send_routing_notification") as mock_send:
        run_email_workflow(llm, vectorstore, session_factory, email_id, "a@b.com", "subj", "something ambiguous")

    mock_send.assert_not_called()


def test_low_confidence_goes_to_human_review(session_factory, vectorstore):
    email_id = _make_email(session_factory, body="something ambiguous")
    low_conf_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.3, "reasoning": "Weak match."}
    )
    llm = FakeLLMClient(responses=[low_conf_json, "Ambiguous request."])

    state = run_email_workflow(
        llm, vectorstore, session_factory, email_id, "a@b.com", "subj", "something ambiguous"
    )

    assert state["status"] == "REVIEW_REQUIRED"

    db = session_factory()
    review = repositories.get_review(db, email_id)
    assert review is not None
    assert review.reviewer_decision == "PENDING"
    assert review.predicted_team_id == 101
    db.close()


def test_classification_failure_marks_email_failed(session_factory, vectorstore):
    email_id = _make_email(session_factory, body="some content")
    llm = FakeLLMClient(responses=["not json", "still not json"])

    state = run_email_workflow(llm, vectorstore, session_factory, email_id, "a@b.com", "subj", "some content")

    assert state["status"] == "FAILED"
    db = session_factory()
    assert repositories.get_email(db, email_id).status == "FAILED"
    assert repositories.get_routing_result(db, email_id) is None
    db.close()


def test_empty_email_fails_before_calling_llm(session_factory, vectorstore):
    email_id = _make_email(session_factory, body="  ")
    llm = FakeLLMClient(responses=[])  # would raise if ever called

    state = run_email_workflow(llm, vectorstore, session_factory, email_id, "a@b.com", "", "  ")

    assert state["status"] == "FAILED"
    assert llm.calls == []  # confirms we never reached retrieval/classification