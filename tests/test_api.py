"""
API endpoint tests.

Uses the `client` fixture (see conftest.py): a real FastAPI TestClient
wired to an isolated in-memory SQLite database. The LLM client and
vectorstore are overridden with fakes via `app.dependency_overrides`, so
these tests exercise the full request -> background task -> DB write ->
response cycle with zero network calls.
"""

import json
from unittest.mock import patch

from langchain_core.documents import Document

from app.core.dependencies import get_llm_client, get_vectorstore
from app.integrations.imap_client import IMAPConnectionError
from app.llm.client import FakeLLMClient
from app.main import app
from app.rag.embeddings import DeterministicFakeEmbeddings
from app.rag.vectorstore import build_vectorstore

SAMPLE_DOCS = [
    Document(
        page_content="Team: IT Support. Handles login problems and password resets.",
        metadata={"department": "IT", "team": "IT Support", "team_id": 101},
    ),
]


def _override_ai_deps(vectorstore, llm_responses):
    fake_llm = FakeLLMClient(responses=llm_responses)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_vectorstore] = lambda: vectorstore
    return fake_llm


def test_submit_email_returns_pending_immediately(client, tmp_path):
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    valid_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.9, "reasoning": "login issue"}
    )
    _override_ai_deps(vs, [valid_json, "Cannot log in."])

    response = client.post("/emails", json={"sender": "a@b.com", "subject": "help", "body": "I can't log in"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "PENDING"  # response returns before background processing completes
    assert "id" in body


def test_full_pipeline_high_confidence_routes(client, tmp_path):
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    valid_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.9, "reasoning": "login issue"}
    )
    _override_ai_deps(vs, [valid_json, "Cannot log in."])

    create_resp = client.post("/emails", json={"sender": "a@b.com", "subject": "help", "body": "I can't log in"})
    email_id = create_resp.json()["id"]

    detail = client.get(f"/emails/{email_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "ROUTED"
    assert detail.json()["routing_result"]["team_id"] == 101

    classification = client.get(f"/emails/{email_id}/classification")
    assert classification.status_code == 200
    assert classification.json()["team_id"] == 101

    summary = client.get(f"/emails/{email_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["summary"] == "Cannot log in."


def test_full_pipeline_low_confidence_creates_pending_review(client, tmp_path):
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    low_conf_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.3, "reasoning": "unsure"}
    )
    _override_ai_deps(vs, [low_conf_json, "Ambiguous."])

    create_resp = client.post("/emails", json={"sender": "a@b.com", "subject": "q", "body": "something unclear"})
    email_id = create_resp.json()["id"]

    detail = client.get(f"/emails/{email_id}")
    assert detail.json()["status"] == "REVIEW_REQUIRED"

    pending = client.get("/reviews/pending")
    assert pending.status_code == 200
    assert len(pending.json()) == 1
    assert pending.json()[0]["email_id"] == email_id

    decision = client.post(f"/reviews/{email_id}", json={"final_team": "Billing", "final_team_id": 201})
    assert decision.status_code == 200
    assert decision.json()["reviewer_decision"] == "CORRECTED"

    # Now resolved, should no longer appear in pending
    pending_after = client.get("/reviews/pending")
    assert pending_after.json() == []

    resolved_email = client.get(f"/emails/{email_id}")
    assert resolved_email.json()["status"] == "RESOLVED"


def test_get_nonexistent_email_returns_404(client):
    response = client.get("/emails/99999")
    assert response.status_code == 404


def test_classification_before_processing_returns_409(client, db_session):
    from app.database import repositories

    email = repositories.create_email(db_session, sender="a@b.com", subject="x", body="y")
    response = client.get(f"/emails/{email.id}/classification")
    assert response.status_code == 409


def test_review_nonexistent_email_returns_404(client):
    response = client.post("/reviews/99999", json={"final_team": "Billing", "final_team_id": 201})
    assert response.status_code == 404


def test_analytics_endpoint_shape(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "total_emails", "auto_routed", "human_reviewed",
        "review_required", "average_confidence", "by_department", "by_team",
    }


def test_submit_email_validates_empty_body(client, tmp_path):
    # Stub the AI dependencies so this test is purely about Pydantic body
    # validation, not about whether a real vectorstore/LLM is configured.
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    _override_ai_deps(vs, [])

    response = client.post("/emails", json={"sender": "a@b.com", "subject": "x", "body": ""})
    assert response.status_code == 422  # Pydantic min_length validation


def test_fetch_endpoint_calls_imap_and_returns_summary(client, tmp_path):
    """
    POST /emails/fetch should delegate to fetch_and_process_unseen_emails
    and return its summary. We patch that function directly (already
    covered by its own unit tests in test_imap_integration.py) so this
    test is purely about the endpoint's wiring: DI, response shape, and
    error translation.
    """
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    _override_ai_deps(vs, [])

    with patch("app.api.emails.fetch_and_process_unseen_emails") as mock_fetch:
        mock_fetch.return_value = {"fetched": 3, "processed": 2, "failed": 1}
        response = client.post("/emails/fetch")

    assert response.status_code == 200
    assert response.json() == {"fetched": 3, "processed": 2, "failed": 1}
    mock_fetch.assert_called_once()


def test_fetch_endpoint_returns_503_when_imap_not_configured(client, tmp_path):
    embedding = DeterministicFakeEmbeddings()
    vs = build_vectorstore(SAMPLE_DOCS, embedding, persist_directory=str(tmp_path / "chroma"))
    _override_ai_deps(vs, [])

    with patch("app.api.emails.fetch_and_process_unseen_emails") as mock_fetch:
        mock_fetch.side_effect = IMAPConnectionError("IMAP is not configured.")
        response = client.post("/emails/fetch")

    assert response.status_code == 503
    assert "IMAP is not configured" in response.json()["detail"]


def test_missing_vectorstore_index_returns_actionable_503(client):
    """
    Regression test: hitting POST /emails before the Chroma index has been
    built must return a clear 503 with instructions, not an opaque 500.

    We override get_vectorstore() with a fake that raises FileNotFoundError
    directly (rather than calling the real load_vectorstore(), which now
    also depends on a local embedding model being downloadable/cached —
    a real network dependency we don't want this test coupled to).
    """
    def raise_not_found():
        raise FileNotFoundError("No Chroma index found")

    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient(responses=[])
    app.dependency_overrides[get_vectorstore] = raise_not_found

    response = client.post("/emails", json={"sender": "a@b.com", "subject": "x", "body": "some content"})
    assert response.status_code == 503
    assert "build_index" in response.json()["detail"]