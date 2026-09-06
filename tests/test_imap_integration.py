"""
Tests for the real-email (IMAP) integration.

`imaplib.IMAP4_SSL` is mocked entirely — these tests never touch a real
mailbox. What's actually under test:

1. fetch_and_process_unseen_emails() — the seen/unseen bookkeeping logic:
   does it call process_fn for each unseen message, and does it mark
   \\Seen only when process_fn succeeds?
2. process_parsed_email_sync() — does it correctly create an Email row
   from a parsed-IMAP-message dict and hand it to the same process_email()
   the API uses?
"""

import json
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import models  # noqa: F401 - registers models on Base
from app.database import repositories
from app.database.database import Base
from app.integrations.imap_client import IMAPConnectionError, fetch_and_process_unseen_emails
from app.llm.client import FakeLLMClient
from app.rag.embeddings import DeterministicFakeEmbeddings
from app.rag.vectorstore import build_vectorstore
from app.services.email_service import process_parsed_email_sync


def _raw_email_bytes(subject="Test", body="Hello") -> bytes:
    msg = EmailMessage()
    msg["From"] = "customer@gmail.com"
    msg["To"] = "router@example.com"
    msg["Subject"] = subject
    msg.set_content(body)
    return bytes(msg)


# ---------- fetch_and_process_unseen_emails ----------

def test_fetch_raises_when_imap_not_configured():
    with patch("app.integrations.imap_client.settings") as mock_settings:
        mock_settings.imap_host = None
        mock_settings.imap_user = None
        mock_settings.imap_password = None
        with pytest.raises(IMAPConnectionError):
            fetch_and_process_unseen_emails(process_fn=lambda parsed: True)


@patch("app.integrations.imap_client.imaplib.IMAP4_SSL")
def test_fetch_marks_seen_only_on_successful_processing(mock_imap_class):
    mock_conn = MagicMock()
    mock_imap_class.return_value.__enter__.return_value = mock_conn
    mock_conn.search.return_value = ("OK", [b"1 2"])
    mock_conn.fetch.side_effect = [
        ("OK", [(b"1 (BODY[])", _raw_email_bytes(subject="Succeeds"))]),
        ("OK", [(b"2 (BODY[])", _raw_email_bytes(subject="Fails"))]),
    ]

    processed_subjects = []

    def process_fn(parsed: dict) -> bool:
        processed_subjects.append(parsed["subject"])
        return parsed["subject"] == "Succeeds"

    with patch("app.integrations.imap_client.settings") as mock_settings:
        mock_settings.imap_host = "imap.gmail.com"
        mock_settings.imap_port = 993
        mock_settings.imap_user = "user@gmail.com"
        mock_settings.imap_password = "app-password"
        mock_settings.max_attachment_size_mb = 10

        summary = fetch_and_process_unseen_emails(process_fn, limit=20)

    assert processed_subjects == ["Succeeds", "Fails"]
    assert summary == {"fetched": 2, "processed": 1, "failed": 1}

    # \Seen was stored for message 1 (succeeded) but NOT for message 2 (failed)
    seen_calls = [call for call in mock_conn.store.call_args_list if call.args[1] == "+FLAGS"]
    assert len(seen_calls) == 1
    assert seen_calls[0].args[0] == b"1"


@patch("app.integrations.imap_client.imaplib.IMAP4_SSL")
def test_fetch_uses_peek_not_plain_rfc822(mock_imap_class):
    """BODY.PEEK[] must be used so unprocessed messages aren't silently
    marked \\Seen by the IMAP server itself before we decide the outcome."""
    mock_conn = MagicMock()
    mock_imap_class.return_value.__enter__.return_value = mock_conn
    mock_conn.search.return_value = ("OK", [b"1"])
    mock_conn.fetch.return_value = ("OK", [(b"1 (BODY[])", _raw_email_bytes())])

    with patch("app.integrations.imap_client.settings") as mock_settings:
        mock_settings.imap_host = "imap.gmail.com"
        mock_settings.imap_port = 993
        mock_settings.imap_user = "user@gmail.com"
        mock_settings.imap_password = "app-password"
        mock_settings.max_attachment_size_mb = 10

        fetch_and_process_unseen_emails(process_fn=lambda parsed: True)

    fetch_call_args = mock_conn.fetch.call_args
    assert "PEEK" in fetch_call_args.args[1]


@patch("app.integrations.imap_client.imaplib.IMAP4_SSL")
def test_fetch_continues_after_one_message_fails_to_parse(mock_imap_class):
    """A single malformed message must not abort the whole batch."""
    mock_conn = MagicMock()
    mock_imap_class.return_value.__enter__.return_value = mock_conn
    mock_conn.search.return_value = ("OK", [b"1 2"])
    mock_conn.fetch.side_effect = [
        ("OK", [(b"1 (BODY[])", b"not a valid email at all \xff\xfe")]),
        ("OK", [(b"2 (BODY[])", _raw_email_bytes(subject="Good one"))]),
    ]

    processed = []

    with patch("app.integrations.imap_client.settings") as mock_settings:
        mock_settings.imap_host = "imap.gmail.com"
        mock_settings.imap_port = 993
        mock_settings.imap_user = "user@gmail.com"
        mock_settings.imap_password = "app-password"
        mock_settings.max_attachment_size_mb = 10

        summary = fetch_and_process_unseen_emails(
            process_fn=lambda parsed: processed.append(parsed["subject"]) or True
        )

    # message 1 either fails to parse or parses to something processed;
    # either way, message 2 must still be reached and processed.
    assert "Good one" in processed
    assert summary["fetched"] >= 1


# ---------- process_parsed_email_sync ----------

@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def vectorstore(tmp_path):
    docs = [
        Document(
            page_content="Team: IT Support. Handles login problems.",
            metadata={"department": "IT", "team": "IT Support", "team_id": 101},
        )
    ]
    embedding = DeterministicFakeEmbeddings()
    return build_vectorstore(docs, embedding, persist_directory=str(tmp_path / "chroma"))


def test_process_parsed_email_sync_creates_email_and_routes(session_factory, vectorstore):
    parsed = {
        "sender": "customer@gmail.com",
        "subject": "Can't log in",
        "body_text": "I can't log into my account",
        "body_html": "",
        "attachments": [],
    }
    valid_json = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 0.9, "reasoning": "login issue"}
    )
    llm = FakeLLMClient(responses=[valid_json, "Cannot log in."])

    with patch("app.services.email_service.SessionLocal", session_factory):
        success = process_parsed_email_sync(parsed, llm, vectorstore)

    assert success is True

    db = session_factory()
    emails = repositories.list_emails(db)
    assert len(emails) == 1
    assert emails[0].sender == "customer@gmail.com"
    assert emails[0].status == "ROUTED"

    result = repositories.get_routing_result(db, emails[0].id)
    assert result.team_id == 101
    db.close()


def test_process_parsed_email_sync_prefers_html_body_when_no_plain_text(session_factory, vectorstore):
    parsed = {
        "sender": "customer@gmail.com",
        "subject": "HTML only",
        "body_text": "",
        "body_html": "<p>I can't log into my account</p>",
        "attachments": [],
    }
    llm = FakeLLMClient(responses=["not json", "not json"])  # will FAIL classification, that's fine here

    with patch("app.services.email_service.SessionLocal", session_factory):
        success = process_parsed_email_sync(parsed, llm, vectorstore)

    assert success is True  # email was still durably created despite classification failing
    db = session_factory()
    email = repositories.list_emails(db)[0]
    assert "log into" in email.body
    db.close()