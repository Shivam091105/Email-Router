"""
Unit and integration tests for Kafka producer, consumers, and recovery logic.

Uses mocked Kafka broker interactions so the test suite runs deterministically
in local and CI environments without requiring an active Kafka broker.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import models, repositories
from app.database.database import Base
from app.kafka.admin import check_kafka_connection, ensure_kafka_topics
from app.kafka.config import (
    ALL_TOPICS,
    TOPIC_EMAIL_APPROVED,
    TOPIC_EMAIL_FAILED,
    TOPIC_EMAIL_INCOMING,
)
from app.kafka.producer import close_producer, publish_email_event
from app.kafka.security_consumer import _process_message as process_security_message
from app.kafka.processing_consumer import _process_message as process_approved_message


@pytest.fixture
def mem_db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    yield db, session_factory
    db.close()


class TestKafkaProducer:
    def test_publish_email_event_serializes_json_correctly(self):
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 0  # 0 remaining messages = success

        with patch("app.kafka.producer._get_producer", return_value=mock_producer):
            success = publish_email_event(
                topic=TOPIC_EMAIL_INCOMING,
                email_id=42,
                sender="alice@test.com",
                subject="Test Subject Line",
            )

            assert success is True
            assert mock_producer.produce.call_count == 1
            call_kwargs = mock_producer.produce.call_args[1]

            assert call_kwargs["topic"] == TOPIC_EMAIL_INCOMING
            assert call_kwargs["key"] == b"42"

            payload = json.loads(call_kwargs["value"].decode("utf-8"))
            assert payload["email_id"] == 42
            assert payload["sender"] == "alice@test.com"
            assert payload["subject"] == "Test Subject Line"
            assert "timestamp" in payload

    def test_publish_email_event_handles_timeout_gracefully(self):
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 1  # 1 message remaining = timed out

        with patch("app.kafka.producer._get_producer", return_value=mock_producer):
            success = publish_email_event(topic=TOPIC_EMAIL_INCOMING, email_id=99)
            assert success is False

    def test_publish_email_event_handles_exception_gracefully(self):
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = Exception("Broker unavailable")

        with patch("app.kafka.producer._get_producer", return_value=mock_producer):
            success = publish_email_event(topic=TOPIC_EMAIL_INCOMING, email_id=101)
            assert success is False


class TestKafkaRecovery:
    def test_kafka_failed_emails_are_queried_for_recovery(self, mem_db):
        db, _ = mem_db
        e1 = repositories.create_email(db, sender="a@b.com", subject="s1", body="b1", status="KAFKA_FAILED")
        e2 = repositories.create_email(db, sender="a@b.com", subject="s2", body="b2", status="PENDING_SECURITY")
        e3 = repositories.create_email(db, sender="a@b.com", subject="s3", body="b3", status="KAFKA_FAILED")

        failed_emails = repositories.list_emails_by_status(db, "KAFKA_FAILED")
        assert len(failed_emails) == 2
        assert {e.id for e in failed_emails} == {e1.id, e3.id}


class TestSecurityConsumer:
    def test_security_consumer_approves_safe_email_and_publishes(self, mem_db):
        db, session_factory = mem_db
        email = repositories.create_email(
            db, sender="hr@company.com", subject="Meeting", body="Quarterly review meeting at 2pm."
        )

        with patch("app.kafka.security_consumer.SessionLocal", session_factory), \
             patch("app.kafka.security_consumer.publish_email_event") as mock_publish:
            mock_publish.return_value = True

            handled = process_security_message({"email_id": email.id})
            assert handled is True

            # Check DB was updated
            screening = repositories.get_security_screening(db, email.id)
            assert screening is not None
            assert screening.decision == "APPROVED"
            assert screening.risk_score == 0.0

            # Check email status updated
            db.refresh(email)
            assert email.status == "APPROVED"

            # Check published to email-approved
            mock_publish.assert_called_once_with(
                topic=TOPIC_EMAIL_APPROVED,
                email_id=email.id,
                sender="hr@company.com",
                subject="Meeting",
            )

    def test_security_consumer_quarantines_credentials(self, mem_db):
        db, session_factory = mem_db
        email = repositories.create_email(
            db, sender="user@bad.com", subject="Credentials", body="Your AWS key: AKIAIOSFODNN7EXAMPLE"
        )

        with patch("app.kafka.security_consumer.SessionLocal", session_factory), \
             patch("app.kafka.security_consumer.publish_email_event") as mock_publish:

            handled = process_security_message({"email_id": email.id})
            assert handled is True

            screening = repositories.get_security_screening(db, email.id)
            assert screening is not None
            assert screening.decision == "QUARANTINED"
            assert screening.sensitive_content_found is True

            db.refresh(email)
            assert email.status == "QUARANTINED"
            # Should NOT publish to approved topic
            mock_publish.assert_not_called()

    def test_security_consumer_idempotency_skips_duplicate(self, mem_db):
        db, session_factory = mem_db
        email = repositories.create_email(
            db, sender="a@b.com", subject="Safe", body="All clean."
        )
        repositories.create_security_screening(
            db,
            email_id=email.id,
            decision="APPROVED",
            risk_score=0.0,
            explanation="Pre-screened",
        )

        with patch("app.kafka.security_consumer.SessionLocal", session_factory), \
             patch("app.kafka.security_consumer.publish_email_event") as mock_publish:

            handled = process_security_message({"email_id": email.id})
            assert handled is True
            # Did not re-publish or fail
            mock_publish.assert_not_called()


class TestKafkaAdmin:
    def test_check_kafka_connection_returns_false_when_disabled(self):
        with patch("app.kafka.admin.settings.kafka_enabled", False):
            assert check_kafka_connection() is False

    def test_check_kafka_connection_success_with_mock_brokers(self):
        mock_admin = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.brokers = {1: MagicMock()}
        mock_admin.list_topics.return_value = mock_metadata

        with patch("app.kafka.admin.settings.kafka_enabled", True), \
             patch("app.kafka.admin.get_admin_client", return_value=mock_admin):
            assert check_kafka_connection() is True

    def test_ensure_kafka_topics_creates_missing(self):
        mock_admin = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.topics = {"email-incoming": MagicMock()}  # missing approved & failed
        mock_admin.list_topics.return_value = mock_metadata

        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_admin.create_topics.return_value = {
            TOPIC_EMAIL_APPROVED: mock_future,
            TOPIC_EMAIL_FAILED: mock_future,
        }

        with patch("app.kafka.admin.settings.kafka_enabled", True), \
             patch("app.kafka.admin.get_admin_client", return_value=mock_admin):
            success = ensure_kafka_topics()
            assert success is True
            assert mock_admin.create_topics.call_count == 1
