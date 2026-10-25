"""
Security screening Kafka consumer.

Consumes events from ``email-incoming``, runs deterministic security
checks, and routes the email to one of:
- ``email-approved`` topic (safe — proceed to normal classification)
- SECURITY_REVIEW status in PostgreSQL (uncertain — human must decide)
- QUARANTINED status in PostgreSQL (high risk — blocked from processing)

Reliability features:
- Manual offset commits (only after DB writes succeed)
- Bounded retries with delays for transient errors
- Dead-letter topic (``email-failed``) for messages that exhaust retries
- Idempotent processing (checks if screening already exists before
  creating a duplicate)

Run as a standalone process:
    python -m app.kafka.security_consumer
"""

import json
import logging
import signal
import sys
import time

from confluent_kafka import Consumer, KafkaException

from app.core.config import settings
from app.core.logging import configure_logging
from app.database import repositories
from app.database.database import SessionLocal, Base, engine
from app.database import models  # noqa: F401 - ensures models are registered
from app.kafka.config import (
    GROUP_SECURITY_WORKER,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    TOPIC_EMAIL_APPROVED,
    TOPIC_EMAIL_FAILED,
    TOPIC_EMAIL_INCOMING,
)
from app.kafka.producer import publish_email_event
from app.security.scanner import screen_email

configure_logging()
logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Received signal %s — shutting down security consumer", sig)
    _running = False


def _process_message(msg_value: dict) -> bool:
    """
    Process one email-incoming event.  Returns True if the message was
    handled successfully (commit-worthy), False if it should be retried.
    """
    email_id = msg_value.get("email_id")
    if email_id is None:
        logger.error("Malformed message (missing email_id): %s", msg_value)
        return True  # commit — retrying won't fix a malformed message

    db = SessionLocal()
    try:
        # Idempotency: if this email was already screened, skip
        existing = repositories.get_security_screening(db, email_id)
        if existing is not None:
            logger.info("Email %s already screened (decision=%s) — skipping duplicate", email_id, existing.decision)
            return True

        email = repositories.get_email(db, email_id)
        if email is None:
            logger.error("Email %s not found in database — skipping", email_id)
            return True  # commit — the email doesn't exist

        # Run security screening
        decision, scan_result = screen_email(
            subject=email.subject,
            body=email.body,
            sender=email.sender,
            attachment_text=email.attachment_text or "",
            attachment_filename=email.attachment_filename,
        )

        # Persist screening result
        repositories.create_security_screening(
            db,
            email_id=email_id,
            decision=decision.outcome,
            risk_score=decision.risk_score,
            explanation=decision.explanation,
            detection_count=scan_result.total_detections,
            sensitive_content_found=scan_result.sensitive_content_found,
            phishing_signals_found=scan_result.phishing_signals_found,
        )

        # Route based on decision
        if decision.outcome == "APPROVED":
            repositories.update_email_status(db, email_id, "APPROVED")
            # Publish to email-approved for normal processing
            success = publish_email_event(
                topic=TOPIC_EMAIL_APPROVED,
                email_id=email_id,
                sender=email.sender,
                subject=email.subject,
            )
            if not success:
                logger.error(
                    "Failed to publish email_id=%s to email-approved; "
                    "status remains APPROVED — processing worker can also "
                    "pick up APPROVED-status emails via recovery sweep",
                    email_id,
                )

        elif decision.outcome == "REVIEW_REQUIRED":
            repositories.update_email_status(db, email_id, "SECURITY_REVIEW")
            # Create a review record so it appears in the review queue,
            # reusing the existing review infrastructure.
            repositories.create_review(
                db,
                email_id=email_id,
                predicted_team="SECURITY_SCREENING",
                predicted_team_id=0,
                predicted_confidence=0.0,
            )
            logger.info("Email %s sent to security review queue", email_id)

        elif decision.outcome == "QUARANTINED":
            repositories.update_email_status(db, email_id, "QUARANTINED")
            logger.warning("Email %s quarantined: %s", email_id, decision.explanation)

        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing email_id=%s: %s", email_id, exc)
        return False
    finally:
        db.close()


def _publish_to_dead_letter(msg_value: dict, error: str) -> None:
    """Send a message to the dead-letter topic after exhausting retries."""
    try:
        dead_event = {
            "original_event": msg_value,
            "error": error[:500],  # truncate to avoid huge messages
            "timestamp": time.time(),
        }
        from confluent_kafka import Producer
        producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        producer.produce(
            topic=TOPIC_EMAIL_FAILED,
            value=json.dumps(dead_event).encode("utf-8"),
        )
        producer.flush(timeout=5.0)
        logger.info("Published email_id=%s to dead-letter topic", msg_value.get("email_id"))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to publish to dead-letter topic: %s", exc)


def run_consumer() -> None:
    """Main consumer loop — blocks until SIGINT/SIGTERM."""
    global _running

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    conf = {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": GROUP_SECURITY_WORKER,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,  # manual commit after successful processing
    }

    consumer = Consumer(conf)
    consumer.subscribe([TOPIC_EMAIL_INCOMING])
    logger.info("Security consumer started, subscribed to %s", TOPIC_EMAIL_INCOMING)

    retry_counts: dict[str, int] = {}  # message key -> retry count

    try:
        while _running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            try:
                msg_value = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.error("Malformed message at offset %s: %s", msg.offset(), exc)
                consumer.commit(message=msg)
                continue

            msg_key = f"{msg.partition()}-{msg.offset()}"

            success = _process_message(msg_value)
            if success:
                consumer.commit(message=msg)
                retry_counts.pop(msg_key, None)
            else:
                retries = retry_counts.get(msg_key, 0) + 1
                retry_counts[msg_key] = retries

                if retries >= MAX_RETRIES:
                    logger.error(
                        "Message at offset %s exhausted %d retries — sending to dead-letter",
                        msg.offset(), MAX_RETRIES,
                    )
                    _publish_to_dead_letter(msg_value, f"Failed after {MAX_RETRIES} retries")
                    consumer.commit(message=msg)
                    retry_counts.pop(msg_key, None)

                    # Also mark the email as FAILED in the database
                    email_id = msg_value.get("email_id")
                    if email_id:
                        db = SessionLocal()
                        try:
                            repositories.update_email_status(db, email_id, "FAILED")
                        finally:
                            db.close()
                else:
                    logger.warning(
                        "Retry %d/%d for message at offset %s — waiting %.1fs",
                        retries, MAX_RETRIES, msg.offset(), RETRY_DELAY_SECONDS,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)

    finally:
        consumer.close()
        logger.info("Security consumer shut down")


if __name__ == "__main__":
    run_consumer()
