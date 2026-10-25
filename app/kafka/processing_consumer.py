"""
Normal processing Kafka consumer.

Consumes events from ``email-approved``, fetches the email from PostgreSQL,
and runs the existing LangGraph RAG+LLM classification pipeline — exactly
the same ``run_email_workflow()`` function that the original BackgroundTasks
approach used.

Reliability features:
- Manual offset commits (only after processing + DB writes succeed)
- Bounded retries with delays
- Dead-letter topic for messages that exhaust retries
- Idempotent: checks if a routing result already exists before processing

Run as a standalone process:
    python -m app.kafka.processing_consumer
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
from app.core.dependencies import get_llm_client, get_vectorstore
from app.graph.workflow import run_email_workflow
from app.kafka.config import (
    GROUP_PROCESSING_WORKER,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    TOPIC_EMAIL_APPROVED,
    TOPIC_EMAIL_FAILED,
)
from app.security.decision import redact_sensitive_content

configure_logging()
logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Received signal %s — shutting down processing consumer", sig)
    _running = False


def _process_message(msg_value: dict, llm_client, vectorstore) -> bool:
    """
    Process one email-approved event.  Invokes the existing LangGraph
    pipeline.  Returns True on success, False to trigger retry.
    """
    email_id = msg_value.get("email_id")
    if email_id is None:
        logger.error("Malformed message (missing email_id): %s", msg_value)
        return True  # commit — retrying won't fix it

    db = SessionLocal()
    try:
        # Idempotency: if this email already has a routing result, skip
        existing_result = repositories.get_routing_result(db, email_id)
        if existing_result is not None:
            logger.info("Email %s already has a routing result — skipping duplicate", email_id)
            return True

        email = repositories.get_email(db, email_id)
        if email is None:
            logger.error("Email %s not found in database — skipping", email_id)
            return True

        # Check if the email needs content redaction (sensitive info was
        # detected during security screening, but the email was still
        # approved — e.g. low-risk credential pattern in context).
        screening = repositories.get_security_screening(db, email_id)
        body = email.body
        attachment_text = email.attachment_text or ""

        if screening and screening.sensitive_content_found:
            # Redact sensitive content before sending to external LLM
            body = redact_sensitive_content(body)
            attachment_text = redact_sensitive_content(attachment_text)
            logger.info("Redacted sensitive content for email %s before LLM classification", email_id)

        sender = email.sender
        subject = email.subject
    finally:
        db.close()

    # Update status to PROCESSING
    db2 = SessionLocal()
    try:
        repositories.update_email_status(db2, email_id, "PROCESSING")
    finally:
        db2.close()

    try:
        run_email_workflow(
            llm_client=llm_client,
            vectorstore=vectorstore,
            session_factory=SessionLocal,
            email_id=email_id,
            sender=sender,
            subject=subject,
            body=body,
            attachment_text=attachment_text,
        )
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow failed for email_id=%s: %s", email_id, exc)
        db3 = SessionLocal()
        try:
            repositories.update_email_status(db3, email_id, "FAILED")
        finally:
            db3.close()
        return False


def _publish_to_dead_letter(msg_value: dict, error: str) -> None:
    """Send a message to the dead-letter topic after exhausting retries."""
    try:
        dead_event = {
            "original_event": msg_value,
            "error": error[:500],
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

    # Initialize LLM + vectorstore (same singletons the API uses)
    llm_client = get_llm_client()
    vectorstore = get_vectorstore()

    conf = {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": GROUP_PROCESSING_WORKER,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }

    consumer = Consumer(conf)
    consumer.subscribe([TOPIC_EMAIL_APPROVED])
    logger.info("Processing consumer started, subscribed to %s", TOPIC_EMAIL_APPROVED)

    retry_counts: dict[str, int] = {}

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

            success = _process_message(msg_value, llm_client, vectorstore)
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
                else:
                    logger.warning(
                        "Retry %d/%d for message at offset %s — waiting %.1fs",
                        retries, MAX_RETRIES, msg.offset(), RETRY_DELAY_SECONDS,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)

    finally:
        consumer.close()
        logger.info("Processing consumer shut down")


if __name__ == "__main__":
    run_consumer()
