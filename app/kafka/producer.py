"""
Kafka producer.

Publishes small JSON events (email ID + minimal metadata) to Kafka topics.
Full email bodies, secrets, and sensitive content stay in PostgreSQL —
Kafka events are pointers, not payloads.

Failure handling:
- ``publish_email_event`` calls ``producer.flush()`` synchronously so the
  caller knows immediately whether delivery succeeded or failed.
- If Kafka is unreachable, the caller (API endpoint) can fall back to
  updating the email status to ``KAFKA_FAILED`` so a recovery sweep can
  retry later — see the ``recover_kafka_failed_emails`` function.
"""

import json
import logging
from datetime import datetime, timezone

from confluent_kafka import KafkaException, Producer

from app.core.config import settings
from app.kafka.config import PRODUCER_FLUSH_TIMEOUT

logger = logging.getLogger(__name__)

_producer: Producer | None = None


def _get_producer() -> Producer:
    """Lazy-initialised singleton producer."""
    global _producer
    if _producer is None:
        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "client.id": "email-router-api",
            # Idempotent producer: Kafka deduplicates retries at the broker
            # level, so a network hiccup that triggers a retry doesn't create
            # duplicate messages on the topic.  The overhead is negligible for
            # our throughput.
            "enable.idempotence": True,
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 500,
            "socket.timeout.ms": 3000,
            "message.timeout.ms": 5000,
        }
        _producer = Producer(conf)
    return _producer



def _delivery_callback(err, msg):
    """Called once per message when Kafka confirms (or rejects) delivery."""
    if err is not None:
        logger.error("Kafka delivery failed for topic=%s: %s", msg.topic(), err)
    else:
        logger.info(
            "Kafka delivery confirmed: topic=%s partition=%s offset=%s",
            msg.topic(), msg.partition(), msg.offset(),
        )


def publish_email_event(topic: str, email_id: int, sender: str = "", subject: str = "") -> bool:
    """
    Publish a lightweight event to ``topic``.

    Returns True if the broker acknowledged the message within the flush
    timeout, False otherwise.  The caller decides what to do on failure
    (e.g. mark the email as KAFKA_FAILED for later retry).
    """
    event = {
        "email_id": email_id,
        "sender": sender,
        "subject": subject[:100],  # truncate — this is metadata only
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        producer = _get_producer()
        producer.produce(
            topic=topic,
            key=str(email_id).encode("utf-8"),
            value=json.dumps(event).encode("utf-8"),
            callback=_delivery_callback,
        )
        # Flush synchronously so we know NOW whether the message was
        # accepted.  At our scale (single-digit emails/second at most)
        # the latency cost is negligible — well under 100 ms on a local
        # broker.
        remaining = producer.flush(timeout=PRODUCER_FLUSH_TIMEOUT)
        if remaining > 0:
            logger.error(
                "Kafka flush timed out: %d message(s) still in queue for email_id=%s",
                remaining, email_id,
            )
            return False
        return True

    except KafkaException as exc:
        logger.error("Kafka publish failed for email_id=%s on topic=%s: %s", email_id, topic, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error publishing email_id=%s: %s", email_id, exc)
        return False


def close_producer() -> None:
    """Flush remaining messages and release the producer (for clean shutdown)."""
    global _producer
    if _producer is not None:
        _producer.flush(timeout=PRODUCER_FLUSH_TIMEOUT)
        _producer = None
