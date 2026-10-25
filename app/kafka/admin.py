"""
Kafka administrative utilities: connectivity checks and automatic topic creation.

Used at startup so that topics exist before consumers start reading or
producers start publishing, avoiding leader election race conditions.
"""

import logging
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from app.core.config import settings
from app.kafka.config import ALL_TOPICS

logger = logging.getLogger(__name__)


def get_admin_client() -> AdminClient:
    """Returns a configured Kafka AdminClient."""
    return AdminClient({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "socket.timeout.ms": 3000,
    })


def check_kafka_connection(timeout: float = 2.0) -> bool:
    """
    Checks if the Kafka broker is reachable and responsive.
    Returns True if connected, False if unreachable or timed out.
    """
    if not settings.kafka_enabled:
        return False
    try:
        admin = get_admin_client()
        metadata = admin.list_topics(timeout=timeout)
        return metadata is not None and metadata.brokers is not None and len(metadata.brokers) > 0
    except Exception as exc:
        logger.debug("Kafka connection check failed: %s", exc)
        return False


def ensure_kafka_topics(timeout: float = 5.0) -> bool:
    """
    Ensures all required topics exist on the broker.
    Creates any missing topics with 1 partition and replication factor 1
    (appropriate for single-node development / compose environments).
    """
    if not settings.kafka_enabled:
        logger.info("Kafka is disabled; skipping topic creation")
        return False

    try:
        admin = get_admin_client()
        metadata = admin.list_topics(timeout=timeout)
        existing_topics = set(metadata.topics.keys())

        missing = [t for t in ALL_TOPICS if t not in existing_topics]
        if not missing:
            logger.info("All Kafka topics already exist: %s", ALL_TOPICS)
            return True

        new_topics = [
            NewTopic(topic=t, num_partitions=1, replication_factor=1)
            for t in missing
        ]
        fs = admin.create_topics(new_topics, request_timeout=timeout)

        success = True
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                logger.info("Created Kafka topic: %s", topic)
            except Exception as e:
                logger.warning("Failed to create topic %s: %s", topic, e)
                success = False

        return success

    except Exception as exc:
        logger.warning("Could not verify or create Kafka topics: %s", exc)
        return False
