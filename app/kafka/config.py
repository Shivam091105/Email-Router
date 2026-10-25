"""
Kafka topic names, consumer group IDs, and shared constants.

All topic and group names are defined here so producers and consumers
reference the same strings — a typo in a topic name is one of the most
common silent Kafka bugs, and centralizing them prevents it.
"""

# --- Topics ---
TOPIC_EMAIL_INCOMING = "email-incoming"
TOPIC_EMAIL_APPROVED = "email-approved"
TOPIC_EMAIL_FAILED = "email-failed"

ALL_TOPICS = [TOPIC_EMAIL_INCOMING, TOPIC_EMAIL_APPROVED, TOPIC_EMAIL_FAILED]

# --- Consumer groups ---
GROUP_SECURITY_WORKER = "security-worker"
GROUP_PROCESSING_WORKER = "processing-worker"

# --- Retry / reliability ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0

# --- Producer delivery timeout (seconds) ---
PRODUCER_FLUSH_TIMEOUT = 3.0

