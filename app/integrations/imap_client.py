"""
IMAP client — retrieves emails from a real mailbox.

This is the "networking" half of the project: an actual TCP connection to
a mail server, authenticated, speaking the IMAP protocol, to pull down
messages the application didn't create itself.

Deliberately NOT unit-tested with a real server (that would make tests
flaky and dependent on external infrastructure/credentials, which the
project rules explicitly forbid). Instead, tests/test_imap_integration.py
mocks `imaplib` entirely. The parsing logic this module depends on
(`app.integrations.email_parser.parse_raw_email`) IS unit-tested directly,
since that's pure logic with no network involved.
"""

import imaplib
import logging
from typing import Callable

from app.core.config import settings
from app.integrations.email_parser import parse_raw_email

logger = logging.getLogger(__name__)


class IMAPConnectionError(Exception):
    pass


def fetch_and_process_unseen_emails(process_fn: Callable[[dict], bool], limit: int = 20) -> dict:
    """
    Connects to the configured IMAP mailbox, and for each unseen message
    (up to `limit`):

    1. Fetches it via BODY.PEEK[] — this deliberately does NOT mark the
       message \\Seen as a side effect (a plain "(RFC822)" fetch would).
    2. Parses it with parse_raw_email (same parser the rest of the app uses).
    3. Calls `process_fn(parsed_email_dict)`, which is the caller's job of
       actually doing something with it — this module knows nothing about
       the classification pipeline, only about IMAP mechanics.
    4. Marks the message \\Seen ONLY if process_fn returned True.

    If process_fn returns False or raises, the message is left unseen on
    purpose, so the next fetch call will pick it up again. There is no
    retry loop here — per the project's "keep it simple" requirement,
    "retry" just means "it'll still be UNSEEN next time someone triggers
    a fetch."

    Returns a summary dict: {"fetched": int, "processed": int, "failed": int}.
    """
    if not (settings.imap_host and settings.imap_user and settings.imap_password):
        raise IMAPConnectionError(
            "IMAP is not configured. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD in .env."
        )

    summary = {"fetched": 0, "processed": 0, "failed": 0}

    try:
        with imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port) as conn:
            conn.login(settings.imap_user, settings.imap_password)
            conn.select("INBOX")

            status, data = conn.search(None, "UNSEEN")
            if status != "OK":
                raise IMAPConnectionError(f"IMAP SEARCH failed: {status}")

            message_ids = data[0].split()[:limit]
            logger.info("Found %d unseen messages", len(message_ids))

            for msg_id in message_ids:
                status, msg_data = conn.fetch(msg_id, "(BODY.PEEK[])")
                if status != "OK":
                    logger.error("Failed to fetch message %s: %s", msg_id, status)
                    continue

                raw_bytes = msg_data[0][1]
                summary["fetched"] += 1

                try:
                    parsed = parse_raw_email(raw_bytes, settings.max_attachment_size_mb)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to parse message %s: %s", msg_id, exc)
                    summary["failed"] += 1
                    continue  # left unseen; a malformed message won't parse better next time either

                try:
                    success = process_fn(parsed)
                except Exception:  # noqa: BLE001
                    logger.exception("process_fn raised for message %s", msg_id)
                    success = False

                if success:
                    conn.store(msg_id, "+FLAGS", "\\Seen")
                    summary["processed"] += 1
                    logger.info("Marked message %s as Seen after successful processing", msg_id)
                else:
                    summary["failed"] += 1
                    logger.warning("Leaving message %s unseen after failed processing", msg_id)

    except imaplib.IMAP4.error as exc:
        raise IMAPConnectionError(f"IMAP authentication or protocol error: {exc}") from exc

    return summary