"""
IMAP client — retrieves emails from a real mailbox.

This is the "networking" half of the project: an actual TCP connection to
a mail server, authenticated, speaking the IMAP protocol, to pull down
messages the application didn't create itself.

Deliberately NOT unit-tested with a real server (that would make tests
flaky and dependent on external infrastructure/credentials, which the
project rules explicitly forbid). The parsing logic it depends on
(`app.integrations.email_parser.parse_raw_email`) IS unit-tested, since
that's pure logic with no network involved. This module is exercised
manually against a real mailbox — see README's networking section for how
IMAP fits into the overall data flow (Sender -> SMTP -> Mail Server ->
IMAP -> this application).
"""

import imaplib
import logging

from app.core.config import settings
from app.integrations.email_parser import parse_raw_email

logger = logging.getLogger(__name__)


class IMAPConnectionError(Exception):
    pass


def fetch_unseen_emails(limit: int = 20) -> list[dict]:
    """
    Connects to the configured IMAP mailbox, fetches unseen messages
    (up to `limit`), parses each with `parse_raw_email`, and marks them
    as seen. Returns a list of parsed email dicts (see email_parser for
    shape).
    """
    if not (settings.imap_host and settings.imap_user and settings.imap_password):
        raise IMAPConnectionError(
            "IMAP is not configured. Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD in .env."
        )

    parsed_emails: list[dict] = []

    try:
        with imaplib.IMAP4_SSL(settings.imap_host) as conn:
            conn.login(settings.imap_user, settings.imap_password)
            conn.select("INBOX")

            status, data = conn.search(None, "UNSEEN")
            if status != "OK":
                raise IMAPConnectionError(f"IMAP SEARCH failed: {status}")

            message_ids = data[0].split()[:limit]
            logger.info("Found %d unseen messages", len(message_ids))

            for msg_id in message_ids:
                status, msg_data = conn.fetch(msg_id, "(RFC822)")
                if status != "OK":
                    logger.error("Failed to fetch message %s: %s", msg_id, status)
                    continue
                raw_bytes = msg_data[0][1]
                try:
                    parsed = parse_raw_email(raw_bytes, settings.max_attachment_size_mb)
                    parsed_emails.append(parsed)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to parse message %s: %s", msg_id, exc)

    except imaplib.IMAP4.error as exc:
        raise IMAPConnectionError(f"IMAP authentication or protocol error: {exc}") from exc

    return parsed_emails
