"""
MIME email parsing.

Kept as a pure function (`parse_raw_email`) that takes raw RFC 5322 bytes
and returns a plain dict — no IMAP connection involved. This is what
makes it unit-testable: tests build a raw email in memory and check the
parser's output, with zero network dependency. `app/integrations/imap_client.py`
is the network layer that fetches raw bytes and hands them to this function.
"""

import logging
import re
from email import message_from_bytes, policy
from email.message import Message

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 255
UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(filename: str) -> str:
    """
    Strips directory components and unsafe characters from an attachment
    filename before it's ever used to build a file path. Never trust a
    filename that arrived over email.
    """
    name = filename.replace("\\", "/").split("/")[-1]
    name = UNSAFE_FILENAME_CHARS.sub("_", name)
    return name[:MAX_FILENAME_LENGTH] or "attachment"


def _get_body_text(msg: Message) -> tuple[str, str]:
    """Returns (plain_text, html) bodies, preferring the multipart walk."""
    plain, html = "", ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()
            if disposition == "attachment":
                continue
            if content_type == "text/plain" and not plain:
                plain = part.get_content()
            elif content_type == "text/html" and not html:
                html = part.get_content()
    else:
        if msg.get_content_type() == "text/plain":
            plain = msg.get_content()
        elif msg.get_content_type() == "text/html":
            html = msg.get_content()

    return plain.strip(), html.strip()


def parse_raw_email(raw_bytes: bytes, max_attachment_size_mb: int = 10) -> dict:
    """
    Parses a raw RFC 5322 email into:
        {
            "sender": str,
            "recipients": list[str],
            "subject": str,
            "body_text": str,
            "body_html": str,
            "attachments": [{"filename": str, "content": bytes, "content_type": str}, ...],
        }

    Attachments larger than `max_attachment_size_mb` are skipped (not
    included, logged) rather than raising — a single oversized attachment
    shouldn't prevent the rest of the email from being processed.
    """
    msg = message_from_bytes(raw_bytes, policy=policy.default)

    sender = str(msg.get("From", ""))
    recipients_raw = msg.get_all("To", []) + msg.get_all("Cc", [])
    recipients = [str(r) for r in recipients_raw]
    subject = str(msg.get("Subject", ""))

    body_text, body_html = _get_body_text(msg)

    attachments = []
    max_bytes = max_attachment_size_mb * 1024 * 1024
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() != "attachment":
                continue
            filename = part.get_filename() or "attachment"
            content = part.get_content()
            if isinstance(content, str):
                content = content.encode("utf-8")
            if len(content) > max_bytes:
                logger.warning(
                    "Skipping attachment %r: %d bytes exceeds limit of %d bytes",
                    filename, len(content), max_bytes,
                )
                continue
            attachments.append(
                {
                    "filename": sanitize_filename(filename),
                    "content": content,
                    "content_type": part.get_content_type(),
                }
            )

    return {
        "sender": sender,
        "recipients": recipients,
        "subject": subject,
        "body_text": body_text,
        "body_html": body_html,
        "attachments": attachments,
    }
