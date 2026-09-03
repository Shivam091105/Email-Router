"""
SMTP client — notifies the destination team once an email has been routed.

Like imap_client.py, this is intentionally not unit-tested against a real
server. `build_notification_email` (the message-construction logic) is
pure and could be unit-tested if desired; `send_routing_notification` is
the actual network call.
"""

import logging
import smtplib
from email.message import EmailMessage

from app.core.config import settings

logger = logging.getLogger(__name__)


class SMTPSendError(Exception):
    pass


def build_notification_email(
    to_address: str,
    original_sender: str,
    subject: str,
    summary: str,
    team_name: str,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = f"[Routed to {team_name}] {subject}"
    msg["From"] = settings.smtp_user or "noreply@example.com"
    msg["To"] = to_address
    msg.set_content(
        f"A new email has been routed to your team.\n\n"
        f"Original sender: {original_sender}\n"
        f"Subject: {subject}\n\n"
        f"Summary: {summary}\n"
    )
    return msg


def send_routing_notification(
    to_address: str,
    original_sender: str,
    subject: str,
    summary: str,
    team_name: str,
) -> None:
    if not (settings.smtp_host and settings.smtp_user and settings.smtp_password):
        logger.warning(
            "SMTP not configured (SMTP_HOST/SMTP_USER/SMTP_PASSWORD missing); "
            "skipping notification email to %s", to_address,
        )
        return

    msg = build_notification_email(to_address, original_sender, subject, summary, team_name)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
        logger.info("Sent routing notification to %s", to_address)
    except smtplib.SMTPException as exc:
        raise SMTPSendError(f"Failed to send notification email: {exc}") from exc
