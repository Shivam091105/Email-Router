"""
Security screening orchestrator.

This is the single entry point that the security consumer calls.
It runs all pattern detections, makes a decision, and returns a structured
result ready to be persisted in PostgreSQL.

The scanner itself has NO database or Kafka dependencies — it's a pure
function of (email content) → (security decision), which makes it trivially
testable without any infrastructure.
"""

import logging

from app.security.patterns import scan_email, ScanResult
from app.security.decision import SecurityDecision, make_decision, redact_sensitive_content

logger = logging.getLogger(__name__)


def screen_email(
    subject: str,
    body: str,
    sender: str = "",
    attachment_text: str = "",
    attachment_filename: str | None = None,
) -> tuple[SecurityDecision, ScanResult]:
    """
    Run full security screening on an email.

    Returns:
        (decision, scan_result) — the decision to act on, and the raw
        scan result for auditing/storage.
    """
    scan_result = scan_email(
        subject=subject,
        body=body,
        sender=sender,
        attachment_text=attachment_text,
        attachment_filename=attachment_filename,
    )

    decision = make_decision(scan_result)

    # Log at appropriate levels — never log detected secret values
    if decision.outcome == "QUARANTINED":
        logger.warning(
            "Email from sender=%s quarantined: risk_score=%.2f, detections=%d",
            sender, decision.risk_score, scan_result.total_detections,
        )
    elif decision.outcome == "REVIEW_REQUIRED":
        logger.info(
            "Email from sender=%s flagged for security review: risk_score=%.2f, detections=%d",
            sender, decision.risk_score, scan_result.total_detections,
        )
    else:
        logger.debug("Email from sender=%s approved by security screening", sender)

    return decision, scan_result


# Re-export for convenience
__all__ = [
    "screen_email",
    "redact_sensitive_content",
    "SecurityDecision",
    "ScanResult",
]
