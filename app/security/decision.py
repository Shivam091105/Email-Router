"""
Security decision model.

Takes a ScanResult and produces a clear, auditable decision:
- ``APPROVED``        — no significant risk, proceed to normal processing.
- ``REVIEW_REQUIRED`` — uncertain or medium risk, route to human review.
- ``QUARANTINED``     — high risk, block automatic routing entirely.

Decision rules are explicit and readable.  This is intentionally NOT an
ML model — it's a rule-based system that an interviewer can read in 30
seconds and understand exactly why each email was treated the way it was.

If sensitive content (OTPs, passwords, API keys) is detected, the email
body is redacted before being sent to the external LLM, so that the LLM
never sees raw credential values.
"""

import re
import logging
from dataclasses import dataclass

from app.security.patterns import ScanResult

logger = logging.getLogger(__name__)


@dataclass
class SecurityDecision:
    """The outcome of security screening for one email."""
    outcome: str            # APPROVED | REVIEW_REQUIRED | QUARANTINED
    risk_score: float       # 0.0 (safe) to 1.0 (dangerous)
    explanation: str        # human-readable, stored in DB
    should_redact: bool     # True if body needs redaction before LLM


def make_decision(scan_result: ScanResult) -> SecurityDecision:
    """
    Deterministic decision based on scan results.

    Rules (applied in priority order):
    1. Any high-severity detection with sensitive content → QUARANTINED
       (we cannot safely forward credentials to the LLM or auto-route)
    2. Any high-severity phishing signal (suspicious URL, risky attachment)
       → QUARANTINED
    3. Multiple medium-severity signals → REVIEW_REQUIRED
    4. Any single detection → REVIEW_REQUIRED
    5. No detections → APPROVED
    """
    high_count = scan_result.high_severity_count
    total = scan_result.total_detections
    has_sensitive = scan_result.sensitive_content_found
    has_phishing = scan_result.phishing_signals_found
    has_risky_attachment = scan_result.risky_attachment

    # Calculate a simple risk score (not a probability — just a heuristic
    # for ordering/display purposes)
    risk_score = min(1.0, (high_count * 0.4) + ((total - high_count) * 0.15))

    # Rule 1: High-severity sensitive content — quarantine
    if has_sensitive and high_count > 0:
        return SecurityDecision(
            outcome="QUARANTINED",
            risk_score=risk_score,
            explanation=(
                f"Email quarantined: sensitive information detected "
                f"({scan_result.summary_text}). Cannot safely forward "
                f"to automated processing."
            ),
            should_redact=True,
        )

    # Rule 2: High-severity phishing signals — quarantine
    if has_phishing and (any(d.severity == "high" for d in scan_result.detections if d.category in ("suspicious_url", "risky_attachment"))):
        return SecurityDecision(
            outcome="QUARANTINED",
            risk_score=risk_score,
            explanation=(
                f"Email quarantined: high-risk phishing indicators "
                f"detected ({scan_result.summary_text})."
            ),
            should_redact=False,
        )

    # Rule 3: Multiple medium+ signals — review
    medium_plus = sum(1 for d in scan_result.detections if d.severity in ("high", "medium"))
    if medium_plus >= 2:
        return SecurityDecision(
            outcome="REVIEW_REQUIRED",
            risk_score=risk_score,
            explanation=(
                f"Email flagged for review: multiple risk signals "
                f"detected ({scan_result.summary_text})."
            ),
            should_redact=has_sensitive,
        )

    # Rule 4: Any detection at all — review (conservative approach)
    if total > 0:
        return SecurityDecision(
            outcome="REVIEW_REQUIRED",
            risk_score=risk_score,
            explanation=(
                f"Email flagged for review: {scan_result.summary_text}."
            ),
            should_redact=has_sensitive,
        )

    # Rule 5: Clean — approved
    return SecurityDecision(
        outcome="APPROVED",
        risk_score=0.0,
        explanation="No security issues detected. Email approved for processing.",
        should_redact=False,
    )


def redact_sensitive_content(text: str) -> str:
    """
    Replace detected sensitive values with [REDACTED] placeholders.

    Used before sending email content to an external LLM, so that
    credentials, OTPs, and API keys are never transmitted to third-party
    services.  The original content stays in PostgreSQL (the source of
    truth) — only the LLM's copy is redacted.
    """
    # Redact OTPs
    text = re.sub(
        r"(?i)((?:verification\s*code|otp|one[- ]time\s*(?:password|code|pin)|"
        r"security\s*code|confirmation\s*code|2fa\s*code|mfa\s*code|"
        r"login\s*code|access\s*code)\s*(?:is|:| )\s*)\d{4,8}\b",
        r"\1[REDACTED]",
        text,
    )

    # Redact passwords
    text = re.sub(
        r"(?i)((?:password|passwd|pwd|passcode)\s*(?:is|:|=)\s*)\S+",
        r"\1[REDACTED]",
        text,
    )

    # Redact API keys / tokens (the value after the label)
    text = re.sub(
        r"(?i)((?:api[_ -]?key|api[_ -]?token|access[_ -]?token|secret[_ -]?key|auth[_ -]?token)\s*(?:is|:|=)\s*)\S{16,}",
        r"\1[REDACTED]",
        text,
    )

    # Redact AWS keys
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]", text)

    # Redact GitHub tokens
    text = re.sub(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}", "[REDACTED_GH_TOKEN]", text)

    # Redact Bearer tokens
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", r"\1[REDACTED]", text)

    # Redact SSN-like patterns
    text = re.sub(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", "[REDACTED_SSN]", text)

    return text
