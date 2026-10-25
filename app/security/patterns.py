"""
Regex-based pattern detectors for sensitive information and phishing signals.

Design philosophy:
- Deterministic checks run BEFORE any LLM call.  They are cheap, fast,
  and explainable — you can point at the exact regex that triggered.
- Patterns are intentionally conservative: we'd rather send a borderline
  email to human review than silently approve something suspicious.
- Detected values are NEVER logged, stored in Kafka events, or included
  in security screening records.  Only the *type* of detection and the
  *count* of matches are recorded.

Limitations (documented honestly):
- These patterns catch common formats, not all formats.
- URL analysis is purely structural — we do not fetch URLs or check
  reputation databases.
- SPF/DKIM/DMARC verification is not implemented because the current
  ingestion flow does not provide raw email headers with authentication
  results.  Noted as a future integration point.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# A. Sensitive information patterns
# ---------------------------------------------------------------------------

# OTPs: 4-8 digit codes, typically preceded by context words
_OTP_CONTEXT = (
    r"(?:verification\s*code|otp|one[- ]time\s*(?:password|code|pin)|"
    r"security\s*code|confirmation\s*code|2fa\s*code|mfa\s*code|"
    r"login\s*code|access\s*code)"
)
OTP_PATTERN = re.compile(
    rf"(?i){_OTP_CONTEXT}\s*(?:is|:| )\s*(\d{{4,8}})\b"
)

# Passwords explicitly stated in plain text
PASSWORD_PATTERN = re.compile(
    r"(?i)(?:password|passwd|pwd|passcode)\s*(?:is|:|=)\s*\S+",
)

# API keys / tokens — common formats (AWS, GitHub, Slack, generic hex/base64)
API_KEY_PATTERNS = [
    re.compile(r"(?i)(?:api[_ -]?key|api[_ -]?token|access[_ -]?token|secret[_ -]?key|auth[_ -]?token)\s*(?:is|:|=)\s*\S{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),                           # AWS access key
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}"),  # GitHub token
    re.compile(r"xox[bporsea]-[0-9a-zA-Z-]{10,}"),             # Slack token
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                         # OpenAI-style key
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"),             # Bearer tokens
]

# Credit card numbers (basic Luhn-eligible 13-19 digit sequences)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)

# SSN-like patterns (US format)
SSN_PATTERN = re.compile(
    r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"
)


# ---------------------------------------------------------------------------
# B. Phishing and suspicious email patterns
# ---------------------------------------------------------------------------

# Suspicious URL patterns: IP-based URLs, URL shorteners, data URIs, and
# common obfuscation techniques
SUSPICIOUS_URL_PATTERNS = [
    # IP-address URLs (http://192.168.1.1/...)
    re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
    # URL shorteners (bit.ly, tinyurl, t.co, etc.)
    re.compile(r"https?://(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|rb\.gy)/\S+"),
    # Unicode / punycode domain obfuscation
    re.compile(r"https?://[^\s]*xn--[^\s]+"),
    # Extremely long URLs (often phishing)
    re.compile(r"https?://\S{200,}"),
    # URLs with @ sign (credential-in-URL trick)
    re.compile(r"https?://[^/]*@"),
    # data: URIs that could embed scripts
    re.compile(r"data:\s*text/html", re.IGNORECASE),
]

# Urgency / credential-request language (supplementary signal only)
URGENCY_PATTERNS = [
    re.compile(r"(?i)\b(?:urgent|immediately|account\s+(?:suspended|locked|disabled)|verify\s+(?:your|now)|click\s+(?:here|now)|act\s+now|limited\s+time|expire[sd]?\s+(?:soon|today|in\s+\d))\b"),
    re.compile(r"(?i)\b(?:confirm\s+your\s+(?:identity|account|password)|update\s+(?:your\s+)?payment|unusual\s+(?:activity|sign[- ]?in)|unauthorized\s+(?:access|transaction))\b"),
]

# Sender domain mismatches (e.g. "From: security@paypal.com" but reply-to
# is a different domain) — checked structurally, not via DNS
FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "icloud.com", "mail.com", "zoho.com", "yandex.com",
}

# Risky attachment extensions (if attachment metadata is present)
RISKY_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".scr", ".pif", ".vbs", ".js", ".wsf",
    ".msi", ".com", ".hta", ".cpl", ".reg", ".ps1",
}


# ---------------------------------------------------------------------------
# Detection result containers
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One specific finding from the security scan."""
    category: str       # e.g. "otp", "password", "api_key", "suspicious_url"
    description: str    # human-readable, NO actual secret values
    severity: str       # "high", "medium", "low"


@dataclass
class ScanResult:
    """Aggregated result of all security checks on one email."""
    detections: list[Detection] = field(default_factory=list)
    sensitive_content_found: bool = False
    phishing_signals_found: bool = False
    risky_attachment: bool = False
    # SPF/DKIM/DMARC are unavailable in the current ingestion flow.
    # Documented as a future integration point.
    auth_check_available: bool = False

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for d in self.detections if d.severity == "high")

    @property
    def summary_text(self) -> str:
        """Concise, secret-free summary for storage in PostgreSQL."""
        if not self.detections:
            return "No security issues detected."
        parts = []
        for d in self.detections:
            parts.append(f"[{d.severity.upper()}] {d.category}: {d.description}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Scanner functions
# ---------------------------------------------------------------------------

def scan_for_sensitive_content(text: str) -> list[Detection]:
    """Check for OTPs, passwords, API keys, and other credentials."""
    detections = []

    if OTP_PATTERN.search(text):
        detections.append(Detection(
            category="otp",
            description="Likely OTP or verification code detected in message body",
            severity="high",
        ))

    if PASSWORD_PATTERN.search(text):
        detections.append(Detection(
            category="password",
            description="Explicit password or credential value found in plain text",
            severity="high",
        ))

    for pattern in API_KEY_PATTERNS:
        if pattern.search(text):
            detections.append(Detection(
                category="api_key",
                description="Likely API key, access token, or secret detected",
                severity="high",
            ))
            break  # one detection is enough — don't enumerate each key

    if SSN_PATTERN.search(text):
        detections.append(Detection(
            category="ssn",
            description="Pattern matching a US Social Security Number format",
            severity="high",
        ))

    # Credit card detection is intentionally conservative — we only flag
    # if the digit sequence appears with a finance-related keyword nearby
    if CREDIT_CARD_PATTERN.search(text) and re.search(r"(?i)\b(?:card|credit|debit|payment|visa|mastercard|amex)\b", text):
        detections.append(Detection(
            category="credit_card",
            description="Possible credit/debit card number detected near payment-related keywords",
            severity="high",
        ))

    return detections


def scan_for_phishing_signals(text: str, sender: str = "", attachment_filename: str | None = None) -> list[Detection]:
    """Check for phishing indicators: suspicious URLs, urgency language, attachment risks."""
    detections = []

    # Suspicious URLs
    for pattern in SUSPICIOUS_URL_PATTERNS:
        if pattern.search(text):
            detections.append(Detection(
                category="suspicious_url",
                description="Suspicious or obfuscated URL detected (IP-based, shortened, or encoded)",
                severity="high",
            ))
            break

    # Urgency / credential-request language (medium severity — supplementary)
    urgency_count = sum(1 for p in URGENCY_PATTERNS if p.search(text))
    if urgency_count >= 2:
        detections.append(Detection(
            category="urgency_language",
            description=f"Multiple urgency/credential-request phrases detected ({urgency_count} patterns matched)",
            severity="medium",
        ))
    elif urgency_count == 1:
        detections.append(Detection(
            category="urgency_language",
            description="Urgency or credential-request language detected (single pattern)",
            severity="low",
        ))

    # Sender domain analysis
    if sender and "@" in sender:
        domain = sender.split("@")[-1].lower().strip(">").strip()
        # Freemail sending as if it's a corporate entity
        if domain in FREEMAIL_DOMAINS:
            # Check if the email claims to be from a corporate entity
            corporate_keywords = re.findall(
                r"(?i)\b(?:bank|paypal|amazon|microsoft|apple|google|support|security|admin|helpdesk)\b",
                text,
            )
            if len(corporate_keywords) >= 2:
                detections.append(Detection(
                    category="sender_mismatch",
                    description=f"Sender uses freemail domain but email references corporate entities",
                    severity="medium",
                ))

    # Attachment risk
    if attachment_filename:
        ext = "." + attachment_filename.rsplit(".", 1)[-1].lower() if "." in attachment_filename else ""
        if ext in RISKY_EXTENSIONS:
            detections.append(Detection(
                category="risky_attachment",
                description=f"Attachment has a potentially dangerous extension ({ext})",
                severity="high",
            ))

    return detections


def scan_email(
    subject: str,
    body: str,
    sender: str = "",
    attachment_text: str = "",
    attachment_filename: str | None = None,
) -> ScanResult:
    """
    Run all security checks on an email.  Returns a ScanResult with all
    detections aggregated.  This is the main entry point called by the
    security consumer.
    """
    full_text = f"{subject}\n{body}\n{attachment_text}".strip()

    sensitive = scan_for_sensitive_content(full_text)
    phishing = scan_for_phishing_signals(full_text, sender=sender, attachment_filename=attachment_filename)

    result = ScanResult(
        detections=sensitive + phishing,
        sensitive_content_found=bool(sensitive),
        phishing_signals_found=bool(phishing),
        risky_attachment=any(d.category == "risky_attachment" for d in phishing),
        auth_check_available=False,  # SPF/DKIM/DMARC not available in current flow
    )
    return result
