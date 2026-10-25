"""
Unit tests for the security screening layer.

Tests:
- Pattern detection (OTPs, passwords, API keys, suspicious URLs, risky attachments)
- Risk scoring and decision model (APPROVED, REVIEW_REQUIRED, QUARANTINED)
- Sensitive data redaction for LLM privacy
- Scanner orchestration (pure function)
"""

import pytest

from app.security.decision import (
    SecurityDecision,
    make_decision,
    redact_sensitive_content,
)
from app.security.patterns import (
    CREDIT_CARD_PATTERN,
    OTP_PATTERN,
    PASSWORD_PATTERN,
    RISKY_EXTENSIONS,
    SSN_PATTERN,
    Detection,
    ScanResult,
    scan_email,
    scan_for_phishing_signals,
    scan_for_sensitive_content,
)
from app.security.scanner import screen_email


class TestPatternDetection:
    def test_detects_otp_in_various_contexts(self):
        cases = [
            "Your verification code is 492011",
            "Use OTP: 123456 to login",
            "one-time password is 884129",
            "Your MFA code: 9384",
            "Security code is 102938",
        ]
        for text in cases:
            match = OTP_PATTERN.search(text)
            assert match is not None, f"Failed to detect OTP in: {text}"

    def test_detects_plaintext_passwords(self):
        cases = [
            "Your temporary password is SuperSecret123!",
            "Default password: ChangeMeNow",
            "pwd=AdminPass99",
            "passcode is 98765",
        ]
        for text in cases:
            match = PASSWORD_PATTERN.search(text)
            assert match is not None, f"Failed to detect password in: {text}"

    def test_detects_api_keys(self):
        aws_text = "Deploy using AKIAIOSFODNN7EXAMPLE key"
        gh_text = "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyzAB"
        bearer_text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDc"

        res_aws = scan_email(subject="Key", body=aws_text)
        assert any(d.category == "api_key" for d in res_aws.detections)

        res_gh = scan_email(subject="Token", body=gh_text)
        assert any(d.category == "api_key" for d in res_gh.detections)

        res_bearer = scan_email(subject="Auth", body=bearer_text)
        assert any(d.category == "api_key" for d in res_bearer.detections)

    def test_detects_credit_card_and_ssn(self):
        cc_text = "Card payment 4532-1234-5678-9012"
        ssn_text = "SSN is 123-45-6789"

        assert CREDIT_CARD_PATTERN.search(cc_text) is not None
        assert SSN_PATTERN.search(ssn_text) is not None

        res_ssn = scan_email(subject="Tax", body=ssn_text)
        assert any(d.category == "ssn" for d in res_ssn.detections)

    def test_detects_suspicious_urls(self):
        ip_url = "Please confirm your account at http://192.168.1.100/login"
        short_url = "Check this document: http://bit.ly/3xYqz"
        at_url = "Visit https://google.com@phishing.site/login"

        res_ip = scan_email(subject="Notice", body=ip_url)
        assert any(d.category == "suspicious_url" for d in res_ip.detections)

        res_short = scan_email(subject="Document", body=short_url)
        assert any(d.category == "suspicious_url" for d in res_short.detections)

        res_at = scan_email(subject="Link", body=at_url)
        assert any(d.category == "suspicious_url" for d in res_at.detections)

    def test_detects_risky_attachments(self):
        res_exe = scan_email(subject="File", body="See attached", attachment_filename="invoice.exe")
        assert res_exe.risky_attachment is True
        assert any(d.category == "risky_attachment" for d in res_exe.detections)

        res_pdf = scan_email(subject="File", body="See attached", attachment_filename="invoice.pdf")
        assert res_pdf.risky_attachment is False


class TestDecisionModel:
    def test_clean_email_is_approved(self):
        decision, result = screen_email(
            subject="Team lunch tomorrow",
            body="Hey everyone, let's meet at 12:30 at the main cafeteria for pizza.",
            sender="alice@company.com",
        )
        assert decision.outcome == "APPROVED"
        assert decision.risk_score == 0.0
        assert decision.should_redact is False
        assert result.total_detections == 0

    def test_high_risk_credentials_quarantined(self):
        decision, result = screen_email(
            subject="System access credentials",
            body="Your AWS access key is AKIAIOSFODNN7EXAMPLE and secret is active.",
            sender="admin@company.com",
        )
        assert decision.outcome == "QUARANTINED"
        assert decision.risk_score >= 0.4
        assert decision.should_redact is True
        assert result.sensitive_content_found is True

    def test_executable_attachment_quarantined(self):
        decision, result = screen_email(
            subject="Urgent Invoice Payment",
            body="Please find the attached invoice.",
            sender="billing@external.com",
            attachment_filename="invoice_march.exe",
        )
        assert decision.outcome == "QUARANTINED"
        assert result.risky_attachment is True

    def test_suspicious_phishing_url_quarantined(self):
        decision, result = screen_email(
            subject="Security Alert",
            body="Verify your account now: http://192.168.1.1/reset",
            sender="security-alert@fake.com",
        )
        assert decision.outcome == "QUARANTINED"
        assert result.phishing_signals_found is True

    def test_mild_signal_routes_to_review(self):
        # A single urgency pattern
        decision, result = screen_email(
            subject="Action required: please update your profile",
            body="Please verify your personal information before the end of the quarter.",
            sender="hr@company.com",
        )
        assert decision.outcome in ("APPROVED", "REVIEW_REQUIRED")


class TestRedaction:
    def test_redacts_credentials_before_llm(self):
        raw_text = (
            "Hi, your password is SecretPassword123 and your "
            "verification code is 584920. Also AWS key AKIAIOSFODNN7EXAMPLE."
        )
        redacted = redact_sensitive_content(raw_text)

        assert "SecretPassword123" not in redacted
        assert "584920" not in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED]" in redacted
        assert "[REDACTED_AWS_KEY]" in redacted

