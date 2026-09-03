"""
Email parser tests.

Builds real RFC 5322 messages in memory with Python's `email` module and
feeds the raw bytes to `parse_raw_email` — no IMAP connection, no network.
"""

from email.message import EmailMessage

from app.integrations.email_parser import parse_raw_email, sanitize_filename


def _build_plain_email(subject="Hello", body="Hello world", to="team@example.com") -> bytes:
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    return bytes(msg)


def _build_email_with_attachment(filename: str, content: bytes) -> bytes:
    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "team@example.com"
    msg["Subject"] = "Invoice attached"
    msg.set_content("Please see attached invoice.")
    msg.add_attachment(content, maintype="application", subtype="pdf", filename=filename)
    return bytes(msg)


def test_parse_plain_email():
    raw = _build_plain_email(subject="Login issue", body="I can't log in.")
    parsed = parse_raw_email(raw)

    assert parsed["sender"] == "sender@example.com"
    assert parsed["recipients"] == ["team@example.com"]
    assert parsed["subject"] == "Login issue"
    assert "I can't log in." in parsed["body_text"]
    assert parsed["attachments"] == []


def test_parse_email_with_attachment_extracts_content():
    raw = _build_email_with_attachment("invoice.pdf", b"%PDF-1.4 fake content")
    parsed = parse_raw_email(raw)

    assert len(parsed["attachments"]) == 1
    attachment = parsed["attachments"][0]
    assert attachment["filename"] == "invoice.pdf"
    assert attachment["content"] == b"%PDF-1.4 fake content"
    assert attachment["content_type"] == "application/pdf"


def test_oversized_attachment_is_skipped_not_raised():
    big_content = b"x" * (2 * 1024 * 1024)  # 2MB
    raw = _build_email_with_attachment("big.pdf", big_content)

    parsed = parse_raw_email(raw, max_attachment_size_mb=1)

    assert parsed["attachments"] == []  # skipped, and the rest of the email still parsed
    assert parsed["subject"] == "Invoice attached"


def test_sanitize_filename_strips_path_and_unsafe_chars():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("invoice report (final)!.pdf") == "invoice_report__final__.pdf"
    assert sanitize_filename("C:\\Users\\me\\file.pdf") == "file.pdf"


def test_sanitize_filename_handles_empty_result():
    assert sanitize_filename("???") == "___" or sanitize_filename("???") != ""
