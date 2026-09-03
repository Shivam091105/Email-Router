"""
Classification service tests.

All LLM calls are mocked via FakeLLMClient — no network, no API token
needed, per the project's testing rules.
"""

import json

import pytest

from app.llm.client import FakeLLMClient
from app.schemas.classification import ClassificationError
from app.services.classification_service import classify_email

VALID_RESPONSE = json.dumps(
    {
        "department": "IT",
        "team": "IT Support",
        "team_id": 101,
        "confidence": 0.91,
        "reasoning": "The email describes a login/account access issue.",
    }
)


def test_classify_email_valid_response():
    llm = FakeLLMClient(responses=[VALID_RESPONSE])
    result = classify_email(
        llm_client=llm,
        subject="Can't log in",
        body="I can't access my account",
        context="[team_id: 101] IT / IT Support\n...",
        valid_team_ids={101, 201},
    )
    assert result.team_id == 101
    assert result.confidence == 0.91
    assert len(llm.calls) == 1


def test_classify_email_retries_after_invalid_json_then_succeeds():
    llm = FakeLLMClient(responses=["not json at all", VALID_RESPONSE])
    result = classify_email(
        llm_client=llm,
        subject="Can't log in",
        body="I can't access my account",
        context="[team_id: 101] IT / IT Support\n...",
        valid_team_ids={101},
    )
    assert result.team_id == 101
    assert len(llm.calls) == 2  # confirms a retry actually happened


def test_classify_email_fails_after_exhausting_retries():
    llm = FakeLLMClient(responses=["garbage", "still garbage"])
    with pytest.raises(ClassificationError):
        classify_email(
            llm_client=llm,
            subject="x",
            body="y",
            context="[team_id: 101] IT / IT Support",
            valid_team_ids={101},
        )


def test_classify_email_rejects_invented_team_id():
    """
    Core guardrail: even if the LLM returns perfectly valid JSON, a
    team_id that wasn't in the retrieved candidates must be rejected —
    we never trust the model to invent an ID.
    """
    invented = json.dumps(
        {
            "department": "IT",
            "team": "IT Support",
            "team_id": 999,  # not in valid_team_ids
            "confidence": 0.9,
            "reasoning": "...",
        }
    )
    llm = FakeLLMClient(responses=[invented, invented])
    with pytest.raises(ClassificationError, match="not among the retrieved candidate"):
        classify_email(
            llm_client=llm,
            subject="x",
            body="y",
            context="[team_id: 101] IT / IT Support",
            valid_team_ids={101},
        )


def test_classify_email_rejects_confidence_out_of_range():
    bad_confidence = json.dumps(
        {"department": "IT", "team": "IT Support", "team_id": 101, "confidence": 1.5, "reasoning": "x"}
    )
    llm = FakeLLMClient(responses=[bad_confidence, bad_confidence])
    with pytest.raises(ClassificationError):
        classify_email(
            llm_client=llm, subject="x", body="y", context="ctx", valid_team_ids={101}
        )


def test_classify_email_handles_markdown_fenced_json():
    fenced = f"```json\n{VALID_RESPONSE}\n```"
    llm = FakeLLMClient(responses=[fenced])
    result = classify_email(
        llm_client=llm, subject="x", body="y", context="ctx", valid_team_ids={101}
    )
    assert result.team_id == 101
