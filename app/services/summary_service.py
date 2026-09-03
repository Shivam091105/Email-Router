"""
Summary generation.

Deliberately simple: one LLM call, no structured output needed since a
summary is just prose. If the LLM call fails, we fall back to a naive
truncation rather than failing the whole workflow — a missing/imperfect
summary is not worth blocking routing over, unlike an invalid
classification (which we treat as fatal).
"""

import logging

from app.llm.client import LLMClient
from app.rag.prompts import build_summary_prompt

logger = logging.getLogger(__name__)

FALLBACK_SUMMARY_LENGTH = 200


def generate_summary(llm_client: LLMClient, subject: str, body: str) -> str:
    prompt = build_summary_prompt(subject=subject, body=body)
    try:
        summary = llm_client.generate(prompt, max_new_tokens=100).strip()
        if summary:
            return summary
        logger.warning("LLM returned an empty summary; falling back to truncation")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Summary generation failed, falling back to truncation: %s", exc)

    truncated = body.strip().replace("\n", " ")[:FALLBACK_SUMMARY_LENGTH]
    suffix = "..." if len(body.strip()) > FALLBACK_SUMMARY_LENGTH else ""
    return f"{truncated}{suffix}"
