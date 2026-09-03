"""
Approach 2: LLM classification WITHOUT RAG.

The LLM is given the full list of team names and departments (so it has
*some* notion of what categories exist — otherwise this wouldn't be a
meaningful classifier at all) but NOT the rich descriptions/example
issues that retrieval would normally surface, and NOT narrowed to a
relevant subset via similarity search. This isolates the effect of
"retrieval-augmented context" from "having categories to choose from at
all" — which is the actual thing Approach 3 (RAG+LLM) adds.
"""

import json
import logging

from app.llm.client import LLMClient
from app.schemas.classification import ClassificationResult

logger = logging.getLogger(__name__)

LLM_ONLY_PROMPT = """You are an email routing assistant. Classify the email below into \
exactly one of these teams (no other information about them is available to you):

{team_list}

Respond with ONLY a JSON object, no other text:
{{
  "department": "<department name>",
  "team": "<team name>",
  "team_id": <integer, must be one of the ids listed above>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Email:
{email_text}

JSON response:"""


def build_team_list_text(departments: list[dict]) -> str:
    lines = []
    for dept in departments:
        for team in dept["teams"]:
            lines.append(f"- team_id {team['team_id']}: {dept['department']} / {team['name']}")
    return "\n".join(lines)


def classify_llm_only(llm_client: LLMClient, email_text: str, departments: list[dict]) -> dict:
    team_list = build_team_list_text(departments)
    prompt = LLM_ONLY_PROMPT.format(team_list=team_list, email_text=email_text)

    raw = llm_client.generate(prompt)
    text = raw.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        logger.warning("LLM-only approach returned no JSON: %r", raw)
        return {"department": "UNKNOWN", "team": "UNKNOWN", "team_id": -1, "confidence": 0.0}

    try:
        parsed = json.loads(text[start : end + 1])
        result = ClassificationResult.model_validate(parsed)
        return result.model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM-only approach produced invalid output: %s (%r)", exc, raw)
        return {"department": "UNKNOWN", "team": "UNKNOWN", "team_id": -1, "confidence": 0.0}
