"""
Classification service.

Orchestrates: build prompt -> call LLM -> parse JSON -> validate with
Pydantic -> enforce that the returned team_id was actually one of the
retrieved candidates (never trust the LLM to only pick from the list just
because we asked nicely — enforce it in code).

Failure handling, per the project requirements: invalid output is never
silently accepted. We retry once with a corrective follow-up prompt; if
it's still invalid, we raise ClassificationError, which the calling
LangGraph node (Phase 4) turns into a FAILED email status.
"""

import json
import logging

from app.llm.client import LLMClient
from app.rag.prompts import build_classification_prompt
from app.schemas.classification import ClassificationError, ClassificationResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 1


def _extract_json(raw_text: str) -> dict:
    """
    Best-effort extraction of a JSON object from the model's raw response.

    Even with explicit instructions, some models wrap JSON in markdown
    fences or add a stray sentence. We strip common wrapping before
    attempting json.loads, but we do NOT try to fix malformed JSON itself
    — that's exactly the kind of "silently continue with invalid data"
    the project rules forbid.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ClassificationError(f"No JSON object found in LLM response: {raw_text!r}")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"LLM response was not valid JSON: {exc}") from exc


def _validate_team_id_is_grounded(result: ClassificationResult, valid_team_ids: set[int]) -> None:
    if result.team_id not in valid_team_ids:
        raise ClassificationError(
            f"LLM returned team_id={result.team_id}, which was not among the "
            f"retrieved candidate team_ids {sorted(valid_team_ids)}. Refusing to "
            f"trust an invented team_id."
        )


def classify_email(
    llm_client: LLMClient,
    subject: str,
    body: str,
    context: str,
    valid_team_ids: set[int],
) -> ClassificationResult:
    """
    Runs classification with up to MAX_RETRIES retries. Raises
    ClassificationError if no valid, grounded result could be obtained.
    """
    prompt = build_classification_prompt(subject=subject, body=body, context=context)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = llm_client.generate(prompt)
            parsed = _extract_json(raw)
            result = ClassificationResult.model_validate(parsed)
            _validate_team_id_is_grounded(result, valid_team_ids)
            return result
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure triggers retry/raise
            last_error = exc
            logger.warning(
                "Classification attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES + 1, exc
            )
            # On retry, make the correction instruction explicit rather than
            # just repeating the same prompt verbatim.
            prompt = (
                build_classification_prompt(subject=subject, body=body, context=context)
                + "\n\nYour previous response was invalid or used a team_id not in the "
                "candidate list. Respond again with ONLY a valid JSON object as instructed."
            )

    logger.error("Classification failed after %d attempts: %s", MAX_RETRIES + 1, last_error)
    raise ClassificationError(str(last_error))
