"""
Runs a classifier function over the labeled evaluation dataset and
computes accuracy/precision/recall/F1, a per-difficulty breakdown, and
(where applicable) the human-review rate at the configured confidence
threshold.

This module doesn't care WHICH approach it's evaluating — `compare_methods.py`
supplies one of three classify_fn callables (keyword baseline, LLM-only,
RAG+LLM) and this module just runs the loop and computes metrics
identically for all three, so the comparison is apples-to-apples.
"""

import json
import logging
from pathlib import Path
from typing import Callable

from app.core.config import settings
from evaluation.metrics import breakdown_by_field, classification_metrics, human_review_rate

logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_emails.json"


def load_evaluation_dataset() -> list[dict]:
    return json.loads(EVAL_DATASET_PATH.read_text())


def evaluate_classifier(
    classify_fn: Callable[[str], dict],
    dataset: list[dict],
    confidence_threshold: float | None = None,
) -> dict:
    """
    `classify_fn(email_text) -> {"team_id": int, "confidence": float, ...}`

    Any exception from classify_fn for a given email is caught and scored
    as an incorrect prediction (team_id=-1) rather than crashing the whole
    evaluation run — one bad LLM response shouldn't stop us from scoring
    the other 99.
    """
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

    y_true, y_pred, statuses, correctness = [], [], [], []

    for item in dataset:
        expected_id = item["expected_team_id"]
        try:
            prediction = classify_fn(item["email"])
        except Exception as exc:  # noqa: BLE001
            logger.error("classify_fn raised for eval item %s: %s", item["id"], exc)
            prediction = {"team_id": -1, "confidence": 0.0}

        predicted_id = prediction.get("team_id", -1)
        confidence = prediction.get("confidence", 0.0)

        y_true.append(expected_id)
        y_pred.append(predicted_id)
        correctness.append(predicted_id == expected_id)
        statuses.append("REVIEW_REQUIRED" if confidence < threshold else "ROUTED")

    metrics = classification_metrics(y_true, y_pred)
    metrics["human_review_rate"] = human_review_rate(statuses)
    metrics["by_difficulty"] = breakdown_by_field(dataset, correctness, "difficulty")
    return metrics


def main() -> None:
    """Ad-hoc CLI: evaluate just the keyword baseline (no LLM/API needed)."""
    from app.rag.loader import load_departments
    from evaluation.keyword_baseline import build_team_keyword_index, classify_keyword_baseline

    dataset = load_evaluation_dataset()
    departments = load_departments()
    team_index = build_team_keyword_index(departments)

    report = evaluate_classifier(lambda text: classify_keyword_baseline(text, team_index), dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
