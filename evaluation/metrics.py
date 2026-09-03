"""
Metric computation for the evaluation suite.

Kept as pure functions operating on plain lists of predictions/labels, so
they can be unit tested with fixture data (tests/test_evaluation.py)
completely independent of whether any real LLM or embedding call ever
happened. The evaluation scripts (evaluate_classification.py, etc.) are
what actually call an LLM; this module never does.
"""

from collections import Counter


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Computes accuracy and macro-averaged precision/recall/F1 over
    predicted vs. true team_ids.

    Macro averaging (rather than micro/weighted) is used because we care
    about performing reasonably across ALL teams, including rare ones in
    a small evaluation set — a classifier that's perfect on the most
    common team but terrible on rare ones should not look "good" overall.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length"
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    labels = sorted(set(y_true) | set(y_pred))
    precisions, recalls, f1s = [], [], []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t != label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall": round(sum(recalls) / len(recalls), 4),
        "f1": round(sum(f1s) / len(f1s), 4),
        "n": n,
    }


def recall_at_k(retrieved_ids_per_query: list[list[int]], expected_id_per_query: list[int]) -> float:
    """
    Fraction of queries where the expected team_id appears ANYWHERE in
    that query's top-k retrieved team_ids. This measures the retrieval
    step alone (before any LLM ever sees the candidates) — if the correct
    team isn't even retrieved, no classifier downstream can get it right.
    """
    assert len(retrieved_ids_per_query) == len(expected_id_per_query)
    n = len(expected_id_per_query)
    if n == 0:
        return 0.0
    hits = sum(
        1 for retrieved, expected in zip(retrieved_ids_per_query, expected_id_per_query) if expected in retrieved
    )
    return round(hits / n, 4)


def human_review_rate(statuses: list[str]) -> float:
    """Fraction of predictions that would be routed to human review (below the confidence threshold)."""
    if not statuses:
        return 0.0
    return round(sum(1 for s in statuses if s == "REVIEW_REQUIRED") / len(statuses), 4)


def breakdown_by_field(dataset: list[dict], correctness: list[bool], field: str) -> dict[str, dict]:
    """
    Accuracy broken down by an arbitrary dataset field (e.g. "difficulty"),
    so a report can show "straightforward: 95%, ambiguous: 40%" instead of
    just one blended number.
    """
    assert len(dataset) == len(correctness)
    counts: dict[str, Counter] = {}
    for item, correct in zip(dataset, correctness):
        key = item[field]
        counts.setdefault(key, Counter())
        counts[key]["total"] += 1
        if correct:
            counts[key]["correct"] += 1

    return {
        key: {
            "accuracy": round(c["correct"] / c["total"], 4) if c["total"] else 0.0,
            "n": c["total"],
        }
        for key, c in counts.items()
    }
