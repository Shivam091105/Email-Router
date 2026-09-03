"""
Evaluation module tests.

These verify the metric *computation* is correct using small fixture
predictions, and that the keyword baseline runs deterministically — they
do NOT call any LLM or embedding API. Real accuracy numbers against the
full dataset and real models are produced by actually running
evaluation/compare_methods.py, not by these tests.
"""

from evaluation.keyword_baseline import build_team_keyword_index, classify_keyword_baseline
from evaluation.metrics import breakdown_by_field, classification_metrics, human_review_rate, recall_at_k


def test_classification_metrics_perfect_predictions():
    y_true = [1, 2, 3, 1]
    y_pred = [1, 2, 3, 1]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["n"] == 4


def test_classification_metrics_all_wrong():
    y_true = [1, 2]
    y_pred = [2, 1]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["f1"] == 0.0


def test_classification_metrics_partial():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]  # 3/4 correct
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75


def test_classification_metrics_empty():
    m = classification_metrics([], [])
    assert m["n"] == 0
    assert m["accuracy"] == 0.0


def test_recall_at_k_all_hits():
    retrieved = [[1, 2, 3], [4, 5]]
    expected = [2, 4]
    assert recall_at_k(retrieved, expected) == 1.0


def test_recall_at_k_partial():
    retrieved = [[1, 2], [4, 5]]
    expected = [1, 99]  # second query's expected id never retrieved
    assert recall_at_k(retrieved, expected) == 0.5


def test_human_review_rate():
    assert human_review_rate(["ROUTED", "ROUTED", "REVIEW_REQUIRED", "REVIEW_REQUIRED"]) == 0.5
    assert human_review_rate([]) == 0.0


def test_breakdown_by_field():
    dataset = [{"difficulty": "easy"}, {"difficulty": "easy"}, {"difficulty": "hard"}]
    correctness = [True, False, False]
    result = breakdown_by_field(dataset, correctness, "difficulty")
    assert result["easy"]["accuracy"] == 0.5
    assert result["easy"]["n"] == 2
    assert result["hard"]["accuracy"] == 0.0


def test_keyword_baseline_picks_best_matching_team():
    departments = [
        {
            "department": "IT",
            "teams": [
                {"team_id": 101, "name": "IT Support", "description": "login password account access issues", "examples": ["cant log in", "password reset"]},
                {"team_id": 201, "name": "Billing", "description": "refunds payments charges invoices", "examples": ["refund request", "double charged"]},
            ],
        }
    ]
    index = build_team_keyword_index(departments)

    result = classify_keyword_baseline("I forgot my password and can't log in to my account", index)
    assert result["team_id"] == 101

    result2 = classify_keyword_baseline("I need a refund, I was charged twice for my invoice", index)
    assert result2["team_id"] == 201


def test_keyword_baseline_handles_no_overlap_gracefully():
    departments = [
        {"department": "IT", "teams": [{"team_id": 101, "name": "IT Support", "description": "login issues", "examples": []}]}
    ]
    index = build_team_keyword_index(departments)
    result = classify_keyword_baseline("completely unrelated words xyz", index)
    assert "team_id" in result  # still returns *a* prediction, doesn't crash
