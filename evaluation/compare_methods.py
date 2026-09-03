"""
Compares all three classification approaches on the same labeled dataset
and writes a markdown report to evaluation/comparison_report.md.

    Approach 1: keyword_baseline.classify_keyword_baseline   (no LLM, no network)
    Approach 2: llm_only.classify_llm_only                   (LLM, no retrieval)
    Approach 3: classification_service.classify_email        (RAG + LLM, production path)

Approaches 2 and 3 make real Hugging Face Inference API calls - one per
email in the dataset, per approach - so running this fully requires
HUGGINGFACEHUB_API_TOKEN in .env and will take a few minutes for a
~80-item dataset. No numbers in this script are hardcoded; every value in
the generated report is computed from an actual run.

Usage:
    python -m evaluation.compare_methods
"""

import json
import logging
from pathlib import Path

from app.core.dependencies import get_llm_client, get_vectorstore
from app.core.logging import configure_logging
from app.rag.loader import load_departments
from app.rag.retriever import format_context, retrieve_teams
from app.services.classification_service import classify_email
from evaluation.evaluate_classification import evaluate_classifier, load_evaluation_dataset
from evaluation.keyword_baseline import build_team_keyword_index, classify_keyword_baseline
from evaluation.llm_only import classify_llm_only

configure_logging()
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent / "comparison_report.md"


def make_keyword_classifier(departments: list[dict]):
    team_index = build_team_keyword_index(departments)
    return lambda text: classify_keyword_baseline(text, team_index)


def make_llm_only_classifier(llm_client, departments: list[dict]):
    return lambda text: classify_llm_only(llm_client, text, departments)


def make_rag_llm_classifier(llm_client, vectorstore, top_k: int = 3):
    def classify(text: str) -> dict:
        results = retrieve_teams(vectorstore, text, k=top_k)
        context = format_context(results)
        valid_ids = {doc.metadata["team_id"] for doc, _score in results}
        result = classify_email(llm_client, subject="", body=text, context=context, valid_team_ids=valid_ids)
        return result.model_dump()

    return classify


def render_markdown_report(results: dict[str, dict]) -> str:
    lines = [
        "# Classification Approach Comparison",
        "",
        "Computed from an actual run of `evaluation/compare_methods.py` against",
        f"`data/evaluation_emails.json` ({next(iter(results.values()))['n']} labeled examples).",
        "No numbers below are hardcoded.",
        "",
        "| Approach | Accuracy | Precision | Recall | F1 | Human Review Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | "
            f"{m['f1']:.2%} | {m['human_review_rate']:.2%} |"
        )

    lines.append("")
    lines.append("## Accuracy by difficulty category")
    lines.append("")
    difficulties = sorted({d for m in results.values() for d in m["by_difficulty"]})
    header = "| Approach | " + " | ".join(difficulties) + " |"
    sep = "|---|" + "---:|" * len(difficulties)
    lines.append(header)
    lines.append(sep)
    for name, m in results.items():
        row = [f"{m['by_difficulty'].get(d, {'accuracy': 0})['accuracy']:.0%}" for d in difficulties]
        lines.append(f"| {name} | " + " | ".join(row) + " |")

    return "\n".join(lines)


def main() -> None:
    dataset = load_evaluation_dataset()
    departments = load_departments()
    logger.info("Evaluating 3 approaches over %d labeled examples", len(dataset))

    results = {}

    logger.info("Running Approach 1: keyword baseline...")
    results["Keyword baseline"] = evaluate_classifier(make_keyword_classifier(departments), dataset)

    llm_client = get_llm_client()
    vectorstore = get_vectorstore()

    logger.info("Running Approach 2: LLM only (no RAG)...")
    results["LLM only"] = evaluate_classifier(make_llm_only_classifier(llm_client, departments), dataset)

    logger.info("Running Approach 3: RAG + LLM...")
    results["RAG + LLM"] = evaluate_classifier(make_rag_llm_classifier(llm_client, vectorstore), dataset)

    report_md = render_markdown_report(results)
    REPORT_PATH.write_text(report_md)
    logger.info("Wrote comparison report to %s", REPORT_PATH)
    print(report_md)
    print("\nFull metrics JSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
