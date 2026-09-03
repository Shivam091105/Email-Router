"""
Evaluates the RAG retrieval step in isolation: for each labeled email,
does the correct team_id appear anywhere in the top-K retrieved
candidates? This is measured before any LLM ever sees the email — if
retrieval doesn't surface the right team, no downstream classifier can
recover from that.

Usage (requires a built Chroma index and HUGGINGFACEHUB_API_TOKEN):
    python -m evaluation.evaluate_retrieval
"""

import json
import logging
from pathlib import Path

from app.core.logging import configure_logging
from app.rag.embeddings import HuggingFaceInferenceEmbeddings
from app.rag.retriever import retrieve_teams
from app.rag.vectorstore import load_vectorstore
from evaluation.metrics import recall_at_k

configure_logging()
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation_emails.json"


def load_evaluation_dataset() -> list[dict]:
    return json.loads(EVAL_DATASET_PATH.read_text())


def evaluate_retrieval(vectorstore, dataset: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    retrieved_per_query: dict[int, list[list[int]]] = {k: [] for k in k_values}
    expected_ids = [item["expected_team_id"] for item in dataset]

    max_k = max(k_values)
    for item in dataset:
        results = retrieve_teams(vectorstore, item["email"], k=max_k)
        all_ids = [doc.metadata["team_id"] for doc, _score in results]
        for k in k_values:
            retrieved_per_query[k].append(all_ids[:k])

    return {f"recall_at_{k}": recall_at_k(retrieved_per_query[k], expected_ids) for k in k_values}


def main() -> None:
    dataset = load_evaluation_dataset()
    logger.info("Loaded %d labeled examples", len(dataset))

    embedding = HuggingFaceInferenceEmbeddings()
    vectorstore = load_vectorstore(embedding)

    report = evaluate_retrieval(vectorstore, dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
