"""
Run this once (and again any time data/departments.json changes) to build
the Chroma vector index that the running application queries.

Usage:
    python -m scripts.build_index

Requires HUGGINGFACEHUB_API_TOKEN to be set in .env, since real
embeddings are computed via the Hugging Face Inference API.
"""

import logging

from app.core.logging import configure_logging
from app.rag.embeddings import HuggingFaceInferenceEmbeddings
from app.rag.loader import load_team_documents
from app.rag.vectorstore import build_vectorstore

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading organizational knowledge base...")
    documents = load_team_documents()

    logger.info("Embedding %d team documents via Hugging Face Inference API...", len(documents))
    embedding = HuggingFaceInferenceEmbeddings()

    build_vectorstore(documents, embedding)
    logger.info("Index build complete.")


if __name__ == "__main__":
    main()
