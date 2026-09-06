"""
Run this once (and again any time data/departments.json changes) to build
the Chroma vector index that the running application queries.

Usage:
    python -m scripts.build_index

Uses local sentence-transformers embeddings by default — no API key, no
network call after the model is first downloaded and cached (~90MB,
one-time).
"""

import logging

from app.core.logging import configure_logging
from app.rag.embeddings import LocalSentenceTransformerEmbeddings
from app.rag.loader import load_team_documents
from app.rag.vectorstore import build_vectorstore

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading organizational knowledge base...")
    documents = load_team_documents()

    logger.info("Embedding %d team documents locally (first run downloads the model)...", len(documents))
    embedding = LocalSentenceTransformerEmbeddings()

    build_vectorstore(documents, embedding)
    logger.info("Index build complete.")


if __name__ == "__main__":
    main()