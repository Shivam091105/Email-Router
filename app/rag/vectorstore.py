"""
Chroma vector store management.

Two entry points:
- `build_vectorstore()`  — embeds a fresh set of Documents and persists
  them to disk. Run this once (via scripts/build_index.py) whenever
  data/departments.json changes.
- `load_vectorstore()`   — reopens an already-built index for querying.
  This is what the running application uses; it should NOT re-embed
  anything, since re-embedding on every app startup would be slow and
  waste API calls.

We keep these as plain functions rather than a class, since there's no
state to manage beyond "where is the index on disk" and "which embedding
function to use" — both are just parameters.
"""

import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_vectorstore(
    documents: list[Document],
    embedding: Embeddings,
    persist_directory: str | None = None,
) -> Chroma:
    """
    Embeds `documents` and writes a fresh Chroma index to disk, replacing
    any existing index at that path (so re-running the build script after
    editing departments.json doesn't leave stale, duplicate, or deleted
    team documents behind).
    """
    persist_dir = persist_directory or settings.chroma_persist_dir
    path = Path(persist_dir)

    if path.exists():
        logger.info("Removing existing Chroma index at %s before rebuilding", path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(path),
        collection_name="teams",
    )
    logger.info("Built Chroma index with %d documents at %s", len(documents), path)
    return vectorstore


def load_vectorstore(
    embedding: Embeddings,
    persist_directory: str | None = None,
) -> Chroma:
    """Opens an existing Chroma index for querying."""
    persist_dir = persist_directory or settings.chroma_persist_dir
    path = Path(persist_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"No Chroma index found at {path}. Run scripts/build_index.py first."
        )

    return Chroma(
        embedding_function=embedding,
        persist_directory=str(path),
        collection_name="teams",
    )
