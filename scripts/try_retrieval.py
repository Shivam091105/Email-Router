"""
Manual, human-in-the-loop check of retrieval quality against the real
knowledge base — not an automated test (that's tests/test_rag.py), but a
quick way to eyeball "does this actually retrieve sensible teams" before
wiring retrieval into the Phase 3 classification prompt.

Uses local sentence-transformers embeddings by default — no API key
needed, so this is judging real semantic retrieval quality, not the fake
test embeddings, with zero external dependency.

Usage:
    python -m scripts.try_retrieval "I can't log into my account"
"""

import sys

from app.core.logging import configure_logging
from app.rag.embeddings import LocalSentenceTransformerEmbeddings
from app.rag.retriever import retrieve_teams
from app.rag.vectorstore import load_vectorstore

configure_logging()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.try_retrieval "some email text"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    embedding = LocalSentenceTransformerEmbeddings()
    vectorstore = load_vectorstore(embedding)

    results = retrieve_teams(vectorstore, query, k=3)

    print(f"\nQuery: {query}\n")
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"{rank}. [score={score:.4f}] team_id={meta['team_id']}  {meta['department']} / {meta['team']}")


if __name__ == "__main__":
    main()