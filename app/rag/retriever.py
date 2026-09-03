"""
Retriever built on top of the Chroma vector store.

This module is intentionally thin: it exposes two things the rest of the
app (and Phase 3's classification prompt) will actually need:

1. `retrieve_teams(query, k)` — the raw ranked list of candidate team
   Documents for a given email, with similarity scores. Useful for
   evaluation (Recall@K in Phase 6/11) and for debugging.

2. `format_context(documents)` — turns those Documents into a single
   string block suitable for dropping into an LLM prompt, so the prompt
   never has to know about LangChain Document objects directly.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document


def retrieve_teams(
    vectorstore: Chroma,
    query: str,
    k: int = 3,
) -> list[tuple[Document, float]]:
    """
    Returns the top-k most similar team documents for `query`, each paired
    with a similarity distance score (lower = more similar, for Chroma's
    default cosine/L2 distance depending on configuration).
    """
    return vectorstore.similarity_search_with_score(query, k=k)


def format_context(results: list[tuple[Document, float]]) -> str:
    """
    Formats retrieved team documents into a single prompt-ready string.

    Each block clearly labels department/team/team_id so the LLM (Phase 3)
    can be instructed to only ever choose a team_id that appears in this
    block — never one it invents.
    """
    blocks = []
    for doc, score in results:
        meta = doc.metadata
        blocks.append(
            f"[team_id: {meta['team_id']}] {meta['department']} / {meta['team']}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)
