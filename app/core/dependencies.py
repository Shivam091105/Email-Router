"""
Application-level singletons.

The LLM client and vector store are expensive-ish to construct and are
shared, read-mostly resources — we build them once at import time rather
than per-request. FastAPI route handlers and the background task both
import `get_llm_client()` / `get_vectorstore()` from here rather than
constructing their own.

This is also the one place that decides "are we using the real Hugging
Face-backed implementations or not" — useful if we ever want a
demo/offline mode that swaps in fakes without touching route code.
"""

import logging

from app.llm.client import HuggingFaceLLMClient, LLMClient
from app.rag.embeddings import HuggingFaceInferenceEmbeddings
from app.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None
_vectorstore = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = HuggingFaceLLMClient()
    return _llm_client


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding = HuggingFaceInferenceEmbeddings()
        _vectorstore = load_vectorstore(embedding)
    return _vectorstore
