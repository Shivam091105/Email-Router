"""
Application-level singletons.

The LLM client and vector store are expensive-ish to construct and are
shared, read-mostly resources — we build them once at import time rather
than per-request. FastAPI route handlers and the background task both
import `get_llm_client()` / `get_vectorstore()` from here rather than
constructing their own.

Defaults: Groq for the LLM (free, no credit card) and local
sentence-transformers for embeddings (free, no API key at all). Hugging
Face Inference API classes remain available in app/llm/client.py and
app/rag/embeddings.py for anyone with HF billing configured, but they are
no longer the default — HF's Inference Providers now require a payment
method on file even for free-tier usage, which defeats the point for a
portfolio project.
"""

import logging

from app.llm.client import GroqLLMClient, LLMClient
from app.rag.embeddings import LocalSentenceTransformerEmbeddings
from app.rag.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None
_vectorstore = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = GroqLLMClient()
    return _llm_client


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding = LocalSentenceTransformerEmbeddings()
        _vectorstore = load_vectorstore(embedding)
    return _vectorstore