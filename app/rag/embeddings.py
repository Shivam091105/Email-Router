"""
Embedding providers.

We depend on LangChain's `Embeddings` interface (two methods:
`embed_documents` and `embed_query`) rather than on any single provider's
SDK directly. That's what makes the vector store and retriever code in
this package provider-agnostic — swapping Hugging Face for OpenAI later
means writing one new class here, nothing else in `app/rag` changes.

Two implementations live here:

1. HuggingFaceInferenceEmbeddings — calls the Hugging Face Inference API
   remotely (via `huggingface_hub.InferenceClient`). No local model
   download, no torch dependency — good for a lightweight portfolio
   project and avoids Python-version/heavy-dependency headaches.

2. DeterministicFakeEmbeddings — a hash-based "embedding" with no ML and
   no network call at all. It exists purely for tests: it's fast,
   reproducible, and lets us test retrieval *logic* (does the vector
   store return the right document for a query, does top-k work, etc.)
   without needing a Hugging Face API token or network access in CI.
   It is NOT semantically meaningful and must never be used outside tests.
"""

import hashlib
import logging

import numpy as np
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceInferenceEmbeddings(Embeddings):
    """Embeddings backed by the Hugging Face Inference API's feature-extraction task."""

    def __init__(self, model_name: str | None = None, api_token: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        self._client = InferenceClient(
            model=self.model_name,
            token=api_token or settings.huggingfacehub_api_token,
        )

    def _embed(self, text: str) -> list[float]:
        try:
            result = self._client.feature_extraction(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Hugging Face embedding call failed for model %s: %s", self.model_name, exc)
            raise

        vector = np.array(result)
        # Some models return per-token embeddings (2D); mean-pool to a
        # single fixed-size sentence vector in that case.
        if vector.ndim == 2:
            vector = vector.mean(axis=0)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class DeterministicFakeEmbeddings(Embeddings):
    """
    Hash-based fake embeddings for tests only.

    Produces a fixed-size vector deterministically derived from the input
    text's words, so the same text always maps to the same vector and
    similar/overlapping text produces closer vectors than unrelated text
    (good enough to exercise retrieval logic in tests without any real
    semantic understanding).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[h % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
