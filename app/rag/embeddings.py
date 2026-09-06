"""
Embedding providers.

We depend on LangChain's `Embeddings` interface (two methods:
`embed_documents` and `embed_query`) rather than on any single provider's
SDK directly. That's what makes the vector store and retriever code in
this package provider-agnostic — swapping providers means writing one
new class here, nothing else in `app/rag` changes.

Three implementations live here:

1. LocalSentenceTransformerEmbeddings — runs the embedding model on your
   own CPU via `sentence-transformers`. No API key, no network call, no
   billing account required. This is now the DEFAULT, because Hugging
   Face's Inference Providers now require a payment method on file even
   for free-tier usage — a real barrier for a portfolio project. The
   trade-off: the first call downloads the model (~90MB, one-time,
   cached locally), and embedding is slightly slower than a remote GPU
   call, but for a 10-team knowledge base this is a non-issue.

2. HuggingFaceInferenceEmbeddings — calls the Hugging Face Inference API
   remotely. Kept for reference / for anyone who does have HF billing
   set up, but no longer the default.

3. DeterministicFakeEmbeddings — a hash-based "embedding" with no ML and
   no network call at all. It exists purely for tests: it's fast,
   reproducible, and lets us test retrieval *logic* (does the vector
   store return the right document for a query, does top-k work, etc.)
   without needing any model download or network access in CI.
   It is NOT semantically meaningful and must never be used outside tests.
"""

import hashlib
import logging

import numpy as np
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Runs a sentence-transformers model locally (CPU by default). No API
    key, no network call after the model is first downloaded and cached.

    Uses langchain_huggingface's HuggingFaceEmbeddings under the hood,
    which wraps sentence-transformers directly — we don't reinvent that
    wrapper, just choose which one gets used by default.
    """

    def __init__(self, model_name: str | None = None):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name or settings.embedding_model_name
        logger.info(
            "Loading local embedding model '%s' (first run downloads and caches it)",
            self.model_name,
        )
        self._model = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


class HuggingFaceInferenceEmbeddings(Embeddings):
    """
    Embeddings backed by the Hugging Face Inference API's feature-extraction
    task. Requires HUGGINGFACEHUB_API_TOKEN and, as of HF's current billing
    model, a payment method on file even to use free monthly credits. Kept
    for anyone with that set up; LocalSentenceTransformerEmbeddings is the
    default (see app/core/dependencies.py).
    """

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