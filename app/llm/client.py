"""
LLM client abstraction.

Mirrors the pattern used for embeddings in app/rag/embeddings.py: an
abstract interface (`LLMClient`), a real implementation backed by the
Hugging Face Inference API, and a fake implementation for tests. Nothing
downstream (classification_service, summary_service, LangGraph nodes)
imports huggingface_hub directly — they depend only on `LLMClient.generate`.

This is also the seam where a future OpenAI or Ollama provider would be
added: a new class implementing `generate(prompt) -> str`, nothing else
in the codebase changes.
"""

from abc import ABC, abstractmethod

from huggingface_hub import InferenceClient

from app.core.config import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Returns the raw text completion for `prompt`."""


class HuggingFaceLLMClient(LLMClient):
    """Calls a chat-capable model via the Hugging Face Inference API."""

    def __init__(self, model_name: str | None = None, api_token: str | None = None):
        self.model_name = model_name or settings.llm_model_name
        self._client = InferenceClient(
            model=self.model_name,
            token=api_token or settings.huggingfacehub_api_token,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content


class FakeLLMClient(LLMClient):
    """
    Test-only client. Returns whatever fixed response(s) it was
    constructed with, in order, with no network call. Lets tests exercise
    retry/validation logic by queuing up an invalid response followed by
    a valid one, etc.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("FakeLLMClient ran out of queued responses")
        return self._responses.pop(0)
