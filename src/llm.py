"""
Provider-agnostic LLM and embedding clients.

Works with any OpenAI-compatible API. The provider is chosen by environment
variables, so switching provider needs no code changes:

    LLM_BASE_URL     https://api.openai.com/v1 (default when unset)
    LLM_API_KEY      provider API key
    LLM_CHAT_MODEL   default: gpt-4o-mini
    LLM_EMBED_MODEL  default: text-embedding-3-small

The embedding model must stay text-embedding-3-small (1536 dims): the vectors
already stored in pgvector were produced by it, and a different model would
make similarity search meaningless without re-running ingest.
"""

import os
from typing import Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from openai import AsyncOpenAI, OpenAI
from pydantic import PrivateAttr


def _base_url() -> str | None:
    return os.environ.get("LLM_BASE_URL") or None


def _is_openai() -> bool:
    url = _base_url()
    return url is None or "api.openai.com" in url


def chat_model_name() -> str:
    return os.environ.get("LLM_CHAT_MODEL") or "gpt-4o-mini"


def embed_model_name() -> str:
    return os.environ.get("LLM_EMBED_MODEL") or "text-embedding-3-small"


def chat_llm(**kwargs: Any) -> ChatOpenAI:
    if _is_openai():
        # ADR-016: routes repeated system-prompt prefixes to the same cache server.
        # Not sent to other providers, which may reject unknown request fields.
        kwargs.setdefault("model_kwargs", {"prompt_cache_key": "langchain-prompt-caching"})
    return ChatOpenAI(
        model=chat_model_name(),
        base_url=_base_url(),
        api_key=os.environ["LLM_API_KEY"],
        **kwargs,
    )


def langchain_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False sends raw text instead of tiktoken IDs,
    # which non-OpenAI endpoints don't accept.
    return OpenAIEmbeddings(
        model=embed_model_name(),
        base_url=_base_url(),
        api_key=os.environ["LLM_API_KEY"],
        check_embedding_ctx_length=False,
    )


class CompatEmbedding(BaseEmbedding):
    """
    LlamaIndex embedding over any OpenAI-compatible endpoint.

    LlamaIndex's own OpenAIEmbedding rejects model names outside its built-in
    enum, which would block non-OpenAI providers, so this calls the openai
    client directly.
    """

    _client: OpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name=embed_model_name(), embed_batch_size=50, **kwargs)
        creds = {"base_url": _base_url(), "api_key": os.environ["LLM_API_KEY"]}
        self._client = OpenAI(**creds)
        self._aclient = AsyncOpenAI(**creds)

    @classmethod
    def class_name(cls) -> str:
        return "CompatEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]

    async def _aembed(self, texts: List[str]) -> List[List[float]]:
        resp = await self._aclient.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts)
