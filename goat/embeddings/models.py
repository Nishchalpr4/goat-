"""Embedding model abstractions — provider-agnostic interfaces for
OpenAI, Cohere, Voyage, Jina, and local models.

Supports: dimension shortening (MRL), batch embedding, multilingual models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.config import EmbeddingModelSpec

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a single text."""
    text: str
    vector: list[float]
    model: str
    dimensions: int
    tokens_used: int = 0


class BaseEmbeddingModel(ABC):
    """Abstract base for embedding models."""

    def __init__(self, spec: EmbeddingModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dimensions(self) -> int:
        return self.spec.dimensions

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]: ...


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model with dimension shortening support.

    Supports text-embedding-3-small and text-embedding-3-large with
    controllable output dimensions via the API's `dimensions` parameter.
    """

    def __init__(self, spec: EmbeddingModelSpec, api_key: str = ""):
        super().__init__(spec)
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key or None)
            except ImportError:
                raise RuntimeError("openai package required for OpenAI embeddings")
        return self._client

    def embed(self, text: str) -> EmbeddingResult:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        client = self._get_client()
        model_name = self.spec.name.replace("openai/", "")

        kwargs = {"model": model_name, "input": texts}
        if self.spec.supports_dimension_shortening and self.spec.dimensions:
            kwargs["dimensions"] = self.spec.dimensions

        response = client.embeddings.create(**kwargs)
        results = []
        for i, embedding_data in enumerate(response.data):
            results.append(EmbeddingResult(
                text=texts[i],
                vector=embedding_data.embedding,
                model=self.spec.name,
                dimensions=len(embedding_data.embedding),
                tokens_used=response.usage.total_tokens // len(texts),
            ))
        return results


class CohereEmbeddingModel(BaseEmbeddingModel):
    """Cohere multilingual embedding model."""

    def __init__(self, spec: EmbeddingModelSpec, api_key: str = ""):
        super().__init__(spec)
        self._api_key = api_key

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        try:
            import cohere
        except ImportError:
            raise RuntimeError("cohere package required for Cohere embeddings")

        client = cohere.Client(api_key=self._api_key)
        model_name = self.spec.name.replace("cohere/", "")
        response = client.embed(
            texts=texts,
            model=model_name,
            input_type="search_document",
        )
        results = []
        for i, vec in enumerate(response.embeddings):
            results.append(EmbeddingResult(
                text=texts[i],
                vector=vec,
                model=self.spec.name,
                dimensions=len(vec),
            ))
        return results


class VoyageEmbeddingModel(BaseEmbeddingModel):
    """Voyage AI embedding model with contextualized chunk embedding support."""

    def __init__(self, spec: EmbeddingModelSpec, api_key: str = ""):
        super().__init__(spec)
        self._api_key = api_key

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        try:
            import voyageai
        except ImportError:
            raise RuntimeError("voyageai package required for Voyage embeddings")

        client = voyageai.Client(api_key=self._api_key)
        model_name = self.spec.name.replace("voyage/", "")
        response = client.embed(texts, model=model_name)
        results = []
        for i, vec in enumerate(response.embeddings):
            results.append(EmbeddingResult(
                text=texts[i],
                vector=vec,
                model=self.spec.name,
                dimensions=len(vec),
            ))
        return results

    def embed_contextualized(self, chunks: list[str],
                              document_context: str) -> list[EmbeddingResult]:
        """Embed chunks with document context (Voyage contextualized chunk embeddings)."""
        try:
            import voyageai
        except ImportError:
            raise RuntimeError("voyageai package required for Voyage embeddings")

        client = voyageai.Client(api_key=self._api_key)
        model_name = self.spec.name.replace("voyage/", "")
        # Voyage contextualized API: embed chunks with surrounding context
        contextualized_texts = [
            f"[Context: {document_context}] {chunk}" for chunk in chunks
        ]
        response = client.embed(contextualized_texts, model=model_name)
        results = []
        for i, vec in enumerate(response.embeddings):
            results.append(EmbeddingResult(
                text=chunks[i],
                vector=vec,
                model=self.spec.name,
                dimensions=len(vec),
            ))
        return results


class LocalEmbeddingModel(BaseEmbeddingModel):
    """Local/self-hosted embedding model (e.g., Jina, E5, BGE-M3 via sentence-transformers)."""

    def __init__(self, spec: EmbeddingModelSpec):
        super().__init__(spec)
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.spec.name.replace("local/", "")
                self._model = SentenceTransformer(model_name)
                logger.info("Loaded local embedding model: %s", model_name)
            except ImportError:
                raise RuntimeError("sentence-transformers required for local embeddings")

    def embed(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True)

        results = []
        for i, vec in enumerate(vectors):
            vec_list = vec.tolist()
            # Apply MRL/dimension truncation if configured
            if self.spec.dimensions and len(vec_list) > self.spec.dimensions:
                vec_list = vec_list[: self.spec.dimensions]
                # Re-normalize after truncation
                norm = sum(x * x for x in vec_list) ** 0.5
                if norm > 0:
                    vec_list = [x / norm for x in vec_list]

            results.append(EmbeddingResult(
                text=texts[i],
                vector=vec_list,
                model=self.spec.name,
                dimensions=len(vec_list),
            ))
        return results


def create_embedding_model(spec: EmbeddingModelSpec,
                           api_key: str = "") -> BaseEmbeddingModel:
    """Factory: create the appropriate embedding model."""
    provider = spec.provider.lower()
    if provider == "openai":
        return OpenAIEmbeddingModel(spec, api_key)
    elif provider == "cohere":
        return CohereEmbeddingModel(spec, api_key)
    elif provider == "voyage":
        return VoyageEmbeddingModel(spec, api_key)
    elif provider == "local":
        return LocalEmbeddingModel(spec)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
