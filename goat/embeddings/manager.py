"""Embedding manager — orchestrates dual-embedding strategy (narrative + entity/schema),
model versioning, batch processing, and dimension-shortening tradeoffs.
"""

from dataclasses import dataclass
from typing import Optional
import logging

from goat.config import EmbeddingConfig, EmbeddingModelSpec
from goat.embeddings.models import (
    BaseEmbeddingModel, EmbeddingResult, create_embedding_model,
)
from goat.models.document import Chunk

logger = logging.getLogger(__name__)


@dataclass
class DualEmbeddingResult:
    """Result of the dual-embedding strategy: narrative + entity/schema conditioned."""
    chunk_id: str
    narrative_embedding: Optional[EmbeddingResult] = None
    entity_schema_embedding: Optional[EmbeddingResult] = None


class EmbeddingManager:
    """Manages dual-embedding pipeline with versioning and cost control.

    Dual-embedding strategy:
      1. Narrative embedding: pure text meaning (for general semantic recall)
      2. Entity+schema embedding: [Company | Ticker | Sector | DocType | Section] + text
         (for entity-disambiguated retrieval)

    Both are stored and retrieved with fusion (RRF / weighted blend).
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None,
                 api_keys: Optional[dict[str, str]] = None):
        self.config = config or EmbeddingConfig()
        self._api_keys = api_keys or {}
        self._narrative_model: Optional[BaseEmbeddingModel] = None
        self._entity_model: Optional[BaseEmbeddingModel] = None

    def _get_model(self, spec: EmbeddingModelSpec) -> BaseEmbeddingModel:
        api_key = self._api_keys.get(spec.provider, "")
        return create_embedding_model(spec, api_key)

    @property
    def narrative_model(self) -> BaseEmbeddingModel:
        if self._narrative_model is None:
            # Apply dimension shortening if configured
            spec = self.config.narrative_model
            if self.config.narrative_reduced_dims:
                spec = EmbeddingModelSpec(
                    name=spec.name, provider=spec.provider,
                    dimensions=self.config.narrative_reduced_dims,
                    max_tokens=spec.max_tokens,
                    supports_dimension_shortening=spec.supports_dimension_shortening,
                    supports_multilingual=spec.supports_multilingual,
                    batch_size=spec.batch_size,
                )
            self._narrative_model = self._get_model(spec)
        return self._narrative_model

    @property
    def entity_model(self) -> BaseEmbeddingModel:
        if self._entity_model is None:
            spec = self.config.entity_schema_model
            if self.config.entity_reduced_dims:
                spec = EmbeddingModelSpec(
                    name=spec.name, provider=spec.provider,
                    dimensions=self.config.entity_reduced_dims,
                    max_tokens=spec.max_tokens,
                    supports_dimension_shortening=spec.supports_dimension_shortening,
                    supports_multilingual=spec.supports_multilingual,
                    batch_size=spec.batch_size,
                )
            self._entity_model = self._get_model(spec)
        return self._entity_model

    def embed_chunk(self, chunk: Chunk) -> DualEmbeddingResult:
        """Embed a single chunk with dual strategy."""
        result = DualEmbeddingResult(chunk_id=chunk.chunk_id)

        # Narrative embedding (pure text)
        result.narrative_embedding = self.narrative_model.embed(chunk.text)

        # Entity+schema conditioned embedding
        result.entity_schema_embedding = self.entity_model.embed(chunk.conditioned_text)

        return result

    def embed_chunks(self, chunks: list[Chunk]) -> list[DualEmbeddingResult]:
        """Batch embed chunks with dual strategy."""
        if not chunks:
            return []

        batch_size = self.config.narrative_model.batch_size
        results = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Narrative embeddings
            narrative_texts = [c.text for c in batch]
            narrative_results = self.narrative_model.embed_batch(narrative_texts)

            # Entity+schema conditioned embeddings
            conditioned_texts = [c.conditioned_text for c in batch]
            entity_results = self.entity_model.embed_batch(conditioned_texts)

            for j, chunk in enumerate(batch):
                results.append(DualEmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    narrative_embedding=narrative_results[j],
                    entity_schema_embedding=entity_results[j],
                ))

            logger.debug("Embedded batch %d-%d / %d chunks",
                         i, i + len(batch), len(chunks))

        return results

    def embed_query(self, query_text: str,
                    entity_context: str = "") -> dict[str, EmbeddingResult]:
        """Embed a query for dual-channel retrieval.

        Returns both narrative and entity-conditioned query embeddings.
        """
        embeddings = {}

        # Narrative query embedding
        embeddings["narrative"] = self.narrative_model.embed(query_text)

        # Entity+schema query embedding (if context available)
        if entity_context:
            conditioned = f"[{entity_context}] {query_text}"
            embeddings["entity_schema"] = self.entity_model.embed(conditioned)
        else:
            embeddings["entity_schema"] = self.entity_model.embed(query_text)

        return embeddings

    def get_model_versions(self) -> dict:
        """Get current model version info for provenance tracking."""
        return {
            "narrative_model": self.config.narrative_model.name,
            "narrative_dimensions": (
                self.config.narrative_reduced_dims or
                self.config.narrative_model.dimensions
            ),
            "entity_schema_model": self.config.entity_schema_model.name,
            "entity_schema_dimensions": (
                self.config.entity_reduced_dims or
                self.config.entity_schema_model.dimensions
            ),
        }
