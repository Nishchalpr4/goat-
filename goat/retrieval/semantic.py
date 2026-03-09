"""Semantic retrieval — dense ANN search over embeddings with metadata filtering.

Uses dual-embedding strategy: narrative and entity+schema conditioned embeddings.
Each channel contributes candidates that are later fused.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.models.retrieval import RetrievalHit, ScoringBreakdown
from goat.storage.vector_store import BaseVectorStore, SearchFilters
from goat.embeddings.manager import EmbeddingManager
from goat.lexical.query_expander import ExpandedQuery

logger = logging.getLogger(__name__)


@dataclass
class SemanticCandidate:
    """A candidate from semantic (ANN) retrieval."""
    chunk_id: str
    score: float  # cosine similarity
    channel: str  # "narrative" or "entity_schema"
    payload: dict = field(default_factory=dict)

    def to_retrieval_hit(self) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=self.chunk_id,
            doc_id=self.payload.get("doc_id", ""),
            text=self.payload.get("text", ""),
            section=self.payload.get("section", ""),
            company_id=self.payload.get("company_id", ""),
            company_name=self.payload.get("company_name", ""),
            ticker=self.payload.get("ticker", ""),
            doc_type=self.payload.get("doc_type", ""),
            language=self.payload.get("language", "en"),
            scoring=ScoringBreakdown(semantic_score=self.score),
        )


class SemanticRetriever:
    """Dense ANN retrieval using dual-embedding channels."""

    def __init__(self, vector_store: BaseVectorStore,
                 embedding_manager: EmbeddingManager,
                 narrative_collection: str = "goat_narrative",
                 entity_collection: str = "goat_entity_schema"):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.narrative_collection = narrative_collection
        self.entity_collection = entity_collection

    def retrieve(self, expanded_query: ExpandedQuery,
                 top_k: int = 100) -> list[SemanticCandidate]:
        """Dual-channel semantic retrieval.

        Retrieves from both narrative and entity+schema collections,
        returns combined candidates (deduplication happens at fusion stage).
        """
        query_text = expanded_query.original_query
        candidates = []

        # Build filters from resolved entities
        filters = self._build_filters(expanded_query)

        # Build entity context for conditioned embedding
        entity_context = ""
        if expanded_query.entity_resolution:
            entity_labels = [
                r.canonical_label
                for r in expanded_query.entity_resolution.resolved
                if r.entry_type in ("ticker", "company_alias")
            ]
            if entity_labels:
                entity_context = " | ".join(entity_labels)

        # Embed query for both channels
        query_embeddings = self.embedding_manager.embed_query(
            query_text, entity_context=entity_context,
        )

        # Channel 1: Narrative retrieval
        if "narrative" in query_embeddings:
            narrative_results = self.vector_store.search(
                collection=self.narrative_collection,
                vector=query_embeddings["narrative"].vector,
                top_k=top_k,
                filters=filters,
            )
            for result in narrative_results:
                candidates.append(SemanticCandidate(
                    chunk_id=result.id,
                    score=result.score,
                    channel="narrative",
                    payload=result.payload,
                ))

        # Channel 2: Entity+schema conditioned retrieval
        if "entity_schema" in query_embeddings:
            entity_results = self.vector_store.search(
                collection=self.entity_collection,
                vector=query_embeddings["entity_schema"].vector,
                top_k=top_k,
                filters=filters,
            )
            for result in entity_results:
                candidates.append(SemanticCandidate(
                    chunk_id=result.id,
                    score=result.score,
                    channel="entity_schema",
                    payload=result.payload,
                ))

        return candidates

    def retrieve_simple(self, query: str, top_k: int = 100,
                        collection: Optional[str] = None,
                        filters: Optional[SearchFilters] = None,
                        ) -> list[SemanticCandidate]:
        """Single-channel retrieval for benchmarking."""
        coll = collection or self.narrative_collection
        embedding = self.embedding_manager.narrative_model.embed(query)
        results = self.vector_store.search(
            collection=coll,
            vector=embedding.vector,
            top_k=top_k,
            filters=filters,
        )
        return [
            SemanticCandidate(
                chunk_id=r.id, score=r.score,
                channel="narrative", payload=r.payload,
            )
            for r in results
        ]

    def _build_filters(self, expanded_query: ExpandedQuery) -> Optional[SearchFilters]:
        """Build vector search filters from query expansion results."""
        filters = SearchFilters()
        has_filter = False

        query_filters = expanded_query.filters
        if "company_id" in query_filters:
            company_ids = query_filters["company_id"]
            if isinstance(company_ids, list):
                if len(company_ids) == 1:
                    filters.company_id = company_ids[0]
                else:
                    filters.company_ids = company_ids
            else:
                filters.company_id = company_ids
            has_filter = True

        return filters if has_filter else None
