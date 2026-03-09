"""Retrieval result models with explainable scoring breakdown."""

from dataclasses import dataclass, field
from typing import Optional

from goat.models.provenance import ProvenanceRecord


@dataclass
class ScoringBreakdown:
    """Explainable scoring: why this result was ranked here.

    Shows contribution from each retrieval channel and any boosting factors.
    """
    # Channel contributions
    lexical_score: float = 0.0  # BM25/FTS match score
    lexical_match_terms: list[str] = field(default_factory=list)
    trigram_similarity: float = 0.0  # for alias/ticker fuzzy matching
    semantic_score: float = 0.0  # embedding cosine similarity (rank-based in fusion)
    graph_score: float = 0.0  # graph-based relevance contribution
    # Fusion
    fused_score: float = 0.0  # after RRF / weighted blend
    fusion_method: str = "rrf"
    # Reranking
    rerank_score: Optional[float] = None  # set if reranking was applied
    reranker_model: str = ""
    # Boosts / penalties
    recency_boost: float = 0.0
    entity_match_boost: float = 0.0  # exact entity resolution match
    # Final
    final_score: float = 0.0
    rank: int = 0

    @property
    def was_reranked(self) -> bool:
        return self.rerank_score is not None


@dataclass
class RetrievalHit:
    """A single retrieval result with full provenance and scoring breakdown."""
    chunk_id: str
    doc_id: str
    text: str
    # Context
    company_id: str = ""
    company_name: str = ""
    ticker: str = ""
    section: str = ""
    doc_type: str = ""
    language: str = "en"
    # Scoring
    scoring: ScoringBreakdown = field(default_factory=ScoringBreakdown)
    # Provenance
    provenance: ProvenanceRecord = field(default_factory=lambda: ProvenanceRecord(
        source_doc_id="", source_system="",
    ))

    @property
    def has_provenance(self) -> bool:
        return bool(self.provenance.source_doc_id)


@dataclass
class RetrievalResult:
    """Complete result set from a hybrid retrieval query."""
    query_text: str
    query_id: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    # Pipeline metadata
    tier_used: str = "A"  # A/B/C
    fusion_method: str = "rrf"
    reranking_applied: bool = False
    graph_expansion_applied: bool = False
    # Diagnostics
    lexical_candidates: int = 0
    semantic_candidates: int = 0
    fused_candidates: int = 0
    total_latency_ms: float = 0.0
    stage_latencies: dict = field(default_factory=dict)
    # Entities resolved
    resolved_entities: list[dict] = field(default_factory=list)
    resolved_schema_targets: list[dict] = field(default_factory=list)
    # Provenance summary
    provenance_coverage: float = 0.0  # fraction of hits with source provenance

    @property
    def top_hit(self) -> Optional[RetrievalHit]:
        return self.hits[0] if self.hits else None

    def compute_provenance_coverage(self) -> float:
        if not self.hits:
            return 0.0
        with_prov = sum(1 for h in self.hits if h.has_provenance)
        self.provenance_coverage = with_prov / len(self.hits)
        return self.provenance_coverage
