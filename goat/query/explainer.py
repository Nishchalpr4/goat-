"""Explainable scoring — produces human-readable explanations
of why each result was retrieved and how it was scored.

Implements the transparency requirement from the architecture:
every retrieved chunk must carry a ScoringBreakdown showing
lexical, semantic, graph, fusion, rerank, and recency components.
"""

from dataclasses import dataclass
import logging

from goat.models.retrieval import RetrievalHit, RetrievalResult, ScoringBreakdown

logger = logging.getLogger(__name__)


@dataclass
class ExplainedHit:
    """A retrieval hit with human-readable explanation."""
    rank: int
    chunk_id: str
    text_preview: str
    final_score: float
    explanation: str
    scoring: ScoringBreakdown


class QueryExplainer:
    """Generates human-readable explanations of retrieval scoring."""

    def explain_result(self, result: RetrievalResult,
                       top_k: int = 10) -> list[ExplainedHit]:
        """Explain the top-k results from a retrieval."""
        explained = []
        for hit in result.hits[:top_k]:
            explanation = self._explain_hit(hit)
            explained.append(ExplainedHit(
                rank=hit.scoring.rank,
                chunk_id=hit.chunk_id,
                text_preview=hit.text[:200] if hit.text else "",
                final_score=hit.scoring.final_score,
                explanation=explanation,
                scoring=hit.scoring,
            ))
        return explained

    def _explain_hit(self, hit: RetrievalHit) -> str:
        """Generate explanation for a single hit."""
        s = hit.scoring
        parts = []

        # Score components
        components = []
        if s.lexical_score > 0:
            components.append(f"BM25={s.lexical_score:.3f}")
        if s.semantic_score > 0:
            components.append(f"dense={s.semantic_score:.3f}")
        if s.fused_score > 0:
            components.append(f"fused={s.fused_score:.3f}")
        if s.rerank_score > 0:
            components.append(f"rerank={s.rerank_score:.3f}")
        if s.graph_score > 0:
            components.append(f"graph={s.graph_score:.3f}")

        if components:
            parts.append("Scores: " + ", ".join(components))

        # Boosts
        boosts = []
        if s.recency_boost > 0:
            boosts.append(f"recency=+{s.recency_boost:.3f}")
        if s.entity_match_boost > 0:
            boosts.append(f"entity=+{s.entity_match_boost:.3f}")

        if boosts:
            parts.append("Boosts: " + ", ".join(boosts))

        parts.append(f"Final={s.final_score:.4f} (rank #{s.rank})")

        # Source info
        if hit.source_doc_id:
            parts.append(f"Source: {hit.source_doc_id}")

        return " | ".join(parts)

    def format_diagnostics(self, result: RetrievalResult) -> str:
        """Format retrieval diagnostics as readable text."""
        lines = [
            f"Query: {result.query_text[:100]}",
            f"Tier: {result.tier}",
            f"Total hits: {len(result.hits)}",
        ]

        if result.latencies:
            latency_strs = [f"{k}={v:.0f}ms" for k, v in
                            result.latencies.items()]
            lines.append(f"Latencies: {', '.join(latency_strs)}")

        if result.diagnostics:
            for key, value in result.diagnostics.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)
