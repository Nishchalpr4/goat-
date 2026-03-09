"""Hybrid retrieval orchestrator — coordinates lexical + semantic retrieval,
fusion, and selective reranking across the three tiers.

Tier A (fast path): lexical + dense retrieval + fusion (no rerank)
Tier B (precision path): + reranker for ambiguous/high-stakes queries
Tier C (reasoning path): + graph-guided expansion and provenance synthesis
"""

import time
import uuid
from typing import Optional
import logging

from goat.config import RetrievalConfig
from goat.models.retrieval import RetrievalResult, RetrievalHit
from goat.retrieval.lexical import LexicalRetriever
from goat.retrieval.semantic import SemanticRetriever
from goat.retrieval.fusion import (
    reciprocal_rank_fusion, weighted_score_blend, FusedCandidate,
)
from goat.retrieval.reranker import BaseReranker, NoOpReranker
from goat.lexical.query_expander import ExpandedQuery

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates the full hybrid retrieval pipeline.

    Pipeline:
      1. Lexical retrieval (BM25/FTS + trigram) → ranked list
      2. Semantic retrieval (dual-channel ANN) → ranked lists
      3. Fusion (RRF or weighted blend)
      4. Selective reranking (Tier B/C only)
      5. Score finalization + provenance attachment
    """

    def __init__(
        self,
        lexical_retriever: LexicalRetriever,
        semantic_retriever: SemanticRetriever,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.lexical = lexical_retriever
        self.semantic = semantic_retriever
        self.reranker = reranker or NoOpReranker()
        self.config = config or RetrievalConfig()

    def retrieve(self, expanded_query: ExpandedQuery,
                 tier: Optional[str] = None,
                 top_k: int = 20) -> RetrievalResult:
        """Execute the full hybrid retrieval pipeline."""
        t0 = time.perf_counter()
        stage_latencies = {}
        tier = tier or self.config.default_tier
        tier_config = self.config.tier_thresholds.get(tier, {})

        query_id = str(uuid.uuid4())[:12]
        result = RetrievalResult(
            query_text=expanded_query.original_query,
            query_id=query_id,
            tier_used=tier,
            fusion_method=self.config.fusion_method,
        )

        # Record resolved entities
        if expanded_query.entity_resolution:
            result.resolved_entities = [
                {"surface": r.surface_form, "canonical_id": r.canonical_id,
                 "label": r.canonical_label, "method": r.resolution_method}
                for r in expanded_query.entity_resolution.resolved
            ]
        if expanded_query.schema_linking:
            result.resolved_schema_targets = [
                {"term": l.user_term, "concept": l.canonical_concept,
                 "taxonomy": l.taxonomy}
                for l in expanded_query.schema_linking.links
            ]

        # Stage 1: Lexical retrieval
        t1 = time.perf_counter()
        lexical_candidates = self.lexical.retrieve(
            expanded_query, top_k=self.config.lexical_top_k,
        )
        stage_latencies["lexical"] = (time.perf_counter() - t1) * 1000
        result.lexical_candidates = len(lexical_candidates)

        # Stage 2: Semantic retrieval
        t2 = time.perf_counter()
        semantic_candidates = self.semantic.retrieve(
            expanded_query, top_k=self.config.semantic_top_k,
        )
        stage_latencies["semantic"] = (time.perf_counter() - t2) * 1000
        result.semantic_candidates = len(semantic_candidates)

        # Stage 3: Fusion
        t3 = time.perf_counter()
        ranked_lists = self._prepare_ranked_lists(
            lexical_candidates, semantic_candidates,
        )
        if self.config.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(ranked_lists, k=self.config.rrf_k)
        else:
            weights = {
                "lexical": self.config.lexical_weight,
                "narrative": self.config.semantic_weight * 0.6,
                "entity_schema": self.config.semantic_weight * 0.4,
            }
            fused = weighted_score_blend(ranked_lists, weights)
        stage_latencies["fusion"] = (time.perf_counter() - t3) * 1000
        result.fused_candidates = len(fused)

        # Stage 4: Selective reranking
        if tier_config.get("rerank", False) and not isinstance(self.reranker, NoOpReranker):
            t4 = time.perf_counter()
            fused = self.reranker.rerank(
                expanded_query.original_query, fused,
                top_n=self.config.rerank_top_n,
            )
            stage_latencies["rerank"] = (time.perf_counter() - t4) * 1000
            result.reranking_applied = True
        else:
            # Assign final scores from fusion
            for i, c in enumerate(fused):
                c.hit.scoring.final_score = c.fused_score
                c.hit.scoring.rank = i + 1

        # Stage 5: Finalize
        result.hits = [c.hit for c in fused[:top_k]]
        result.stage_latencies = stage_latencies
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        result.compute_provenance_coverage()

        logger.info(
            "Hybrid retrieval [tier=%s]: %d hits in %.1fms "
            "(lexical=%d, semantic=%d, fused=%d, reranked=%s)",
            tier, len(result.hits), result.total_latency_ms,
            result.lexical_candidates, result.semantic_candidates,
            result.fused_candidates, result.reranking_applied,
        )

        return result

    def _prepare_ranked_lists(self, lexical_candidates, semantic_candidates) -> dict:
        """Convert retrieval candidates into ranked-list dicts for fusion."""
        lists = {}

        # Lexical list
        lists["lexical"] = [
            {
                "chunk_id": c.chunk_id,
                "score": c.score,
                "doc_id": c.doc_id,
                "text": c.text,
                "section": c.section,
                "company_id": c.company_id,
                "doc_type": c.doc_type,
            }
            for c in lexical_candidates
        ]

        # Semantic lists (separate by channel)
        narrative_list = []
        entity_list = []
        for c in semantic_candidates:
            item = {
                "chunk_id": c.chunk_id,
                "score": c.score,
                **c.payload,
            }
            if c.channel == "narrative":
                narrative_list.append(item)
            else:
                entity_list.append(item)

        if narrative_list:
            lists["narrative"] = sorted(narrative_list, key=lambda x: -x["score"])
        if entity_list:
            lists["entity_schema"] = sorted(entity_list, key=lambda x: -x["score"])

        return lists
