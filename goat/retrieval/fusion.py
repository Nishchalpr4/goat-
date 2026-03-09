"""Result fusion — combines ranked lists from multiple retrieval channels.

Implements:
  - Reciprocal Rank Fusion (RRF) — simple, well-cited, used in Elastic and other systems
  - Weighted score blending — linear combination of normalized scores
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.models.retrieval import RetrievalHit, ScoringBreakdown

logger = logging.getLogger(__name__)


@dataclass
class FusedCandidate:
    """A candidate after fusion with scoring breakdown."""
    chunk_id: str
    fused_score: float
    hit: RetrievalHit
    source_ranks: dict = field(default_factory=dict)   # channel → rank
    source_scores: dict = field(default_factory=dict)  # channel → raw score


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list],  # channel_name → sorted list of (id, score, data)
    k: int = 60,
) -> list[FusedCandidate]:
    """Reciprocal Rank Fusion (RRF) over multiple ranked lists.

    RRF score for document d = Σ 1 / (k + rank_i(d))
    where rank_i(d) is the rank of d in ranked list i.

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods" (SIGIR 2009).
    """
    # Accumulate RRF scores by chunk_id
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}  # chunk_id → {channel: rank}
    raw_scores: dict[str, dict[str, float]] = {}
    data_store: dict[str, dict] = {}  # chunk_id → best data

    for channel, items in ranked_lists.items():
        for rank, item in enumerate(items, start=1):
            chunk_id = item["chunk_id"]
            score = item.get("score", 0.0)

            rrf_contribution = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contribution

            if chunk_id not in ranks:
                ranks[chunk_id] = {}
                raw_scores[chunk_id] = {}
            ranks[chunk_id][channel] = rank
            raw_scores[chunk_id][channel] = score

            # Keep the fullest data
            if chunk_id not in data_store or len(item) > len(data_store[chunk_id]):
                data_store[chunk_id] = item

    # Build fused candidates
    candidates = []
    for chunk_id, fused_score in sorted(scores.items(), key=lambda x: -x[1]):
        data = data_store[chunk_id]
        scoring = ScoringBreakdown(
            lexical_score=raw_scores[chunk_id].get("lexical", 0.0),
            semantic_score=raw_scores[chunk_id].get("narrative", 0.0)
                          + raw_scores[chunk_id].get("entity_schema", 0.0),
            fused_score=fused_score,
            fusion_method="rrf",
        )
        hit = RetrievalHit(
            chunk_id=chunk_id,
            doc_id=data.get("doc_id", ""),
            text=data.get("text", ""),
            section=data.get("section", ""),
            company_id=data.get("company_id", ""),
            company_name=data.get("company_name", ""),
            ticker=data.get("ticker", ""),
            doc_type=data.get("doc_type", ""),
            language=data.get("language", "en"),
            scoring=scoring,
        )
        candidates.append(FusedCandidate(
            chunk_id=chunk_id,
            fused_score=fused_score,
            hit=hit,
            source_ranks=ranks[chunk_id],
            source_scores=raw_scores[chunk_id],
        ))

    return candidates


def weighted_score_blend(
    ranked_lists: dict[str, list],
    weights: dict[str, float],
) -> list[FusedCandidate]:
    """Weighted linear combination of normalized scores.

    Normalizes each list's scores to [0, 1] via min-max, then blends.
    """
    # Normalize scores within each channel
    normalized: dict[str, dict[str, float]] = {}
    for channel, items in ranked_lists.items():
        if not items:
            continue
        scores = [item.get("score", 0.0) for item in items]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        for item in items:
            chunk_id = item["chunk_id"]
            norm_score = (item.get("score", 0.0) - min_s) / range_s
            if chunk_id not in normalized:
                normalized[chunk_id] = {}
            normalized[chunk_id][channel] = norm_score

    # Blend
    blended: dict[str, float] = {}
    data_store: dict[str, dict] = {}
    for channel, items in ranked_lists.items():
        for item in items:
            chunk_id = item["chunk_id"]
            if chunk_id not in data_store or len(item) > len(data_store[chunk_id]):
                data_store[chunk_id] = item

    for chunk_id, channel_scores in normalized.items():
        score = sum(
            channel_scores.get(ch, 0.0) * weights.get(ch, 0.0)
            for ch in set(list(channel_scores.keys()) + list(weights.keys()))
        )
        blended[chunk_id] = score

    # Build candidates
    candidates = []
    for chunk_id, fused_score in sorted(blended.items(), key=lambda x: -x[1]):
        data = data_store.get(chunk_id, {})
        scoring = ScoringBreakdown(
            fused_score=fused_score,
            fusion_method="weighted_blend",
        )
        hit = RetrievalHit(
            chunk_id=chunk_id,
            doc_id=data.get("doc_id", ""),
            text=data.get("text", ""),
            section=data.get("section", ""),
            company_id=data.get("company_id", ""),
            scoring=scoring,
        )
        candidates.append(FusedCandidate(
            chunk_id=chunk_id, fused_score=fused_score,
            hit=hit, source_scores=normalized.get(chunk_id, {}),
        ))

    return candidates
