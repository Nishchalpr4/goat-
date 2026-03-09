"""Intent detection and retrieval tier routing.

Classifies user queries into intent categories and determines
the appropriate retrieval tier:
  - Tier A (fast): Lexical + dense + fusion → simple factual queries
  - Tier B (precision): + reranking → precision-critical queries
  - Tier C (reasoning): + graph expansion → relational / global queries
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re
import logging

from goat.query.parser import ParsedQuery

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """High-level query intent categories."""
    FIND_EVIDENCE = "find_evidence"      # "What did AAPL say about AI?"
    COMPARE = "compare"                  # "Compare AAPL vs MSFT margins"
    EXPLAIN = "explain"                  # "Why did revenue decline?"
    COMPUTE = "compute"                  # "What is the 3-year CAGR?"
    SUMMARIZE = "summarize"              # "Summarize AAPL's latest 10-K"
    TREND = "trend"                      # "How has margin changed?"
    SCREEN = "screen"                    # "Which companies have >30% margins?"
    UNKNOWN = "unknown"


class RetrievalTier(Enum):
    """Retrieval complexity tiers."""
    TIER_A = "A"  # fast: lexical + dense + fusion
    TIER_B = "B"  # precision: + reranking
    TIER_C = "C"  # reasoning: + graph expansion


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: QueryIntent
    tier: RetrievalTier
    confidence: float
    reasoning: str
    # Whether multi-company comparison mode should be used
    is_multi_company: bool = False
    # Whether graph expansion is needed
    needs_graph: bool = False
    # Suggested sub-intents (for compound queries)
    sub_intents: list[QueryIntent] = None

    def __post_init__(self):
        if self.sub_intents is None:
            self.sub_intents = []


# Keyword patterns for intent classification
_INTENT_SIGNALS = {
    QueryIntent.COMPARE: [
        r'\bcompare\b', r'\bvs\.?\b', r'\bversus\b', r'\bagainst\b',
        r'\bbetter\b', r'\bworse\b', r'\boutperform\b', r'\bunderperform\b',
        r'\bbenchmark\b', r'\brelative\s+to\b',
    ],
    QueryIntent.EXPLAIN: [
        r'\bwhy\b', r'\bexplain\b', r'\breason\b', r'\bcause\b',
        r'\bdriver\b', r'\bdue\s+to\b', r'\bbecause\b', r'\battribut\b',
        r'\bcontribut\b', r'\bimpact\b',
    ],
    QueryIntent.COMPUTE: [
        r'\bcalculate\b', r'\bcompute\b', r'\bwhat\s+is\s+the\b.*\b(?:ratio|rate|cagr|growth|margin)\b',
        r'\baverage\b', r'\bmedian\b', r'\bsum\b', r'\btotal\b',
    ],
    QueryIntent.SUMMARIZE: [
        r'\bsummar\w+\b', r'\boverview\b', r'\bhighlight\b',
        r'\bkey\s+(?:point|takeaway|finding)\b', r'\btldr\b',
    ],
    QueryIntent.TREND: [
        r'\btrend\b', r'\bover\s+time\b', r'\bhistorical\b',
        r'\btrajectory\b', r'\bchange\b.*\b(?:over|since|from)\b',
        r'\bimproving\b', r'\bdeteriorating\b', r'\bYoY\b', r'\bQoQ\b',
    ],
    QueryIntent.SCREEN: [
        r'\bwhich\s+compan\w+\b', r'\bscreen\b', r'\bfilter\b',
        r'\bfind\s+(?:all|companies|stocks)\b', r'\blist\b.*\bcompan\w+\b',
        r'\b(?:greater|more|less|above|below)\s+than\b',
    ],
}

# Tier mapping by intent
_INTENT_TIER_MAP = {
    QueryIntent.FIND_EVIDENCE: RetrievalTier.TIER_A,
    QueryIntent.COMPARE: RetrievalTier.TIER_B,
    QueryIntent.EXPLAIN: RetrievalTier.TIER_C,
    QueryIntent.COMPUTE: RetrievalTier.TIER_A,
    QueryIntent.SUMMARIZE: RetrievalTier.TIER_B,
    QueryIntent.TREND: RetrievalTier.TIER_B,
    QueryIntent.SCREEN: RetrievalTier.TIER_B,
    QueryIntent.UNKNOWN: RetrievalTier.TIER_A,
}


class IntentDetector:
    """Classifies query intent and determines retrieval tier."""

    def __init__(self):
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in _INTENT_SIGNALS.items()
        }

    def detect(self, parsed: ParsedQuery) -> IntentResult:
        """Detect intent from a parsed query."""
        text = parsed.normalized_query
        scores: dict[QueryIntent, float] = {}

        # Score each intent based on pattern matches
        for intent, patterns in self._compiled.items():
            match_count = sum(1 for p in patterns if p.search(text))
            if match_count > 0:
                scores[intent] = match_count / len(patterns)

        # Determine primary intent
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] * 2, 1.0)  # scale up
        else:
            best_intent = QueryIntent.FIND_EVIDENCE
            confidence = 0.5

        # Determine tier
        tier = _INTENT_TIER_MAP.get(best_intent, RetrievalTier.TIER_A)

        # Multi-company detection
        is_multi = parsed.entity_count > 1 or parsed.comparison_type == "compare"

        # Upgrade tier if multi-company comparison
        if is_multi and tier == RetrievalTier.TIER_A:
            tier = RetrievalTier.TIER_B

        # Graph expansion needed for EXPLAIN, TREND with relationships
        needs_graph = best_intent in (
            QueryIntent.EXPLAIN, QueryIntent.TREND,
        ) or tier == RetrievalTier.TIER_C

        reasoning = self._build_reasoning(best_intent, scores, tier,
                                           is_multi, needs_graph)

        return IntentResult(
            intent=best_intent,
            tier=tier,
            confidence=confidence,
            reasoning=reasoning,
            is_multi_company=is_multi,
            needs_graph=needs_graph,
        )

    def _build_reasoning(self, intent: QueryIntent,
                          scores: dict[QueryIntent, float],
                          tier: RetrievalTier,
                          is_multi: bool,
                          needs_graph: bool) -> str:
        """Build an explanation for the intent decision."""
        parts = [f"Intent={intent.value} (tier {tier.value})"]
        if scores:
            top_scores = sorted(scores.items(), key=lambda x: -x[1])[:3]
            signal_str = ", ".join(
                f"{i.value}={s:.2f}" for i, s in top_scores
            )
            parts.append(f"signals: [{signal_str}]")
        if is_multi:
            parts.append("multi-company")
        if needs_graph:
            parts.append("graph-expansion")
        return "; ".join(parts)
