"""Query plan generation — translates intent + parsed query
into an executable plan that the pipeline orchestrator follows.

A QueryPlan describes:
  - Which retrieval stages to run (lexical, semantic, fusion, rerank, graph)
  - Filter parameters (company IDs, periods, document types)
  - Fusion strategy and weights
  - Whether to invoke graph expansion
  - Top-k settings per stage
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.config import Config, default_config
from goat.query.parser import ParsedQuery
from goat.query.intent import IntentResult, RetrievalTier, QueryIntent
from goat.lexical.entity_resolver import ResolutionResult

logger = logging.getLogger(__name__)


@dataclass
class PlanStage:
    """A single stage in the query plan."""
    name: str
    enabled: bool = True
    top_k: int = 100
    params: dict = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Executable plan for a single query."""
    # Identification
    query_id: str = ""
    # The text to actually retrieve against (post-expansion)
    retrieval_text: str = ""
    # Filters
    company_ids: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    # Plan stages
    lexical_stage: PlanStage = field(
        default_factory=lambda: PlanStage(name="lexical"))
    semantic_stage: PlanStage = field(
        default_factory=lambda: PlanStage(name="semantic"))
    fusion_stage: PlanStage = field(
        default_factory=lambda: PlanStage(name="fusion"))
    rerank_stage: PlanStage = field(
        default_factory=lambda: PlanStage(name="rerank", enabled=False))
    graph_stage: PlanStage = field(
        default_factory=lambda: PlanStage(name="graph", enabled=False))
    # Metadata
    tier: str = "A"
    intent: str = "find_evidence"
    is_multi_company: bool = False


class QueryPlanner:
    """Generates executable query plans from parsed queries and intent."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.retrieval_cfg = self.config.retrieval

    def plan(self, parsed: ParsedQuery, intent: IntentResult,
             resolution: Optional[ResolutionResult] = None,
             expanded_query: Optional[str] = None) -> QueryPlan:
        """Generate a query plan."""
        qp = QueryPlan(
            retrieval_text=expanded_query or parsed.normalized_query,
            tier=intent.tier.value,
            intent=intent.intent.value,
            is_multi_company=intent.is_multi_company,
        )

        # Apply filters from entity resolution
        if resolution:
            qp.company_ids = list(resolution.company_ids)

        # Apply period filters
        qp.periods = [p.normalized for p in parsed.periods if p.normalized]

        # Configure stages based on tier
        self._configure_stages(qp, intent)

        return qp

    def _configure_stages(self, qp: QueryPlan,
                           intent: IntentResult) -> None:
        """Configure plan stages based on retrieval tier."""
        cfg = self.retrieval_cfg

        # Tier A: lexical + semantic + fusion
        qp.lexical_stage.enabled = True
        qp.lexical_stage.top_k = cfg.lexical_top_k
        qp.semantic_stage.enabled = True
        qp.semantic_stage.top_k = cfg.semantic_top_k

        fusion_top_k = max(cfg.lexical_top_k, cfg.semantic_top_k)
        qp.fusion_stage.enabled = True
        qp.fusion_stage.top_k = fusion_top_k
        qp.fusion_stage.params = {"rrf_k": cfg.rrf_k}

        # Tier B: + rerank
        if intent.tier in (RetrievalTier.TIER_B, RetrievalTier.TIER_C):
            qp.rerank_stage.enabled = True
            qp.rerank_stage.top_k = min(cfg.rerank_top_n, fusion_top_k)

        # Tier C: + graph expansion
        if intent.tier == RetrievalTier.TIER_C or intent.needs_graph:
            qp.graph_stage.enabled = True
            qp.graph_stage.top_k = 50
            qp.graph_stage.params = {
                "depth": 2,
                "include_zones": ["entity", "data", "knowledge"],
            }

        # Adjust for specific intents
        if intent.intent == QueryIntent.COMPUTE:
            # Computation queries need more structured data
            qp.semantic_stage.top_k = 50  # Less text needed
            qp.graph_stage.enabled = True
            qp.graph_stage.params["include_zones"] = ["data"]

        if intent.intent == QueryIntent.SCREEN:
            # Screen queries need broad coverage
            qp.lexical_stage.top_k = 200
            qp.semantic_stage.top_k = 200
            qp.fusion_stage.top_k = 200
