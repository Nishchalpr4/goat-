"""Main query processing pipeline — orchestrates the full
parse → intent → resolve → expand → plan → retrieve → explain flow.

This is the primary entry point for answering user queries.
It coordinates all subsystems:
  1. Parse the raw query (parser)
  2. Detect intent and select tier (intent)
  3. Resolve entities (entity_resolver)
  4. Expand query (query_expander)
  5. Generate execution plan (planner)
  6. Execute retrieval (hybrid retriever)
  7. Optionally expand via graph (graph traverser)
  8. Produce explainable results (explainer)
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid
import logging

from goat.config import Config, default_config
from goat.query.parser import QueryParser, ParsedQuery
from goat.query.intent import IntentDetector, IntentResult, RetrievalTier, QueryIntent
from goat.query.planner import QueryPlanner, QueryPlan
from goat.query.explainer import QueryExplainer, ExplainedHit
from goat.lexical.entity_resolver import EntityResolver, ResolutionResult
from goat.lexical.query_expander import QueryExpander
from goat.models.retrieval import RetrievalResult
from goat.models.lexicon import Lexicon

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    """Complete response to a user query."""
    query_id: str
    raw_query: str
    parsed: ParsedQuery
    intent: IntentResult
    resolution: ResolutionResult
    plan: QueryPlan
    result: Optional[RetrievalResult] = None
    explained: list[ExplainedHit] = field(default_factory=list)
    graph_context: Optional[dict] = None
    total_time_ms: float = 0.0
    error: Optional[str] = None


class QueryPipeline:
    """End-to-end query processing pipeline.

    Usage:
        pipeline = QueryPipeline(config=config)
        pipeline.set_lexicon(lexicon)
        pipeline.set_retriever(hybrid_retriever)
        pipeline.set_graph_traverser(graph_traverser)

        response = pipeline.process("What drives AAPL's margin expansion?")
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        # Core components
        self.parser = QueryParser()
        self.intent_detector = IntentDetector()
        self.planner = QueryPlanner(self.config)
        self.explainer = QueryExplainer()
        # These are set via dependency injection
        self._entity_resolver: Optional[EntityResolver] = None
        self._query_expander: Optional[QueryExpander] = None
        self._retriever = None  # HybridRetriever
        self._graph_traverser = None  # GraphTraverser

    def set_lexicon(self, lexicon: Lexicon) -> None:
        """Initialize entity resolver and query expander with a lexicon."""
        self._entity_resolver = EntityResolver(lexicon)
        self._query_expander = QueryExpander(lexicon)

    def set_retriever(self, retriever) -> None:
        """Set the hybrid retriever for executing retrieval plans."""
        self._retriever = retriever

    def set_graph_traverser(self, traverser) -> None:
        """Set the graph traverser for Tier C expansion."""
        self._graph_traverser = traverser

    def process(self, query: str) -> QueryResponse:
        """Process a user query end-to-end."""
        start = time.perf_counter()
        query_id = uuid.uuid4().hex[:12]

        response = QueryResponse(
            query_id=query_id,
            raw_query=query,
            parsed=ParsedQuery(raw_query=query, normalized_query=query),
            intent=IntentResult(
                intent=QueryIntent.UNKNOWN,
                tier=RetrievalTier.TIER_A,
                confidence=0.0,
                reasoning="not processed",
            ),
            resolution=ResolutionResult(),
            plan=QueryPlan(),
        )

        try:
            # Stage 1: Parse
            response.parsed = self.parser.parse(query)
            logger.info("Parsed query: %d tokens, %d periods, %d entities",
                        len(response.parsed.tokens),
                        len(response.parsed.periods),
                        response.parsed.entity_count)

            # Stage 2: Detect intent
            response.intent = self.intent_detector.detect(response.parsed)
            logger.info("Intent: %s (tier %s, confidence %.2f)",
                        response.intent.intent.value,
                        response.intent.tier.value,
                        response.intent.confidence)

            # Stage 3: Resolve entities
            if self._entity_resolver:
                response.resolution = self._entity_resolver.resolve(
                    response.parsed.normalized_query
                )

            # Stage 4: Expand query
            expanded_query = response.parsed.normalized_query
            if self._query_expander:
                expansion = self._query_expander.expand(query)
                expanded_query = expansion.expanded_query

            # Stage 5: Generate plan
            response.plan = self.planner.plan(
                response.parsed, response.intent,
                response.resolution, expanded_query,
            )
            response.plan.query_id = query_id

            # Stage 6: Execute retrieval
            if self._retriever:
                response.result = self._execute_retrieval(
                    response.plan, response.intent,
                )

            # Stage 7: Graph expansion (Tier C)
            if (response.plan.graph_stage.enabled
                    and self._graph_traverser
                    and response.resolution.company_ids):
                response.graph_context = self._execute_graph_expansion(
                    response.resolution.company_ids,
                    response.plan.graph_stage.params,
                )

            # Stage 8: Explain results
            if response.result:
                response.explained = self.explainer.explain_result(
                    response.result
                )

        except Exception as e:
            logger.exception("Query pipeline error: %s", e)
            response.error = str(e)

        response.total_time_ms = (time.perf_counter() - start) * 1000
        return response

    def _execute_retrieval(self, plan: QueryPlan,
                            intent: IntentResult) -> RetrievalResult:
        """Execute the retrieval plan via the hybrid retriever."""
        metadata_filter = {}
        if plan.company_ids:
            metadata_filter["company_id"] = plan.company_ids
        if plan.periods:
            metadata_filter["period"] = plan.periods
        if plan.doc_types:
            metadata_filter["doc_type"] = plan.doc_types

        return self._retriever.retrieve(
            query=plan.retrieval_text,
            top_k=plan.fusion_stage.top_k,
            tier=intent.tier.value,
            metadata_filter=metadata_filter if metadata_filter else None,
        )

    def _execute_graph_expansion(self, company_ids: list[str],
                                  params: dict) -> dict:
        """Execute graph-guided expansion for Tier C queries."""
        result = self._graph_traverser.expand_for_query(
            entity_ids=company_ids,
            depth=params.get("depth", 2),
            include_zones=params.get("include_zones"),
        )
        return {
            "related_entities": result.related_entities,
            "related_facts": result.related_facts,
            "related_claims": result.related_claims,
            "expansion_texts": result.expansion_texts,
        }
