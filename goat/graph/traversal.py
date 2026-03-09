"""Graph-guided retrieval and expansion — traverses the knowledge graph
to pull related evidence, entity context, and provenance for
relational/global queries (Tier C).

Handles:
  - Entity-centered expansion (get related companies, facts, claims)
  - Relationship path finding (company → acquired → target)
  - Provenance-aware aggregation (trace claims to source documents)
  - Community-based evidence gathering
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.storage.graph_store import GraphStore, GraphTraversalResult, GraphNode

logger = logging.getLogger(__name__)


@dataclass
class GraphExpansionResult:
    """Result of graph-guided query expansion."""
    # Related entities found
    related_entities: list[dict] = field(default_factory=list)
    # Facts connected to query entities
    related_facts: list[dict] = field(default_factory=list)
    # Claims connected to query entities
    related_claims: list[dict] = field(default_factory=list)
    # Provenance chains discovered
    provenance_chains: list[dict] = field(default_factory=list)
    # Additional text for retrieval augmentation
    expansion_texts: list[str] = field(default_factory=list)
    # Graph paths traversed
    paths: list[list[str]] = field(default_factory=list)


class GraphTraverser:
    """Graph-guided retrieval for augmenting hybrid search results."""

    def __init__(self, graph_store: GraphStore):
        self.graph = graph_store

    def expand_for_query(self, entity_ids: list[str],
                         depth: int = 2,
                         include_zones: Optional[list[str]] = None,
                         ) -> GraphExpansionResult:
        """Expand query context by traversing from resolved entities.

        Used in Tier C (reasoning path) to augment retrieval with
        relational and provenance context.
        """
        result = GraphExpansionResult()
        zones = include_zones or ["entity", "data", "knowledge", "provenance"]

        for entity_id in entity_ids:
            traversal = self.graph.traverse(
                start_id=entity_id, max_depth=depth, zones=zones,
            )
            self._process_traversal(traversal, result)

        return result

    def _process_traversal(self, traversal: GraphTraversalResult,
                           result: GraphExpansionResult) -> None:
        """Process traversal results into structured expansion data."""
        for node in traversal.nodes:
            zone = node.zone
            if zone == "entity":
                result.related_entities.append({
                    "id": node.node_id,
                    "type": node.node_type,
                    "label": node.label,
                    **node.properties,
                })
            elif zone == "data":
                result.related_facts.append({
                    "id": node.node_id,
                    "label": node.label,
                    **node.properties,
                })
                # Generate textualized fact for retrieval augmentation
                metric = node.properties.get("metric", "")
                value = node.properties.get("value", "")
                period = node.properties.get("period", "")
                if metric and value:
                    result.expansion_texts.append(
                        f"Metric={metric}; Period={period}; Value={value}"
                    )
            elif zone == "knowledge":
                result.related_claims.append({
                    "id": node.node_id,
                    "type": node.node_type,
                    "text": node.label,
                    **node.properties,
                })
                result.expansion_texts.append(node.label)
            elif zone == "provenance":
                result.provenance_chains.append({
                    "id": node.node_id,
                    "type": node.node_type,
                    "label": node.label,
                    **node.properties,
                })

        result.paths.extend(traversal.paths)

    def find_related_companies(self, company_id: str,
                                edge_types: Optional[list[str]] = None,
                                ) -> list[dict]:
        """Find companies related to a given company via entity-zone edges."""
        if not edge_types:
            edge_types = ["competes_with", "has_subsidiary", "acquired",
                          "partnered_with", "invested_in"]

        edges = self.graph.get_edges(company_id, direction="both")
        related = []
        seen = set()

        for edge in edges:
            if edge.edge_type not in edge_types:
                continue
            other_id = edge.target_id if edge.source_id == company_id else edge.source_id
            if other_id in seen:
                continue
            seen.add(other_id)
            node = self.graph.get_node(other_id)
            if node and node.node_type == "company":
                related.append({
                    "id": node.node_id,
                    "name": node.label,
                    "relationship": edge.edge_type,
                    **node.properties,
                })

        return related

    def get_fact_timeline(self, company_id: str,
                          metric: str) -> list[dict]:
        """Get a time-ordered series of facts for a metric."""
        edges = self.graph.get_edges(company_id, direction="out", edge_type="has_fact")
        facts = []
        for edge in edges:
            node = self.graph.get_node(edge.target_id)
            if node and node.properties.get("metric") == metric:
                facts.append({
                    "period": node.properties.get("period", ""),
                    "value": node.properties.get("value"),
                    "unit": node.properties.get("unit", ""),
                })
        # Sort by period
        facts.sort(key=lambda f: f.get("period", ""))
        return facts

    def get_provenance_chain(self, entity_id: str) -> list[dict]:
        """Trace the provenance chain for a specific entity/fact/claim."""
        traversal = self.graph.traverse(
            start_id=entity_id, max_depth=3,
            zones=["provenance"],
        )
        chain = []
        for node in traversal.nodes:
            chain.append({
                "id": node.node_id,
                "type": node.node_type,
                "label": node.label,
                **node.properties,
            })
        return chain
