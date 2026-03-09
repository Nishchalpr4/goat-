"""Graph RAG store — multi-zone graph for entity identity, relationships,
metric definitions, lexical variants, and provenance edges.

Zone semantics:
  - Entity zone: companies, subsidiaries, executives, products, regulators, tickers
  - Data zone: normalized numeric facts, financial statements, ratios, time series pointers
  - Knowledge zone: extracted relations/claims (acquisitions, guidance revisions, events)
  - Lexical zone: aliases, abbreviations, translations, domain synonyms, schema mappings
  - Provenance zone: document source, span offsets, extraction run metadata (PROV triples)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging

from goat.config import GraphStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the Graph RAG store."""
    node_id: str
    zone: str  # "entity", "data", "knowledge", "lexical", "provenance"
    node_type: str  # "company", "executive", "fact", "claim", "alias", "document"
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the Graph RAG store."""
    source_id: str
    target_id: str
    edge_type: str  # "has_subsidiary", "acquired", "alias_of", "derived_from", etc.
    properties: dict = field(default_factory=dict)
    zone: str = ""  # zone this edge belongs to


@dataclass
class GraphTraversalResult:
    """Result of a graph traversal query."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)  # sequences of node IDs


class GraphStore:
    """Multi-zone graph store using NetworkX as the default backend.

    Designed for single-process / moderate-scale usage.
    For production scale, swap to Neo4j or Memgraph.
    """

    def __init__(self, config: Optional[GraphStoreConfig] = None):
        self.config = config or GraphStoreConfig()
        self._graph = None

    def connect(self):
        """Initialize the graph backend."""
        if self.config.backend == "networkx":
            import networkx as nx
            self._graph = nx.MultiDiGraph()
            logger.info("Graph store initialized (NetworkX in-memory)")
        else:
            logger.warning("Backend '%s' not yet implemented, using NetworkX",
                           self.config.backend)
            import networkx as nx
            self._graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        if self._graph is None:
            self.connect()
        self._graph.add_node(
            node.node_id,
            zone=node.zone,
            node_type=node.node_type,
            label=node.label,
            **node.properties,
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if self._graph is None or node_id not in self._graph:
            return None
        data = self._graph.nodes[node_id]
        return GraphNode(
            node_id=node_id,
            zone=data.get("zone", ""),
            node_type=data.get("node_type", ""),
            label=data.get("label", ""),
            properties={k: v for k, v in data.items()
                        if k not in ("zone", "node_type", "label")},
        )

    def find_nodes(self, zone: Optional[str] = None,
                   node_type: Optional[str] = None,
                   **properties) -> list[GraphNode]:
        """Find nodes matching criteria."""
        if self._graph is None:
            return []
        results = []
        for nid, data in self._graph.nodes(data=True):
            if zone and data.get("zone") != zone:
                continue
            if node_type and data.get("node_type") != node_type:
                continue
            match = all(data.get(k) == v for k, v in properties.items())
            if match:
                results.append(GraphNode(
                    node_id=nid,
                    zone=data.get("zone", ""),
                    node_type=data.get("node_type", ""),
                    label=data.get("label", ""),
                    properties={k: v for k, v in data.items()
                                if k not in ("zone", "node_type", "label")},
                ))
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if self._graph is None:
            self.connect()
        self._graph.add_edge(
            edge.source_id, edge.target_id,
            edge_type=edge.edge_type,
            zone=edge.zone,
            **edge.properties,
        )

    def get_edges(self, node_id: str, direction: str = "both",
                  edge_type: Optional[str] = None) -> list[GraphEdge]:
        """Get edges for a node."""
        if self._graph is None:
            return []
        edges = []
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if edge_type and data.get("edge_type") != edge_type:
                    continue
                edges.append(GraphEdge(
                    source_id=node_id, target_id=target,
                    edge_type=data.get("edge_type", ""),
                    properties={k: v for k, v in data.items()
                                if k not in ("edge_type", "zone")},
                    zone=data.get("zone", ""),
                ))
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if edge_type and data.get("edge_type") != edge_type:
                    continue
                edges.append(GraphEdge(
                    source_id=source, target_id=node_id,
                    edge_type=data.get("edge_type", ""),
                    properties={k: v for k, v in data.items()
                                if k not in ("edge_type", "zone")},
                    zone=data.get("zone", ""),
                ))
        return edges

    # ------------------------------------------------------------------
    # Zone-specific operations
    # ------------------------------------------------------------------

    def add_entity(self, entity_id: str, entity_type: str,
                   label: str, **properties) -> GraphNode:
        """Add a node to the entity zone."""
        node = GraphNode(
            node_id=entity_id, zone="entity",
            node_type=entity_type, label=label,
            properties=properties,
        )
        self.add_node(node)
        return node

    def add_fact(self, fact_id: str, company_id: str,
                 metric: str, value: Any, period: str,
                 **properties) -> GraphNode:
        """Add a normalized fact to the data zone."""
        node = GraphNode(
            node_id=fact_id, zone="data",
            node_type="fact", label=f"{metric}={value}",
            properties={"metric": metric, "value": value,
                         "period": period, **properties},
        )
        self.add_node(node)
        # Link fact to company
        self.add_edge(GraphEdge(
            source_id=company_id, target_id=fact_id,
            edge_type="has_fact", zone="data",
        ))
        return node

    def add_claim(self, claim_id: str, claim_text: str,
                  source_entities: list[str],
                  **properties) -> GraphNode:
        """Add an extracted claim/relation to the knowledge zone."""
        node = GraphNode(
            node_id=claim_id, zone="knowledge",
            node_type="claim", label=claim_text,
            properties=properties,
        )
        self.add_node(node)
        for ent_id in source_entities:
            self.add_edge(GraphEdge(
                source_id=ent_id, target_id=claim_id,
                edge_type="mentioned_in_claim", zone="knowledge",
            ))
        return node

    def add_alias(self, alias_text: str, canonical_id: str,
                  language: str = "en") -> None:
        """Add a lexical alias edge to the lexical zone."""
        alias_node_id = f"alias:{canonical_id}:{alias_text}"
        node = GraphNode(
            node_id=alias_node_id, zone="lexical",
            node_type="alias", label=alias_text,
            properties={"language": language},
        )
        self.add_node(node)
        self.add_edge(GraphEdge(
            source_id=alias_node_id, target_id=canonical_id,
            edge_type="alias_of", zone="lexical",
        ))

    def add_provenance_edge(self, entity_id: str, source_doc_id: str,
                            span_start: int = 0, span_end: int = 0,
                            **properties) -> None:
        """Add a provenance edge to the provenance zone."""
        # Ensure document node exists
        doc_node = self.get_node(source_doc_id)
        if not doc_node:
            self.add_node(GraphNode(
                node_id=source_doc_id, zone="provenance",
                node_type="document", label=source_doc_id,
            ))
        self.add_edge(GraphEdge(
            source_id=entity_id, target_id=source_doc_id,
            edge_type="derived_from", zone="provenance",
            properties={"span_start": span_start, "span_end": span_end,
                         **properties},
        ))

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(self, start_id: str, max_depth: int = 2,
                 edge_types: Optional[list[str]] = None,
                 zones: Optional[list[str]] = None) -> GraphTraversalResult:
        """BFS traversal from a starting node."""
        if self._graph is None or start_id not in self._graph:
            return GraphTraversalResult()

        visited_nodes = set()
        visited_edges = []
        paths = []
        queue = [(start_id, [start_id], 0)]

        while queue:
            current, path, depth = queue.pop(0)
            if current in visited_nodes and depth > 0:
                continue
            visited_nodes.add(current)

            if depth >= max_depth:
                paths.append(path)
                continue

            for _, target, data in self._graph.out_edges(current, data=True):
                etype = data.get("edge_type", "")
                ezone = data.get("zone", "")
                if edge_types and etype not in edge_types:
                    continue
                if zones and ezone not in zones:
                    continue
                visited_edges.append(GraphEdge(
                    source_id=current, target_id=target,
                    edge_type=etype, zone=ezone,
                    properties={k: v for k, v in data.items()
                                if k not in ("edge_type", "zone")},
                ))
                queue.append((target, path + [target], depth + 1))

        result_nodes = [self.get_node(nid) for nid in visited_nodes]
        return GraphTraversalResult(
            nodes=[n for n in result_nodes if n is not None],
            edges=visited_edges,
            paths=paths,
        )

    def get_company_subgraph(self, company_id: str, depth: int = 2) -> GraphTraversalResult:
        """Get all connected nodes for a company (entity + data + knowledge + provenance)."""
        return self.traverse(company_id, max_depth=depth)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes() if self._graph else 0

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges() if self._graph else 0

    def zone_stats(self) -> dict[str, int]:
        """Count nodes per zone."""
        if not self._graph:
            return {}
        counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            zone = data.get("zone", "unknown")
            counts[zone] = counts.get(zone, 0) + 1
        return counts
