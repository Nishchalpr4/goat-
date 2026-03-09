"""Community detection and summarization for Graph RAG.

Implements the "global queries" path: partition the graph into
communities, generate summaries per community, then aggregate
community-level answers into a final response.

Uses NetworkX community detection algorithms:
  - Louvain (primary, resolution-tunable)
  - Label propagation (fast fallback)
  - Greedy modularity (deterministic)
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    from networkx.algorithms.community import (
        louvain_communities,
        label_propagation_communities,
        greedy_modularity_communities,
    )
    HAS_NX = True
except ImportError:
    HAS_NX = False


@dataclass
class Community:
    """A detected community within a graph zone."""
    community_id: str
    zone: str
    node_ids: list[str] = field(default_factory=list)
    # Human-readable summary (generated post-detection)
    summary: str = ""
    # Representative entities (top-degree nodes)
    key_entities: list[str] = field(default_factory=list)
    # Modularity contribution
    modularity_score: float = 0.0


@dataclass
class CommunityReport:
    """Collection of communities with metadata."""
    communities: list[Community] = field(default_factory=list)
    algorithm: str = "louvain"
    resolution: float = 1.0
    total_nodes: int = 0
    total_edges: int = 0
    modularity: float = 0.0


class CommunityDetector:
    """Detects communities in the knowledge graph and produces summaries."""

    def __init__(self, algorithm: str = "louvain", resolution: float = 1.0):
        if not HAS_NX:
            raise ImportError("networkx is required for community detection")
        self.algorithm = algorithm
        self.resolution = resolution

    def detect(self, graph: nx.Graph,
               zone: Optional[str] = None) -> CommunityReport:
        """Run community detection on a graph or zone subgraph.

        Args:
            graph: NetworkX graph (or DiGraph — will be converted)
            zone: If provided, filter to nodes in this zone only
        """
        # Convert DiGraph to undirected for community detection
        if isinstance(graph, nx.DiGraph):
            g = graph.to_undirected()
        else:
            g = graph.copy()

        # Filter by zone if specified
        if zone:
            zone_nodes = [n for n, d in g.nodes(data=True)
                          if d.get("zone") == zone]
            g = g.subgraph(zone_nodes).copy()

        if len(g) == 0:
            return CommunityReport(algorithm=self.algorithm,
                                   resolution=self.resolution)

        report = CommunityReport(
            algorithm=self.algorithm,
            resolution=self.resolution,
            total_nodes=g.number_of_nodes(),
            total_edges=g.number_of_edges(),
        )

        communities = self._run_detection(g)

        for i, node_set in enumerate(communities):
            node_list = sorted(node_set)
            cid = self._community_id(zone or "all", i, node_list)
            key_entities = self._top_degree_nodes(g, node_list, top_k=5)

            comm = Community(
                community_id=cid,
                zone=zone or "all",
                node_ids=node_list,
                key_entities=key_entities,
            )
            report.communities.append(comm)

        # Compute modularity
        if report.communities:
            partition = [set(c.node_ids) for c in report.communities]
            try:
                report.modularity = nx.community.modularity(g, partition)
            except Exception:
                report.modularity = 0.0

        return report

    def _run_detection(self, g: nx.Graph) -> list[set]:
        """Execute the selected community detection algorithm."""
        if self.algorithm == "louvain":
            return louvain_communities(g, resolution=self.resolution,
                                       seed=42)
        elif self.algorithm == "label_propagation":
            return list(label_propagation_communities(g))
        elif self.algorithm == "greedy_modularity":
            return list(greedy_modularity_communities(g))
        else:
            # Default to Louvain
            return louvain_communities(g, resolution=self.resolution,
                                       seed=42)

    @staticmethod
    def _top_degree_nodes(g: nx.Graph, node_ids: list[str],
                          top_k: int = 5) -> list[str]:
        """Return the top-k highest degree nodes in a community."""
        sub = g.subgraph(node_ids)
        degree_list = sorted(sub.degree(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in degree_list[:top_k]]

    @staticmethod
    def _community_id(zone: str, index: int, node_ids: list[str]) -> str:
        """Generate a stable community ID."""
        content = f"{zone}:{index}:{','.join(node_ids[:10])}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def summarize_community(self, community: Community,
                            graph: nx.Graph) -> str:
        """Generate a textual summary of a community from its node/edge data.

        This produces a structured representation suitable for downstream
        LLM summarization or direct display.
        """
        sub = graph.subgraph(community.node_ids)
        lines = [f"Community {community.community_id[:8]} "
                 f"({len(community.node_ids)} nodes):"]

        # Group nodes by type
        type_groups: dict[str, list[str]] = {}
        for node_id, data in sub.nodes(data=True):
            ntype = data.get("type", "unknown")
            label = data.get("label", node_id)
            type_groups.setdefault(ntype, []).append(label)

        for ntype, labels in sorted(type_groups.items()):
            lines.append(f"  {ntype}: {', '.join(labels[:10])}")
            if len(labels) > 10:
                lines.append(f"    ... and {len(labels) - 10} more")

        # Key relationships
        edge_types: dict[str, int] = {}
        for u, v, data in sub.edges(data=True):
            etype = data.get("type", "related_to")
            edge_types[etype] = edge_types.get(etype, 0) + 1

        if edge_types:
            lines.append("  Relationships:")
            for etype, count in sorted(edge_types.items(),
                                        key=lambda x: -x[1]):
                lines.append(f"    {etype}: {count}")

        summary = "\n".join(lines)
        community.summary = summary
        return summary

    def generate_community_report(self, report: CommunityReport,
                                   graph: nx.Graph) -> list[str]:
        """Generate summaries for all communities in a report."""
        summaries = []
        for comm in report.communities:
            summary = self.summarize_community(comm, graph)
            summaries.append(summary)
        return summaries
