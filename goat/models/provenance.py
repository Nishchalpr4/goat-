"""W3C PROV-based provenance models for auditability and traceability.

Aligned with PROV-DM (Data Model) and PROV-O (Ontology) concepts:
  - Entity: a thing with provenance (document, chunk, fact, embedding)
  - Activity: an action that produces/uses entities (ingestion run, embedding, extraction)
  - Agent: an actor responsible for an activity (system version, model, user)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ProvenanceEntity:
    """A PROV Entity — something whose provenance we track."""
    entity_id: str
    entity_type: str  # "document", "chunk", "fact", "embedding", "graph_node"
    attributes: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProvenanceActivity:
    """A PROV Activity — an action that generated or used entities."""
    activity_id: str
    activity_type: str  # "ingestion", "chunking", "embedding", "extraction", "indexing"
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    parameters: dict = field(default_factory=dict)  # config snapshot


@dataclass
class ProvenanceAgent:
    """A PROV Agent — the actor responsible for an activity."""
    agent_id: str
    agent_type: str  # "system", "model", "user", "extractor"
    version: str = ""
    attributes: dict = field(default_factory=dict)


@dataclass
class ProvenanceRelation:
    """A PROV relationship between entities, activities, and agents.

    Relation types (from PROV-DM):
      - wasGeneratedBy: entity → activity
      - used: activity → entity
      - wasDerivedFrom: entity → entity
      - wasAttributedTo: entity → agent
      - wasAssociatedWith: activity → agent
      - actedOnBehalfOf: agent → agent
    """
    relation_type: str
    source_id: str
    target_id: str
    attributes: dict = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProvenanceRecord:
    """Complete provenance for a single retrieval hit or generated fact."""
    source_doc_id: str
    source_system: str  # "EDGAR", "reuters", "internal"
    span_offsets: tuple[int, int] = (0, 0)  # (start, end) in source
    ingestion_run_id: str = ""
    extractor_version: str = ""
    embed_model_version: str = ""
    index_version: str = ""
    # PROV edges for this record
    entities: list[ProvenanceEntity] = field(default_factory=list)
    activities: list[ProvenanceActivity] = field(default_factory=list)
    agents: list[ProvenanceAgent] = field(default_factory=list)
    relations: list[ProvenanceRelation] = field(default_factory=list)

    def add_generation(self, entity_id: str, activity_id: str) -> None:
        self.relations.append(ProvenanceRelation(
            relation_type="wasGeneratedBy",
            source_id=entity_id,
            target_id=activity_id,
        ))

    def add_derivation(self, derived_id: str, source_id: str) -> None:
        self.relations.append(ProvenanceRelation(
            relation_type="wasDerivedFrom",
            source_id=derived_id,
            target_id=source_id,
        ))

    def add_attribution(self, entity_id: str, agent_id: str) -> None:
        self.relations.append(ProvenanceRelation(
            relation_type="wasAttributedTo",
            source_id=entity_id,
            target_id=agent_id,
        ))

    def to_prov_triples(self) -> list[tuple[str, str, str]]:
        """Export as (subject, predicate, object) triples."""
        return [
            (r.source_id, r.relation_type, r.target_id)
            for r in self.relations
        ]
