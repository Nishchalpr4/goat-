"""Graph RAG zones — defines the multi-zone architecture for entity identity,
data facts, knowledge claims, lexical variants, and provenance.

Each zone has specific node/edge types and semantics aligned with the
architectural specification.
"""

from dataclasses import dataclass, field
from enum import Enum


class Zone(Enum):
    """The five semantic zones of the Graph RAG store."""
    ENTITY = "entity"
    DATA = "data"
    KNOWLEDGE = "knowledge"
    LEXICAL = "lexical"
    PROVENANCE = "provenance"


# ---------------------------------------------------------------------------
# Entity zone node types
# ---------------------------------------------------------------------------

class EntityNodeType(Enum):
    COMPANY = "company"
    SUBSIDIARY = "subsidiary"
    EXECUTIVE = "executive"
    PRODUCT = "product"
    REGULATOR = "regulator"
    TICKER = "ticker"
    EXCHANGE = "exchange"
    INVESTOR = "investor"
    INDUSTRY = "industry"
    SECTOR = "sector"


class EntityEdgeType(Enum):
    HAS_SUBSIDIARY = "has_subsidiary"
    EMPLOYED_BY = "employed_by"
    PRODUCES = "produces"
    REGULATED_BY = "regulated_by"
    LISTED_ON = "listed_on"
    COMPETES_WITH = "competes_with"
    INVESTED_IN = "invested_in"
    BELONGS_TO_SECTOR = "belongs_to_sector"
    BELONGS_TO_INDUSTRY = "belongs_to_industry"


# ---------------------------------------------------------------------------
# Data zone node types (normalized numeric facts)
# ---------------------------------------------------------------------------

class DataNodeType(Enum):
    FINANCIAL_FACT = "financial_fact"      # revenue, net_income, etc.
    RATIO = "ratio"                        # P/E, D/E, etc.
    GUIDANCE = "guidance"                  # forward-looking metric
    TIME_SERIES = "time_series"            # pointer to time series data


class DataEdgeType(Enum):
    HAS_FACT = "has_fact"
    HAS_RATIO = "has_ratio"
    GAVE_GUIDANCE = "gave_guidance"
    FACT_FOR_PERIOD = "fact_for_period"


# ---------------------------------------------------------------------------
# Knowledge zone node types (extracted relations/claims)
# ---------------------------------------------------------------------------

class KnowledgeNodeType(Enum):
    CLAIM = "claim"                  # extracted factual claim
    EVENT = "event"                  # corporate event (acquisition, IPO, etc.)
    RISK = "risk"                    # identified risk factor
    GUIDANCE_CHANGE = "guidance_change"
    ANALYST_OPINION = "analyst_opinion"


class KnowledgeEdgeType(Enum):
    ACQUIRED = "acquired"
    MERGED_WITH = "merged_with"
    PARTNERED_WITH = "partnered_with"
    DIVESTED = "divested"
    MENTIONED_IN_CLAIM = "mentioned_in_claim"
    IMPACTS = "impacts"
    REVISED_GUIDANCE = "revised_guidance"
    RISK_AFFECTS = "risk_affects"


# ---------------------------------------------------------------------------
# Lexical zone node types
# ---------------------------------------------------------------------------

class LexicalNodeType(Enum):
    ALIAS = "alias"
    ABBREVIATION = "abbreviation"
    TRANSLATION = "translation"
    DOMAIN_SYNONYM = "domain_synonym"
    SCHEMA_MAPPING = "schema_mapping"


class LexicalEdgeType(Enum):
    ALIAS_OF = "alias_of"
    ABBREVIATES = "abbreviates"
    TRANSLATES = "translates"
    SYNONYM_OF = "synonym_of"
    MAPS_TO_CONCEPT = "maps_to_concept"


# ---------------------------------------------------------------------------
# Provenance zone node types
# ---------------------------------------------------------------------------

class ProvenanceNodeType(Enum):
    DOCUMENT = "document"
    EXTRACTION_RUN = "extraction_run"
    EMBEDDING_RUN = "embedding_run"
    MODEL = "model"
    SYSTEM = "system"


class ProvenanceEdgeType(Enum):
    DERIVED_FROM = "derived_from"          # entity → source document
    WAS_GENERATED_BY = "wasGeneratedBy"    # PROV-DM
    USED = "used"                          # PROV-DM
    WAS_ATTRIBUTED_TO = "wasAttributedTo"  # PROV-DM
    WAS_ASSOCIATED_WITH = "wasAssociatedWith"  # PROV-DM


# ---------------------------------------------------------------------------
# Zone configuration
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    """Configuration for a specific graph zone."""
    zone: Zone
    node_types: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    description: str = ""


ZONE_CONFIGS = {
    Zone.ENTITY: ZoneConfig(
        zone=Zone.ENTITY,
        node_types=[t.value for t in EntityNodeType],
        edge_types=[t.value for t in EntityEdgeType],
        description="Companies, subsidiaries, executives, products, regulators, tickers",
    ),
    Zone.DATA: ZoneConfig(
        zone=Zone.DATA,
        node_types=[t.value for t in DataNodeType],
        edge_types=[t.value for t in DataEdgeType],
        description="Normalized numeric facts, financial statements, ratios, guidance",
    ),
    Zone.KNOWLEDGE: ZoneConfig(
        zone=Zone.KNOWLEDGE,
        node_types=[t.value for t in KnowledgeNodeType],
        edge_types=[t.value for t in KnowledgeEdgeType],
        description="Extracted relations/claims, events, risks, guidance changes",
    ),
    Zone.LEXICAL: ZoneConfig(
        zone=Zone.LEXICAL,
        node_types=[t.value for t in LexicalNodeType],
        edge_types=[t.value for t in LexicalEdgeType],
        description="Aliases, abbreviations, translations, domain synonyms, schema mappings",
    ),
    Zone.PROVENANCE: ZoneConfig(
        zone=Zone.PROVENANCE,
        node_types=[t.value for t in ProvenanceNodeType],
        edge_types=[t.value for t in ProvenanceEdgeType],
        description="Document source, span offsets, extraction run metadata, PROV triples",
    ),
}
