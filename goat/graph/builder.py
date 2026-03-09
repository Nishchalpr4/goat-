"""Graph builder — constructs the multi-zone graph from documents, entities,
and extracted facts/relations.

Populates all five zones:
  - Entity zone from company/entity data
  - Data zone from financial facts
  - Knowledge zone from extracted claims/events
  - Lexical zone from lexicon entries
  - Provenance zone from ingestion/extraction metadata
"""

from typing import Optional
import logging
import uuid

from goat.storage.graph_store import GraphStore, GraphNode, GraphEdge
from goat.models.company import Company
from goat.models.document import Document, Chunk
from goat.models.lexicon import Lexicon
from goat.models.provenance import ProvenanceRecord

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the multi-zone graph from structured and extracted data."""

    def __init__(self, graph_store: GraphStore):
        self.graph = graph_store

    def build_company_entity(self, company: Company) -> None:
        """Add a company and its related entities to the entity zone."""
        # Company node
        self.graph.add_entity(
            entity_id=company.canonical_id,
            entity_type="company",
            label=company.name,
            primary_ticker=company.primary_ticker or "",
            sector=company.metadata.sector,
            industry=company.metadata.industry,
            country=company.metadata.country,
            market_cap=company.metadata.market_cap,
        )

        # Ticker nodes
        for ident in company.identifiers:
            if ident.id_type == "ticker":
                ticker_node_id = f"ticker:{ident.value}"
                self.graph.add_entity(
                    entity_id=ticker_node_id,
                    entity_type="ticker",
                    label=ident.value,
                    exchange=ident.exchange or "",
                )
                self.graph.add_edge(GraphEdge(
                    source_id=company.canonical_id,
                    target_id=ticker_node_id,
                    edge_type="listed_on",
                    zone="entity",
                ))

        # Sector/Industry nodes
        if company.metadata.sector:
            sector_id = f"sector:{company.metadata.sector}"
            self.graph.add_entity(
                entity_id=sector_id, entity_type="sector",
                label=company.metadata.sector,
            )
            self.graph.add_edge(GraphEdge(
                source_id=company.canonical_id,
                target_id=sector_id,
                edge_type="belongs_to_sector",
                zone="entity",
            ))

        if company.metadata.industry:
            industry_id = f"industry:{company.metadata.industry}"
            self.graph.add_entity(
                entity_id=industry_id, entity_type="industry",
                label=company.metadata.industry,
            )
            self.graph.add_edge(GraphEdge(
                source_id=company.canonical_id,
                target_id=industry_id,
                edge_type="belongs_to_industry",
                zone="entity",
            ))

        # Executive nodes
        for mgmt in company.management:
            exec_id = f"exec:{company.canonical_id}:{mgmt.name}"
            self.graph.add_entity(
                entity_id=exec_id, entity_type="executive",
                label=mgmt.name, title=mgmt.title,
                insider_ownership_pct=mgmt.insider_ownership_pct,
            )
            self.graph.add_edge(GraphEdge(
                source_id=exec_id,
                target_id=company.canonical_id,
                edge_type="employed_by",
                zone="entity",
            ))

        # Aliases → lexical zone
        for alias in company.aliases:
            self.graph.add_alias(
                alias_text=alias,
                canonical_id=company.canonical_id,
            )

        logger.debug("Built entity subgraph for %s", company.name)

    def build_financial_facts(self, company: Company) -> None:
        """Add financial facts to the data zone."""
        financials = company.financials

        for stmt in financials.income_statements:
            # Revenue
            if stmt.revenue:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:revenue:{stmt.year}",
                    company_id=company.canonical_id,
                    metric="revenue",
                    value=stmt.revenue,
                    period=f"FY{stmt.year}",
                    unit="USD",
                )
            # Net income
            if stmt.net_income:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:net_income:{stmt.year}",
                    company_id=company.canonical_id,
                    metric="net_income",
                    value=stmt.net_income,
                    period=f"FY{stmt.year}",
                    unit="USD",
                )
            # EBITDA
            if stmt.ebitda:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:ebitda:{stmt.year}",
                    company_id=company.canonical_id,
                    metric="ebitda",
                    value=stmt.ebitda,
                    period=f"FY{stmt.year}",
                    unit="USD",
                )
            # Margins as ratios
            if stmt.gross_margin is not None:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:gross_margin:{stmt.year}",
                    company_id=company.canonical_id,
                    metric="gross_margin",
                    value=round(stmt.gross_margin, 4),
                    period=f"FY{stmt.year}",
                    unit="ratio",
                )

        for bs in financials.balance_sheets:
            if bs.total_assets:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:total_assets:{bs.year}",
                    company_id=company.canonical_id,
                    metric="total_assets",
                    value=bs.total_assets,
                    period=f"FY{bs.year}",
                    unit="USD",
                )
            if bs.debt_to_equity is not None:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:debt_to_equity:{bs.year}",
                    company_id=company.canonical_id,
                    metric="debt_to_equity",
                    value=round(bs.debt_to_equity, 4),
                    period=f"FY{bs.year}",
                    unit="ratio",
                )

        for cf in financials.cash_flow_statements:
            if cf.free_cash_flow:
                self.graph.add_fact(
                    fact_id=f"fact:{company.canonical_id}:fcf:{cf.year}",
                    company_id=company.canonical_id,
                    metric="free_cash_flow",
                    value=cf.free_cash_flow,
                    period=f"FY{cf.year}",
                    unit="USD",
                )

        logger.debug("Built data zone for %s (%d years)",
                      company.name, len(financials.years))

    def build_lexicon_graph(self, lexicon: Lexicon) -> None:
        """Populate the lexical zone from a lexicon snapshot."""
        for entry in lexicon.entries:
            alias_node_id = f"lex:{entry.entry_id}"
            self.graph.add_node(GraphNode(
                node_id=alias_node_id,
                zone="lexical",
                node_type=entry.entry_type.value,
                label=entry.surface_form,
                properties={
                    "canonical_id": entry.canonical_id,
                    "language": entry.language,
                    "confidence": entry.confidence,
                },
            ))
            self.graph.add_edge(GraphEdge(
                source_id=alias_node_id,
                target_id=entry.canonical_id,
                edge_type="alias_of",
                zone="lexical",
            ))

        for mapping in lexicon.schema_mappings:
            mapping_node_id = f"schema:{mapping.canonical_concept}:{mapping.surface_form}"
            self.graph.add_node(GraphNode(
                node_id=mapping_node_id,
                zone="lexical",
                node_type="schema_mapping",
                label=mapping.surface_form,
                properties={
                    "canonical_concept": mapping.canonical_concept,
                    "taxonomy": mapping.taxonomy,
                },
            ))

        logger.debug("Built lexical zone (%d entries, %d schema mappings)",
                      len(lexicon.entries), len(lexicon.schema_mappings))

    def build_document_provenance(self, document: Document,
                                   chunks: list[Chunk]) -> None:
        """Add provenance edges for a document and its chunks."""
        # Document node
        self.graph.add_node(GraphNode(
            node_id=document.doc_id,
            zone="provenance",
            node_type="document",
            label=document.title,
            properties={
                "source_system": document.metadata.source_system,
                "source_url": document.metadata.source_url,
                "doc_type": document.metadata.doc_type,
                "filing_date": str(document.metadata.filing_date) if document.metadata.filing_date else "",
                "regulator_doc_id": document.metadata.regulator_doc_id,
            },
        ))

        # Ingestion run node
        if document.ingestion_run_id:
            run_id = document.ingestion_run_id
            self.graph.add_node(GraphNode(
                node_id=run_id,
                zone="provenance",
                node_type="extraction_run",
                label=f"ingestion:{run_id}",
            ))
            self.graph.add_edge(GraphEdge(
                source_id=document.doc_id,
                target_id=run_id,
                edge_type="wasGeneratedBy",
                zone="provenance",
            ))

        # Chunk → document edges
        for chunk in chunks:
            self.graph.add_provenance_edge(
                entity_id=chunk.chunk_id,
                source_doc_id=document.doc_id,
                span_start=chunk.start_offset,
                span_end=chunk.end_offset,
            )

    def add_claim(self, claim_text: str, company_ids: list[str],
                  claim_type: str = "claim",
                  source_doc_id: str = "",
                  **properties) -> str:
        """Add an extracted claim/relation to the knowledge zone."""
        claim_id = f"claim:{uuid.uuid4().hex[:12]}"
        self.graph.add_claim(
            claim_id=claim_id,
            claim_text=claim_text,
            source_entities=company_ids,
            claim_type=claim_type,
            **properties,
        )
        if source_doc_id:
            self.graph.add_provenance_edge(
                entity_id=claim_id,
                source_doc_id=source_doc_id,
            )
        return claim_id
