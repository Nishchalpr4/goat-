"""Document and chunk models for ingestion and retrieval."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class DocumentType(Enum):
    FILING_10K = "10-K"
    FILING_10Q = "10-Q"
    FILING_8K = "8-K"
    FILING_20F = "20-F"
    EARNINGS_TRANSCRIPT = "transcript"
    PRESS_RELEASE = "press_release"
    NEWS_ARTICLE = "news"
    ANALYST_REPORT = "analyst_report"
    REGULATORY = "regulatory"
    OTHER = "other"


@dataclass
class DocumentMetadata:
    """Metadata attached to a source document."""
    company_id: str = ""
    doc_type: str = "other"
    filing_date: Optional[datetime] = None
    period_end: Optional[datetime] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    language: str = "en"
    source_system: str = ""  # "EDGAR", "press_wire", "internal", etc.
    source_url: str = ""
    regulator_doc_id: str = ""  # e.g. EDGAR accession number
    section: str = ""  # e.g. "Risk Factors", "MD&A"


@dataclass
class Document:
    """A source document prior to chunking."""
    doc_id: str
    title: str
    content: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    created_at: datetime = field(default_factory=datetime.utcnow)
    ingestion_run_id: str = ""

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class Chunk:
    """A chunk of a document, the unit for embedding and retrieval.

    Carries both raw text and optional entity/schema context for
    the dual-embedding strategy (narrative vs entity+schema conditioned).
    """
    chunk_id: str
    doc_id: str
    text: str
    # Positional info in source document
    start_offset: int = 0
    end_offset: int = 0
    chunk_index: int = 0  # position within the document's chunk sequence
    # Structural segmentation
    section: str = ""  # "Risk Factors", "Item 7", etc.
    subsection: str = ""
    # Entity/schema context for conditioned embeddings
    company_id: str = ""
    company_name: str = ""
    ticker: str = ""
    sector: str = ""
    doc_type: str = ""
    # Embedding references (set after embedding)
    narrative_embedding_id: Optional[str] = None
    entity_schema_embedding_id: Optional[str] = None
    # Provenance
    ingestion_run_id: str = ""
    embed_model_version: str = ""

    @property
    def entity_context_prefix(self) -> str:
        """Build the entity+schema prefix for conditioned embeddings."""
        parts = []
        if self.company_name:
            parts.append(self.company_name)
        if self.ticker:
            parts.append(self.ticker)
        if self.sector:
            parts.append(self.sector)
        if self.doc_type:
            parts.append(self.doc_type)
        if self.section:
            parts.append(self.section)
        return " | ".join(parts)

    @property
    def conditioned_text(self) -> str:
        """Full text with entity context for entity+schema embedding."""
        prefix = self.entity_context_prefix
        if prefix:
            return f"[{prefix}] {self.text}"
        return self.text
