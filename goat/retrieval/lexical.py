"""Lexical retrieval — BM25/FTS via Postgres full-text search + trigram matching.

Provides:
  - BM25-like retrieval using Postgres tsvector/tsquery + ts_rank_cd
  - Trigram similarity for ticker/alias resolution
  - Field-weighted scoring (BM25F-style via section/title boosting)
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

from goat.models.retrieval import RetrievalHit, ScoringBreakdown
from goat.storage.postgres import PostgresStore
from goat.lexical.query_expander import ExpandedQuery

logger = logging.getLogger(__name__)


@dataclass
class LexicalCandidate:
    """A candidate from lexical retrieval with BM25-style scoring."""
    chunk_id: str
    doc_id: str
    text: str
    score: float  # BM25/ts_rank_cd score
    match_terms: list[str] = field(default_factory=list)
    section: str = ""
    company_id: str = ""
    doc_type: str = ""
    language: str = "en"

    def to_retrieval_hit(self) -> RetrievalHit:
        return RetrievalHit(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            text=self.text,
            section=self.section,
            company_id=self.company_id,
            doc_type=self.doc_type,
            language=self.language,
            scoring=ScoringBreakdown(
                lexical_score=self.score,
                lexical_match_terms=self.match_terms,
            ),
        )


class LexicalRetriever:
    """BM25/FTS retrieval using Postgres full-text search."""

    # BM25F-style field weights (section/title matches boost score)
    SECTION_BOOSTS = {
        "Risk Factors": 1.2,
        "MD&A": 1.3,
        "Business": 1.1,
        "Financial Statements": 1.0,
        "Q&A Session": 1.1,
        "Forward-Looking Statements": 1.15,
    }

    def __init__(self, postgres: PostgresStore):
        self.postgres = postgres

    def retrieve(self, expanded_query: ExpandedQuery,
                 top_k: int = 100) -> list[LexicalCandidate]:
        """Execute lexical retrieval using expanded query.

        Uses both FTS (tsvector) and trigram matching for identifiers.
        """
        candidates = []

        # 1. Full-text search with expanded query
        query_text = expanded_query.expanded_query or expanded_query.original_query
        company_filter = None
        if expanded_query.has_entity_filters:
            company_ids = expanded_query.filters.get("company_id", [])
            if len(company_ids) == 1:
                company_filter = company_ids[0]

        fts_results = self.postgres.fts_search(
            query=query_text,
            top_k=top_k,
            company_id=company_filter,
        )

        for row in fts_results:
            score = row.get("rank", 0.0)
            section = row.get("section", "")
            # Apply BM25F-style section boost
            boost = self.SECTION_BOOSTS.get(section, 1.0)
            score *= boost

            candidates.append(LexicalCandidate(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                score=score,
                section=section,
                company_id=row.get("company_id", ""),
                doc_type=row.get("doc_type", ""),
            ))

        # 2. Trigram matching for ticker/alias-heavy queries
        if expanded_query.entity_resolution:
            for resolved in expanded_query.entity_resolution.resolved:
                if resolved.resolution_method.startswith("trigram"):
                    # Search chunks mentioning this entity via trigram
                    trgm_results = self.postgres.trigram_search(
                        query=resolved.surface_form,
                        table="chunks",
                        column="text",
                        limit=20,
                    )
                    for row in trgm_results:
                        # Avoid duplicates
                        if any(c.chunk_id == row.get("chunk_id") for c in candidates):
                            continue
                        candidates.append(LexicalCandidate(
                            chunk_id=row.get("chunk_id", ""),
                            doc_id=row.get("doc_id", ""),
                            text=row.get("text", ""),
                            score=row.get("sim", 0.0) * 0.5,  # lower weight
                            section=row.get("section", ""),
                            company_id=row.get("company_id", ""),
                            doc_type=row.get("doc_type", ""),
                        ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def retrieve_simple(self, query: str, top_k: int = 100,
                        company_id: Optional[str] = None) -> list[LexicalCandidate]:
        """Simple FTS retrieval without query expansion (for benchmarking)."""
        fts_results = self.postgres.fts_search(
            query=query, top_k=top_k, company_id=company_id,
        )
        return [
            LexicalCandidate(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"],
                text=r["text"], score=r.get("rank", 0.0),
                section=r.get("section", ""),
                company_id=r.get("company_id", ""),
                doc_type=r.get("doc_type", ""),
            )
            for r in fts_results
        ]
