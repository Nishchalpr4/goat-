"""Schema linker — maps user terms to canonical metric/schema concepts.

Grounded in XBRL taxonomy concepts where possible:
  "net income" → us-gaap:NetIncomeLoss
  "turnover" → us-gaap:Revenues (or ifrs-full:Revenue)
  "FCF" → us-gaap:FreeCashFlow (custom mapping)
"""

from dataclasses import dataclass, field
from typing import Optional

from goat.models.lexicon import Lexicon, SchemaTermMapping, AbbreviationEntry
from goat.lexical.entity_resolver import trigram_similarity


@dataclass
class SchemaLink:
    """A resolved link from a user term to a canonical schema concept."""
    user_term: str
    canonical_concept: str  # e.g. "us-gaap:NetIncomeLoss"
    taxonomy: str  # "us-gaap", "ifrs-full", "custom"
    concept_label: str  # human-readable label
    resolution_method: str  # "exact", "abbreviation", "fuzzy", "embedding"
    confidence: float = 1.0


@dataclass
class SchemaLinkingResult:
    """Result of schema-linking a set of terms."""
    links: list[SchemaLink] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)

    @property
    def canonical_concepts(self) -> list[str]:
        return [link.canonical_concept for link in self.links]


class SchemaLinker:
    """Links user-facing metric terms to canonical schema concepts."""

    def __init__(self, lexicon: Lexicon, similarity_threshold: float = 0.4):
        self.lexicon = lexicon
        self.similarity_threshold = similarity_threshold

    def link(self, term: str) -> Optional[SchemaLink]:
        """Link a single term to a canonical schema concept."""
        cleaned = term.strip()
        if not cleaned:
            return None

        # 1. Check abbreviation table first
        abbr = self.lexicon.expand_abbreviation(cleaned)
        if abbr and abbr.canonical_metric_id:
            return SchemaLink(
                user_term=cleaned,
                canonical_concept=abbr.canonical_metric_id,
                taxonomy="custom",
                concept_label=abbr.expansion,
                resolution_method="abbreviation",
                confidence=1.0,
            )

        # 2. Exact schema term mapping
        mappings = self.lexicon.map_to_schema(cleaned)
        if mappings:
            best = max(mappings, key=lambda m: m.confidence)
            return SchemaLink(
                user_term=cleaned,
                canonical_concept=best.canonical_concept,
                taxonomy=best.taxonomy,
                concept_label=best.concept_label,
                resolution_method="exact",
                confidence=best.confidence,
            )

        # 3. Fuzzy matching against schema surface forms
        best_score = 0.0
        best_mapping: Optional[SchemaTermMapping] = None
        for mapping in self.lexicon.schema_mappings:
            sim = trigram_similarity(cleaned, mapping.surface_form)
            if sim > best_score:
                best_score = sim
                best_mapping = mapping

        if best_mapping and best_score >= self.similarity_threshold:
            return SchemaLink(
                user_term=cleaned,
                canonical_concept=best_mapping.canonical_concept,
                taxonomy=best_mapping.taxonomy,
                concept_label=best_mapping.concept_label,
                resolution_method="fuzzy",
                confidence=best_score,
            )

        return None

    def link_batch(self, terms: list[str]) -> SchemaLinkingResult:
        """Link multiple terms to canonical concepts."""
        result = SchemaLinkingResult()
        for term in terms:
            link = self.link(term)
            if link:
                result.links.append(link)
            else:
                result.unlinked.append(term)
        return result
