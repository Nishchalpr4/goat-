"""Lexicon models — synonyms, aliases, abbreviations, and schema term mappings.

These are the "lexical assets" stored in Postgres as versioned, slowly changing
dimension tables that power entity resolution, query expansion, and schema linking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class LexiconEntryType(Enum):
    TICKER = "ticker"
    COMPANY_ALIAS = "company_alias"
    ABBREVIATION = "abbreviation"
    SYNONYM = "synonym"
    XBRL_CONCEPT = "xbrl_concept"
    SECTOR_TERM = "sector_term"
    MULTILINGUAL_ALIAS = "multilingual_alias"
    METRIC_ALIAS = "metric_alias"


@dataclass
class LexiconEntry:
    """A single lexicon entry mapping a surface form to a canonical target."""
    entry_id: str
    surface_form: str  # what the user/document says ("FCF", "BRK.B", "Net income")
    canonical_id: str  # stable ID of the canonical entity/concept
    canonical_label: str  # human-readable label ("Free Cash Flow", "Berkshire Hathaway B")
    entry_type: LexiconEntryType = LexiconEntryType.SYNONYM
    language: str = "en"
    confidence: float = 1.0  # 1.0 = exact, <1.0 = fuzzy/inferred
    source: str = "manual"  # "manual", "xbrl_taxonomy", "auto_extracted", "user_feedback"
    version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class TickerMapping:
    """Specialized mapping for ticker symbols with exchange variants."""
    canonical_id: str
    ticker: str
    exchange: str = ""
    # Surface form variants (e.g., "BRK.B", "BRK-B", "BRK B", "BRKB")
    variants: list[str] = field(default_factory=list)
    company_name: str = ""
    is_active: bool = True


@dataclass
class AbbreviationEntry:
    """Financial abbreviation with expansion and XBRL mapping."""
    abbreviation: str  # "FCF", "EPS", "TTM", "NIM", "CAC"
    expansion: str  # "Free Cash Flow", "Earnings Per Share"
    canonical_metric_id: str = ""  # link to XBRL concept or internal metric ID
    category: str = ""  # "profitability", "valuation", "liquidity", etc.
    context_hint: str = ""  # disambiguator when abbreviation is overloaded


@dataclass
class SchemaTermMapping:
    """Mapping between human language and canonical metric/schema terms.

    Anchored to XBRL taxonomy concepts where possible, enabling
    "Net income" / "Profit/Loss" / "Gewinn" → us-gaap:NetIncomeLoss.
    """
    surface_form: str  # "net income", "profit attributable", "Gewinn"
    canonical_concept: str  # "us-gaap:NetIncomeLoss"
    taxonomy: str = "us-gaap"  # "us-gaap", "ifrs-full", "custom"
    concept_label: str = ""  # official XBRL label
    language: str = "en"
    confidence: float = 1.0


@dataclass
class Lexicon:
    """A versioned, complete lexicon snapshot."""
    version: str
    entries: list[LexiconEntry] = field(default_factory=list)
    ticker_mappings: list[TickerMapping] = field(default_factory=list)
    abbreviations: list[AbbreviationEntry] = field(default_factory=list)
    schema_mappings: list[SchemaTermMapping] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def lookup_surface_form(self, surface: str) -> list[LexiconEntry]:
        """Find all entries matching a surface form (case-insensitive)."""
        surface_lower = surface.lower()
        return [e for e in self.entries if e.surface_form.lower() == surface_lower]

    def lookup_canonical(self, canonical_id: str) -> list[LexiconEntry]:
        """Find all surface forms for a canonical ID."""
        return [e for e in self.entries if e.canonical_id == canonical_id]

    def resolve_ticker(self, raw_ticker: str) -> Optional[TickerMapping]:
        """Resolve a possibly noisy ticker string to a canonical mapping."""
        cleaned = raw_ticker.upper().strip()
        for tm in self.ticker_mappings:
            if tm.ticker.upper() == cleaned:
                return tm
            if cleaned in (v.upper() for v in tm.variants):
                return tm
        return None

    def expand_abbreviation(self, abbr: str) -> Optional[AbbreviationEntry]:
        """Expand a financial abbreviation."""
        abbr_upper = abbr.upper().strip()
        for ae in self.abbreviations:
            if ae.abbreviation.upper() == abbr_upper:
                return ae
        return None

    def map_to_schema(self, term: str) -> list[SchemaTermMapping]:
        """Map a human term to canonical schema concepts."""
        term_lower = term.lower().strip()
        return [m for m in self.schema_mappings if m.surface_form.lower() == term_lower]
