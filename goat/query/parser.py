"""Query parsing, normalization, and structural analysis.

Takes a raw user query and produces a normalized, annotated
ParsedQuery with identified entities, periods, metrics, and
structural cues (negation, comparison operators, etc.).
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import logging

from goat.lexical.normalizer import TextNormalizer
from goat.lexical.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Common period patterns
_PERIOD_PATTERNS = [
    # Fiscal year: FY2023, FY 2023, fiscal year 2023
    (r'\b(?:FY|fiscal\s*year)\s*(\d{4})\b', 'FY'),
    # Quarter: Q1 2023, Q1'23, 1Q23, 1Q 2023
    (r"\b[Qq]([1-4])\s*['’]?\s*(\d{2,4})\b", 'Q'),
    (r"\b([1-4])[Qq]\s*['’]?\s*(\d{2,4})\b", 'Q'),
    # Calendar year
    (r'\b(20\d{2})\b', 'CY'),
    # Relative periods
    (r'\b(last|prior|previous|trailing)\s+(year|quarter|month)\b', 'REL'),
    (r'\b(TTM|LTM|NTM)\b', 'REL'),
    (r'\b(YoY|QoQ|MoM)\b', 'COMP'),
]

# Comparison operators in natural language
_COMPARISON_PATTERNS = [
    (r'\bcompare\b|\bvs\.?\b|\bversus\b|\bagainst\b', 'compare'),
    (r'\bgreater than\b|\babove\b|\bexceeds?\b|\b>\b', 'gt'),
    (r'\bless than\b|\bbelow\b|\bunder\b|\b<\b', 'lt'),
    (r'\bbetween\b', 'between'),
    (r'\btrend\b|\bover time\b|\bhistorical\b', 'trend'),
]

# Negation patterns
_NEGATION_PATTERNS = [
    r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bwithout\b',
    r'\bexclud(?:e|ing|es)\b', r'\bdeclin(?:e|ing|ed)\b',
]


@dataclass
class ExtractedPeriod:
    """A time period extracted from the query."""
    raw: str
    period_type: str  # FY, Q, CY, REL, COMP
    year: Optional[int] = None
    quarter: Optional[int] = None
    normalized: str = ""


@dataclass
class ParsedQuery:
    """Structured representation of a parsed user query."""
    raw_query: str
    normalized_query: str
    tokens: list[str] = field(default_factory=list)
    # Extracted periods
    periods: list[ExtractedPeriod] = field(default_factory=list)
    # Detected comparison mode
    comparison_type: Optional[str] = None
    # Whether negation is present
    has_negation: bool = False
    # Raw entity mentions (pre-resolution)
    entity_mentions: list[str] = field(default_factory=list)
    # Raw metric mentions
    metric_mentions: list[str] = field(default_factory=list)
    # Number of distinct entities mentioned (used for intent routing)
    entity_count: int = 0


class QueryParser:
    """Parses raw user queries into structured ParsedQuery objects."""

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        self._period_re = [(re.compile(p, re.IGNORECASE), t)
                           for p, t in _PERIOD_PATTERNS]
        self._comparison_re = [(re.compile(p, re.IGNORECASE), t)
                               for p, t in _COMPARISON_PATTERNS]
        self._negation_re = re.compile(
            '|'.join(_NEGATION_PATTERNS), re.IGNORECASE
        )

    def parse(self, query: str) -> ParsedQuery:
        """Parse a raw query into a structured ParsedQuery."""
        norm_result = self.normalizer.normalize(query)
        normalized = norm_result.normalized
        result = self.tokenizer.tokenize(normalized, field_type="narrative")
        tokens = [t.text for t in result.tokens]

        parsed = ParsedQuery(
            raw_query=query,
            normalized_query=normalized,
            tokens=tokens,
        )

        self._extract_periods(normalized, parsed)
        self._detect_comparison(normalized, parsed)
        self._detect_negation(normalized, parsed)
        self._extract_entity_mentions(tokens, parsed)
        self._extract_metric_mentions(tokens, parsed)

        return parsed

    def _extract_periods(self, text: str, parsed: ParsedQuery) -> None:
        """Extract time periods from query text."""
        for pattern, period_type in self._period_re:
            for match in pattern.finditer(text):
                raw = match.group(0)
                period = ExtractedPeriod(raw=raw, period_type=period_type)

                if period_type == 'FY':
                    period.year = int(match.group(1))
                    period.normalized = f"FY{period.year}"
                elif period_type == 'Q':
                    groups = match.groups()
                    if len(groups) >= 2:
                        period.quarter = int(groups[0])
                        yr = groups[1]
                        period.year = int(yr) if len(yr) == 4 else 2000 + int(yr)
                        period.normalized = f"Q{period.quarter} {period.year}"
                elif period_type == 'CY':
                    period.year = int(match.group(1))
                    period.normalized = str(period.year)
                elif period_type in ('REL', 'COMP'):
                    period.normalized = raw.upper()

                parsed.periods.append(period)

    def _detect_comparison(self, text: str, parsed: ParsedQuery) -> None:
        """Detect comparison operators in the query."""
        for pattern, comp_type in self._comparison_re:
            if pattern.search(text):
                parsed.comparison_type = comp_type
                break

    def _detect_negation(self, text: str, parsed: ParsedQuery) -> None:
        """Detect negation words in the query."""
        parsed.has_negation = bool(self._negation_re.search(text))

    def _extract_entity_mentions(self, tokens: list[str],
                                  parsed: ParsedQuery) -> None:
        """Extract potential entity mentions (tickers, company names).

        Heuristic: tokens that look like tickers (1-5 uppercase letters)
        or capitalized multi-word sequences.
        """
        ticker_re = re.compile(r'^[A-Z]{1,5}$')
        for token in tokens:
            if ticker_re.match(token):
                parsed.entity_mentions.append(token)

        parsed.entity_count = len(parsed.entity_mentions)

    def _extract_metric_mentions(self, tokens: list[str],
                                  parsed: ParsedQuery) -> None:
        """Extract potential metric references from tokens."""
        # Financial metric keywords
        metric_keywords = {
            "revenue", "sales", "income", "profit", "margin", "eps",
            "ebitda", "ebit", "fcf", "cash", "debt", "equity",
            "assets", "liabilities", "roe", "roa", "roic",
            "pe", "pb", "ps", "ev", "market", "cap", "dividend",
            "yield", "growth", "cagr", "gross", "operating", "net",
        }
        for token in tokens:
            if token.lower() in metric_keywords:
                parsed.metric_mentions.append(token)
