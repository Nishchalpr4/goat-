"""Field-aware tokenization for investment documents.

Three tokenizer strategies matching the architecture:
  1. Identifier tokenizer — tickers, ISINs, CUSIPs, executive names
  2. Narrative tokenizer — MD&A, risk factors, transcripts (stemming/lemma)
  3. Schema tokenizer — controlled vocabulary metric fields (XBRL concepts)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TokenType(Enum):
    IDENTIFIER = "identifier"    # ticker, ISIN, name
    NARRATIVE = "narrative"      # prose text
    SCHEMA = "schema"            # metric/concept term
    NUMBER = "number"            # numeric value
    PERIOD = "period"            # FY2025, Q3, TTM
    CURRENCY = "currency"        # $, USD, EUR


@dataclass
class Token:
    """A single token with type and metadata."""
    text: str
    token_type: TokenType
    start: int = 0
    end: int = 0
    normalized: str = ""
    canonical_id: Optional[str] = None  # set after entity/schema resolution


@dataclass
class TokenizationResult:
    """Result of tokenizing a text with field awareness."""
    tokens: list[Token] = field(default_factory=list)
    raw_text: str = ""
    field_type: str = ""

    @property
    def identifiers(self) -> list[Token]:
        return [t for t in self.tokens if t.token_type == TokenType.IDENTIFIER]

    @property
    def periods(self) -> list[Token]:
        return [t for t in self.tokens if t.token_type == TokenType.PERIOD]

    @property
    def numbers(self) -> list[Token]:
        return [t for t in self.tokens if t.token_type == TokenType.NUMBER]


class Tokenizer:
    """Field-aware tokenizer for investment analysis text."""

    # Patterns for special token types
    _TICKER_PATTERN = re.compile(
        r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b"  # AAPL, BRK.B, TSM
    )
    _PERIOD_PATTERN = re.compile(
        r"\b(?:FY|CY|Q[1-4]|H[12]|TTM|LTM|YTD|MRQ)\s*\d{0,4}\b",
        re.IGNORECASE,
    )
    _NUMBER_PATTERN = re.compile(
        r"(?<!\w)[-+]?\$?\d[\d,]*\.?\d*[BMKbmk%]?(?!\w)"
    )
    _CURRENCY_PATTERN = re.compile(
        r"\b(?:USD|EUR|GBP|JPY|CNY|INR|KRW|CHF|CAD|AUD|HKD)\b"
    )
    _ISIN_PATTERN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    _CUSIP_PATTERN = re.compile(r"\b[A-Z0-9]{9}\b")

    # Common financial abbreviations (not to be stemmed)
    _FINANCIAL_ABBREVS = {
        "EPS", "FCF", "EBITDA", "EBIT", "ROE", "ROA", "ROIC", "ROI",
        "P/E", "PE", "P/B", "PB", "P/S", "PS", "EV", "DCF", "WACC",
        "NIM", "NPL", "CAC", "LTV", "ARR", "MRR", "GMV", "TAM",
        "SAM", "SOM", "CAGR", "YOY", "QOQ", "MOM", "BPS", "SGA",
        "CAPEX", "OPEX", "IPO", "M&A", "ESG", "GDP", "CPI", "PPI",
        "BPS", "TTM", "LTM", "MRQ", "FY", "CY", "NAV", "AUM",
    }

    # Simple stemming rules for English narrative text
    _STEM_SUFFIXES = [
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("iveness", "ive"),
        ("ously", "ous"),
        ("ation", "ate"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ally", "al"),
        ("ting", "t"),
        ("ing", ""),
        ("ied", "y"),
        ("ies", "y"),
        ("ely", "e"),
        ("ed", ""),
        ("ly", ""),
        ("er", ""),
        ("es", ""),
        ("s", ""),
    ]

    def tokenize(self, text: str, field_type: str = "narrative") -> TokenizationResult:
        """Tokenize text with awareness of field type."""
        if field_type == "identifier":
            return self._tokenize_identifier(text)
        elif field_type == "schema":
            return self._tokenize_schema(text)
        else:
            return self._tokenize_narrative(text)

    def _tokenize_identifier(self, text: str) -> TokenizationResult:
        """Tokenize identifiers: preserve punctuation variants, no stemming."""
        tokens = []
        # Check for ISIN
        for m in self._ISIN_PATTERN.finditer(text):
            tokens.append(Token(
                text=m.group(), token_type=TokenType.IDENTIFIER,
                start=m.start(), end=m.end(), normalized=m.group().upper(),
            ))
        # Check for tickers
        for m in self._TICKER_PATTERN.finditer(text):
            tokens.append(Token(
                text=m.group(), token_type=TokenType.IDENTIFIER,
                start=m.start(), end=m.end(), normalized=m.group().upper(),
            ))
        # Remaining word tokens
        for m in re.finditer(r"\S+", text):
            if not any(t.start <= m.start() < t.end for t in tokens):
                tokens.append(Token(
                    text=m.group(), token_type=TokenType.IDENTIFIER,
                    start=m.start(), end=m.end(),
                    normalized=m.group().strip().upper(),
                ))
        tokens.sort(key=lambda t: t.start)
        return TokenizationResult(tokens=tokens, raw_text=text, field_type="identifier")

    def _tokenize_schema(self, text: str) -> TokenizationResult:
        """Tokenize schema/metric fields as controlled vocabulary.

        No stemming — treat as exact canonical terms.
        """
        tokens = []
        # Lowercase, strip, split on separators
        cleaned = text.lower().strip()
        parts = re.split(r"[:/\-_\s]+", cleaned)
        offset = 0
        for part in parts:
            if part:
                tokens.append(Token(
                    text=part, token_type=TokenType.SCHEMA,
                    start=offset, end=offset + len(part),
                    normalized=part,
                ))
            offset += len(part) + 1
        return TokenizationResult(tokens=tokens, raw_text=text, field_type="schema")

    def _tokenize_narrative(self, text: str) -> TokenizationResult:
        """Tokenize narrative text with special-token detection and light stemming."""
        tokens = []
        pos = 0

        # First pass: extract special tokens (numbers, periods, currencies, abbreviations)
        special_spans = []

        for m in self._PERIOD_PATTERN.finditer(text):
            tokens.append(Token(
                text=m.group(), token_type=TokenType.PERIOD,
                start=m.start(), end=m.end(),
                normalized=m.group().upper().replace(" ", ""),
            ))
            special_spans.append((m.start(), m.end()))

        for m in self._NUMBER_PATTERN.finditer(text):
            if not any(s <= m.start() < e for s, e in special_spans):
                tokens.append(Token(
                    text=m.group(), token_type=TokenType.NUMBER,
                    start=m.start(), end=m.end(),
                    normalized=m.group().replace(",", ""),
                ))
                special_spans.append((m.start(), m.end()))

        for m in self._CURRENCY_PATTERN.finditer(text):
            if not any(s <= m.start() < e for s, e in special_spans):
                tokens.append(Token(
                    text=m.group(), token_type=TokenType.CURRENCY,
                    start=m.start(), end=m.end(),
                    normalized=m.group().upper(),
                ))
                special_spans.append((m.start(), m.end()))

        # Second pass: word tokens
        for m in re.finditer(r"\b[\w&/]+\b", text):
            if any(s <= m.start() < e for s, e in special_spans):
                continue
            word = m.group()
            upper_word = word.upper()

            if upper_word in self._FINANCIAL_ABBREVS:
                tokens.append(Token(
                    text=word, token_type=TokenType.IDENTIFIER,
                    start=m.start(), end=m.end(),
                    normalized=upper_word,
                ))
            else:
                normalized = self._simple_stem(word.lower())
                tokens.append(Token(
                    text=word, token_type=TokenType.NARRATIVE,
                    start=m.start(), end=m.end(),
                    normalized=normalized,
                ))

        tokens.sort(key=lambda t: t.start)
        return TokenizationResult(tokens=tokens, raw_text=text, field_type="narrative")

    def _simple_stem(self, word: str) -> str:
        """Very simple suffix-stripping stemmer (not Porter — intentionally light)."""
        if len(word) <= 3:
            return word
        for suffix, replacement in self._STEM_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
                return word[: -len(suffix)] + replacement
        return word
