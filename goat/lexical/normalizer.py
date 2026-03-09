"""Text normalization for investment-analysis documents and queries.

Handles: encoding normalization, language detection, case folding,
whitespace/punctuation cleanup, and field-type-aware processing.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizationResult:
    """Result of normalizing a text string."""
    original: str
    normalized: str
    language: str
    encoding_issues: list[str]


class TextNormalizer:
    """Language-aware text normalizer for investment documents and queries."""

    # Common Unicode replacements in financial docs
    _UNICODE_REPLACEMENTS = {
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\u200e": "",    # left-to-right mark
        "\u200f": "",    # right-to-left mark
        "\ufeff": "",    # BOM
    }

    # Currency symbols to preserve
    _CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹", "₩", "₽", "₪", "₫", "₿"}

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language

    def normalize(self, text: str, language: Optional[str] = None) -> NormalizationResult:
        """Full normalization pipeline for a text string."""
        issues = []
        lang = language or self.default_language

        # Step 1: Unicode normalization (NFC)
        normalized = unicodedata.normalize("NFC", text)

        # Step 2: Replace known problematic Unicode characters
        for char, replacement in self._UNICODE_REPLACEMENTS.items():
            if char in normalized:
                normalized = normalized.replace(char, replacement)

        # Step 3: Remove control characters (keep newlines, tabs)
        cleaned = []
        for ch in normalized:
            if unicodedata.category(ch).startswith("C") and ch not in ("\n", "\r", "\t"):
                issues.append(f"removed control char U+{ord(ch):04X}")
            else:
                cleaned.append(ch)
        normalized = "".join(cleaned)

        # Step 4: Normalize whitespace within lines
        normalized = re.sub(r"[^\S\n]+", " ", normalized)

        # Step 5: Collapse excessive newlines
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        normalized = normalized.strip()

        return NormalizationResult(
            original=text,
            normalized=normalized,
            language=lang,
            encoding_issues=issues,
        )

    def normalize_identifier(self, identifier: str) -> str:
        """Normalize a ticker/ISIN/CUSIP-style identifier.

        Preserves dots and hyphens (significant in tickers like BRK.B),
        strips surrounding whitespace, uppercases.
        """
        cleaned = identifier.strip().upper()
        # Remove surrounding brackets/parens often seen in source data
        cleaned = re.sub(r"^[\[\(]+|[\]\)]+$", "", cleaned)
        return cleaned

    def normalize_metric_name(self, name: str) -> str:
        """Normalize a financial metric name to a canonical form.

        "Net  Income (Loss)" → "net income loss"
        "EBITDA (adj.)" → "ebitda adj"
        """
        lowered = name.lower().strip()
        # Remove parentheses but keep content
        lowered = re.sub(r"[()]", " ", lowered)
        # Remove dots that aren't decimal points
        lowered = re.sub(r"\.(?!\d)", " ", lowered)
        # Collapse whitespace
        return re.sub(r"\s+", " ", lowered).strip()

    def fold_case(self, text: str, preserve_identifiers: bool = True) -> str:
        """Case-fold text while optionally preserving uppercase identifiers.

        If preserve_identifiers=True, tokens that are ALL CAPS and <= 6 chars
        (likely tickers or abbreviations) are kept uppercase.
        """
        if not preserve_identifiers:
            return text.lower()

        tokens = text.split()
        result = []
        for token in tokens:
            stripped = token.strip(".,;:!?()[]{}\"'")
            if stripped.isupper() and len(stripped) <= 6 and stripped.isalpha():
                result.append(token)  # keep as-is (likely abbreviation/ticker)
            else:
                result.append(token.lower())
        return " ".join(result)
