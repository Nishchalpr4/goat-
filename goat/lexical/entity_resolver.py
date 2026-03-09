"""Entity resolver — maps noisy surface forms (tickers, company names,
abbreviations) to canonical entity IDs using lexicon lookup + trigram
similarity for fuzzy matching.

Primary resolution chain:
  1. Exact match in lexicon (case-insensitive)
  2. Trigram similarity against ticker variants
  3. Trigram similarity against company aliases
  4. Return unresolved token for downstream handling
"""

from dataclasses import dataclass, field
from typing import Optional

from goat.models.lexicon import Lexicon, TickerMapping, LexiconEntry


@dataclass
class ResolvedEntity:
    """Result of resolving a surface form to a canonical entity."""
    surface_form: str
    canonical_id: str
    canonical_label: str
    resolution_method: str  # "exact_ticker", "exact_alias", "trigram_ticker", "trigram_alias"
    confidence: float = 1.0
    match_score: float = 1.0  # trigram similarity when applicable
    entry_type: str = ""


@dataclass
class ResolutionResult:
    """Batch resolution result for a query or document."""
    resolved: list[ResolvedEntity] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    # Resolved entities promoted to hard filters
    company_ids: list[str] = field(default_factory=list)
    metric_ids: list[str] = field(default_factory=list)


def trigram_set(s: str) -> set[str]:
    """Compute the set of trigrams for a string (pg_trgm compatible)."""
    padded = f"  {s.lower()} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    """Compute trigram similarity between two strings (equivalent to pg_trgm similarity())."""
    ta = trigram_set(a)
    tb = trigram_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


class EntityResolver:
    """Resolves surface forms to canonical entities using lexicon + trigram matching."""

    def __init__(self, lexicon: Lexicon, similarity_threshold: float = 0.3):
        self.lexicon = lexicon
        self.similarity_threshold = similarity_threshold

    def resolve(self, surface_form: str) -> Optional[ResolvedEntity]:
        """Resolve a single surface form to a canonical entity."""
        cleaned = surface_form.strip()
        if not cleaned:
            return None

        # 1. Exact ticker match
        ticker_match = self.lexicon.resolve_ticker(cleaned)
        if ticker_match:
            return ResolvedEntity(
                surface_form=cleaned,
                canonical_id=ticker_match.canonical_id,
                canonical_label=ticker_match.company_name,
                resolution_method="exact_ticker",
                confidence=1.0,
                match_score=1.0,
                entry_type="ticker",
            )

        # 2. Exact lexicon entry match
        entries = self.lexicon.lookup_surface_form(cleaned)
        if entries:
            best = max(entries, key=lambda e: e.confidence)
            return ResolvedEntity(
                surface_form=cleaned,
                canonical_id=best.canonical_id,
                canonical_label=best.canonical_label,
                resolution_method="exact_alias",
                confidence=best.confidence,
                match_score=1.0,
                entry_type=best.entry_type.value,
            )

        # 3. Trigram fuzzy match against ticker variants
        best_ticker = self._fuzzy_ticker_match(cleaned)
        if best_ticker:
            return best_ticker

        # 4. Trigram fuzzy match against lexicon entries
        best_alias = self._fuzzy_alias_match(cleaned)
        if best_alias:
            return best_alias

        return None

    def resolve_batch(self, surface_forms: list[str]) -> ResolutionResult:
        """Resolve multiple surface forms, returning a combined result."""
        result = ResolutionResult()
        seen_company_ids = set()
        seen_metric_ids = set()

        for sf in surface_forms:
            resolved = self.resolve(sf)
            if resolved:
                result.resolved.append(resolved)
                if resolved.entry_type in ("ticker", "company_alias"):
                    if resolved.canonical_id not in seen_company_ids:
                        result.company_ids.append(resolved.canonical_id)
                        seen_company_ids.add(resolved.canonical_id)
                elif resolved.entry_type in ("abbreviation", "xbrl_concept", "metric_alias"):
                    if resolved.canonical_id not in seen_metric_ids:
                        result.metric_ids.append(resolved.canonical_id)
                        seen_metric_ids.add(resolved.canonical_id)
            else:
                result.unresolved.append(sf)

        return result

    def _fuzzy_ticker_match(self, surface: str) -> Optional[ResolvedEntity]:
        """Trigram similarity search against all ticker variants."""
        best_score = 0.0
        best_match: Optional[TickerMapping] = None

        for tm in self.lexicon.ticker_mappings:
            candidates = [tm.ticker] + tm.variants
            for variant in candidates:
                sim = trigram_similarity(surface, variant)
                if sim > best_score:
                    best_score = sim
                    best_match = tm

        if best_match and best_score >= self.similarity_threshold:
            return ResolvedEntity(
                surface_form=surface,
                canonical_id=best_match.canonical_id,
                canonical_label=best_match.company_name,
                resolution_method="trigram_ticker",
                confidence=best_score,
                match_score=best_score,
                entry_type="ticker",
            )
        return None

    def _fuzzy_alias_match(self, surface: str) -> Optional[ResolvedEntity]:
        """Trigram similarity search against all lexicon surface forms."""
        best_score = 0.0
        best_entry: Optional[LexiconEntry] = None

        for entry in self.lexicon.entries:
            sim = trigram_similarity(surface, entry.surface_form)
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            return ResolvedEntity(
                surface_form=surface,
                canonical_id=best_entry.canonical_id,
                canonical_label=best_entry.canonical_label,
                resolution_method="trigram_alias",
                confidence=best_score * best_entry.confidence,
                match_score=best_score,
                entry_type=best_entry.entry_type.value,
            )
        return None
