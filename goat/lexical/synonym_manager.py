"""Synonym manager — manages the versioned lexicon of synonyms, aliases,
abbreviations, multilingual variants, and domain-specific terms.

Supports:
  - CRUD operations on lexicon entries
  - Versioned snapshots for reproducibility
  - Import from structured sources (XBRL taxonomy, CSV)
  - Bidirectional synonym expansion
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path

from goat.models.lexicon import (
    Lexicon, LexiconEntry, LexiconEntryType,
    TickerMapping, AbbreviationEntry, SchemaTermMapping,
)


class SynonymManager:
    """Manages the versioned lexicon of synonyms and aliases."""

    def __init__(self, lexicon: Optional[Lexicon] = None):
        self.lexicon = lexicon or Lexicon(version="v1")

    def add_entry(
        self,
        surface_form: str,
        canonical_id: str,
        canonical_label: str,
        entry_type: LexiconEntryType = LexiconEntryType.SYNONYM,
        language: str = "en",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> LexiconEntry:
        """Add a new synonym/alias entry."""
        entry = LexiconEntry(
            entry_id=f"{self.lexicon.version}:{canonical_id}:{surface_form}",
            surface_form=surface_form,
            canonical_id=canonical_id,
            canonical_label=canonical_label,
            entry_type=entry_type,
            language=language,
            confidence=confidence,
            source=source,
            version=self.lexicon.version,
        )
        self.lexicon.entries.append(entry)
        return entry

    def add_ticker(
        self,
        canonical_id: str,
        ticker: str,
        company_name: str,
        exchange: str = "",
        variants: Optional[list[str]] = None,
    ) -> TickerMapping:
        """Add a ticker mapping with its variants."""
        mapping = TickerMapping(
            canonical_id=canonical_id,
            ticker=ticker,
            exchange=exchange,
            variants=variants or [],
            company_name=company_name,
        )
        self.lexicon.ticker_mappings.append(mapping)
        # Also add as lexicon entries for general lookup
        self.add_entry(
            surface_form=ticker,
            canonical_id=canonical_id,
            canonical_label=company_name,
            entry_type=LexiconEntryType.TICKER,
        )
        for variant in (variants or []):
            self.add_entry(
                surface_form=variant,
                canonical_id=canonical_id,
                canonical_label=company_name,
                entry_type=LexiconEntryType.TICKER,
                confidence=0.95,
            )
        return mapping

    def add_abbreviation(
        self,
        abbreviation: str,
        expansion: str,
        canonical_metric_id: str = "",
        category: str = "",
    ) -> AbbreviationEntry:
        """Add a financial abbreviation."""
        entry = AbbreviationEntry(
            abbreviation=abbreviation,
            expansion=expansion,
            canonical_metric_id=canonical_metric_id,
            category=category,
        )
        self.lexicon.abbreviations.append(entry)
        # Also add as lexicon entry
        if canonical_metric_id:
            self.add_entry(
                surface_form=abbreviation,
                canonical_id=canonical_metric_id,
                canonical_label=expansion,
                entry_type=LexiconEntryType.ABBREVIATION,
            )
        return entry

    def add_schema_mapping(
        self,
        surface_form: str,
        canonical_concept: str,
        taxonomy: str = "us-gaap",
        concept_label: str = "",
        language: str = "en",
    ) -> SchemaTermMapping:
        """Add a schema term mapping (typically from XBRL taxonomy)."""
        mapping = SchemaTermMapping(
            surface_form=surface_form,
            canonical_concept=canonical_concept,
            taxonomy=taxonomy,
            concept_label=concept_label or surface_form,
            language=language,
        )
        self.lexicon.schema_mappings.append(mapping)
        return mapping

    def get_synonyms(self, canonical_id: str) -> list[str]:
        """Get all surface forms for a canonical entity/concept."""
        return [e.surface_form for e in self.lexicon.lookup_canonical(canonical_id)]

    def bump_version(self, new_version: str) -> None:
        """Create a new version of the lexicon."""
        self.lexicon.version = new_version
        self.lexicon.created_at = datetime.utcnow()

    def export_json(self, path: Path) -> None:
        """Export the lexicon to a JSON file for snapshotting."""
        data = {
            "version": self.lexicon.version,
            "created_at": self.lexicon.created_at.isoformat(),
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "surface_form": e.surface_form,
                    "canonical_id": e.canonical_id,
                    "canonical_label": e.canonical_label,
                    "entry_type": e.entry_type.value,
                    "language": e.language,
                    "confidence": e.confidence,
                    "source": e.source,
                }
                for e in self.lexicon.entries
            ],
            "ticker_mappings": [
                {
                    "canonical_id": t.canonical_id,
                    "ticker": t.ticker,
                    "exchange": t.exchange,
                    "variants": t.variants,
                    "company_name": t.company_name,
                }
                for t in self.lexicon.ticker_mappings
            ],
            "abbreviations": [
                {
                    "abbreviation": a.abbreviation,
                    "expansion": a.expansion,
                    "canonical_metric_id": a.canonical_metric_id,
                    "category": a.category,
                }
                for a in self.lexicon.abbreviations
            ],
            "schema_mappings": [
                {
                    "surface_form": s.surface_form,
                    "canonical_concept": s.canonical_concept,
                    "taxonomy": s.taxonomy,
                    "concept_label": s.concept_label,
                    "language": s.language,
                }
                for s in self.lexicon.schema_mappings
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_json(self, path: Path) -> None:
        """Import a lexicon snapshot from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.lexicon.version = data.get("version", "v1")

        for e in data.get("entries", []):
            self.lexicon.entries.append(LexiconEntry(
                entry_id=e["entry_id"],
                surface_form=e["surface_form"],
                canonical_id=e["canonical_id"],
                canonical_label=e["canonical_label"],
                entry_type=LexiconEntryType(e["entry_type"]),
                language=e.get("language", "en"),
                confidence=e.get("confidence", 1.0),
                source=e.get("source", "imported"),
                version=self.lexicon.version,
            ))

        for t in data.get("ticker_mappings", []):
            self.lexicon.ticker_mappings.append(TickerMapping(
                canonical_id=t["canonical_id"],
                ticker=t["ticker"],
                exchange=t.get("exchange", ""),
                variants=t.get("variants", []),
                company_name=t.get("company_name", ""),
            ))

        for a in data.get("abbreviations", []):
            self.lexicon.abbreviations.append(AbbreviationEntry(
                abbreviation=a["abbreviation"],
                expansion=a["expansion"],
                canonical_metric_id=a.get("canonical_metric_id", ""),
                category=a.get("category", ""),
            ))

        for s in data.get("schema_mappings", []):
            self.lexicon.schema_mappings.append(SchemaTermMapping(
                surface_form=s["surface_form"],
                canonical_concept=s["canonical_concept"],
                taxonomy=s.get("taxonomy", "us-gaap"),
                concept_label=s.get("concept_label", ""),
                language=s.get("language", "en"),
            ))

    def load_xbrl_taxonomy(self, concepts: list[dict]) -> int:
        """Load schema mappings from XBRL taxonomy concept definitions.

        Each concept dict should have: name, label, definition (optional).
        Example: {"name": "us-gaap:Revenues", "label": "Revenues", "definition": "..."}
        """
        count = 0
        for concept in concepts:
            name = concept.get("name", "")
            label = concept.get("label", "")
            if name and label:
                self.add_schema_mapping(
                    surface_form=label.lower(),
                    canonical_concept=name,
                    concept_label=label,
                    taxonomy=name.split(":")[0] if ":" in name else "custom",
                )
                count += 1
                # Also add common variations
                if "," in label:
                    # "Revenue, Net" → also map "net revenue"
                    reversed_label = " ".join(reversed(label.split(", ")))
                    self.add_schema_mapping(
                        surface_form=reversed_label.lower(),
                        canonical_concept=name,
                        concept_label=label,
                        taxonomy=name.split(":")[0] if ":" in name else "custom",
                    )
                    count += 1
        return count
