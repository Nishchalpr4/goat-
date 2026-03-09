"""Query expander — expands user queries using lexical synonym injection,
abbreviation expansion, and schema term alignment.

Produces expanded lexical representations for BM25/FTS retrieval and
optional SPLADE-like sparse expansion hints.
"""

from dataclasses import dataclass, field
from typing import Optional

from goat.models.lexicon import Lexicon
from goat.lexical.entity_resolver import EntityResolver, ResolutionResult
from goat.lexical.schema_linker import SchemaLinker, SchemaLinkingResult
from goat.lexical.tokenizer import Tokenizer, TokenType


@dataclass
class ExpandedQuery:
    """Result of query expansion with multiple representations."""
    original_query: str
    # Expanded lexical query for BM25/FTS
    expanded_terms: list[str] = field(default_factory=list)
    expanded_query: str = ""
    # Synonym injections applied
    synonym_expansions: list[dict] = field(default_factory=list)
    # Abbreviation expansions applied
    abbreviation_expansions: list[dict] = field(default_factory=list)
    # Entity resolution results (for hard filters)
    entity_resolution: Optional[ResolutionResult] = None
    # Schema linking results (for metric alignment)
    schema_linking: Optional[SchemaLinkingResult] = None
    # Filter directives extracted
    filters: dict = field(default_factory=dict)  # company_id, sector, year, etc.

    @property
    def has_entity_filters(self) -> bool:
        return bool(self.entity_resolution and self.entity_resolution.company_ids)

    @property
    def has_metric_targets(self) -> bool:
        return bool(self.schema_linking and self.schema_linking.links)


class QueryExpander:
    """Expands user queries using lexicon, entity resolution, and schema linking."""

    def __init__(
        self,
        lexicon: Lexicon,
        entity_resolver: Optional[EntityResolver] = None,
        schema_linker: Optional[SchemaLinker] = None,
    ):
        self.lexicon = lexicon
        self.tokenizer = Tokenizer()
        self.entity_resolver = entity_resolver or EntityResolver(lexicon)
        self.schema_linker = schema_linker or SchemaLinker(lexicon)

    def expand(self, query: str) -> ExpandedQuery:
        """Full query expansion pipeline."""
        result = ExpandedQuery(original_query=query)

        # Step 1: Tokenize the query
        tokenization = self.tokenizer.tokenize(query, field_type="narrative")
        terms = []

        # Step 2: Collect identifier tokens for entity resolution
        identifier_tokens = [
            t.text for t in tokenization.tokens
            if t.token_type == TokenType.IDENTIFIER
        ]

        # Step 3: Entity resolution
        if identifier_tokens:
            result.entity_resolution = self.entity_resolver.resolve_batch(identifier_tokens)
            # Elevate resolved companies to filters
            if result.entity_resolution.company_ids:
                result.filters["company_id"] = result.entity_resolution.company_ids

        # Step 4: Extract period filters
        period_tokens = [t.normalized for t in tokenization.periods]
        if period_tokens:
            result.filters["periods"] = period_tokens

        # Step 5: Abbreviation expansion
        for token in tokenization.tokens:
            if token.token_type == TokenType.IDENTIFIER:
                abbr = self.lexicon.expand_abbreviation(token.text)
                if abbr:
                    result.abbreviation_expansions.append({
                        "original": token.text,
                        "expansion": abbr.expansion,
                        "metric_id": abbr.canonical_metric_id,
                    })
                    terms.append(token.text)
                    terms.append(abbr.expansion)
                else:
                    terms.append(token.text)
            elif token.token_type == TokenType.NARRATIVE:
                terms.append(token.text)

        # Step 6: Schema linking for metric-related terms
        narrative_terms = [
            t.text for t in tokenization.tokens
            if t.token_type == TokenType.NARRATIVE
        ]
        # Also try multi-word phrases (2-grams and 3-grams)
        text_parts = query.lower().split()
        phrases = []
        for n in (2, 3):
            for i in range(len(text_parts) - n + 1):
                phrases.append(" ".join(text_parts[i : i + n]))

        schema_candidates = narrative_terms + phrases
        if schema_candidates:
            result.schema_linking = self.schema_linker.link_batch(schema_candidates)
            # Add concept labels as expansion terms
            for link in result.schema_linking.links:
                if link.concept_label not in terms:
                    terms.append(link.concept_label)
                    result.synonym_expansions.append({
                        "original": link.user_term,
                        "expansion": link.concept_label,
                        "concept": link.canonical_concept,
                    })

        # Step 7: Synonym injection from lexicon
        for token in tokenization.tokens:
            if token.token_type == TokenType.NARRATIVE:
                entries = self.lexicon.lookup_surface_form(token.text)
                for entry in entries:
                    synonyms = self.lexicon.lookup_canonical(entry.canonical_id)
                    for syn in synonyms:
                        if syn.surface_form.lower() != token.text.lower():
                            if syn.surface_form not in terms:
                                terms.append(syn.surface_form)
                                result.synonym_expansions.append({
                                    "original": token.text,
                                    "expansion": syn.surface_form,
                                    "canonical_id": entry.canonical_id,
                                })

        result.expanded_terms = terms
        result.expanded_query = " ".join(terms)
        return result
