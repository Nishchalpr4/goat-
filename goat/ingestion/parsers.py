"""Document parsers for various filing/transcript/news formats.

Handles:
  - SEC filings (10-K, 10-Q, 8-K) — plaintext and XBRL
  - Earnings call transcripts
  - News articles
  - Analyst reports
  - CSV/JSON structured data
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import re
import json
import csv
import io
import logging

from goat.models.document import Document, DocumentType, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of document parsing."""
    document: Document
    sections: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    metadata_extracted: dict = field(default_factory=dict)


# SEC filing section patterns
_10K_SECTIONS = {
    "item_1": r"(?:ITEM\s*1[.\s]|Part\s+I\b).*?Business",
    "item_1a": r"ITEM\s*1A[.\s].*?Risk\s*Factors",
    "item_2": r"ITEM\s*2[.\s].*?Properties",
    "item_3": r"ITEM\s*3[.\s].*?Legal\s*Proceedings",
    "item_5": r"ITEM\s*5[.\s].*?Market.*?(?:Registrant|Equity)",
    "item_6": r"ITEM\s*6[.\s].*?(?:Selected|Reserved)",
    "item_7": r"ITEM\s*7[.\s].*?(?:MD&?A|Management.*?Discussion)",
    "item_7a": r"ITEM\s*7A[.\s].*?(?:Quantitative|Market\s*Risk)",
    "item_8": r"ITEM\s*8[.\s].*?Financial\s*Statements",
    "item_9": r"ITEM\s*9[.\s].*?(?:Changes|Disagreements)",
}

# Earnings transcript section patterns
_TRANSCRIPT_SECTIONS = {
    "prepared_remarks": r"(?:prepared\s+remarks|opening\s+remarks|presentation)",
    "ceo_remarks": r"(?:CEO|Chief\s+Executive\s+Officer).*?(?:remarks|comments)",
    "cfo_remarks": r"(?:CFO|Chief\s+Financial\s+Officer).*?(?:remarks|comments)",
    "qa_session": r"(?:Q(?:uestion)?[\s&]+A(?:nswer)?|question.and.answer)\s*(?:session)?",
}


class BaseParser:
    """Base class for document parsers."""

    def parse(self, content: str, metadata: Optional[dict] = None) -> ParseResult:
        raise NotImplementedError

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning shared across parsers."""
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        # Remove form feeds and other control chars
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()


class SECFilingParser(BaseParser):
    """Parser for SEC filings (10-K, 10-Q, 8-K)."""

    def parse(self, content: str, metadata: Optional[dict] = None) -> ParseResult:
        metadata = metadata or {}
        cleaned = self._clean_text(content)

        doc_type = self._detect_filing_type(cleaned, metadata)
        sections = self._extract_sections(cleaned, doc_type)

        doc = Document(
            doc_id=metadata.get("doc_id", ""),
            content=cleaned,
            metadata=DocumentMetadata(
                doc_type=doc_type,
                company_id=metadata.get("company_id", ""),
                ticker=metadata.get("ticker", ""),
                filing_date=metadata.get("filing_date", ""),
                period=metadata.get("period", ""),
                source_url=metadata.get("source_url", ""),
            ),
        )

        return ParseResult(
            document=doc,
            sections=sections,
            metadata_extracted={"doc_type": doc_type.value,
                                "section_count": len(sections)},
        )

    def _detect_filing_type(self, content: str,
                             metadata: dict) -> DocumentType:
        """Detect SEC filing type from content or metadata."""
        if "doc_type" in metadata:
            try:
                return DocumentType(metadata["doc_type"])
            except ValueError:
                pass

        content_lower = content[:2000].lower()
        if "annual report" in content_lower or "form 10-k" in content_lower:
            return DocumentType.FILING_10K
        elif "quarterly report" in content_lower or "form 10-q" in content_lower:
            return DocumentType.FILING_10Q
        elif "form 8-k" in content_lower:
            return DocumentType.FILING_8K
        return DocumentType.FILING_10K

    def _extract_sections(self, content: str,
                           doc_type: DocumentType) -> list[dict]:
        """Extract named sections from a filing."""
        patterns = _10K_SECTIONS if doc_type == DocumentType.FILING_10K else {}
        sections = []

        for section_name, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.start()
                sections.append({
                    "name": section_name,
                    "start": start,
                    "header": match.group(0)[:100],
                })

        # Sort by position and assign end boundaries
        sections.sort(key=lambda s: s["start"])
        for i, section in enumerate(sections):
            end = sections[i + 1]["start"] if i + 1 < len(sections) else len(content)
            section["end"] = end
            section["content"] = content[section["start"]:end]
            section["length"] = end - section["start"]

        return sections


class TranscriptParser(BaseParser):
    """Parser for earnings call transcripts."""

    def parse(self, content: str, metadata: Optional[dict] = None) -> ParseResult:
        metadata = metadata or {}
        cleaned = self._clean_text(content)

        sections = self._extract_sections(cleaned)
        speakers = self._extract_speakers(cleaned)

        doc = Document(
            doc_id=metadata.get("doc_id", ""),
            content=cleaned,
            metadata=DocumentMetadata(
                doc_type=DocumentType.EARNINGS_TRANSCRIPT,
                company_id=metadata.get("company_id", ""),
                ticker=metadata.get("ticker", ""),
                filing_date=metadata.get("date", ""),
                period=metadata.get("period", ""),
            ),
        )

        return ParseResult(
            document=doc,
            sections=sections,
            metadata_extracted={"speakers": speakers,
                                "section_count": len(sections)},
        )

    def _extract_sections(self, content: str) -> list[dict]:
        """Extract sections from transcript."""
        sections = []
        for section_name, pattern in _TRANSCRIPT_SECTIONS.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                sections.append({
                    "name": section_name,
                    "start": match.start(),
                    "header": match.group(0)[:100],
                })

        sections.sort(key=lambda s: s["start"])
        for i, section in enumerate(sections):
            end = sections[i + 1]["start"] if i + 1 < len(sections) else len(content)
            section["end"] = end
            section["content"] = content[section["start"]:end]

        return sections

    def _extract_speakers(self, content: str) -> list[str]:
        """Extract speaker names from transcript."""
        # Common patterns: "John Smith -- CEO" or "John Smith - CEO"
        pattern = re.compile(
            r'^([A-Z][a-z]+ [A-Z][a-z]+)\s*[-–—]+\s*(.+)$',
            re.MULTILINE,
        )
        speakers = []
        seen = set()
        for match in pattern.finditer(content):
            name = match.group(1).strip()
            if name not in seen:
                seen.add(name)
                speakers.append(name)
        return speakers


class NewsParser(BaseParser):
    """Parser for news articles and press releases."""

    def parse(self, content: str, metadata: Optional[dict] = None) -> ParseResult:
        metadata = metadata or {}
        cleaned = self._clean_text(content)

        doc = Document(
            doc_id=metadata.get("doc_id", ""),
            content=cleaned,
            metadata=DocumentMetadata(
                doc_type=DocumentType.NEWS_ARTICLE,
                company_id=metadata.get("company_id", ""),
                ticker=metadata.get("ticker", ""),
                filing_date=metadata.get("date", ""),
                source_url=metadata.get("url", ""),
            ),
        )

        return ParseResult(
            document=doc,
            sections=[{"name": "full", "content": cleaned}],
        )


class StructuredDataParser(BaseParser):
    """Parser for CSV/JSON structured financial data."""

    def parse_json(self, content: str,
                   metadata: Optional[dict] = None) -> list[dict]:
        """Parse JSON data (company lists, financial data)."""
        data = json.loads(content)
        if isinstance(data, dict):
            return [data]
        return data

    def parse_csv(self, content: str,
                  metadata: Optional[dict] = None) -> list[dict]:
        """Parse CSV data into list of dicts."""
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)

    def parse(self, content: str, metadata: Optional[dict] = None) -> ParseResult:
        """Parse structured data — returns empty ParseResult, use parse_json/parse_csv."""
        return ParseResult(
            document=Document(doc_id="", content=content,
                              metadata=DocumentMetadata()),
        )


def create_parser(doc_type: str) -> BaseParser:
    """Factory for document parsers."""
    parsers = {
        "10-K": SECFilingParser,
        "10-Q": SECFilingParser,
        "8-K": SECFilingParser,
        "sec_filing": SECFilingParser,
        "transcript": TranscriptParser,
        "earnings_transcript": TranscriptParser,
        "news": NewsParser,
        "news_article": NewsParser,
    }
    parser_cls = parsers.get(doc_type, NewsParser)
    return parser_cls()
