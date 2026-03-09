"""Document chunker — splits documents into retrieval-ready chunks with
structural segmentation and contextual metadata.

Respects section boundaries (e.g., "Risk Factors", "MD&A") and maintains
chunk overlap for continuity.
"""

import re
from dataclasses import dataclass
from typing import Optional
import uuid

from goat.config import ChunkingConfig
from goat.models.document import Document, Chunk


# Common section patterns in SEC filings
_SECTION_PATTERNS = [
    (re.compile(r"(?:Item|ITEM)\s+1[A.]?\s*[-–—:]?\s*Risk\s+Factors", re.IGNORECASE), "Risk Factors"),
    (re.compile(r"(?:Item|ITEM)\s+7[A.]?\s*[-–—:]?\s*Management.?s\s+Discussion", re.IGNORECASE), "MD&A"),
    (re.compile(r"(?:Item|ITEM)\s+1\s*[-–—:]?\s*Business", re.IGNORECASE), "Business"),
    (re.compile(r"(?:Item|ITEM)\s+2\s*[-–—:]?\s*Properties", re.IGNORECASE), "Properties"),
    (re.compile(r"(?:Item|ITEM)\s+3\s*[-–—:]?\s*Legal", re.IGNORECASE), "Legal Proceedings"),
    (re.compile(r"(?:Item|ITEM)\s+5\s*[-–—:]?\s*Market", re.IGNORECASE), "Market for Stock"),
    (re.compile(r"(?:Item|ITEM)\s+6\s*[-–—:]?\s*Selected\s+Financial", re.IGNORECASE), "Selected Financial Data"),
    (re.compile(r"(?:Item|ITEM)\s+8\s*[-–—:]?\s*Financial\s+Statements", re.IGNORECASE), "Financial Statements"),
    (re.compile(r"(?:Item|ITEM)\s+9\s*[-–—:]?\s*Changes", re.IGNORECASE), "Changes and Disagreements"),
    (re.compile(r"Forward[- ]Looking\s+Statements", re.IGNORECASE), "Forward-Looking Statements"),
    (re.compile(r"(?:Notes?\s+to|NOTES?\s+TO)\s+(?:Consolidated\s+)?Financial\s+Statements", re.IGNORECASE), "Notes to Financial Statements"),
    # Earnings transcript sections
    (re.compile(r"Prepared\s+Remarks", re.IGNORECASE), "Prepared Remarks"),
    (re.compile(r"(?:Q(?:uestion)?[&\s]+A(?:nswer)?|Q&A)\s*(?:Session)?", re.IGNORECASE), "Q&A Session"),
    (re.compile(r"Opening\s+Remarks", re.IGNORECASE), "Opening Remarks"),
    (re.compile(r"(?:Closing|Concluding)\s+Remarks", re.IGNORECASE), "Closing Remarks"),
]


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3 for subword tokenization)."""
    return int(len(text.split()) * 1.3)


class DocumentChunker:
    """Chunks documents with structural awareness and overlap."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into retrieval-ready chunks."""
        if self.config.structural_segmentation:
            return self._structural_chunk(document)
        else:
            return self._simple_chunk(document)

    def _structural_chunk(self, document: Document) -> list[Chunk]:
        """Chunk with section boundary awareness."""
        text = document.content
        sections = self._detect_sections(text)

        if not sections:
            # No sections detected, fall back to simple chunking
            return self._simple_chunk(document)

        chunks = []
        chunk_index = 0

        for section_name, start, end in sections:
            section_text = text[start:end].strip()
            if not section_text:
                continue

            # Split this section into chunks
            section_chunks = self._split_text(
                section_text, section_name, document,
                base_offset=start, start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _simple_chunk(self, document: Document) -> list[Chunk]:
        """Simple fixed-size chunking with overlap."""
        return self._split_text(
            document.content, document.metadata.section or "",
            document, base_offset=0, start_index=0,
        )

    def _split_text(self, text: str, section: str, document: Document,
                    base_offset: int = 0, start_index: int = 0) -> list[Chunk]:
        """Split text into chunks of target token size with overlap."""
        words = text.split()
        if not words:
            return []

        # Convert token counts to word counts (rough inverse of _estimate_tokens)
        words_per_chunk = max(1, int(self.config.chunk_size / 1.3))
        overlap_words = max(0, int(self.config.chunk_overlap / 1.3))
        min_words = max(1, int(self.config.min_chunk_size / 1.3))

        chunks = []
        idx = 0
        chunk_number = start_index

        while idx < len(words):
            end = min(idx + words_per_chunk, len(words))
            chunk_words = words[idx:end]

            if len(chunk_words) < min_words and chunks:
                # Too small — append to previous chunk
                break

            chunk_text = " ".join(chunk_words)
            # Calculate byte offsets in original text
            start_pos = text.find(chunk_words[0], max(0, base_offset + idx - 1))
            if start_pos == -1:
                start_pos = base_offset

            chunk = Chunk(
                chunk_id=f"{document.doc_id}:chunk:{chunk_number}",
                doc_id=document.doc_id,
                text=chunk_text,
                start_offset=base_offset + start_pos,
                end_offset=base_offset + start_pos + len(chunk_text),
                chunk_index=chunk_number,
                section=section,
                company_id=document.metadata.company_id,
                company_name="",  # set by caller
                ticker="",  # set by caller
                sector="",  # set by caller
                doc_type=document.metadata.doc_type,
                ingestion_run_id=document.ingestion_run_id,
            )
            chunks.append(chunk)
            chunk_number += 1

            idx = end - overlap_words if end < len(words) else end

        return chunks

    def _detect_sections(self, text: str) -> list[tuple[str, int, int]]:
        """Detect section boundaries in document text."""
        matches = []
        for pattern, section_name in _SECTION_PATTERNS:
            for m in pattern.finditer(text):
                matches.append((section_name, m.start()))

        if not matches:
            return []

        # Sort by position
        matches.sort(key=lambda x: x[1])

        # Convert to (name, start, end) tuples
        sections = []
        for i, (name, start) in enumerate(matches):
            end = matches[i + 1][1] if i + 1 < len(matches) else len(text)
            sections.append((name, start, end))

        return sections
