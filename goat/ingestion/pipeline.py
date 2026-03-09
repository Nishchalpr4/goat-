"""Ingestion pipeline orchestrator — coordinates parsing, chunking,
embedding, and indexing of documents into all storage backends.

Flow:
  1. Parse document (parsers)
  2. Validate (validator)
  3. Chunk with structural awareness (chunker)
  4. Generate dual embeddings (embedding manager)
  5. Index into vector store (narrative + entity collections)
  6. Index into Postgres FTS (chunks table with tsvector)
  7. Build graph nodes/edges (graph builder)
  8. Record provenance (provenance store)
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
import uuid
import time
import logging

from goat.config import Config, default_config
from goat.models.document import Document, DocumentMetadata, DocumentType, Chunk
from goat.models.company import Company
from goat.models.provenance import ProvenanceRecord, ProvenanceEntity, ProvenanceActivity
from goat.ingestion.parsers import create_parser, ParseResult
from goat.ingestion.validator import DataValidator, ValidationResult
from goat.ingestion.loader import DataLoader
from goat.embeddings.chunker import DocumentChunker
from goat.embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    graph_nodes_created: int = 0
    total_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline(config)
        pipeline.set_embedding_manager(embed_mgr)
        pipeline.set_vector_store(vector_store)
        pipeline.set_postgres(pg_store)
        pipeline.set_graph_builder(graph_builder)

        stats = pipeline.ingest_document(content, metadata)
        stats = pipeline.ingest_file(path, metadata)
        stats = pipeline.ingest_company_batch(companies_json_path)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.validator = DataValidator()
        self.chunker = DocumentChunker(config=self.config.ingestion.chunking)
        self.loader = DataLoader()
        # These are set via dependency injection
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._vector_store = None
        self._postgres = None
        self._graph_builder = None

    def set_embedding_manager(self, manager: EmbeddingManager) -> None:
        self._embedding_manager = manager

    def set_vector_store(self, store) -> None:
        self._vector_store = store

    def set_postgres(self, store) -> None:
        self._postgres = store

    def set_graph_builder(self, builder) -> None:
        self._graph_builder = builder

    def ingest_document(self, content: str,
                         metadata: Optional[dict] = None) -> IngestionStats:
        """Ingest a single document from raw text."""
        start = time.perf_counter()
        stats = IngestionStats()
        metadata = metadata or {}

        try:
            # Step 1: Parse
            doc_type = metadata.get("doc_type", "news")
            parser = create_parser(doc_type)
            parse_result = parser.parse(content, metadata)

            # Assign doc_id if not present
            if not parse_result.document.doc_id:
                parse_result.document.doc_id = uuid.uuid4().hex[:16]

            # Step 2: Validate
            validation = self.validator.validate_document(parse_result.document)
            if not validation.valid:
                for issue in validation.issues:
                    if issue.severity == "error":
                        stats.errors.append(f"{issue.field}: {issue.message}")
                stats.documents_failed += 1
                return stats
            for issue in validation.issues:
                if issue.severity == "warning":
                    stats.warnings.append(f"{issue.field}: {issue.message}")

            # Step 3: Chunk
            chunks = self._create_chunks(parse_result)
            stats.chunks_created = len(chunks)

            # Step 4: Generate embeddings
            if self._embedding_manager and chunks:
                self._embed_chunks(chunks, stats)

            # Step 5-6: Index into stores
            if self._vector_store and chunks:
                self._index_vectors(chunks, stats)

            if self._postgres and chunks:
                self._index_postgres(chunks, parse_result.document, stats)

            # Step 7: Build graph
            if self._graph_builder:
                self._build_graph(parse_result, stats)

            stats.documents_processed = 1

        except Exception as e:
            logger.exception("Ingestion failed: %s", e)
            stats.documents_failed = 1
            stats.errors.append(str(e))

        stats.total_time_ms = (time.perf_counter() - start) * 1000
        return stats

    def ingest_file(self, path: Union[str, Path],
                     metadata: Optional[dict] = None) -> IngestionStats:
        """Ingest a document from a file path."""
        filepath = Path(path)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        meta = metadata or {}
        if "doc_id" not in meta:
            meta["doc_id"] = filepath.stem

        return self.ingest_document(content, meta)

    def ingest_company_batch(self, companies_path: Union[str, Path],
                              ) -> IngestionStats:
        """Ingest a batch of companies from a JSON file."""
        start = time.perf_counter()
        stats = IngestionStats()

        companies = self.loader.load_companies_json(companies_path)

        for company in companies:
            validation = self.validator.validate_company(company)
            if not validation.valid:
                stats.documents_failed += 1
                for issue in validation.issues:
                    stats.errors.append(
                        f"{company.canonical_id}: {issue.message}"
                    )
                continue

            # Build graph entities
            if self._graph_builder:
                self._graph_builder.build_company_entity(company)
                stats.graph_nodes_created += 1

            stats.documents_processed += 1

        stats.total_time_ms = (time.perf_counter() - start) * 1000
        logger.info("Batch ingested %d companies (%d failed)",
                    stats.documents_processed, stats.documents_failed)
        return stats

    def _create_chunks(self, parse_result: ParseResult) -> list[Chunk]:
        """Chunk a parsed document."""
        doc = parse_result.document
        chunks = self.chunker.structural_chunk(
            text=doc.content,
            doc_id=doc.doc_id,
            metadata={
                "company_id": doc.metadata.company_id if doc.metadata else "",
                "ticker": doc.metadata.ticker if doc.metadata else "",
                "doc_type": (doc.metadata.doc_type.value
                             if doc.metadata and doc.metadata.doc_type else ""),
                "period": doc.metadata.period if doc.metadata else "",
            },
        )
        return chunks

    def _embed_chunks(self, chunks: list[Chunk],
                       stats: IngestionStats) -> None:
        """Generate embeddings for chunks via the embedding manager."""
        self._embedding_manager.embed_chunks(chunks)
        stats.embeddings_generated = len(chunks) * 2  # dual embeddings

    def _index_vectors(self, chunks: list[Chunk],
                        stats: IngestionStats) -> None:
        """Index chunk embeddings into vector store."""
        for chunk in chunks:
            if chunk.embedding:
                self._vector_store.upsert(
                    collection="narrative",
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "section": chunk.section,
                        "company_id": chunk.metadata.get("company_id", ""),
                    },
                )
            if chunk.entity_embedding:
                self._vector_store.upsert(
                    collection="entity_schema",
                    id=f"{chunk.chunk_id}_ent",
                    vector=chunk.entity_embedding,
                    payload={
                        "text": chunk.conditioned_text,
                        "doc_id": chunk.doc_id,
                        "company_id": chunk.metadata.get("company_id", ""),
                    },
                )

    def _index_postgres(self, chunks: list[Chunk], doc: Document,
                         stats: IngestionStats) -> None:
        """Index chunks into Postgres for FTS."""
        for chunk in chunks:
            self._postgres.insert_chunk(chunk)

    def _build_graph(self, parse_result: ParseResult,
                      stats: IngestionStats) -> None:
        """Build graph nodes from parsed document."""
        doc = parse_result.document
        if doc.metadata:
            self._graph_builder.build_document_provenance(doc)
            stats.graph_nodes_created += 1
