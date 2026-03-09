"""Postgres storage backend — canonical data, lexical indices, FTS, and trigram similarity.

Responsibilities:
  - Authoritative canonical IDs and company data
  - Lexicon/synonym tables (versioned, slowly changing dimension)
  - Full-text search via tsvector + GIN indexes
  - Trigram similarity via pg_trgm + GIN/GiST for identifiers/aliases
  - Query logs and click/accept signals
  - Token statistics (DF/IDF proxies)
"""

from dataclasses import dataclass
from typing import Optional
import logging

from goat.config import PostgresConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL DDL for the lexical-semantics schema
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Canonical companies
CREATE TABLE IF NOT EXISTS companies (
    canonical_id    TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    primary_ticker  TEXT,
    sector          TEXT DEFAULT '',
    industry        TEXT DEFAULT '',
    country         TEXT DEFAULT '',
    region          TEXT DEFAULT '',
    market_cap      NUMERIC,
    metadata_json   JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies (primary_ticker);
CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies (sector);
CREATE INDEX IF NOT EXISTS idx_companies_name_trgm ON companies USING gin (name gin_trgm_ops);

-- Lexicon entries (versioned synonym/alias table)
CREATE TABLE IF NOT EXISTS lexicon_entries (
    entry_id        TEXT PRIMARY KEY,
    surface_form    TEXT NOT NULL,
    canonical_id    TEXT NOT NULL,
    canonical_label TEXT NOT NULL,
    entry_type      TEXT NOT NULL,  -- ticker, abbreviation, synonym, xbrl_concept, etc.
    language        TEXT DEFAULT 'en',
    confidence      REAL DEFAULT 1.0,
    source          TEXT DEFAULT 'manual',
    version         TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lexicon_surface ON lexicon_entries (lower(surface_form));
CREATE INDEX IF NOT EXISTS idx_lexicon_canonical ON lexicon_entries (canonical_id);
CREATE INDEX IF NOT EXISTS idx_lexicon_surface_trgm ON lexicon_entries USING gin (surface_form gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_lexicon_type ON lexicon_entries (entry_type);

-- Ticker mappings with variant tracking
CREATE TABLE IF NOT EXISTS ticker_mappings (
    canonical_id    TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    exchange        TEXT DEFAULT '',
    variants        TEXT[] DEFAULT '{}',
    company_name    TEXT NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (canonical_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_ticker_ticker ON ticker_mappings (upper(ticker));
CREATE INDEX IF NOT EXISTS idx_ticker_trgm ON ticker_mappings USING gin (ticker gin_trgm_ops);

-- Abbreviation definitions
CREATE TABLE IF NOT EXISTS abbreviations (
    abbreviation        TEXT PRIMARY KEY,
    expansion           TEXT NOT NULL,
    canonical_metric_id TEXT DEFAULT '',
    category            TEXT DEFAULT ''
);

-- Schema term mappings (XBRL concept alignment)
CREATE TABLE IF NOT EXISTS schema_term_mappings (
    id                  SERIAL PRIMARY KEY,
    surface_form        TEXT NOT NULL,
    canonical_concept   TEXT NOT NULL,
    taxonomy            TEXT DEFAULT 'us-gaap',
    concept_label       TEXT DEFAULT '',
    language            TEXT DEFAULT 'en',
    confidence          REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_schema_surface ON schema_term_mappings (lower(surface_form));
CREATE INDEX IF NOT EXISTS idx_schema_concept ON schema_term_mappings (canonical_concept);

-- Document chunks with FTS index
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    text                TEXT NOT NULL,
    section             TEXT DEFAULT '',
    company_id          TEXT DEFAULT '',
    doc_type            TEXT DEFAULT '',
    language            TEXT DEFAULT 'en',
    -- Full-text search vector (auto-populated by trigger or on insert)
    tsv                 TSVECTOR,
    -- Provenance
    ingestion_run_id    TEXT DEFAULT '',
    embed_model_version TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks (company_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON chunks USING gin (text gin_trgm_ops);

-- Query logs for learning-to-rank and monitoring
CREATE TABLE IF NOT EXISTS query_logs (
    query_id        TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    expanded_query  TEXT DEFAULT '',
    resolved_entities JSONB DEFAULT '[]',
    tier_used       TEXT DEFAULT 'A',
    result_count    INTEGER DEFAULT 0,
    latency_ms      REAL DEFAULT 0,
    user_id         TEXT DEFAULT '',
    feedback        TEXT DEFAULT '',  -- 'accepted', 'rejected', 'ignored'
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_querylog_time ON query_logs (created_at);

-- Provenance records
CREATE TABLE IF NOT EXISTS provenance (
    record_id           SERIAL PRIMARY KEY,
    entity_id           TEXT NOT NULL,
    entity_type         TEXT NOT NULL,
    source_doc_id       TEXT DEFAULT '',
    source_system       TEXT DEFAULT '',
    span_start          INTEGER DEFAULT 0,
    span_end            INTEGER DEFAULT 0,
    ingestion_run_id    TEXT DEFAULT '',
    extractor_version   TEXT DEFAULT '',
    embed_model_version TEXT DEFAULT '',
    index_version       TEXT DEFAULT '',
    relation_type       TEXT DEFAULT '',
    related_id          TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prov_entity ON provenance (entity_id);
CREATE INDEX IF NOT EXISTS idx_prov_source ON provenance (source_doc_id);
"""


class PostgresStore:
    """Postgres storage backend for canonical data and lexical indices.

    Wraps psycopg (async or sync) for production use, but provides
    a clean interface that can also be backed by in-memory structures
    for testing.
    """

    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig()
        self._conn = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Establish connection to Postgres."""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                dbname=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )
            logger.info("Connected to Postgres at %s:%s/%s",
                        self.config.host, self.config.port, self.config.database)
        except ImportError:
            logger.warning("psycopg2 not installed — Postgres features unavailable")
        except Exception as exc:
            logger.error("Failed to connect to Postgres: %s", exc)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize_schema(self):
        """Create all tables and indexes."""
        if not self._conn:
            logger.warning("No Postgres connection — skipping schema init")
            return
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)
        self._conn.commit()
        logger.info("Postgres schema initialized")

    # ------------------------------------------------------------------
    # Full-text search (tsvector + GIN)
    # ------------------------------------------------------------------

    def fts_search(self, query: str, top_k: int = 100,
                   company_id: Optional[str] = None,
                   doc_type: Optional[str] = None,
                   language: Optional[str] = None) -> list[dict]:
        """BM25-like full-text search using Postgres FTS.

        Uses plainto_tsquery for safe query parsing (no injection via tsquery syntax).
        """
        if not self._conn:
            return []

        conditions = ["tsv @@ plainto_tsquery(%s, %s)"]
        params: list = [self.config.fts_language, query]

        if company_id:
            conditions.append("company_id = %s")
            params.append(company_id)
        if doc_type:
            conditions.append("doc_type = %s")
            params.append(doc_type)
        if language:
            conditions.append("language = %s")
            params.append(language)

        where = " AND ".join(conditions)
        params.append(top_k)

        sql = f"""
            SELECT chunk_id, doc_id, text, section, company_id, doc_type,
                   ts_rank_cd(tsv, plainto_tsquery(%s, %s)) AS rank
            FROM chunks
            WHERE {where}
            ORDER BY rank DESC
            LIMIT %s
        """
        # ts_rank_cd needs the query again
        rank_params = [self.config.fts_language, query] + params

        with self._conn.cursor() as cur:
            cur.execute(sql, rank_params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Trigram similarity search (pg_trgm)
    # ------------------------------------------------------------------

    def trigram_search(self, query: str, table: str = "lexicon_entries",
                       column: str = "surface_form",
                       threshold: Optional[float] = None,
                       limit: int = 20) -> list[dict]:
        """Fuzzy search using pg_trgm similarity.

        Uses parameterized queries to prevent SQL injection.
        """
        if not self._conn:
            return []

        thresh = threshold or self.config.trigram_similarity_threshold
        # Whitelist allowed table/column combinations
        allowed = {
            ("lexicon_entries", "surface_form"),
            ("companies", "name"),
            ("ticker_mappings", "ticker"),
            ("chunks", "text"),
        }
        if (table, column) not in allowed:
            raise ValueError(f"Trigram search not allowed on {table}.{column}")

        sql = f"""
            SELECT *, similarity({column}, %s) AS sim
            FROM {table}
            WHERE similarity({column}, %s) >= %s
            ORDER BY sim DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (query, query, thresh, limit))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Lexicon CRUD
    # ------------------------------------------------------------------

    def upsert_lexicon_entry(self, entry: dict) -> None:
        """Insert or update a lexicon entry."""
        if not self._conn:
            return
        sql = """
            INSERT INTO lexicon_entries
                (entry_id, surface_form, canonical_id, canonical_label,
                 entry_type, language, confidence, source, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entry_id) DO UPDATE SET
                surface_form = EXCLUDED.surface_form,
                canonical_label = EXCLUDED.canonical_label,
                confidence = EXCLUDED.confidence,
                version = EXCLUDED.version
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                entry["entry_id"], entry["surface_form"], entry["canonical_id"],
                entry["canonical_label"], entry["entry_type"], entry.get("language", "en"),
                entry.get("confidence", 1.0), entry.get("source", "manual"),
                entry.get("version", "v1"),
            ))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chunk storage with auto tsvector
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: dict) -> None:
        """Insert a document chunk with auto-generated tsvector."""
        if not self._conn:
            return
        sql = """
            INSERT INTO chunks
                (chunk_id, doc_id, text, section, company_id, doc_type,
                 language, tsv, ingestion_run_id, embed_model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s,
                    to_tsvector(%s, %s), %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
        """
        lang = chunk.get("language", "english")
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                chunk["chunk_id"], chunk["doc_id"], chunk["text"],
                chunk.get("section", ""), chunk.get("company_id", ""),
                chunk.get("doc_type", ""), lang,
                self.config.fts_language, chunk["text"],
                chunk.get("ingestion_run_id", ""),
                chunk.get("embed_model_version", ""),
            ))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------

    def log_query(self, log_entry: dict) -> None:
        """Log a query for monitoring and learning-to-rank."""
        if not self._conn:
            return
        sql = """
            INSERT INTO query_logs
                (query_id, query_text, expanded_query, resolved_entities,
                 tier_used, result_count, latency_ms, user_id)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)
        """
        import json
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                log_entry["query_id"], log_entry["query_text"],
                log_entry.get("expanded_query", ""),
                json.dumps(log_entry.get("resolved_entities", [])),
                log_entry.get("tier_used", "A"),
                log_entry.get("result_count", 0),
                log_entry.get("latency_ms", 0),
                log_entry.get("user_id", ""),
            ))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Provenance storage
    # ------------------------------------------------------------------

    def store_provenance(self, record: dict) -> None:
        """Store a provenance record."""
        if not self._conn:
            return
        sql = """
            INSERT INTO provenance
                (entity_id, entity_type, source_doc_id, source_system,
                 span_start, span_end, ingestion_run_id, extractor_version,
                 embed_model_version, index_version, relation_type, related_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                record["entity_id"], record["entity_type"],
                record.get("source_doc_id", ""),
                record.get("source_system", ""),
                record.get("span_start", 0), record.get("span_end", 0),
                record.get("ingestion_run_id", ""),
                record.get("extractor_version", ""),
                record.get("embed_model_version", ""),
                record.get("index_version", ""),
                record.get("relation_type", ""),
                record.get("related_id", ""),
            ))
        self._conn.commit()
