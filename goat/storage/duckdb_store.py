"""DuckDB storage backend — analytics, evaluation, and optional in-process vector search.

Responsibilities:
  - Offline evaluation and analytics (BEIR/MTEB-style harnesses)
  - Dataset snapshots and reproducibility
  - Token statistics aggregation (DF/IDF proxies)
  - Optional VSS extension for embedded vector experiments
"""

import logging
from typing import Optional
from pathlib import Path

from goat.config import DuckDBConfig

logger = logging.getLogger(__name__)


class DuckDBStore:
    """DuckDB analytical and evaluation backend."""

    def __init__(self, config: Optional[DuckDBConfig] = None):
        self.config = config or DuckDBConfig()
        self._conn = None

    def connect(self):
        """Connect to DuckDB (creates file if needed)."""
        try:
            import duckdb
            db_path = self.config.database_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(db_path)
            self._conn.execute(f"SET threads = {self.config.threads}")
            if self.config.enable_vss:
                try:
                    self._conn.execute("INSTALL vss; LOAD vss;")
                    logger.info("DuckDB VSS extension loaded")
                except Exception as exc:
                    logger.warning("VSS extension not available: %s", exc)
            logger.info("Connected to DuckDB at %s", db_path)
        except ImportError:
            logger.warning("duckdb not installed — analytics features unavailable")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize_schema(self):
        """Create analytical tables."""
        if not self._conn:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_queries (
                query_id        TEXT,
                query_text      TEXT,
                expected_doc_ids TEXT[],
                category        TEXT DEFAULT '',
                language        TEXT DEFAULT 'en',
                difficulty      TEXT DEFAULT 'medium'
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                run_id          TEXT,
                query_id        TEXT,
                retrieval_method TEXT,  -- 'lexical', 'semantic', 'hybrid', etc.
                retrieved_ids   TEXT[],
                scores          DOUBLE[],
                recall_at_5     DOUBLE,
                recall_at_10    DOUBLE,
                recall_at_20    DOUBLE,
                ndcg_at_10      DOUBLE,
                mrr             DOUBLE,
                latency_ms      DOUBLE,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS token_stats (
                token           TEXT,
                doc_freq        INTEGER DEFAULT 0,  -- number of docs containing token
                total_freq      INTEGER DEFAULT 0,  -- total occurrences
                language        TEXT DEFAULT 'en',
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                snapshot_id     TEXT,
                snapshot_type   TEXT,  -- 'lexicon', 'embeddings', 'index', 'graph'
                version         TEXT,
                record_count    INTEGER,
                metadata_json   TEXT DEFAULT '{}',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("DuckDB schema initialized")

    # ------------------------------------------------------------------
    # Evaluation harness (offline retrieval benchmarks)
    # ------------------------------------------------------------------

    def store_eval_query(self, query_id: str, query_text: str,
                         expected_doc_ids: list[str], category: str = "",
                         language: str = "en") -> None:
        """Store a gold evaluation query."""
        if not self._conn:
            return
        self._conn.execute(
            "INSERT INTO eval_queries VALUES (?, ?, ?, ?, ?, 'medium')",
            [query_id, query_text, expected_doc_ids, category, language],
        )

    def store_eval_result(self, run_id: str, query_id: str,
                          method: str, retrieved_ids: list[str],
                          scores: list[float], metrics: dict,
                          latency_ms: float) -> None:
        """Store evaluation results for a retrieval run."""
        if not self._conn:
            return
        self._conn.execute(
            """INSERT INTO eval_results
               (run_id, query_id, retrieval_method, retrieved_ids, scores,
                recall_at_5, recall_at_10, recall_at_20, ndcg_at_10, mrr, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [run_id, query_id, method, retrieved_ids, scores,
             metrics.get("recall@5", 0), metrics.get("recall@10", 0),
             metrics.get("recall@20", 0), metrics.get("ndcg@10", 0),
             metrics.get("mrr", 0), latency_ms],
        )

    def get_eval_summary(self, run_id: str) -> list[dict]:
        """Get aggregated evaluation metrics for a run."""
        if not self._conn:
            return []
        result = self._conn.execute("""
            SELECT retrieval_method,
                   COUNT(*) as n_queries,
                   AVG(recall_at_5) as avg_recall_5,
                   AVG(recall_at_10) as avg_recall_10,
                   AVG(recall_at_20) as avg_recall_20,
                   AVG(ndcg_at_10) as avg_ndcg_10,
                   AVG(mrr) as avg_mrr,
                   AVG(latency_ms) as avg_latency,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency
            FROM eval_results
            WHERE run_id = ?
            GROUP BY retrieval_method
        """, [run_id]).fetchall()
        columns = ["method", "n_queries", "avg_recall@5", "avg_recall@10",
                    "avg_recall@20", "avg_ndcg@10", "avg_mrr",
                    "avg_latency_ms", "p95_latency_ms"]
        return [dict(zip(columns, row)) for row in result]

    # ------------------------------------------------------------------
    # Token statistics (DF/IDF proxies)
    # ------------------------------------------------------------------

    def update_token_stats(self, token_counts: dict[str, dict]) -> None:
        """Bulk update token statistics.

        token_counts: {token: {"doc_freq": int, "total_freq": int}}
        """
        if not self._conn:
            return
        rows = [(tok, stats["doc_freq"], stats["total_freq"])
                for tok, stats in token_counts.items()]
        self._conn.executemany(
            """INSERT INTO token_stats (token, doc_freq, total_freq)
               VALUES (?, ?, ?)
               ON CONFLICT (token) DO UPDATE SET
                   doc_freq = EXCLUDED.doc_freq,
                   total_freq = EXCLUDED.total_freq,
                   updated_at = CURRENT_TIMESTAMP""",
            rows,
        )

    def get_idf(self, token: str, total_docs: int) -> float:
        """Compute IDF for a token."""
        if not self._conn or total_docs == 0:
            return 0.0
        result = self._conn.execute(
            "SELECT doc_freq FROM token_stats WHERE token = ?", [token]
        ).fetchone()
        if not result or result[0] == 0:
            return 0.0
        import math
        return math.log((total_docs - result[0] + 0.5) / (result[0] + 0.5) + 1)

    # ------------------------------------------------------------------
    # Dataset snapshots
    # ------------------------------------------------------------------

    def record_snapshot(self, snapshot_id: str, snapshot_type: str,
                        version: str, record_count: int,
                        metadata: Optional[dict] = None) -> None:
        """Record a dataset snapshot for reproducibility."""
        if not self._conn:
            return
        import json
        self._conn.execute(
            "INSERT INTO dataset_snapshots VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [snapshot_id, snapshot_type, version, record_count,
             json.dumps(metadata or {})],
        )
