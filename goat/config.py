"""Configuration and settings for the GOAT lexical-semantics investment analysis system.

Covers: storage backends, embedding models, retrieval pipeline, Graph RAG zones,
scoring, operational versioning, and query processing settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Storage backend configs
# ---------------------------------------------------------------------------

@dataclass
class PostgresConfig:
    """Postgres connection and FTS/trigram settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "goat"
    user: str = "goat"
    password: str = ""  # set via env var GOAT_PG_PASSWORD
    schema: str = "public"
    fts_language: str = "english"
    trigram_similarity_threshold: float = 0.3
    pool_min: int = 2
    pool_max: int = 10


@dataclass
class DuckDBConfig:
    """DuckDB analytical / offline evaluation settings."""
    database_path: str = "data/goat_analytics.duckdb"
    enable_vss: bool = False  # experimental in-process vector index
    threads: int = 4


@dataclass
class VectorDBConfig:
    """Vector store settings (provider-agnostic)."""
    provider: str = "qdrant"  # qdrant | weaviate | pinecone | milvus
    host: str = "localhost"
    port: int = 6333
    api_key: str = ""  # set via env var GOAT_VECTOR_API_KEY
    collection_prefix: str = "goat"
    # ANN index params
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 128
    # metadata filtering
    default_payload_indexes: list = field(default_factory=lambda: [
        "company_id", "sector", "doc_type", "year", "language",
    ])


@dataclass
class GraphStoreConfig:
    """Graph RAG store settings."""
    backend: str = "networkx"  # networkx | neo4j | memgraph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    # Zone names (multi-zone architecture)
    zones: list = field(default_factory=lambda: [
        "entity", "data", "knowledge", "lexical", "provenance",
    ])


# ---------------------------------------------------------------------------
# Embedding configs
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingModelSpec:
    """Specification for a single embedding model."""
    name: str = "openai/text-embedding-3-small"
    provider: str = "openai"  # openai | cohere | voyage | jina | local
    dimensions: int = 1536
    max_tokens: int = 8192
    supports_dimension_shortening: bool = True
    supports_multilingual: bool = True
    batch_size: int = 64
    # For contextualized chunk embeddings (Voyage-style)
    contextualized: bool = False


@dataclass
class EmbeddingConfig:
    """Multi-model embedding strategy."""
    # Primary model for narrative text
    narrative_model: EmbeddingModelSpec = field(
        default_factory=lambda: EmbeddingModelSpec(
            name="openai/text-embedding-3-small",
            provider="openai",
            dimensions=1536,
        )
    )
    # Entity+schema conditioned model (dual-embedding strategy)
    entity_schema_model: EmbeddingModelSpec = field(
        default_factory=lambda: EmbeddingModelSpec(
            name="openai/text-embedding-3-large",
            provider="openai",
            dimensions=3072,
            supports_dimension_shortening=True,
        )
    )
    # Reduced dimensions for cost/perf tradeoff (MRL / dimension shortening)
    narrative_reduced_dims: Optional[int] = 512
    entity_reduced_dims: Optional[int] = 1024


# ---------------------------------------------------------------------------
# Retrieval pipeline configs
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Hybrid retrieval pipeline settings."""
    # First-stage candidate counts
    lexical_top_k: int = 100
    semantic_top_k: int = 100
    # Fusion
    fusion_method: str = "rrf"  # rrf | weighted_blend
    rrf_k: int = 60  # RRF constant
    lexical_weight: float = 0.4  # for weighted_blend
    semantic_weight: float = 0.6
    # Reranking
    rerank_enabled: bool = True
    rerank_top_n: int = 50  # rerank only top-N after fusion
    reranker_model: str = "voyage/rerank-2"
    reranker_provider: str = "voyage"
    # Tiering
    default_tier: str = "A"  # A=fast | B=precision | C=reasoning
    tier_thresholds: dict = field(default_factory=lambda: {
        "A": {"rerank": False, "graph_expand": False},
        "B": {"rerank": True, "graph_expand": False},
        "C": {"rerank": True, "graph_expand": True},
    })


# ---------------------------------------------------------------------------
# Scoring configs
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerWeights:
    """Weights for each analysis dimension in composite scoring (must sum to 1.0)."""
    financial: float = 0.20
    valuation: float = 0.15
    moat: float = 0.15
    growth: float = 0.15
    risk: float = 0.10
    management: float = 0.10
    industry: float = 0.05
    esg: float = 0.05
    dividend: float = 0.05

    def validate(self) -> bool:
        total = (
            self.financial + self.valuation + self.moat + self.growth +
            self.risk + self.management + self.industry + self.esg + self.dividend
        )
        return abs(total - 1.0) < 0.001


@dataclass
class ScoringConfig:
    """Scoring and grading settings."""
    min_score: float = 0.0
    max_score: float = 10.0
    grade_thresholds: dict = field(default_factory=lambda: {
        "A+": 9.0, "A": 8.0, "B+": 7.0, "B": 6.0,
        "C+": 5.0, "C": 4.0, "D": 3.0, "F": 0.0,
    })


# ---------------------------------------------------------------------------
# Ingestion configs
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Document chunking settings."""
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64
    structural_segmentation: bool = True  # respect section boundaries
    min_chunk_size: int = 50


@dataclass
class IngestionConfig:
    """Ingestion pipeline settings."""
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    supported_formats: list = field(default_factory=lambda: [
        "json", "csv", "10-K", "10-Q", "8-K", "transcript", "press_release",
    ])
    language_detection: bool = True
    default_language: str = "en"
    edgar_user_agent: str = ""  # required by SEC EDGAR API


# ---------------------------------------------------------------------------
# Operations configs
# ---------------------------------------------------------------------------

@dataclass
class VersioningConfig:
    """Version tracking for reproducibility."""
    embed_model_version: str = "v1"
    vector_index_version: str = "v1"
    lexicon_version: str = "v1"
    graph_extraction_version: str = "v1"


@dataclass
class MonitoringConfig:
    """Production monitoring settings."""
    recall_at_k: list = field(default_factory=lambda: [5, 10, 20, 50])
    ndcg_at_k: list = field(default_factory=lambda: [10])
    latency_percentiles: list = field(default_factory=lambda: [50, 95, 99])
    provenance_coverage_target: float = 0.90


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Global configuration for the GOAT system."""
    # Storage
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    duckdb: DuckDBConfig = field(default_factory=DuckDBConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    # Embeddings
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # Retrieval
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # Scoring
    weights: AnalyzerWeights = field(default_factory=AnalyzerWeights)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    # Ingestion
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    # Operations
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    # General
    currency: str = "USD"
    fiscal_year_end_month: int = 12
    enabled_analyzers: list = field(default_factory=lambda: [
        "financial", "valuation", "moat", "growth", "risk",
        "management", "industry", "esg", "dividend",
    ])


default_config = Config()
