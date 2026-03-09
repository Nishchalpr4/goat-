# GOAT — Growth-Oriented Analysis Toolkit

A scalable investment analysis AI system with lexical semantics, hybrid retrieval (BM25 + dense + fusion + reranking), Graph RAG multi-zone architecture, and W3C PROV provenance tracking.

## Architecture

The system is designed around **lexical semantics as a first-class subsystem** — every user query passes through entity resolution, schema linking, synonym expansion, and abbreviation handling before retrieval. This ensures financial terms like "revenue", "top line", "net sales", and `us-gaap:Revenues` all resolve to the same canonical concept.

### Key Capabilities

- **Hybrid Retrieval**: BM25/FTS (Postgres tsvector) + Dense ANN (HNSW) + RRF fusion + selective reranking
- **Dual-Embedding Strategy**: Narrative embeddings + entity/schema-conditioned embeddings in separate vector collections
- **Graph RAG (5 zones)**: Entity, Data, Knowledge, Lexical, Provenance — enabling relational reasoning across companies
- **XBRL Schema Linking**: Maps natural language financial terms to canonical GAAP concepts
- **W3C PROV Provenance**: Every retrieved chunk carries full provenance (wasGeneratedBy, wasDerivedFrom, wasAttributedTo)
- **Tiered Retrieval**: Tier A (fast), Tier B (precision+rerank), Tier C (reasoning+graph expansion)
- **Batch Scale**: Ingest and analyze 100s of companies with structured data loading

## Quick Start

```bash
pip install -r requirements.txt

# Query the system
python -m goat query "What drives AAPL margin expansion?"

# Ingest documents
python -m goat ingest --file report.txt --type 10-K --ticker AAPL

# Batch ingest companies
python -m goat ingest --companies data/sample_companies.json

# Company overview
python -m goat company AAPL

# Compare companies
python -m goat compare AAPL MSFT GOOGL

# System status
python -m goat status

# Run evaluation
python -m goat eval --dataset eval_data/
```

## Project Structure

```
goat/
├── __main__.py              # CLI entry point
├── config.py                # Master configuration
├── models/                  # Core data models
│   ├── company.py           # Company entity (identifiers, aliases, metadata)
│   ├── financials.py        # Income statement, balance sheet, cash flow
│   ├── document.py          # Document & chunk models (dual-embedding ready)
│   ├── provenance.py        # W3C PROV-DM models
│   ├── lexicon.py           # Lexicon entries, ticker mappings, schema terms
│   ├── retrieval.py         # Retrieval results with scoring breakdowns
│   └── scores.py            # Investment analysis scoring
├── lexical/                 # Lexical semantics subsystem
│   ├── normalizer.py        # Unicode normalization, case folding
│   ├── tokenizer.py         # Field-aware tokenization (identifier/narrative/schema)
│   ├── entity_resolver.py   # 4-stage resolution: exact→lexicon→trigram→alias
│   ├── schema_linker.py     # XBRL concept mapping (3-stage linking)
│   ├── synonym_manager.py   # Versioned lexicon CRUD, import/export
│   └── query_expander.py    # 7-step expansion pipeline
├── storage/                 # Storage backends
│   ├── postgres.py          # FTS (tsvector/GIN), trigram (pg_trgm), canonical data
│   ├── duckdb_store.py      # Analytics, evaluation harness, token stats
│   ├── vector_store.py      # Vector DB abstraction (Qdrant + in-memory)
│   └── graph_store.py       # Multi-zone graph store (NetworkX)
├── embeddings/              # Embedding management
│   ├── models.py            # Multi-provider (OpenAI, Cohere, Voyage, local)
│   ├── manager.py           # Dual-embedding orchestrator
│   └── chunker.py           # Structural document chunking
├── retrieval/               # Hybrid retrieval engine
│   ├── lexical.py           # BM25/FTS with section boosts
│   ├── semantic.py          # Dual-channel dense ANN
│   ├── fusion.py            # RRF (k=60) and weighted blend
│   ├── reranker.py          # Voyage/Cohere/cross-encoder reranking
│   └── hybrid.py            # 5-stage pipeline orchestrator
├── graph/                   # Graph RAG
│   ├── zones.py             # 5-zone definitions with typed nodes/edges
│   ├── builder.py           # Graph construction from structured data
│   ├── traversal.py         # Graph-guided retrieval expansion
│   └── community.py         # Community detection (Louvain/LP/greedy)
├── query/                   # Query processing pipeline
│   ├── parser.py            # Period extraction, entity/metric detection
│   ├── intent.py            # Intent classification & tier routing
│   ├── planner.py           # Execution plan generation
│   ├── explainer.py         # Explainable scoring output
│   └── pipeline.py          # End-to-end orchestrator (8 stages)
├── ingestion/               # Data ingestion
│   ├── parsers.py           # SEC filing, transcript, news parsers
│   ├── loader.py            # JSON/CSV batch loading
│   ├── validator.py         # Data validation & sanity checks
│   ├── edgar.py             # SEC EDGAR API client
│   └── pipeline.py          # 8-step ingestion orchestrator
├── operations/              # Operational tooling
│   ├── versioning.py        # Model/index/lexicon version tracking
│   ├── monitoring.py        # Latency, quality, drift monitoring
│   └── evaluation.py        # Precision@k, NDCG@k, MAP, MRR harness
└── reporting/               # Output & reporting
    ├── terminal.py          # Terminal reports & dashboards
    ├── json_export.py       # JSON export
    └── csv_export.py        # CSV export
data/
├── taxonomies/
│   └── us_gaap_metrics.json # XBRL metric definitions (27 metrics)
├── lexicons/
│   ├── abbreviations.json   # Financial abbreviations (46 entries)
│   └── ticker_aliases.json  # Company ticker mappings (30 companies)
└── sample_companies.json    # Sample company data (5 companies)
```

## Query Processing Pipeline

```
User Query
    │
    ▼
1. Parse (tokenize, extract periods/entities/metrics)
    │
    ▼
2. Intent Detection (find_evidence / compare / explain / compute / summarize / trend / screen)
    │
    ▼
3. Entity Resolution (4-stage: exact ticker → lexicon → trigram ticker → trigram alias)
    │
    ▼
4. Query Expansion (synonym injection, abbreviation expansion, schema alignment)
    │
    ▼
5. Plan Generation (select tier, configure stages, set filters)
    │
    ▼
6. Hybrid Retrieval
    ├── Lexical (BM25/FTS + trigram)
    ├── Semantic (dual-channel dense ANN)
    ├── Fusion (RRF, k=60)
    ├── Rerank (Tier B+C only)
    └── Graph Expansion (Tier C only)
    │
    ▼
7. Explainable Scoring (full breakdown per result)
    │
    ▼
8. Response (hits + provenance + diagnostics)
```

## Retrieval Tiers

| Tier | Path | Use Case |
|------|------|----------|
| A (fast) | Lexical + Dense + RRF | Simple factual queries |
| B (precision) | + Reranking | Comparison, trend, summarization |
| C (reasoning) | + Graph Expansion | Explain causality, relational queries |

## Embedding Strategy

Two separate vector collections for each chunk:
- **Narrative**: Pure text embedding for general semantic similarity
- **Entity+Schema**: Conditioned text `[Company|Ticker|Sector|DocType|Section] text` for entity-aware retrieval

Both channels are retrieved in parallel and fused via RRF.

## Graph RAG Zones

| Zone | Purpose | Node Types | Edge Types |
|------|---------|------------|------------|
| Entity | Company identity & relationships | company, person, ticker, sector | subsidiary, competes, executive |
| Data | Structured financial facts | fact, metric, period | has_fact, has_metric |
| Knowledge | Claims & insights | claim, insight, thesis | supports, contradicts |
| Lexical | Synonyms & schema mappings | synonym, alias, abbreviation | is_alias, maps_to |
| Provenance | Source tracking (W3C PROV) | document, activity, agent | generated_by, derived_from |
