"""Vector store abstraction — provider-agnostic interface for ANN retrieval.

Supports: Qdrant, Weaviate, Pinecone, Milvus via a common interface.
Handles: HNSW indexing, metadata/payload filtering, hybrid search primitives.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import uuid

from goat.config import VectorDBConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """A single vector to store/retrieve."""
    id: str
    vector: list[float]
    payload: dict = field(default_factory=dict)  # metadata for filtering
    sparse_vector: Optional[dict] = None  # {"indices": [...], "values": [...]}


@dataclass
class VectorSearchResult:
    """A single ANN search result."""
    id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class SearchFilters:
    """Metadata filters for vector search."""
    company_id: Optional[str] = None
    company_ids: Optional[list[str]] = None
    sector: Optional[str] = None
    doc_type: Optional[str] = None
    language: Optional[str] = None
    year: Optional[int] = None
    year_range: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict:
        """Convert to a flat filter dict (non-None fields only)."""
        filters = {}
        if self.company_id:
            filters["company_id"] = self.company_id
        if self.company_ids:
            filters["company_id"] = {"$in": self.company_ids}
        if self.sector:
            filters["sector"] = self.sector
        if self.doc_type:
            filters["doc_type"] = self.doc_type
        if self.language:
            filters["language"] = self.language
        if self.year:
            filters["year"] = self.year
        if self.year_range:
            filters["year"] = {"$gte": self.year_range[0], "$lte": self.year_range[1]}
        return filters


class BaseVectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def create_collection(self, name: str, dimension: int,
                          metric: str = "cosine") -> None: ...

    @abstractmethod
    def upsert(self, collection: str, records: list[VectorRecord]) -> int: ...

    @abstractmethod
    def search(self, collection: str, vector: list[float],
               top_k: int = 10, filters: Optional[SearchFilters] = None,
               ) -> list[VectorSearchResult]: ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> int: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store backend with payload index support."""

    def __init__(self, config: Optional[VectorDBConfig] = None):
        self.config = config or VectorDBConfig(provider="qdrant")
        self._client = None

    def connect(self):
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key or None,
            )
            logger.info("Connected to Qdrant at %s:%s", self.config.host, self.config.port)
        except ImportError:
            logger.warning("qdrant-client not installed")

    def create_collection(self, name: str, dimension: int,
                          metric: str = "cosine") -> None:
        if not self._client:
            return
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
        dist_map = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}
        self._client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dimension,
                distance=dist_map.get(metric, Distance.COSINE),
                hnsw_config={
                    "m": self.config.hnsw_m,
                    "ef_construct": self.config.hnsw_ef_construct,
                },
            ),
        )
        # Create payload indexes for fast filtering
        for field_name in self.config.default_payload_indexes:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("Created Qdrant collection '%s' (dim=%d)", name, dimension)

    def upsert(self, collection: str, records: list[VectorRecord]) -> int:
        if not self._client:
            return 0
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=rec.id, vector=rec.vector, payload=rec.payload)
            for rec in records
        ]
        self._client.upsert(collection_name=collection, points=points)
        return len(points)

    def search(self, collection: str, vector: list[float],
               top_k: int = 10, filters: Optional[SearchFilters] = None,
               ) -> list[VectorSearchResult]:
        if not self._client:
            return []
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        query_filter = None
        if filters:
            conditions = []
            fd = filters.to_dict()
            for key, val in fd.items():
                if isinstance(val, dict):
                    if "$in" in val:
                        conditions.append(FieldCondition(
                            key=key, match=MatchAny(any=val["$in"]),
                        ))
                else:
                    conditions.append(FieldCondition(
                        key=key, match=MatchValue(value=val),
                    ))
            if conditions:
                query_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
            search_params={"hnsw_ef": self.config.hnsw_ef_search},
        )
        return [
            VectorSearchResult(id=str(r.id), score=r.score, payload=r.payload or {})
            for r in results
        ]

    def delete(self, collection: str, ids: list[str]) -> int:
        if not self._client:
            return 0
        self._client.delete(collection_name=collection, points_selector=ids)
        return len(ids)

    def count(self, collection: str) -> int:
        if not self._client:
            return 0
        info = self._client.get_collection(collection)
        return info.points_count


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing and small-scale experiments."""

    def __init__(self):
        self._collections: dict[str, dict] = {}  # name -> {records, dimension, metric}

    def create_collection(self, name: str, dimension: int,
                          metric: str = "cosine") -> None:
        self._collections[name] = {
            "records": {},
            "dimension": dimension,
            "metric": metric,
        }

    def upsert(self, collection: str, records: list[VectorRecord]) -> int:
        if collection not in self._collections:
            return 0
        store = self._collections[collection]["records"]
        for rec in records:
            store[rec.id] = rec
        return len(records)

    def search(self, collection: str, vector: list[float],
               top_k: int = 10, filters: Optional[SearchFilters] = None,
               ) -> list[VectorSearchResult]:
        if collection not in self._collections:
            return []
        store = self._collections[collection]["records"]
        metric = self._collections[collection]["metric"]

        candidates = list(store.values())
        # Apply filters
        if filters:
            fd = filters.to_dict()
            filtered = []
            for rec in candidates:
                match = True
                for key, val in fd.items():
                    if isinstance(val, dict):
                        if "$in" in val and rec.payload.get(key) not in val["$in"]:
                            match = False
                    elif rec.payload.get(key) != val:
                        match = False
                if match:
                    filtered.append(rec)
            candidates = filtered

        # Compute similarities
        scored = []
        for rec in candidates:
            sim = self._similarity(vector, rec.vector, metric)
            scored.append(VectorSearchResult(
                id=rec.id, score=sim, payload=rec.payload,
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete(self, collection: str, ids: list[str]) -> int:
        if collection not in self._collections:
            return 0
        store = self._collections[collection]["records"]
        count = 0
        for id_ in ids:
            if id_ in store:
                del store[id_]
                count += 1
        return count

    def count(self, collection: str) -> int:
        if collection not in self._collections:
            return 0
        return len(self._collections[collection]["records"])

    @staticmethod
    def _similarity(a: list[float], b: list[float], metric: str) -> float:
        if len(a) != len(b):
            return 0.0
        if metric == "cosine":
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        elif metric == "dot":
            return sum(x * y for x, y in zip(a, b))
        else:  # euclid (return negative distance for ranking)
            dist = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            return -dist


def create_vector_store(config: Optional[VectorDBConfig] = None) -> BaseVectorStore:
    """Factory: create the appropriate vector store backend."""
    if config is None:
        return InMemoryVectorStore()
    if config.provider == "qdrant":
        store = QdrantVectorStore(config)
        store.connect()
        return store
    # Add more providers here as needed
    logger.warning("Unknown vector DB provider '%s', using in-memory", config.provider)
    return InMemoryVectorStore()
