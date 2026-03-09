"""Selective reranking — applies cross-encoder or vendor reranker to fused candidates.

Applied only when needed (Tier B/C) to control latency and cost.
Supports: Voyage reranker, Cohere reranker, cross-encoder models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

from goat.retrieval.fusion import FusedCandidate

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking a candidate."""
    chunk_id: str
    rerank_score: float
    original_rank: int
    new_rank: int


class BaseReranker(ABC):
    """Abstract base for rerankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[FusedCandidate],
               top_n: int = 50) -> list[FusedCandidate]: ...


class VoyageReranker(BaseReranker):
    """Voyage AI reranker."""

    def __init__(self, model: str = "rerank-2", api_key: str = ""):
        self.model = model
        self._api_key = api_key

    def rerank(self, query: str, candidates: list[FusedCandidate],
               top_n: int = 50) -> list[FusedCandidate]:
        try:
            import voyageai
        except ImportError:
            logger.warning("voyageai not installed — skipping reranking")
            return candidates[:top_n]

        client = voyageai.Client(api_key=self._api_key)
        texts = [c.hit.text for c in candidates[:top_n]]

        response = client.rerank(
            query=query,
            documents=texts,
            model=self.model,
        )

        # Map reranked results back to candidates
        reranked = []
        for i, result in enumerate(response.results):
            idx = result.index
            candidate = candidates[idx]
            candidate.hit.scoring.rerank_score = result.relevance_score
            candidate.hit.scoring.reranker_model = f"voyage/{self.model}"
            candidate.hit.scoring.final_score = result.relevance_score
            candidate.hit.scoring.rank = i + 1
            reranked.append(candidate)

        return reranked


class CohereReranker(BaseReranker):
    """Cohere reranker."""

    def __init__(self, model: str = "rerank-english-v3.0", api_key: str = ""):
        self.model = model
        self._api_key = api_key

    def rerank(self, query: str, candidates: list[FusedCandidate],
               top_n: int = 50) -> list[FusedCandidate]:
        try:
            import cohere
        except ImportError:
            logger.warning("cohere not installed — skipping reranking")
            return candidates[:top_n]

        client = cohere.Client(api_key=self._api_key)
        texts = [c.hit.text for c in candidates[:top_n]]

        response = client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_n=top_n,
        )

        reranked = []
        for i, result in enumerate(response.results):
            idx = result.index
            candidate = candidates[idx]
            candidate.hit.scoring.rerank_score = result.relevance_score
            candidate.hit.scoring.reranker_model = f"cohere/{self.model}"
            candidate.hit.scoring.final_score = result.relevance_score
            candidate.hit.scoring.rank = i + 1
            reranked.append(candidate)

        return reranked


class LocalCrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder: %s", self.model_name)

    def rerank(self, query: str, candidates: list[FusedCandidate],
               top_n: int = 50) -> list[FusedCandidate]:
        self._load_model()

        subset = candidates[:top_n]
        pairs = [(query, c.hit.text) for c in subset]
        scores = self._model.predict(pairs)

        # Assign scores and sort
        for i, score in enumerate(scores):
            subset[i].hit.scoring.rerank_score = float(score)
            subset[i].hit.scoring.reranker_model = self.model_name

        subset.sort(key=lambda c: c.hit.scoring.rerank_score, reverse=True)
        for i, c in enumerate(subset):
            c.hit.scoring.final_score = c.hit.scoring.rerank_score
            c.hit.scoring.rank = i + 1

        return subset


class NoOpReranker(BaseReranker):
    """Passthrough reranker (scores = fused scores)."""

    def rerank(self, query: str, candidates: list[FusedCandidate],
               top_n: int = 50) -> list[FusedCandidate]:
        subset = candidates[:top_n]
        for i, c in enumerate(subset):
            c.hit.scoring.final_score = c.fused_score
            c.hit.scoring.rank = i + 1
        return subset


def create_reranker(provider: str = "noop",
                    model: str = "", api_key: str = "") -> BaseReranker:
    """Factory: create the appropriate reranker."""
    if provider == "voyage":
        return VoyageReranker(model=model or "rerank-2", api_key=api_key)
    elif provider == "cohere":
        return CohereReranker(model=model or "rerank-english-v3.0", api_key=api_key)
    elif provider == "local":
        return LocalCrossEncoderReranker(model_name=model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    else:
        return NoOpReranker()
