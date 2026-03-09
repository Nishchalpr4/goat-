"""Monitoring for retrieval quality, embedding drift, and system health.

Tracks:
  - Query latency distributions (p50, p95, p99)
  - Retrieval quality signals (click-through proxy, score distributions)
  - Embedding drift detection (centroid drift, IDF distribution shifts)
  - Token/vocabulary coverage metrics
  - Error rates by component
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


@dataclass
class QualitySnapshot:
    """Point-in-time quality metrics."""
    timestamp: float = 0.0
    avg_top1_score: float = 0.0
    avg_top10_score: float = 0.0
    empty_result_rate: float = 0.0
    tier_distribution: dict[str, int] = field(default_factory=dict)
    intent_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report on detected drift."""
    metric: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    is_significant: bool
    timestamp: float = 0.0


class MonitoringService:
    """Monitors retrieval system health and quality."""

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        # Latency tracking
        self._latencies: dict[str, deque] = {
            "total": deque(maxlen=window_size),
            "lexical": deque(maxlen=window_size),
            "semantic": deque(maxlen=window_size),
            "fusion": deque(maxlen=window_size),
            "rerank": deque(maxlen=window_size),
            "graph": deque(maxlen=window_size),
        }
        # Score tracking
        self._top1_scores: deque = deque(maxlen=window_size)
        self._top10_scores: deque = deque(maxlen=window_size)
        self._empty_results: deque = deque(maxlen=window_size)
        # Distribution tracking
        self._tier_counts: dict[str, int] = {}
        self._intent_counts: dict[str, int] = {}
        # Error tracking
        self._error_counts: dict[str, int] = {}
        # Drift baselines
        self._baselines: dict[str, float] = {}
        # Embedding centroid tracking
        self._centroid_samples: deque = deque(maxlen=100)

    def record_query(self, latencies: dict[str, float],
                      tier: str,
                      intent: str,
                      top_scores: list[float],
                      error: Optional[str] = None) -> None:
        """Record metrics from a query execution."""
        # Latencies
        for component, ms in latencies.items():
            if component in self._latencies:
                self._latencies[component].append(ms)

        # Tier and intent
        self._tier_counts[tier] = self._tier_counts.get(tier, 0) + 1
        self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1

        # Scores
        if top_scores:
            self._top1_scores.append(top_scores[0])
            avg_top10 = sum(top_scores[:10]) / min(len(top_scores), 10)
            self._top10_scores.append(avg_top10)
            self._empty_results.append(0)
        else:
            self._empty_results.append(1)

        # Errors
        if error:
            self._error_counts[error] = self._error_counts.get(error, 0) + 1

    def get_latency_stats(self, component: str = "total") -> LatencyStats:
        """Get latency stats for a component."""
        values = list(self._latencies.get(component, []))
        if not values:
            return LatencyStats()

        values.sort()
        n = len(values)

        return LatencyStats(
            count=n,
            mean_ms=sum(values) / n,
            p50_ms=values[n // 2],
            p95_ms=values[int(n * 0.95)] if n > 20 else values[-1],
            p99_ms=values[int(n * 0.99)] if n > 100 else values[-1],
            max_ms=values[-1],
        )

    def get_quality_snapshot(self) -> QualitySnapshot:
        """Get current quality metrics."""
        top1_list = list(self._top1_scores)
        top10_list = list(self._top10_scores)
        empty_list = list(self._empty_results)

        return QualitySnapshot(
            timestamp=time.time(),
            avg_top1_score=sum(top1_list) / len(top1_list) if top1_list else 0,
            avg_top10_score=sum(top10_list) / len(top10_list) if top10_list else 0,
            empty_result_rate=(
                sum(empty_list) / len(empty_list) if empty_list else 0
            ),
            tier_distribution=dict(self._tier_counts),
            intent_distribution=dict(self._intent_counts),
        )

    def set_baseline(self, metric: str, value: float) -> None:
        """Set a baseline value for drift detection."""
        self._baselines[metric] = value

    def check_drift(self, metric: str,
                     current_value: float,
                     threshold: float = 0.1) -> DriftReport:
        """Check if a metric has drifted from its baseline."""
        baseline = self._baselines.get(metric, current_value)
        if baseline == 0:
            drift_magnitude = abs(current_value)
        else:
            drift_magnitude = abs(current_value - baseline) / abs(baseline)

        return DriftReport(
            metric=metric,
            baseline_value=baseline,
            current_value=current_value,
            drift_magnitude=drift_magnitude,
            is_significant=drift_magnitude > threshold,
            timestamp=time.time(),
        )

    def record_embedding_sample(self, centroid: list[float]) -> None:
        """Record a centroid sample for drift tracking."""
        self._centroid_samples.append(centroid)

    def check_embedding_drift(self, threshold: float = 0.05) -> Optional[DriftReport]:
        """Check for embedding distribution drift using centroid comparison."""
        samples = list(self._centroid_samples)
        if len(samples) < 10:
            return None

        # Compare first-half vs second-half centroids
        mid = len(samples) // 2
        first_half = samples[:mid]
        second_half = samples[mid:]

        first_centroid = self._mean_vector(first_half)
        second_centroid = self._mean_vector(second_half)

        if first_centroid and second_centroid:
            cosine_dist = 1.0 - self._cosine_sim(first_centroid, second_centroid)
            return DriftReport(
                metric="embedding_centroid_drift",
                baseline_value=0.0,
                current_value=cosine_dist,
                drift_magnitude=cosine_dist,
                is_significant=cosine_dist > threshold,
                timestamp=time.time(),
            )

        return None

    def get_error_summary(self) -> dict[str, int]:
        """Get error counts by type."""
        return dict(self._error_counts)

    @staticmethod
    def _mean_vector(vectors: list[list[float]]) -> Optional[list[float]]:
        """Compute element-wise mean of a list of vectors."""
        if not vectors:
            return None
        dim = len(vectors[0])
        result = [0.0] * dim
        for vec in vectors:
            for i in range(min(dim, len(vec))):
                result[i] += vec[i]
        n = len(vectors)
        return [x / n for x in result]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
