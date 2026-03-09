"""Offline evaluation harness — evaluates retrieval quality
using standard IR metrics.

Supports:
  - Precision@k, Recall@k, NDCG@k, MAP
  - MRR (Mean Reciprocal Rank)
  - Custom domain-specific metrics (financial relevance scoring)
  - BEIR-compatible dataset format
  - Evaluation result persistence (via DuckDB)
"""

from dataclasses import dataclass, field
from typing import Optional
import math
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelevanceJudgment:
    """A single relevance judgment for evaluation."""
    query_id: str
    doc_id: str
    relevance: int  # 0=not relevant, 1=relevant, 2=highly relevant


@dataclass
class EvalQuery:
    """A query with associated relevance judgments."""
    query_id: str
    query_text: str
    judgments: list[RelevanceJudgment] = field(default_factory=list)

    @property
    def relevant_docs(self) -> set[str]:
        return {j.doc_id for j in self.judgments if j.relevance > 0}

    @property
    def highly_relevant_docs(self) -> set[str]:
        return {j.doc_id for j in self.judgments if j.relevance > 1}


@dataclass
class EvalMetrics:
    """Computed evaluation metrics for a single query."""
    query_id: str
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    mrr: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0


@dataclass
class EvalSummary:
    """Aggregated evaluation metrics across all queries."""
    num_queries: int = 0
    avg_precision: dict[int, float] = field(default_factory=dict)
    avg_recall: dict[int, float] = field(default_factory=dict)
    avg_ndcg: dict[int, float] = field(default_factory=dict)
    mean_map: float = 0.0
    mean_mrr: float = 0.0
    eval_time_ms: float = 0.0


class EvaluationHarness:
    """Evaluates retrieval quality using standard IR metrics."""

    def __init__(self, k_values: Optional[list[int]] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20, 50, 100]

    def evaluate_query(self, query: EvalQuery,
                        retrieved_ids: list[str]) -> EvalMetrics:
        """Evaluate retrieval for a single query."""
        relevant = query.relevant_docs
        metrics = EvalMetrics(
            query_id=query.query_id,
            num_relevant=len(relevant),
            num_retrieved=len(retrieved_ids),
        )

        if not relevant or not retrieved_ids:
            return metrics

        # Precision@k and Recall@k
        for k in self.k_values:
            top_k = retrieved_ids[:k]
            hits = sum(1 for d in top_k if d in relevant)
            metrics.precision_at_k[k] = hits / k
            metrics.recall_at_k[k] = hits / len(relevant)

        # NDCG@k
        relevance_map = {j.doc_id: j.relevance for j in query.judgments}
        for k in self.k_values:
            metrics.ndcg_at_k[k] = self._ndcg_at_k(
                retrieved_ids, relevance_map, k
            )

        # MAP
        metrics.map_score = self._average_precision(retrieved_ids, relevant)

        # MRR
        metrics.mrr = self._reciprocal_rank(retrieved_ids, relevant)

        return metrics

    def evaluate_batch(self, queries: list[EvalQuery],
                        retrieval_fn) -> EvalSummary:
        """Evaluate across a batch of queries.

        Args:
            queries: List of evaluation queries with judgments
            retrieval_fn: Callable(query_text) -> list[doc_id]
        """
        start = time.perf_counter()
        all_metrics = []

        for query in queries:
            retrieved = retrieval_fn(query.query_text)
            metrics = self.evaluate_query(query, retrieved)
            all_metrics.append(metrics)

        summary = self._aggregate(all_metrics)
        summary.eval_time_ms = (time.perf_counter() - start) * 1000
        return summary

    def _aggregate(self, metrics_list: list[EvalMetrics]) -> EvalSummary:
        """Aggregate metrics across queries."""
        if not metrics_list:
            return EvalSummary()

        n = len(metrics_list)
        summary = EvalSummary(num_queries=n)

        for k in self.k_values:
            summary.avg_precision[k] = (
                sum(m.precision_at_k.get(k, 0) for m in metrics_list) / n
            )
            summary.avg_recall[k] = (
                sum(m.recall_at_k.get(k, 0) for m in metrics_list) / n
            )
            summary.avg_ndcg[k] = (
                sum(m.ndcg_at_k.get(k, 0) for m in metrics_list) / n
            )

        summary.mean_map = sum(m.map_score for m in metrics_list) / n
        summary.mean_mrr = sum(m.mrr for m in metrics_list) / n

        return summary

    @staticmethod
    def _ndcg_at_k(retrieved: list[str],
                    relevance: dict[str, int],
                    k: int) -> float:
        """Compute NDCG@k."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = relevance.get(doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because log2(1)=0

        # Ideal DCG
        ideal_rels = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum(
            (2**rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _average_precision(retrieved: list[str],
                            relevant: set[str]) -> float:
        """Compute Average Precision."""
        if not relevant:
            return 0.0

        hits = 0
        sum_prec = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                hits += 1
                sum_prec += hits / (i + 1)

        return sum_prec / len(relevant)

    @staticmethod
    def _reciprocal_rank(retrieved: list[str],
                          relevant: set[str]) -> float:
        """Compute Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def load_beir_dataset(self, queries_path: str,
                           qrels_path: str) -> list[EvalQuery]:
        """Load evaluation data in BEIR-compatible format.

        queries file: TSV with query_id, query_text
        qrels file: TSV with query_id, corpus_id, relevance
        """
        import csv

        queries = {}
        with open(queries_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    queries[row[0]] = EvalQuery(
                        query_id=row[0], query_text=row[1]
                    )

        with open(qrels_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    qid, doc_id = row[0], row[1]
                    try:
                        rel = int(row[2])
                    except ValueError:
                        continue
                    if qid in queries:
                        queries[qid].judgments.append(
                            RelevanceJudgment(qid, doc_id, rel)
                        )

        return list(queries.values())
