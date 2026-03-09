"""JSON export for analysis results, retrieval outputs, and reports."""

import json
from typing import Union
from pathlib import Path
import logging

from goat.models.company import Company
from goat.models.financials import Financials
from goat.query.pipeline import QueryResponse
from goat.operations.evaluation import EvalSummary

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports analysis data as JSON."""

    def export_company(self, company: Company,
                        financials: Financials = None,
                        path: Union[str, Path] = None) -> dict:
        """Export company data as JSON dict."""
        data = {
            "canonical_id": company.canonical_id,
            "name": company.name,
            "identifiers": [
                {"type": i.type, "value": i.value, "exchange": i.exchange}
                for i in company.identifiers
            ],
            "aliases": company.aliases,
        }
        if company.metadata:
            data["metadata"] = {
                "sector": company.metadata.sector,
                "industry": company.metadata.industry,
                "country": company.metadata.country,
                "market_cap": company.metadata.market_cap,
            }
        if financials:
            data["financials"] = self._financials_to_dict(financials)

        if path:
            self._write(data, path)
        return data

    def export_query_response(self, response: QueryResponse,
                               path: Union[str, Path] = None) -> dict:
        """Export query response as JSON."""
        data = {
            "query_id": response.query_id,
            "raw_query": response.raw_query,
            "intent": response.intent.intent.value,
            "tier": response.intent.tier.value,
            "confidence": response.intent.confidence,
            "total_time_ms": response.total_time_ms,
            "resolved_companies": list(response.resolution.company_ids),
        }

        if response.explained:
            data["results"] = [
                {
                    "rank": h.rank,
                    "chunk_id": h.chunk_id,
                    "score": h.final_score,
                    "text_preview": h.text_preview,
                    "explanation": h.explanation,
                }
                for h in response.explained
            ]

        if response.error:
            data["error"] = response.error

        if path:
            self._write(data, path)
        return data

    def export_eval_summary(self, summary: EvalSummary,
                             path: Union[str, Path] = None) -> dict:
        """Export evaluation summary as JSON."""
        data = {
            "num_queries": summary.num_queries,
            "mean_map": summary.mean_map,
            "mean_mrr": summary.mean_mrr,
            "precision": {str(k): v for k, v in summary.avg_precision.items()},
            "recall": {str(k): v for k, v in summary.avg_recall.items()},
            "ndcg": {str(k): v for k, v in summary.avg_ndcg.items()},
            "eval_time_ms": summary.eval_time_ms,
        }
        if path:
            self._write(data, path)
        return data

    def _financials_to_dict(self, fin: Financials) -> dict:
        """Convert Financials to serializable dict."""
        result = {}
        for stmt in fin.income_statements:
            result.setdefault(stmt.period, {})["income_statement"] = {
                "revenue": stmt.revenue,
                "cost_of_revenue": stmt.cost_of_revenue,
                "gross_profit": stmt.gross_profit,
                "operating_income": stmt.operating_income,
                "net_income": stmt.net_income,
                "eps_diluted": stmt.eps_diluted,
                "gross_margin": stmt.gross_margin,
                "operating_margin": stmt.operating_margin,
                "net_margin": stmt.net_margin,
            }
        for stmt in fin.balance_sheets:
            result.setdefault(stmt.period, {})["balance_sheet"] = {
                "total_assets": stmt.total_assets,
                "total_liabilities": stmt.total_liabilities,
                "total_equity": stmt.total_equity,
                "current_ratio": stmt.current_ratio,
                "debt_to_equity": stmt.debt_to_equity,
            }
        for stmt in fin.cash_flows:
            result.setdefault(stmt.period, {})["cash_flow"] = {
                "operating_cash_flow": stmt.operating_cash_flow,
                "investing_cash_flow": stmt.investing_cash_flow,
                "financing_cash_flow": stmt.financing_cash_flow,
                "free_cash_flow": stmt.free_cash_flow,
                "capital_expenditures": stmt.capital_expenditures,
            }
        return result

    @staticmethod
    def _write(data: dict, path: Union[str, Path]) -> None:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported JSON to %s", filepath)
