"""CSV export for tabular financial data and comparison reports."""

import csv
from typing import Union
from pathlib import Path
import logging

from goat.models.company import Company
from goat.models.financials import Financials

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports financial and analysis data as CSV."""

    def export_financials(self, company: Company,
                           financials: Financials,
                           path: Union[str, Path]) -> None:
        """Export income statement data as CSV."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for stmt in financials.income_statements:
            rows.append({
                "company_id": company.canonical_id,
                "ticker": company.primary_ticker or "",
                "period": stmt.period,
                "revenue": stmt.revenue,
                "cost_of_revenue": stmt.cost_of_revenue,
                "gross_profit": stmt.gross_profit,
                "operating_income": stmt.operating_income,
                "net_income": stmt.net_income,
                "eps_diluted": stmt.eps_diluted,
                "gross_margin": stmt.gross_margin,
                "operating_margin": stmt.operating_margin,
                "net_margin": stmt.net_margin,
            })

        if rows:
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Exported %d rows to %s", len(rows), filepath)

    def export_comparison(self, companies: list[Company],
                           financials_map: dict[str, Financials],
                           path: Union[str, Path]) -> None:
        """Export multi-company comparison as CSV."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for company in companies:
            fin = financials_map.get(company.canonical_id)
            if not fin or not fin.income_statements:
                continue
            latest = fin.income_statements[-1]
            rows.append({
                "ticker": company.primary_ticker or company.canonical_id,
                "name": company.name,
                "sector": company.metadata.sector if company.metadata else "",
                "period": latest.period,
                "revenue": latest.revenue,
                "net_income": latest.net_income,
                "gross_margin": latest.gross_margin,
                "operating_margin": latest.operating_margin,
                "net_margin": latest.net_margin,
            })

        if rows:
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Exported comparison of %d companies to %s",
                        len(rows), filepath)
