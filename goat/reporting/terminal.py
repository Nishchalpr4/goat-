"""Terminal-based reporting for investment analysis results.

Produces formatted terminal output for:
  - Single-company deep-dive reports
  - Multi-company comparison tables
  - Retrieval result displays with scoring breakdowns
  - System status and monitoring dashboards
"""

import sys
from typing import Optional
import logging

from goat.models.company import Company
from goat.models.financials import Financials
from goat.models.retrieval import RetrievalResult
from goat.query.explainer import ExplainedHit
from goat.query.pipeline import QueryResponse
from goat.operations.monitoring import LatencyStats, QualitySnapshot

logger = logging.getLogger(__name__)


class TerminalReporter:
    """Renders analysis results to the terminal."""

    def __init__(self, width: int = 100, use_color: bool = True):
        self.width = width
        self.use_color = use_color and sys.stdout.isatty()

    # --- Formatting helpers ---

    def _header(self, text: str) -> str:
        return f"\n{'=' * self.width}\n  {text}\n{'=' * self.width}"

    def _subheader(self, text: str) -> str:
        return f"\n{'─' * self.width}\n  {text}\n{'─' * self.width}"

    def _row(self, label: str, value: str, width: int = 30) -> str:
        return f"  {label:<{width}} {value}"

    def _bar(self, value: float, max_val: float = 1.0,
             bar_width: int = 30) -> str:
        filled = int((value / max_val) * bar_width) if max_val > 0 else 0
        filled = max(0, min(filled, bar_width))
        return f"[{'█' * filled}{'░' * (bar_width - filled)}] {value:.2f}"

    def _pct(self, value: float) -> str:
        if value is None:
            return "N/A"
        return f"{value:.1%}"

    def _money(self, value: float) -> str:
        if value is None:
            return "N/A"
        if abs(value) >= 1e9:
            return f"${value / 1e9:.1f}B"
        if abs(value) >= 1e6:
            return f"${value / 1e6:.1f}M"
        if abs(value) >= 1e3:
            return f"${value / 1e3:.1f}K"
        return f"${value:.0f}"

    # --- Company reports ---

    def print_company_overview(self, company: Company,
                                financials: Optional[Financials] = None) -> str:
        """Render a single-company overview."""
        lines = [self._header(f"{company.name} ({company.primary_ticker})")]

        # Identifiers
        lines.append(self._subheader("Identifiers"))
        for ident in company.identifiers:
            lines.append(self._row(ident.id_type, ident.value))

        # Metadata
        if company.metadata:
            lines.append(self._subheader("Company Info"))
            meta = company.metadata
            if meta.sector:
                lines.append(self._row("Sector", meta.sector))
            if meta.industry:
                lines.append(self._row("Industry", meta.industry))
            if meta.country:
                lines.append(self._row("Country", meta.country))
            if meta.market_cap:
                lines.append(self._row("Market Cap", self._money(meta.market_cap)))

        # Latest financials
        if financials and financials.income_statements:
            latest_is = financials.income_statements[-1]
            lines.append(self._subheader(f"Income Statement ({latest_is.period})"))
            lines.append(self._row("Revenue", self._money(latest_is.revenue)))
            lines.append(self._row("Gross Profit", self._money(latest_is.gross_profit)))
            lines.append(self._row("Operating Income", self._money(latest_is.operating_income)))
            lines.append(self._row("Net Income", self._money(latest_is.net_income)))
            lines.append(self._row("Gross Margin", self._pct(latest_is.gross_margin)))
            lines.append(self._row("Operating Margin", self._pct(latest_is.operating_margin)))
            lines.append(self._row("Net Margin", self._pct(latest_is.net_margin)))

        if financials and financials.balance_sheets:
            latest_bs = financials.balance_sheets[-1]
            lines.append(self._subheader(f"Balance Sheet ({latest_bs.period})"))
            lines.append(self._row("Total Assets", self._money(latest_bs.total_assets)))
            lines.append(self._row("Total Liabilities", self._money(latest_bs.total_liabilities)))
            lines.append(self._row("Total Equity", self._money(latest_bs.total_equity)))
            lines.append(self._row("Current Ratio", f"{latest_bs.current_ratio:.2f}"
                                    if latest_bs.current_ratio else "N/A"))
            lines.append(self._row("Debt/Equity", f"{latest_bs.debt_to_equity:.2f}"
                                    if latest_bs.debt_to_equity else "N/A"))

        output = "\n".join(lines)
        print(output)
        return output

    def print_comparison_table(self, companies: list[Company],
                                financials_map: dict[str, Financials]) -> str:
        """Render a multi-company comparison table."""
        if not companies:
            return ""

        tickers = [c.primary_ticker or c.canonical_id for c in companies]
        col_width = max(12, max(len(t) for t in tickers) + 2)

        lines = [self._header("Company Comparison")]

        # Header row
        header = f"  {'Metric':<25}"
        for ticker in tickers:
            header += f" {ticker:>{col_width}}"
        lines.append(header)
        lines.append("  " + "─" * (25 + (col_width + 1) * len(tickers)))

        # Revenue row
        row = f"  {'Revenue':<25}"
        for company in companies:
            fin = financials_map.get(company.canonical_id)
            if fin and fin.income_statements:
                row += f" {self._money(fin.income_statements[-1].revenue):>{col_width}}"
            else:
                row += f" {'N/A':>{col_width}}"
        lines.append(row)

        # Net Income
        row = f"  {'Net Income':<25}"
        for company in companies:
            fin = financials_map.get(company.canonical_id)
            if fin and fin.income_statements:
                row += f" {self._money(fin.income_statements[-1].net_income):>{col_width}}"
            else:
                row += f" {'N/A':>{col_width}}"
        lines.append(row)

        # Gross Margin
        row = f"  {'Gross Margin':<25}"
        for company in companies:
            fin = financials_map.get(company.canonical_id)
            if fin and fin.income_statements:
                row += f" {self._pct(fin.income_statements[-1].gross_margin):>{col_width}}"
            else:
                row += f" {'N/A':>{col_width}}"
        lines.append(row)

        # Net Margin
        row = f"  {'Net Margin':<25}"
        for company in companies:
            fin = financials_map.get(company.canonical_id)
            if fin and fin.income_statements:
                row += f" {self._pct(fin.income_statements[-1].net_margin):>{col_width}}"
            else:
                row += f" {'N/A':>{col_width}}"
        lines.append(row)

        output = "\n".join(lines)
        print(output)
        return output

    # --- Query response ---

    def print_query_response(self, response: QueryResponse) -> str:
        """Render a query response with results and explanations."""
        lines = [self._header(f"Query: {response.raw_query}")]

        # Intent & tier
        lines.append(self._row("Intent", response.intent.intent.value))
        lines.append(self._row("Tier", response.intent.tier.value))
        lines.append(self._row("Confidence", f"{response.intent.confidence:.2f}"))
        lines.append(self._row("Time", f"{response.total_time_ms:.0f}ms"))

        # Resolved entities
        if response.resolution.company_ids:
            lines.append(self._row("Companies",
                                    ", ".join(response.resolution.company_ids)))

        # Results
        if response.explained:
            lines.append(self._subheader(f"Top {len(response.explained)} Results"))
            for hit in response.explained:
                lines.append(f"\n  #{hit.rank} (score={hit.final_score:.4f})")
                lines.append(f"    {hit.text_preview}")
                lines.append(f"    {hit.explanation}")

        if response.error:
            lines.append(self._subheader("Error"))
            lines.append(f"  {response.error}")

        output = "\n".join(lines)
        print(output)
        return output

    # --- Monitoring ---

    def print_system_status(self, latency: LatencyStats,
                             quality: QualitySnapshot) -> str:
        """Render system monitoring dashboard."""
        lines = [self._header("System Status")]

        lines.append(self._subheader("Latency"))
        lines.append(self._row("Queries", str(latency.count)))
        lines.append(self._row("Mean", f"{latency.mean_ms:.0f}ms"))
        lines.append(self._row("P50", f"{latency.p50_ms:.0f}ms"))
        lines.append(self._row("P95", f"{latency.p95_ms:.0f}ms"))
        lines.append(self._row("P99", f"{latency.p99_ms:.0f}ms"))

        lines.append(self._subheader("Quality"))
        lines.append(self._row("Avg Top-1 Score",
                                f"{quality.avg_top1_score:.4f}"))
        lines.append(self._row("Avg Top-10 Score",
                                f"{quality.avg_top10_score:.4f}"))
        lines.append(self._row("Empty Result Rate",
                                self._pct(quality.empty_result_rate)))

        if quality.tier_distribution:
            lines.append(self._subheader("Tier Distribution"))
            for tier, count in sorted(quality.tier_distribution.items()):
                lines.append(self._row(f"Tier {tier}", str(count)))

        output = "\n".join(lines)
        print(output)
        return output
