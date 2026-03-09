"""Batch data loading for companies, financials, and lexicons.

Supports loading from:
  - JSON files (company data, financials, lexicon entries)
  - CSV files (tabular financial data, screening data)
  - Directories (batch processing of multiple files)
"""

from pathlib import Path
from typing import Optional, Union
import json
import csv
import logging

from goat.models.company import Company, Identifier, CompanyMetadata
from goat.models.financials import (
    Financials, IncomeStatement, BalanceSheet, CashFlowStatement,
)
from goat.models.lexicon import (
    Lexicon, LexiconEntry, LexiconEntryType,
    TickerMapping, AbbreviationEntry, SchemaTermMapping,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads structured data from files into domain models."""

    def load_companies_json(self, path: Union[str, Path]) -> list[Company]:
        """Load companies from a JSON file.

        Expected format:
        [
          {
            "canonical_id": "AAPL",
            "name": "Apple Inc.",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "cik": "0000320193",
            "aliases": ["Apple", "AAPL"]
          }
        ]
        """
        data = self._read_json(path)
        if not isinstance(data, list):
            data = [data]

        companies = []
        for item in data:
            identifiers = []
            if item.get("ticker"):
                identifiers.append(
                    Identifier(id_type="ticker", value=item["ticker"],
                               exchange=item.get("exchange", ""))
                )
            if item.get("cik"):
                identifiers.append(
                    Identifier(id_type="CIK", value=item["cik"])
                )
            if item.get("isin"):
                identifiers.append(
                    Identifier(id_type="ISIN", value=item["isin"])
                )

            company = Company(
                canonical_id=item.get("canonical_id", item.get("ticker", "")),
                name=item.get("name", ""),
                identifiers=identifiers,
                aliases=item.get("aliases", []),
                metadata=CompanyMetadata(
                    sector=item.get("sector", ""),
                    industry=item.get("industry", ""),
                    market_cap=item.get("market_cap"),
                    country=item.get("country", "US"),
                    description=item.get("description", ""),
                ),
            )
            companies.append(company)

        logger.info("Loaded %d companies from %s", len(companies), path)
        return companies

    def load_financials_json(self, path: Union[str, Path],
                              company_id: str) -> Financials:
        """Load financial statements from JSON.

        Expected format:
        {
          "2023": {
            "income_statement": { "revenue": 383285, ... },
            "balance_sheet": { "total_assets": 352583, ... },
            "cash_flow": { "operating_cash_flow": 110543, ... }
          }
        }
        """
        data = self._read_json(path)
        financials = Financials(company_id=company_id)

        for year_str, year_data in data.items():
            year = int(year_str)

            if "income_statement" in year_data:
                is_data = year_data["income_statement"]
                financials.income_statements.append(
                    IncomeStatement(
                        period=f"FY{year}",
                        revenue=is_data.get("revenue", 0),
                        cost_of_revenue=is_data.get("cost_of_revenue", 0),
                        gross_profit=is_data.get("gross_profit", 0),
                        operating_expenses=is_data.get("operating_expenses", 0),
                        operating_income=is_data.get("operating_income", 0),
                        net_income=is_data.get("net_income", 0),
                        eps_basic=is_data.get("eps_basic", 0),
                        eps_diluted=is_data.get("eps_diluted", 0),
                        shares_outstanding=is_data.get("shares_outstanding", 0),
                    )
                )

            if "balance_sheet" in year_data:
                bs_data = year_data["balance_sheet"]
                financials.balance_sheets.append(
                    BalanceSheet(
                        period=f"FY{year}",
                        total_assets=bs_data.get("total_assets", 0),
                        total_liabilities=bs_data.get("total_liabilities", 0),
                        total_equity=bs_data.get("total_equity", 0),
                        current_assets=bs_data.get("current_assets", 0),
                        current_liabilities=bs_data.get("current_liabilities", 0),
                        long_term_debt=bs_data.get("long_term_debt", 0),
                        cash_and_equivalents=bs_data.get("cash_and_equivalents", 0),
                    )
                )

            if "cash_flow" in year_data:
                cf_data = year_data["cash_flow"]
                financials.cash_flows.append(
                    CashFlowStatement(
                        period=f"FY{year}",
                        operating_cash_flow=cf_data.get("operating_cash_flow", 0),
                        investing_cash_flow=cf_data.get("investing_cash_flow", 0),
                        financing_cash_flow=cf_data.get("financing_cash_flow", 0),
                        capital_expenditures=cf_data.get("capital_expenditures", 0),
                        free_cash_flow=cf_data.get("free_cash_flow", 0),
                    )
                )

        return financials

    def load_financials_csv(self, path: Union[str, Path],
                             company_id: str) -> Financials:
        """Load financial data from CSV.

        Expected columns: period, metric, value
        """
        rows = self._read_csv(path)
        financials = Financials(company_id=company_id)

        # Group by period
        by_period: dict[str, dict[str, float]] = {}
        for row in rows:
            period = row.get("period", "")
            metric = row.get("metric", "")
            try:
                value = float(row.get("value", 0))
            except (ValueError, TypeError):
                continue
            by_period.setdefault(period, {})[metric] = value

        for period, metrics in by_period.items():
            if "revenue" in metrics:
                financials.income_statements.append(
                    IncomeStatement(
                        period=period,
                        revenue=metrics.get("revenue", 0),
                        cost_of_revenue=metrics.get("cost_of_revenue", 0),
                        gross_profit=metrics.get("gross_profit", 0),
                        operating_income=metrics.get("operating_income", 0),
                        net_income=metrics.get("net_income", 0),
                    )
                )

        return financials

    def load_lexicon_json(self, path: Union[str, Path]) -> Lexicon:
        """Load a lexicon from JSON.

        Expected format:
        {
          "version": "1.0",
          "entries": [...],
          "tickers": [...],
          "abbreviations": [...],
          "schema_mappings": [...]
        }
        """
        data = self._read_json(path)
        lexicon = Lexicon(version=data.get("version", "1.0"))

        for entry_data in data.get("entries", []):
            entry = LexiconEntry(
                canonical=entry_data["canonical"],
                surface_forms=entry_data.get("surface_forms", []),
                entry_type=LexiconEntryType(
                    entry_data.get("type", "synonym")
                ),
            )
            lexicon.entries.append(entry)

        for ticker_data in data.get("tickers", []):
            mapping = TickerMapping(
                ticker=ticker_data["ticker"],
                company_name=ticker_data.get("company_name", ""),
                variants=ticker_data.get("variants", []),
            )
            lexicon.tickers.append(mapping)

        for abbr_data in data.get("abbreviations", []):
            abbr = AbbreviationEntry(
                abbreviation=abbr_data["abbreviation"],
                expansion=abbr_data["expansion"],
                context=abbr_data.get("context", ""),
            )
            lexicon.abbreviations.append(abbr)

        for schema_data in data.get("schema_mappings", []):
            mapping = SchemaTermMapping(
                surface_form=schema_data["surface_form"],
                canonical_concept=schema_data["canonical_concept"],
                xbrl_element=schema_data.get("xbrl_element", ""),
                taxonomy=schema_data.get("taxonomy", "us-gaap"),
            )
            lexicon.schema_mappings.append(mapping)

        return lexicon

    def load_directory(self, directory: Union[str, Path],
                       pattern: str = "*.json") -> list[dict]:
        """Load all files matching pattern from a directory."""
        dirpath = Path(directory)
        results = []
        for filepath in sorted(dirpath.glob(pattern)):
            try:
                data = self._read_json(filepath)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", filepath, e)
        return results

    @staticmethod
    def _read_json(path: Union[str, Path]) -> Union[dict, list]:
        """Read and parse a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _read_csv(path: Union[str, Path]) -> list[dict]:
        """Read a CSV file into list of dicts."""
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
