"""Data validation for ingested documents and financial data.

Validates:
  - Required fields presence
  - Data type correctness
  - Financial data sanity checks (e.g., revenue > 0)
  - Cross-field consistency (e.g., gross_profit = revenue - COGS)
"""

from dataclasses import dataclass, field
import logging

from goat.models.company import Company
from goat.models.financials import IncomeStatement, BalanceSheet, CashFlowStatement
from goat.models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: str  # "error", "warning", "info"
    field: str
    message: str
    value: object = None


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, field_name: str, message: str, value=None):
        self.valid = False
        self.issues.append(ValidationIssue("error", field_name, message, value))

    def add_warning(self, field_name: str, message: str, value=None):
        self.issues.append(ValidationIssue("warning", field_name, message, value))


class DataValidator:
    """Validates data before ingestion."""

    def validate_company(self, company: Company) -> ValidationResult:
        """Validate a company record."""
        result = ValidationResult()

        if not company.canonical_id:
            result.add_error("canonical_id", "Missing canonical_id")
        if not company.name:
            result.add_error("name", "Missing company name")
        if not company.identifiers:
            result.add_warning("identifiers", "No identifiers provided")

        return result

    def validate_income_statement(self, stmt: IncomeStatement) -> ValidationResult:
        """Validate an income statement."""
        result = ValidationResult()

        if not stmt.period:
            result.add_error("period", "Missing period")
        if stmt.revenue < 0:
            result.add_warning("revenue", "Negative revenue", stmt.revenue)

        # Cross-field checks
        if stmt.gross_profit != 0 and stmt.revenue != 0:
            expected_gp = stmt.revenue - stmt.cost_of_revenue
            if stmt.cost_of_revenue != 0 and abs(stmt.gross_profit - expected_gp) > 1:
                result.add_warning(
                    "gross_profit",
                    f"Gross profit ({stmt.gross_profit}) != "
                    f"Revenue ({stmt.revenue}) - COGS ({stmt.cost_of_revenue})",
                )

        # Margin sanity
        if stmt.revenue > 0:
            net_margin = stmt.net_income / stmt.revenue
            if abs(net_margin) > 2.0:
                result.add_warning(
                    "net_margin",
                    f"Extreme net margin: {net_margin:.1%}",
                )

        return result

    def validate_balance_sheet(self, bs: BalanceSheet) -> ValidationResult:
        """Validate a balance sheet."""
        result = ValidationResult()

        if not bs.period:
            result.add_error("period", "Missing period")

        # Accounting identity
        if bs.total_assets > 0 and bs.total_liabilities > 0:
            expected_equity = bs.total_assets - bs.total_liabilities
            if bs.total_equity != 0 and abs(bs.total_equity - expected_equity) > abs(bs.total_assets * 0.01):
                result.add_warning(
                    "accounting_identity",
                    f"Assets ({bs.total_assets}) != "
                    f"Liabilities ({bs.total_liabilities}) + Equity ({bs.total_equity})",
                )

        return result

    def validate_document(self, doc: Document) -> ValidationResult:
        """Validate a document for ingestion."""
        result = ValidationResult()

        if not doc.doc_id:
            result.add_error("doc_id", "Missing document ID")
        if not doc.content:
            result.add_error("content", "Empty document content")
        elif len(doc.content) < 50:
            result.add_warning("content", "Very short document content",
                               len(doc.content))

        if doc.metadata:
            if not doc.metadata.company_id and not doc.metadata.ticker:
                result.add_warning("metadata",
                                   "No company_id or ticker in metadata")

        return result

    def validate_batch(self, companies: list[Company]) -> list[ValidationResult]:
        """Validate a batch of companies."""
        results = []
        seen_ids = set()
        for company in companies:
            vr = self.validate_company(company)
            if company.canonical_id in seen_ids:
                vr.add_error("canonical_id",
                             f"Duplicate canonical_id: {company.canonical_id}")
            seen_ids.add(company.canonical_id)
            results.append(vr)
        return results
