"""Financial statements and metrics data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IncomeStatement:
    """Annual income statement data."""
    year: int
    revenue: float = 0.0
    cost_of_revenue: float = 0.0
    gross_profit: float = 0.0
    operating_expenses: float = 0.0
    operating_income: float = 0.0
    interest_expense: float = 0.0
    income_before_tax: float = 0.0
    income_tax: float = 0.0
    net_income: float = 0.0
    ebitda: float = 0.0
    eps: float = 0.0
    shares_outstanding: float = 0.0
    rd_expense: float = 0.0
    sga_expense: float = 0.0

    @property
    def gross_margin(self) -> Optional[float]:
        return self.gross_profit / self.revenue if self.revenue else None

    @property
    def operating_margin(self) -> Optional[float]:
        return self.operating_income / self.revenue if self.revenue else None

    @property
    def net_margin(self) -> Optional[float]:
        return self.net_income / self.revenue if self.revenue else None

    @property
    def tax_rate(self) -> Optional[float]:
        return self.income_tax / self.income_before_tax if self.income_before_tax else None


@dataclass
class BalanceSheet:
    """Annual balance sheet data."""
    year: int
    cash_and_equivalents: float = 0.0
    short_term_investments: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    total_current_assets: float = 0.0
    property_plant_equipment: float = 0.0
    goodwill: float = 0.0
    intangible_assets: float = 0.0
    total_assets: float = 0.0
    accounts_payable: float = 0.0
    short_term_debt: float = 0.0
    total_current_liabilities: float = 0.0
    long_term_debt: float = 0.0
    total_liabilities: float = 0.0
    total_shareholders_equity: float = 0.0
    retained_earnings: float = 0.0
    total_equity: float = 0.0

    @property
    def current_ratio(self) -> Optional[float]:
        return self.total_current_assets / self.total_current_liabilities if self.total_current_liabilities else None

    @property
    def debt_to_equity(self) -> Optional[float]:
        total_debt = self.short_term_debt + self.long_term_debt
        return total_debt / self.total_equity if self.total_equity else None

    @property
    def net_cash(self) -> float:
        return self.cash_and_equivalents + self.short_term_investments - self.short_term_debt - self.long_term_debt


@dataclass
class CashFlowStatement:
    """Annual cash flow statement data."""
    year: int
    operating_cash_flow: float = 0.0
    capital_expenditures: float = 0.0
    acquisitions: float = 0.0
    investments: float = 0.0
    dividends_paid: float = 0.0
    share_repurchases: float = 0.0
    debt_issued: float = 0.0
    debt_repaid: float = 0.0
    free_cash_flow: float = 0.0
    depreciation_amortization: float = 0.0

    @property
    def capex_to_ocf(self) -> Optional[float]:
        return abs(self.capital_expenditures) / self.operating_cash_flow if self.operating_cash_flow else None


@dataclass
class Financials:
    """Collection of financial statements over multiple years."""
    income_statements: list[IncomeStatement] = field(default_factory=list)
    balance_sheets: list[BalanceSheet] = field(default_factory=list)
    cash_flow_statements: list[CashFlowStatement] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        return sorted({s.year for s in self.income_statements})

    @property
    def latest_income(self) -> Optional[IncomeStatement]:
        return max(self.income_statements, key=lambda s: s.year) if self.income_statements else None

    @property
    def latest_balance(self) -> Optional[BalanceSheet]:
        return max(self.balance_sheets, key=lambda s: s.year) if self.balance_sheets else None

    @property
    def latest_cashflow(self) -> Optional[CashFlowStatement]:
        return max(self.cash_flow_statements, key=lambda s: s.year) if self.cash_flow_statements else None

    def income_for_year(self, year: int) -> Optional[IncomeStatement]:
        return next((s for s in self.income_statements if s.year == year), None)

    def balance_for_year(self, year: int) -> Optional[BalanceSheet]:
        return next((s for s in self.balance_sheets if s.year == year), None)

    def cashflow_for_year(self, year: int) -> Optional[CashFlowStatement]:
        return next((s for s in self.cash_flow_statements if s.year == year), None)
