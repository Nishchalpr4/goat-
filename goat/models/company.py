"""Company entity model — the canonical representation of an analyzed company."""

from dataclasses import dataclass, field
from typing import Optional

from goat.models.financials import Financials


@dataclass
class Identifier:
    """A single company identifier (ticker, ISIN, CUSIP, CIK, etc.)."""
    id_type: str  # "ticker", "isin", "cusip", "cik", "lei"
    value: str
    exchange: Optional[str] = None  # e.g. "NYSE", "NASDAQ", "LSE"
    primary: bool = False


@dataclass
class ManagementMember:
    """Key executive or board member."""
    name: str
    title: str
    since_year: Optional[int] = None
    compensation: Optional[float] = None
    insider_ownership_pct: Optional[float] = None


@dataclass
class CompanyMetadata:
    """Descriptive metadata for a company."""
    sector: str = ""
    industry: str = ""
    sub_industry: str = ""
    country: str = ""
    region: str = ""
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    ipo_year: Optional[int] = None
    fiscal_year_end_month: int = 12
    website: str = ""
    description: str = ""


@dataclass
class Company:
    """Canonical company entity — the root object for all analysis."""
    canonical_id: str  # stable internal ID (e.g. uuid)
    name: str
    identifiers: list[Identifier] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # "Apple Inc.", "Apple", "AAPL"
    metadata: CompanyMetadata = field(default_factory=CompanyMetadata)
    financials: Financials = field(default_factory=Financials)
    management: list[ManagementMember] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def primary_ticker(self) -> Optional[str]:
        for ident in self.identifiers:
            if ident.id_type == "ticker" and ident.primary:
                return ident.value
        tickers = [i for i in self.identifiers if i.id_type == "ticker"]
        return tickers[0].value if tickers else None

    @property
    def cik(self) -> Optional[str]:
        for ident in self.identifiers:
            if ident.id_type == "cik":
                return ident.value
        return None

    def add_identifier(self, id_type: str, value: str, **kwargs) -> None:
        self.identifiers.append(Identifier(id_type=id_type, value=value, **kwargs))

    def merge_aliases(self, new_aliases: list[str]) -> None:
        """Add aliases that don't already exist (case-insensitive dedup)."""
        existing = {a.lower() for a in self.aliases}
        for alias in new_aliases:
            if alias.lower() not in existing:
                self.aliases.append(alias)
                existing.add(alias.lower())
