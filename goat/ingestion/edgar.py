"""EDGAR API client for fetching SEC filings.

Provides:
  - Company lookup by ticker/CIK
  - Filing index retrieval (10-K, 10-Q, 8-K)
  - Filing document download
  - Rate-limited requests (SEC mandates ≤10 req/sec)

Uses the SEC EDGAR FULL-TEXT SEARCH API and EFTS.
Requires a User-Agent header per SEC EDGAR policy.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import json
import logging
import urllib.request
import urllib.parse
import urllib.error

logger = logging.getLogger(__name__)

# SEC requires a User-Agent with contact info
_DEFAULT_USER_AGENT = "GOAT-Investment-Analyzer admin@example.com"
_BASE_URL = "https://efts.sec.gov/LATEST"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
_RATE_LIMIT_DELAY = 0.12  # ~8 req/sec to stay under 10/sec limit


@dataclass
class EdgarFiling:
    """Metadata for a single SEC filing."""
    accession_number: str
    filing_type: str
    filing_date: str
    primary_document: str
    description: str = ""
    company_name: str = ""
    cik: str = ""
    period_of_report: str = ""


@dataclass
class EdgarSearchResult:
    """Result from EDGAR full-text search."""
    filings: list[EdgarFiling] = field(default_factory=list)
    total_hits: int = 0


class EdgarClient:
    """Client for SEC EDGAR API.

    Note: Set user_agent to your name and email per SEC policy.
    """

    def __init__(self, user_agent: str = _DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self._last_request_time = 0.0

    def get_company_filings(self, cik: str,
                             filing_type: str = "10-K",
                             count: int = 10) -> list[EdgarFiling]:
        """Get recent filings for a company by CIK.

        Args:
            cik: SEC CIK number (with or without leading zeros)
            filing_type: Filing type filter (10-K, 10-Q, 8-K, etc.)
            count: Maximum number of filings to return
        """
        # Pad CIK to 10 digits
        cik_padded = cik.lstrip("0").zfill(10)
        url = f"{_SUBMISSIONS_URL}/CIK{cik_padded}.json"

        data = self._get_json(url)
        if not data:
            return []

        filings = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        company_name = data.get("name", "")

        for i in range(min(len(forms), len(dates))):
            if forms[i] == filing_type:
                filing = EdgarFiling(
                    accession_number=accessions[i] if i < len(accessions) else "",
                    filing_type=forms[i],
                    filing_date=dates[i],
                    primary_document=primary_docs[i] if i < len(primary_docs) else "",
                    description=descriptions[i] if i < len(descriptions) else "",
                    company_name=company_name,
                    cik=cik_padded,
                )
                filings.append(filing)
                if len(filings) >= count:
                    break

        return filings

    def search_filings(self, query: str,
                        filing_type: Optional[str] = None,
                        date_range: Optional[tuple[str, str]] = None,
                        count: int = 10) -> EdgarSearchResult:
        """Full-text search of EDGAR filings.

        Args:
            query: Search query text
            filing_type: Optional filter (e.g., "10-K")
            date_range: Optional (start, end) dates as "YYYY-MM-DD"
            count: Max results
        """
        params = {
            "q": query,
            "dateRange": "custom" if date_range else None,
            "startdt": date_range[0] if date_range else None,
            "enddt": date_range[1] if date_range else None,
            "forms": filing_type or None,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        url = f"{_BASE_URL}/search-index?{urllib.parse.urlencode(params)}"
        data = self._get_json(url)

        if not data:
            return EdgarSearchResult()

        hits = data.get("hits", {})
        total = hits.get("total", {}).get("value", 0)

        filings = []
        for hit in hits.get("hits", [])[:count]:
            source = hit.get("_source", {})
            filing = EdgarFiling(
                accession_number=source.get("file_num", ""),
                filing_type=source.get("form_type", ""),
                filing_date=source.get("file_date", ""),
                primary_document="",
                company_name=source.get("entity_name", ""),
                period_of_report=source.get("period_of_report", ""),
            )
            filings.append(filing)

        return EdgarSearchResult(filings=filings, total_hits=total)

    def download_filing(self, filing: EdgarFiling) -> Optional[str]:
        """Download the primary document of a filing."""
        if not filing.accession_number or not filing.primary_document:
            return None

        acc_no_clean = filing.accession_number.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{filing.cik.lstrip('0')}/{acc_no_clean}/"
               f"{filing.primary_document}")

        return self._get_text(url)

    def _get_json(self, url: str) -> Optional[dict]:
        """Make a rate-limited GET request and parse JSON."""
        self._rate_limit()
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", self.user_agent)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.warning("EDGAR HTTP error %d for %s", e.code, url)
            return None
        except Exception as e:
            logger.warning("EDGAR request failed for %s: %s", url, e)
            return None

    def _get_text(self, url: str) -> Optional[str]:
        """Make a rate-limited GET request and return text."""
        self._rate_limit()
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", self.user_agent)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("EDGAR download failed for %s: %s", url, e)
            return None

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()
