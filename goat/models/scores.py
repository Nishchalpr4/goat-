"""Scoring models for investment analysis dimensions."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """A single qualitative or quantitative signal from analysis."""
    name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    detail: str = ""
    source: str = ""  # where this signal came from


@dataclass
class AnalysisResult:
    """Result from a single analyzer dimension."""
    analyzer: str  # "financial", "valuation", "moat", etc.
    score: float  # 0–10 scale
    grade: str = ""
    confidence: float = 1.0
    details: dict = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    narrative: str = ""  # human-readable summary

    @property
    def bullish_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.direction == "bullish"]

    @property
    def bearish_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.direction == "bearish"]


@dataclass
class CompositeScore:
    """Weighted composite score across all analysis dimensions."""
    overall_score: float = 0.0
    overall_grade: str = ""
    dimension_scores: dict[str, float] = field(default_factory=dict)
    dimension_grades: dict[str, str] = field(default_factory=dict)
    dimension_results: list[AnalysisResult] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)

    @property
    def top_strengths(self) -> list[AnalysisResult]:
        """Top 3 highest-scoring dimensions."""
        return sorted(self.dimension_results, key=lambda r: r.score, reverse=True)[:3]

    @property
    def top_weaknesses(self) -> list[AnalysisResult]:
        """Bottom 3 lowest-scoring dimensions."""
        return sorted(self.dimension_results, key=lambda r: r.score)[:3]


@dataclass
class CompanyRanking:
    """A company's position in a multi-company comparison."""
    company_id: str
    company_name: str
    ticker: str = ""
    composite: CompositeScore = field(default_factory=CompositeScore)
    rank: int = 0
    percentile: float = 0.0  # within the comparison set
    peer_group: str = ""  # sector/industry label
