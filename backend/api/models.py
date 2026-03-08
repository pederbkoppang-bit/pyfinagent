"""
Pydantic models for API requests/responses and internal data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Recommendation(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


# ── Scoring ──────────────────────────────────────────────────────────

class ScoringMatrix(BaseModel):
    pillar_1_corporate: float = Field(..., ge=1, le=10)
    pillar_2_industry: float = Field(..., ge=1, le=10)
    pillar_3_valuation: float = Field(..., ge=1, le=10)
    pillar_4_sentiment: float = Field(..., ge=1, le=10)
    pillar_5_governance: float = Field(..., ge=1, le=10)


class RecommendationDetail(BaseModel):
    action: str
    justification: str


class SynthesisReport(BaseModel):
    scoring_matrix: ScoringMatrix
    recommendation: RecommendationDetail
    final_summary: str
    key_risks: list[str]
    final_weighted_score: Optional[float] = None


# ── API Models ───────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Z0-9.]+$")


class AnalysisResponse(BaseModel):
    analysis_id: str
    ticker: str
    status: AnalysisStatus


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    ticker: str
    status: AnalysisStatus
    current_step: Optional[str] = None
    steps_completed: list[str] = []
    error: Optional[str] = None
    report: Optional[SynthesisReport] = None


class ReportSummary(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    analysis_date: datetime
    final_score: float
    recommendation: str
    summary: str


class PerformanceStats(BaseModel):
    total_recommendations: int = 0
    wins: int = 0
    losses: int = 0
    avg_return: float = 0
    win_rate: float = 0
    benchmark_beat_rate: float = 0


# ── Agent Step Events (for status streaming) ────────────────────────

class AgentStepEvent(BaseModel):
    step: str
    agent: str
    status: str  # "started", "completed", "failed"
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[dict] = None
