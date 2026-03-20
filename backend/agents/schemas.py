"""Pydantic schemas for Gemini structured output enforcement (Phase 3).

Guarantees syntactically valid JSON from every JSON-producing agent via
response_mime_type="application/json" + response_schema=PydanticModel.

Complements Fact Ledger (Phase 2) which enforces semantic correctness.
Structured output guarantees JSON STRUCTURE; Fact Ledger guarantees VALUE ACCURACY.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Citation Model (Phase 6) ────────────────────────────────────

class Citation(BaseModel):
    """A traceable data citation linking a claim to its source system.

    source values: YFIN (Yahoo Finance), SEC (EDGAR), FRED (Federal Reserve),
    AV (Alpha Vantage), or a URL for grounding web sources.
    """
    claim: str = Field(description="The specific claim being cited, e.g. 'P/E ratio is above sector average'")
    source: str = Field(description="Source tag: YFIN, SEC, FRED, AV, or a URL for web grounding sources")
    value: str = Field(description="The exact data value cited, e.g. '28.5x' or '4.33%'")


# ── Synthesis Agent (Step 11) ────────────────────────────────────

class ScoringMatrix(BaseModel):
    pillar_1_corporate: float = Field(description="Corporate Quality score (1-10)")
    pillar_2_industry: float = Field(description="Industry Position score (1-10)")
    pillar_3_valuation: float = Field(description="Valuation score (1-10)")
    pillar_4_sentiment: float = Field(description="Sentiment score (1-10)")
    pillar_5_governance: float = Field(description="Governance score (1-10)")


class Recommendation(BaseModel):
    action: str = Field(description="Strong Buy, Buy, Hold, Sell, or Strong Sell")
    justification: str = Field(description="2-3 sentence justification")


class SynthesisReport(BaseModel):
    scoring_matrix: ScoringMatrix
    recommendation: Recommendation
    final_summary: str = Field(description="3-5 paragraph investment thesis")
    key_risks: list[str] = Field(description="Top 3 key risks")
    citations: list[Citation] = Field(
        default_factory=list,
        description="5-10 key claims with traceable data sources from FACT_LEDGER or grounding",
    )


# ── Critic Agent (Step 12) ───────────────────────────────────────

class CriticIssue(BaseModel):
    type: str = Field(description="hallucination, logic, missing_field, or minor")
    severity: str = Field(description="major or minor")
    description: str


class CriticVerdict(BaseModel):
    verdict: Literal["PASS", "REVISE"]
    issues: list[CriticIssue] = Field(default_factory=list)
    corrected_report: Optional[SynthesisReport] = None


# ── Devil's Advocate (Step 8) ────────────────────────────────────

class DevilsAdvocateResult(BaseModel):
    challenges: list[str] = Field(description="Challenges to both bull and bear cases")
    hidden_risks: list[str] = Field(description="Hidden risks not addressed")
    confidence_adjustment: float = Field(description="Suggested adjustment (-1.0 to 1.0)")
    groupthink_flag: bool = Field(description="Whether groupthink was detected")
    summary: str


# ── Moderator (Step 8) ──────────────────────────────────────────

class Contradiction(BaseModel):
    topic: str
    bull_view: str
    bear_view: str
    resolution: str


class Dissent(BaseModel):
    agent: str
    position: str
    reason: str


class ModeratorConsensus(BaseModel):
    consensus: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    consensus_confidence: float = Field(description="0.0-1.0")
    contradictions: list[Contradiction] = Field(default_factory=list)
    dissent_registry: list[Dissent] = Field(default_factory=list)


# ── Risk Analysts (Step 12c) ────────────────────────────────────

class RiskAnalystArgument(BaseModel):
    position: str = Field(description="The analyst's position argument")
    confidence: float = Field(description="0.0-1.0")
    max_position_pct: float = Field(description="Recommended max position %")


# ── Risk Judge (Step 12c) ───────────────────────────────────────

class RiskLimits(BaseModel):
    stop_loss_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    position_size_pct: Optional[float] = None


class RiskJudgeVerdict(BaseModel):
    decision: Literal["APPROVE_FULL", "APPROVE_REDUCED", "APPROVE_HEDGED", "REJECT"]
    risk_adjusted_confidence: float = Field(description="0.0-1.0")
    recommended_position_pct: float
    risk_level: Literal["LOW", "MODERATE", "HIGH", "EXTREME"]
    reasoning: str
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)
    summary: str
