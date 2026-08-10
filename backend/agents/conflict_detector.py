"""
Knowledge Conflict Detector — compares LLM parametric knowledge against real-time data.

Identifies discrepancies between what the LLM "believes" (from training data) and what
current market data shows. These conflicts highlight where the model's knowledge is stale.
"""

import logging
from dataclasses import asdict, dataclass, field
from backend.services.recommendation_vocab import canonical_recommendation, STRONG_BUY, BUY, SELL  # phase-86.22

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeConflict:
    field: str
    llm_belief: str
    actual_value: str
    severity: str  # LOW, MEDIUM, HIGH
    category: str  # price, valuation, fundamentals, market_position
    explanation: str


@dataclass
class ConflictReport:
    conflicts: list[KnowledgeConflict] = field(default_factory=list)
    conflict_count: int = 0
    overall_reliability: str = "HIGH"  # HIGH, MEDIUM, LOW

    def to_dict(self) -> dict:
        return {
            "conflicts": [asdict(c) for c in self.conflicts],
            "conflict_count": self.conflict_count,
            "overall_reliability": self.overall_reliability,
        }


def detect_conflicts(
    ticker: str,
    synthesis_report: dict,
    quant_data: dict,
    enrichment_signals: dict,
) -> dict:
    """
    Compare the synthesis report's claims against real-time quant data.

    Returns a ConflictReport dict with all detected knowledge conflicts.
    """
    report = ConflictReport()
    conflicts: list[KnowledgeConflict] = []

    yf = quant_data.get("yf_data", {})
    valuation = yf.get("valuation", {})
    health = yf.get("health", {})

    # 1. Check score consistency
    conflicts.extend(_check_score_consistency(synthesis_report))

    # 2. Check recommendation vs score alignment
    conflicts.extend(_check_recommendation_alignment(synthesis_report))

    # 3. Check if synthesis mentions outdated price/valuation data
    conflicts.extend(_check_valuation_conflicts(synthesis_report, valuation))

    # 4. Check enrichment signal conflicts (contradictory signals not noted)
    conflicts.extend(_check_signal_contradictions(enrichment_signals))

    report.conflicts = conflicts
    report.conflict_count = len(conflicts)

    if len(conflicts) >= 3:
        report.overall_reliability = "LOW"
    elif len(conflicts) >= 1:
        report.overall_reliability = "MEDIUM"
    else:
        report.overall_reliability = "HIGH"

    logger.info(f"Conflict Detector for {ticker}: {len(conflicts)} conflicts, reliability={report.overall_reliability}")
    return report.to_dict()


def _check_score_consistency(report: dict) -> list[KnowledgeConflict]:
    conflicts = []
    matrix = report.get("scoring_matrix", {})
    weighted_score = report.get("final_weighted_score", 0)

    if not matrix or not weighted_score:
        return conflicts

    # Check if individual pillar scores are wildly inconsistent
    pillar_values = [v for v in matrix.values() if isinstance(v, (int, float))]
    if pillar_values:
        min_p = min(pillar_values)
        max_p = max(pillar_values)
        if max_p - min_p > 5.0:
            conflicts.append(KnowledgeConflict(
                field="scoring_matrix",
                llm_belief=f"Pillar scores range from {min_p} to {max_p}",
                actual_value=f"Weighted score: {weighted_score}",
                severity="MEDIUM",
                category="fundamentals",
                explanation=f"Pillar scores have extreme dispersion (range: {max_p - min_p:.1f}). "
                            "This suggests the model is uncertain about the stock's quality.",
            ))

    return conflicts


def _check_recommendation_alignment(report: dict) -> list[KnowledgeConflict]:
    conflicts = []
    rec = report.get("recommendation", {})
    # phase-86.22: was `.upper()` feeding a chain of SUBSTRING tests below.
    # Two independent defects in one line: the separator gap meant the
    # producer's "Strong Buy" never matched "STRONG_BUY"; and because the chain
    # then fell through to `elif "BUY" in rec_label`, a missed STRONG BUY was
    # graded at the WEAKER 5.5 threshold instead of 7.0 -- so the strictest
    # check silently became the loosest. Substring matching is its own hazard
    # here regardless: "STRONG_SELL" contains "SELL".
    rec_canonical = canonical_recommendation(
        rec.get("recommendation", "") or rec.get("label", ""))
    rec_label = rec_canonical or ""
    score = report.get("final_weighted_score", 0) or 0

    if not rec_label or not score:
        return conflicts

    # Strong Buy with low score
    if rec_canonical == STRONG_BUY and score < 7.0:
        conflicts.append(KnowledgeConflict(
            field="recommendation",
            llm_belief=f"STRONG_BUY recommendation",
            actual_value=f"Weighted score: {score}",
            severity="HIGH",
            category="fundamentals",
            explanation=f"STRONG_BUY typically requires score ≥ 7.0, but weighted score is {score}. "
                        "The model's recommendation conflicts with its own scoring.",
        ))
    elif rec_canonical == BUY and score < 5.5:
        conflicts.append(KnowledgeConflict(
            field="recommendation",
            llm_belief=f"BUY recommendation",
            actual_value=f"Weighted score: {score}",
            severity="HIGH",
            category="fundamentals",
            explanation=f"BUY recommendation with weighted score {score} (< 5.5) indicates internal inconsistency.",
        ))
    elif rec_canonical == SELL and score > 6.0:
        conflicts.append(KnowledgeConflict(
            field="recommendation",
            llm_belief=f"SELL recommendation",
            actual_value=f"Weighted score: {score}",
            severity="HIGH",
            category="fundamentals",
            explanation=f"SELL recommendation conflicts with above-average score of {score}.",
        ))

    return conflicts


def _check_valuation_conflicts(report: dict, valuation: dict) -> list[KnowledgeConflict]:
    conflicts = []
    if not valuation:
        return conflicts

    current_price = valuation.get("Current Price")
    pe_trailing = valuation.get("P/E (Trailing)")
    pe_forward = valuation.get("P/E (Forward)")

    # If synthesis summary mentions specific price but it doesn't match current
    summary = report.get("final_summary", "")
    rec = report.get("recommendation", {})
    rec_text = rec.get("rationale", "") or rec.get("recommendation", "")

    if pe_trailing and pe_forward and isinstance(pe_trailing, (int, float)) and isinstance(pe_forward, (int, float)):
        if pe_trailing > 0 and pe_forward > 0:
            ratio = pe_trailing / pe_forward
            if ratio > 1.5:
                conflicts.append(KnowledgeConflict(
                    field="valuation",
                    llm_belief="Current P/E vs Forward P/E gap is substantial",
                    actual_value=f"Trailing P/E: {pe_trailing:.1f}, Forward P/E: {pe_forward:.1f} (ratio: {ratio:.2f})",
                    severity="LOW",
                    category="valuation",
                    explanation="Large gap between trailing and forward P/E suggests significant earnings growth expectations. "
                                "If the model doesn't account for this, its valuation assessment may be stale.",
                ))

    return conflicts


def _check_signal_contradictions(signals: dict) -> list[KnowledgeConflict]:
    """Check for strongly contradictory enrichment signals that weren't surfaced."""
    conflicts = []

    bullish_signals = []
    bearish_signals = []

    for key, sig in signals.items():
        s = (sig.get("signal", "") or "").upper()
        if "STRONG_BULL" in s or "BREAKOUT" in s or "DOUBLE_TAILWIND" in s:
            bullish_signals.append(key)
        elif "STRONG_BEAR" in s or "EXTREME_RISK" in s or "ANOMALY_RISK" in s:
            bearish_signals.append(key)

    if bullish_signals and bearish_signals:
        conflicts.append(KnowledgeConflict(
            field="enrichment_signals",
            llm_belief=f"Strong bullish: {bullish_signals}",
            actual_value=f"Strong bearish: {bearish_signals}",
            severity="MEDIUM",
            category="market_position",
            explanation=f"Strongly contradictory signals exist simultaneously — "
                        f"bullish ({', '.join(bullish_signals)}) vs bearish ({', '.join(bearish_signals)}). "
                        "This represents genuine market uncertainty that should be weighted heavily.",
        ))

    return conflicts
