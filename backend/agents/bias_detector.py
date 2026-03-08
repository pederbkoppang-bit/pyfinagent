"""
LLM Bias Detector — checks for systematic biases in AI-generated analysis.

Detects:
  - Tech-sector bias (large-cap / tech favoritism)
  - Confirmation bias (ignoring contradictory signals)
  - Recency bias (over-weighting recent data)
  - Anchoring bias (over-relying on a single data point)
  - Source diversity issues (narrow evidence base)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Historical recommendation base rates (approximate market distribution)
BASE_RATES = {
    "STRONG_BUY": 0.08,
    "BUY": 0.30,
    "HOLD": 0.35,
    "SELL": 0.20,
    "STRONG_SELL": 0.07,
}

TECH_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "CRM", "ADBE", "ORCL", "NFLX", "AVGO", "QCOM",
    "NOW", "SNOW", "PLTR", "UBER", "SQ", "SHOP", "COIN", "PYPL",
}

LARGE_CAP_THRESHOLD = 100_000_000_000  # $100B


@dataclass
class BiasFlag:
    bias_type: str  # tech_bias, confirmation_bias, recency_bias, anchoring, source_diversity
    severity: str   # LOW, MEDIUM, HIGH
    description: str
    evidence: str
    adjustment_suggestion: Optional[str] = None


@dataclass
class BiasReport:
    flags: list[BiasFlag] = field(default_factory=list)
    raw_score: Optional[float] = None
    adjusted_score: Optional[float] = None
    bias_count: int = 0

    def to_dict(self) -> dict:
        return {
            "flags": [asdict(f) for f in self.flags],
            "raw_score": self.raw_score,
            "adjusted_score": self.adjusted_score,
            "bias_count": self.bias_count,
        }


def detect_biases(
    ticker: str,
    recommendation: str,
    score: float,
    enrichment_signals: dict,
    debate_result: dict,
    quant_data: dict,
) -> dict:
    """
    Run bias detection on a completed analysis.

    Returns a BiasReport dict with all detected bias flags.
    """
    report = BiasReport(raw_score=score)
    flags: list[BiasFlag] = []

    # 1. Tech-sector / large-cap bias
    flags.extend(_check_tech_bias(ticker, recommendation, score, quant_data))

    # 2. Confirmation bias
    flags.extend(_check_confirmation_bias(recommendation, enrichment_signals, debate_result))

    # 3. Recency bias (if report leans heavily on recent short-term data)
    flags.extend(_check_recency_bias(enrichment_signals))

    # 4. Anchoring bias (single dominant signal)
    flags.extend(_check_anchoring_bias(enrichment_signals))

    # 5. Source diversity
    flags.extend(_check_source_diversity(enrichment_signals))

    report.flags = flags
    report.bias_count = len(flags)

    # Compute adjusted score
    high_count = sum(1 for f in flags if f.severity == "HIGH")
    medium_count = sum(1 for f in flags if f.severity == "MEDIUM")
    penalty = high_count * 0.3 + medium_count * 0.1
    report.adjusted_score = round(max(0.0, min(10.0, score - penalty)), 2)

    logger.info(f"Bias Detector for {ticker}: {len(flags)} flags, raw={score} adjusted={report.adjusted_score}")
    return report.to_dict()


def _check_tech_bias(ticker: str, recommendation: str, score: float, quant_data: dict) -> list[BiasFlag]:
    flags = []
    is_tech = ticker.upper() in TECH_TICKERS

    market_cap = 0
    try:
        yf = quant_data.get("yf_data", {})
        market_cap = yf.get("valuation", {}).get("Market Cap", 0) or 0
    except (AttributeError, TypeError):
        pass

    is_large_cap = market_cap > LARGE_CAP_THRESHOLD

    if is_tech and recommendation.upper() in ("STRONG_BUY", "BUY") and score >= 7.5:
        flags.append(BiasFlag(
            bias_type="tech_bias",
            severity="MEDIUM",
            description=f"{ticker} is a well-known tech stock. LLMs tend to favor tech/large-cap names.",
            evidence=f"Recommendation: {recommendation}, Score: {score}. Historical LLM analyses show ~15% higher scores for FAANG+ stocks.",
            adjustment_suggestion="Cross-check with sector-neutral valuation metrics. Compare score against non-tech peers.",
        ))

    if is_large_cap and not is_tech and recommendation.upper() in ("STRONG_BUY", "BUY") and score >= 8.0:
        flags.append(BiasFlag(
            bias_type="tech_bias",
            severity="LOW",
            description=f"{ticker} is a large-cap (${ market_cap / 1e9:.0f}B). LLMs may favor well-known companies.",
            evidence=f"Market cap ${market_cap / 1e9:.0f}B exceeds large-cap threshold.",
            adjustment_suggestion="Verify the bullish thesis holds for smaller competitors in the same sector.",
        ))

    return flags


def _check_confirmation_bias(recommendation: str, signals: dict, debate: dict) -> list[BiasFlag]:
    flags = []

    # Count bullish vs bearish signals
    bullish = 0
    bearish = 0
    for key, sig in signals.items():
        s = (sig.get("signal", "") or "").upper()
        if "BULL" in s or "RISING" in s or "BREAKOUT" in s or "TAILWIND" in s or "OPPORTUNITY" in s:
            bullish += 1
        elif "BEAR" in s or "DECLIN" in s or "RISK" in s or "LAGGING" in s:
            bearish += 1

    rec = recommendation.upper()
    is_bullish_rec = rec in ("STRONG_BUY", "BUY")
    is_bearish_rec = rec in ("STRONG_SELL", "SELL")

    # Bullish recommendation despite significant bearish signals
    if is_bullish_rec and bearish >= 3:
        flags.append(BiasFlag(
            bias_type="confirmation_bias",
            severity="HIGH",
            description=f"Bullish recommendation ({recommendation}) despite {bearish} bearish signals out of {bullish + bearish} total.",
            evidence=f"Bullish signals: {bullish}, Bearish signals: {bearish}. Model may be discounting contrarian evidence.",
            adjustment_suggestion="Review each bearish signal individually. Consider if any represent material risks being ignored.",
        ))
    elif is_bearish_rec and bullish >= 3:
        flags.append(BiasFlag(
            bias_type="confirmation_bias",
            severity="HIGH",
            description=f"Bearish recommendation ({recommendation}) despite {bullish} bullish signals.",
            evidence=f"Bullish signals: {bullish}, Bearish signals: {bearish}.",
            adjustment_suggestion="Review bullish signals individually. Ensure not overlooking positive catalysts.",
        ))

    # Check debate dissent — many dissenters suggests confirmation bias
    dissent = debate.get("dissent_registry", [])
    if len(dissent) >= 3:
        dissent_agents = ", ".join(d.get("agent", "?") for d in dissent[:5])
        flags.append(BiasFlag(
            bias_type="confirmation_bias",
            severity="MEDIUM",
            description=f"{len(dissent)} agents dissented from the consensus. Significant disagreement was overridden.",
            evidence=f"Dissenting agents: {dissent_agents}",
            adjustment_suggestion="Review the dissent registry and consider whether minority views represent material risks.",
        ))

    return flags


def _check_recency_bias(signals: dict) -> list[BiasFlag]:
    flags = []
    # Check if alt_data (Google Trends, which is very short-term) is the dominant signal
    alt_signal = (signals.get("alt_data", {}).get("signal", "") or "").upper()
    if "RISING_STRONG" in alt_signal:
        flags.append(BiasFlag(
            bias_type="recency_bias",
            severity="LOW",
            description="Google Trends shows 'RISING_STRONG', a very short-term indicator that may not reflect fundamentals.",
            evidence=f"Alt data signal: {alt_signal}. Short-term search interest surges often revert within weeks.",
            adjustment_suggestion="Weigh long-term fundamentals (10-K analysis, competitive position) more heavily than short-term trends.",
        ))
    return flags


def _check_anchoring_bias(signals: dict) -> list[BiasFlag]:
    flags = []
    strong_signals = []
    for key, sig in signals.items():
        s = (sig.get("signal", "") or "").upper()
        if "STRONG" in s or "BREAKOUT" in s or "EXTREME" in s or "DOUBLE" in s:
            strong_signals.append(key)

    # If only 1 strong signal, recommendation might be anchored to it
    if len(strong_signals) == 1:
        flags.append(BiasFlag(
            bias_type="anchoring",
            severity="LOW",
            description=f"Only one extreme signal detected ({strong_signals[0]}). Score may be anchored to this single data point.",
            evidence=f"Strong signal from: {strong_signals[0]}. Other signals are moderate/neutral.",
            adjustment_suggestion="Verify the thesis holds without this single strong signal. Would the recommendation change?",
        ))

    return flags


def _check_source_diversity(signals: dict) -> list[BiasFlag]:
    flags = []
    error_count = sum(1 for sig in signals.values() if (sig.get("signal", "") or "").upper() == "ERROR")
    total = len(signals)

    if error_count >= 3:
        flags.append(BiasFlag(
            bias_type="source_diversity",
            severity="HIGH",
            description=f"{error_count} out of {total} enrichment signals returned errors. Analysis is based on incomplete data.",
            evidence=f"Failed signals: {[k for k, v in signals.items() if (v.get('signal', '') or '').upper() == 'ERROR']}",
            adjustment_suggestion="Treat this analysis with lower confidence. Missing data sources reduce coverage.",
        ))
    elif error_count >= 2:
        flags.append(BiasFlag(
            bias_type="source_diversity",
            severity="MEDIUM",
            description=f"{error_count} enrichment signals failed. Some analysis dimensions are missing.",
            evidence=f"Failed signals: {[k for k, v in signals.items() if (v.get('signal', '') or '').upper() == 'ERROR']}",
            adjustment_suggestion="Consider re-running failed signals or adjusting confidence downward.",
        ))

    return flags
