"""
signal_attribution -- Convert raw analysis dicts into per-trade agent-signal
attribution rows used by the rationale drawer (4.5.5).

Progressive disclosure hierarchy, adapted from the TradingAgents pattern
(Xiao et al., 2024):

    Analyst layer   -- synthesis of market intel, quant, fundamentals
    Debate layer    -- Bull vs Bear arguments
    Trader layer    -- recommendation + final_score
    Risk layer      -- Risk Judge decision + position sizing

Each signal dict has a stable shape: {agent, role, rationale, weight}.

PII / secret protection: any text coming from LLM-produced rationales is
passed through `redact_pii` before persistence. We do not write raw
candidate text to BQ; only the already-summarized fields from the analysis
pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_ANTHROPIC_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")
_GENERIC_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")
_GOOGLE_API_RE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


def redact_pii(text: str) -> str:
    """Redact likely emails + API keys from a rationale string."""
    if not isinstance(text, str) or not text:
        return ""
    t = _EMAIL_RE.sub("[redacted-email]", text)
    t = _ANTHROPIC_KEY_RE.sub("[redacted-key]", t)
    t = _GOOGLE_API_RE.sub("[redacted-key]", t)
    t = _OPENAI_KEY_RE.sub("[redacted-key]", t)
    # generic long token last (broadest; keep specific ones above to avoid double-replacement)
    t = _GENERIC_TOKEN_RE.sub("[redacted-token]", t)
    return t


def _trim(text: Any, limit: int = 500) -> str:
    s = str(text or "")
    s = redact_pii(s).strip()
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


def extract_signals_from_analysis(analysis: dict) -> list[dict]:
    """
    Build the ordered list of signal rows for one analysis dict. Returns an
    empty list on any shape we don't recognize -- callers write an empty
    `signals` array rather than failing the trade.
    """
    if not isinstance(analysis, dict):
        return []

    signals: list[dict] = []

    # ── Analyst layer ────
    analyst = analysis.get("analyst_summary") or analysis.get("synthesis") or ""
    if analyst:
        signals.append({
            "agent": "Analyst",
            "role": "synthesis",
            "rationale": _trim(analyst),
            "weight": 1.0,
        })

    # ── Debate layer (Bull / Bear) ────
    debate = analysis.get("debate") or {}
    if isinstance(debate, dict):
        bull = debate.get("bull_argument") or debate.get("bull") or ""
        bear = debate.get("bear_argument") or debate.get("bear") or ""
        if bull:
            signals.append({
                "agent": "Bull",
                "role": "long_case",
                "rationale": _trim(bull),
                "weight": float(debate.get("bull_weight") or 0.5),
            })
        if bear:
            signals.append({
                "agent": "Bear",
                "role": "short_case",
                "rationale": _trim(bear),
                "weight": float(debate.get("bear_weight") or 0.5),
            })

    # ── Trader layer ────
    rec = str(analysis.get("recommendation", "")).upper() or "HOLD"
    score = analysis.get("final_score")
    # phase-23.1.7: also extract the lite-Claude-analyzer reason at full_report.analysis.reason
    # so the Trader rationale is the actual reasoning sentence, not the literal "Recommendation: BUY".
    trader_note = (
        analysis.get("trader_note")
        or analysis.get("recommendation_reason")
        or (analysis.get("full_report") or {}).get("analysis", {}).get("reason")
        or ""
    )
    signals.append({
        "agent": "Trader",
        "role": "decision",
        "rationale": _trim(trader_note) or f"Recommendation: {rec}",
        "weight": float(score) if isinstance(score, (int, float)) else 0.0,
    })

    # ── Risk layer ────
    risk = analysis.get("risk_assessment") or {}
    if isinstance(risk, dict):
        decision = risk.get("decision") or ""
        # phase-23.1.7: lite shape uses {"reason": "..."}; add as fallback alongside reasoning/rationale.
        reasoning = (
            risk.get("reasoning")
            or risk.get("rationale")
            or risk.get("reason")
            or ""
        )
        pos_pct = risk.get("recommended_position_pct")
        if decision or reasoning:
            # phase-25.B: post-25.A, RiskJudge runs as an independent LLM call;
            # its `reasoning` is structurally distinct from the trader's and
            # `recommended_position_pct` is always > 0 by construction. The
            # lite-path duplicate-detection block that lived here was dead
            # code after cycle 69 (25.A) -- removed.
            risk_rationale = _trim(reasoning) or f"Decision: {decision}"
            risk_weight = float(pos_pct) if isinstance(pos_pct, (int, float)) else 0.0
            signals.append({
                "agent": "RiskJudge",
                "role": "gate",
                "rationale": risk_rationale,
                "weight": risk_weight,
            })

    return signals


def extract_quant_signals(candidate: dict) -> list[dict]:
    """Extract screener-layer quant signals from a `rank_candidates` dict.

    Returns 0-2 rows: a "Quant" signal (momentum/RSI/vol/sector/composite_score)
    and a "SignalStack" overlay signal (conviction + news + source tag from
    cycles 1-5 overlays). Both conform to the {agent, role, rationale, weight}
    shape so the drawer renders them with no schema changes downstream.
    """
    if not isinstance(candidate, dict):
        return []

    signals: list[dict] = []

    # Quant metrics row
    mom_1m = candidate.get("momentum_1m")
    mom_3m = candidate.get("momentum_3m")
    mom_6m = candidate.get("momentum_6m")
    rsi = candidate.get("rsi_14")
    vol = candidate.get("volatility_ann")
    sector = candidate.get("sector", "")
    composite = candidate.get("composite_score")

    quant_parts: list[str] = []
    if isinstance(mom_1m, (int, float)):
        quant_parts.append(f"1m momentum {mom_1m:+.1f}%")
    if isinstance(mom_3m, (int, float)):
        quant_parts.append(f"3m momentum {mom_3m:+.1f}%")
    if isinstance(mom_6m, (int, float)):
        quant_parts.append(f"6m momentum {mom_6m:+.1f}%")
    if isinstance(rsi, (int, float)):
        quant_parts.append(f"RSI14 {rsi:.1f}")
    if isinstance(vol, (int, float)):
        quant_parts.append(f"ann_vol {vol:.2f}")
    if sector:
        quant_parts.append(f"sector {sector}")
    if isinstance(composite, (int, float)):
        quant_parts.append(f"composite_score {composite:.3f}")

    if quant_parts:
        signals.append({
            "agent": "Quant",
            "role": "screener",
            "rationale": _trim("; ".join(quant_parts)),
            "weight": float(composite) if isinstance(composite, (int, float)) else 0.0,
        })

    # SignalStack overlay row (regime + PEAD + news + sector event + meta-scorer conviction)
    stack_parts: list[str] = []
    conviction_score = candidate.get("conviction_score")
    conviction_reason = candidate.get("conviction_reason", "")
    news_event_type = candidate.get("news_event_type", "")
    news_rationale = candidate.get("news_rationale", "")
    regime_tag = candidate.get("regime_tag", "")
    pead_tag = candidate.get("pead_tag", "")
    sector_event_type = candidate.get("sector_event_type", "")
    source_tag = candidate.get("source", "")

    if regime_tag:
        stack_parts.append(f"regime:{regime_tag}")
    if pead_tag:
        stack_parts.append(f"pead:{pead_tag}")
    if isinstance(conviction_score, (int, float)):
        stack_parts.append(f"conviction {conviction_score:.2f}")
    if conviction_reason:
        stack_parts.append(conviction_reason)
    if news_event_type:
        stack_parts.append(f"news:{news_event_type}")
    if news_rationale:
        stack_parts.append(news_rationale)
    if sector_event_type:
        stack_parts.append(f"sector_event:{sector_event_type}")
    if source_tag and source_tag not in ("", "momentum"):
        stack_parts.append(f"source:{source_tag}")

    if stack_parts:
        signals.append({
            "agent": "SignalStack",
            "role": "overlay",
            "rationale": _trim("; ".join(stack_parts)),
            "weight": float(conviction_score) if isinstance(conviction_score, (int, float)) else 0.0,
        })

    return signals


def extract_all_signals(analysis: dict, candidate: dict | None = None) -> list[dict]:
    """Combined extractor: agent rationale (analysis) + quant overlays (candidate).

    Inserts the Quant / SignalStack rows BEFORE the Trader row so the drawer
    ordering is Analyst -> Quant -> SignalStack -> Trader -> Risk. Pass
    `candidate=None` for paths where no screener candidate is available
    (sell side, full Gemini orchestrator).
    """
    signals = extract_signals_from_analysis(analysis)
    if not candidate:
        return signals
    quant_sigs = extract_quant_signals(candidate)
    if not quant_sigs:
        return signals
    trader_idx = next(
        (i for i, s in enumerate(signals) if s.get("agent") == "Trader"),
        len(signals),
    )
    return signals[:trader_idx] + quant_sigs + signals[trader_idx:]


def group_signals_for_drawer(signals: list[dict]) -> dict:
    """
    Reshape flat signals into the progressive-disclosure tree shape the
    frontend drawer renders: {analyst[], debate{bull[], bear[]}, trader[], risk[]}.
    Keeps order and makes the hierarchy explicit.
    """
    out: dict = {
        "analyst": [],
        "debate": {"bull": [], "bear": []},
        "quant": [],          # phase-23.1.7
        "signal_stack": [],   # phase-23.1.7
        "trader": [],
        "risk": [],
    }
    for s in signals or []:
        role = s.get("role")
        agent = s.get("agent")
        if agent == "Analyst":
            out["analyst"].append(s)
        elif agent == "Bull":
            out["debate"]["bull"].append(s)
        elif agent == "Bear":
            out["debate"]["bear"].append(s)
        elif agent == "Quant" or role == "screener":
            out["quant"].append(s)
        elif agent == "SignalStack" or role == "overlay":
            out["signal_stack"].append(s)
        elif agent == "Trader" or role == "decision":
            out["trader"].append(s)
        elif agent == "RiskJudge" or role == "gate":
            out["risk"].append(s)
    return out
