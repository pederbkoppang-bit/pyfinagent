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
    trader_note = analysis.get("trader_note") or analysis.get("recommendation_reason") or ""
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
        reasoning = risk.get("reasoning") or risk.get("rationale") or ""
        pos_pct = risk.get("recommended_position_pct")
        if decision or reasoning:
            signals.append({
                "agent": "RiskJudge",
                "role": "gate",
                "rationale": _trim(reasoning) or f"Decision: {decision}",
                "weight": float(pos_pct) if isinstance(pos_pct, (int, float)) else 0.0,
            })

    return signals


def group_signals_for_drawer(signals: list[dict]) -> dict:
    """
    Reshape flat signals into the progressive-disclosure tree shape the
    frontend drawer renders: {analyst[], debate{bull[], bear[]}, trader[], risk[]}.
    Keeps order and makes the hierarchy explicit.
    """
    out: dict = {"analyst": [], "debate": {"bull": [], "bear": []}, "trader": [], "risk": []}
    for s in signals or []:
        role = s.get("role")
        agent = s.get("agent")
        if agent == "Analyst":
            out["analyst"].append(s)
        elif agent == "Bull":
            out["debate"]["bull"].append(s)
        elif agent == "Bear":
            out["debate"]["bear"].append(s)
        elif agent == "Trader" or role == "decision":
            out["trader"].append(s)
        elif agent == "RiskJudge" or role == "gate":
            out["risk"].append(s)
    return out
