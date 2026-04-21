"""
Prompt compaction helpers for small-context models.

The goal is not to replay full prior transcripts. Instead, these helpers build
small, deterministic state snapshots that preserve the highest-value facts for
later stages like debate carry-forward, critic review, and synthesis revision.
"""

from __future__ import annotations

import json

from backend.utils import json_io
from typing import Any


def approx_token_count(text: str) -> int:
    """Conservative text-to-token estimate used for budget heuristics."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def compact_text(text: str, max_chars: int, suffix: str = "\n\n[Compacted]") -> str:
    """Trim text while preserving both opening and closing context when possible."""
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(suffix) + 32:
        return text[:max_chars]

    available = max_chars - len(suffix)
    head = int(available * 0.72)
    tail = max(0, available - head)
    if tail < 80:
        return text[:available] + suffix
    return text[:head].rstrip() + suffix + text[-tail:].lstrip()


def compact_argument(text: str, max_chars: int) -> str:
    """Compact a debate argument while keeping the opening thesis and closing punchline."""
    return compact_text(text.strip(), max_chars, suffix="\n\n[Earlier details compacted]\n\n")


def compact_trace_summary(
    trace_summary: dict,
    *,
    max_chars: int,
    max_evidence_per_agent: int = 1,
    evidence_chars: int = 120,
) -> dict:
    """Reduce trace payload to the minimum needed for debate positioning."""
    signals = trace_summary.get("signals", {}) if isinstance(trace_summary, dict) else {}
    compacted_signals: dict[str, dict[str, Any]] = {}
    for agent_name, payload in signals.items():
        if not isinstance(payload, dict):
            continue
        compacted_signals[agent_name] = {
            "signal": payload.get("signal", ""),
            "confidence": round(float(payload.get("confidence", 0.0) or 0.0), 2),
            "top_evidence": [
                compact_text(str(item), evidence_chars, suffix="...")
                for item in (payload.get("top_evidence") or [])[:max_evidence_per_agent]
            ],
        }

    compacted = {
        "total_agents": len(compacted_signals),
        "signals": compacted_signals,
    }
    encoded = json.dumps(compacted, separators=(",", ":"), default=str)
    if len(encoded) <= max_chars:
        return compacted

    stripped = {
        "total_agents": len(compacted_signals),
        "signals": {
            agent_name: {
                "signal": payload.get("signal", ""),
                "confidence": payload.get("confidence", 0.0),
            }
            for agent_name, payload in compacted_signals.items()
        },
    }
    encoded = json.dumps(stripped, separators=(",", ":"), default=str)
    if len(encoded) <= max_chars:
        return stripped

    keep = max(4, min(8, len(stripped["signals"])))
    trimmed_items = list(stripped["signals"].items())[:keep]
    return {
        "total_agents": len(compacted_signals),
        "signals": dict(trimmed_items),
    }


def build_compact_debate_history(
    debate_rounds: list[dict],
    *,
    max_chars: int,
    per_argument_chars: int,
) -> str:
    """Build a compact round-by-round history for the moderator."""
    if not debate_rounds:
        return ""

    parts: list[str] = []
    for round_data in debate_rounds:
        parts.append(
            f"ROUND {round_data.get('round', '?')}:\n"
            f"Bull: {compact_argument(str(round_data.get('bull_argument', '')), per_argument_chars)}\n"
            f"Bear: {compact_argument(str(round_data.get('bear_argument', '')), per_argument_chars)}"
        )
    return compact_text("\n\n".join(parts), max_chars, suffix="\n\n[Earlier rounds compacted]")


def compact_da_result(da_result: dict, *, max_chars: int) -> str:
    """Reduce Devil's Advocate payload to key challenges and a short summary."""
    if not isinstance(da_result, dict):
        return compact_text(str(da_result), max_chars)
    compacted = {
        "summary": compact_text(str(da_result.get("summary", "")), 400, suffix="..."),
        "confidence_adjustment": da_result.get("confidence_adjustment", 0.0),
        "groupthink_flag": da_result.get("groupthink_flag", False),
        "challenges": [compact_text(str(item), 180, suffix="...") for item in da_result.get("challenges", [])[:3]],
        "hidden_risks": [compact_text(str(item), 180, suffix="...") for item in da_result.get("hidden_risks", [])[:3]],
    }
    return compact_text(json.dumps(compacted, separators=(",", ":"), default=str), max_chars)


def compact_quant_snapshot(quant_data: dict) -> dict:
    """Build a small critic-friendly quant snapshot from the larger quant payload."""
    yf = quant_data.get("yf_data", {}) if isinstance(quant_data, dict) else {}
    return {
        "ticker": quant_data.get("ticker", "") if isinstance(quant_data, dict) else "",
        "company_name": quant_data.get("company_name", "") if isinstance(quant_data, dict) else "",
        "sector": quant_data.get("sector", "") if isinstance(quant_data, dict) else "",
        "industry": quant_data.get("industry", "") if isinstance(quant_data, dict) else "",
        "valuation": (yf.get("valuation", {}) if isinstance(yf, dict) else {}),
        "efficiency": (yf.get("efficiency", {}) if isinstance(yf, dict) else {}),
        "health": (yf.get("health", {}) if isinstance(yf, dict) else {}),
        "institutional": (yf.get("institutional", {}) if isinstance(yf, dict) else {}),
    }


def compact_report_reference(report_text: str, *, max_chars: int) -> str:
    """Compress a prior synthesis draft into a small typed revision reference."""
    try:
        data = json_io.loads(report_text)
        if isinstance(data, str):
            data = json_io.loads(data)
    except Exception:
        return compact_text(report_text, max_chars, suffix="\n\n[Prior draft compacted]")

    if not isinstance(data, dict):
        return compact_text(report_text, max_chars, suffix="\n\n[Prior draft compacted]")

    recommendation = data.get("recommendation", {}) if isinstance(data.get("recommendation"), dict) else {}
    compacted = {
        "scoring_matrix": data.get("scoring_matrix", {}),
        "recommendation": {
            "action": recommendation.get("action", data.get("recommendation", "")),
            "justification": compact_text(str(recommendation.get("justification", "")), 400, suffix="..."),
        },
        "final_summary": compact_text(str(data.get("final_summary", "")), 900, suffix="..."),
        "key_risks": [compact_text(str(item), 180, suffix="...") for item in data.get("key_risks", [])[:5]],
        "citations": data.get("citations", [])[:5],
    }
    return compact_text(json.dumps(compacted, separators=(",", ":"), default=str), max_chars, suffix="\n\n[Prior draft compacted]")