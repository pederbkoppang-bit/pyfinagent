"""
Daily macro regime filter — LLM-as-judge over a FRED snapshot.

Pulls 7-9 daily/monthly FRED series, hands them to Claude Haiku 4.5 with a
structured-output schema, returns a regime tag + conviction multiplier that
`screener.rank_candidates` applies to its composite score.

Cost target: <$0.05/day. 24-hour file cache prevents re-billing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.tools.fred_data import get_macro_indicators

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_PATH = _CACHE_DIR / "macro_regime.json"
_CACHE_TTL_HOURS = 24

_REGIME_TAGS = ("risk_on", "risk_off", "mixed", "unknown")
_DEFAULT_MULTIPLIERS = {
    "risk_on": 1.15,
    "risk_off": 0.70,
    "mixed": 1.00,
    "unknown": 0.85,
}

_REGIME_SERIES = (
    "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2",
    "FEDFUNDS", "CPIAUCSL", "UNRATE", "INDPRO",
)


class SectorWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overweight: list[str] = Field(
        default_factory=list,
        description="Sector ETF tickers to overweight in this regime (e.g. ['XLU','XLP','XLV']). Max 5.",
    )
    underweight: list[str] = Field(
        default_factory=list,
        description="Sector ETF tickers to underweight in this regime. Max 5.",
    )


class MacroRegimeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(
        description="Free-text explanation max 300 chars. Name the signals that drove the call and any conflicts.",
        max_length=300,
    )
    regime: Literal["risk_on", "risk_off", "mixed", "unknown"]
    conviction: float = Field(
        ge=0.0, le=1.0,
        description="0.0-1.0 confidence. 1.0 = all signals agree; 0.5 = mixed; 0.0 = insufficient data.",
    )
    conviction_multiplier: float = Field(
        ge=0.5, le=1.5,
        description="Applied as scalar in screener.rank_candidates. Defaults: risk_on=1.15, mixed=1.0, risk_off=0.7, unknown=0.85.",
    )
    sector_hints: SectorWeights
    series_used: list[str] = Field(
        description="FRED series IDs that were available (e.g. ['T10Y2Y','VIXCLS','BAMLH0A0HYM2']).",
    )
    computed_at: str = Field(description="ISO-8601 UTC timestamp when this regime was computed.")


def _load_cache() -> Optional[MacroRegimeOutput]:
    if not _CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        computed_at = datetime.fromisoformat(raw["computed_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - computed_at > timedelta(hours=_CACHE_TTL_HOURS):
            return None
        return MacroRegimeOutput.model_validate(raw)
    except Exception as e:
        logger.warning("Macro regime cache unreadable: %s", e)
        return None


_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "maxLength", "minLength",
)


def _strip_unsupported_schema_keys(node):
    """Recursively remove JSON-schema keys Anthropic structured outputs rejects."""
    if isinstance(node, dict):
        for k in _UNSUPPORTED_SCHEMA_KEYS:
            node.pop(k, None)
        for v in node.values():
            _strip_unsupported_schema_keys(v)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_schema_keys(item)
    return node


def _save_cache(regime: MacroRegimeOutput) -> None:
    try:
        _CACHE_PATH.write_text(regime.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Macro regime cache write failed: %s", e)


def _build_prompt(indicators: dict) -> str:
    lines = [
        "Classify the current US macro regime for equity portfolio sizing.",
        "",
        "Available FRED indicators (current value, prior, trend):",
    ]
    for sid in _REGIME_SERIES:
        info = indicators.get(sid)
        if not info or "current" not in info:
            continue
        lines.append(
            f"- {sid} ({info.get('name', sid)}): "
            f"current={info['current']:.3f} previous={info.get('previous', 'n/a')} "
            f"trend={info.get('trend', 'n/a')} as_of={info.get('date', 'n/a')}"
        )
    lines += [
        "",
        "Decision thresholds (canonical, from research-brief phase-23.1.1):",
        "- T10Y2Y < 0 -> recession risk (risk_off lean)",
        "- VIXCLS > 25 -> elevated fear (risk_off); VIXCLS < 18 -> calm (risk_on lean)",
        "- BAMLH0A0HYM2 > 5.0 -> credit stress (risk_off); < 3.5 -> credit easy (risk_on)",
        "- FEDFUNDS rising -> tightening (mild risk_off lean); falling -> easing (risk_on lean)",
        "- UNRATE rising trend -> growth slowing (risk_off lean)",
        "- CPIAUCSL YoY > 3.5% -> inflationary pressure (risk_off lean)",
        "",
        "Sector tilts: risk_off favors XLU/XLP/XLV/XLE (defensives + energy); "
        "risk_on favors XLK/XLY/XLF/XLI (cyclicals); mixed = neutral.",
        "",
        "Return JSON ONLY matching the MacroRegimeOutput schema. "
        "Generate the rationale FIRST, then commit to the regime tag. "
        "Set conviction high (>=0.75) only when 4+ signals align; "
        "use 'mixed' when 2-3 signals conflict; use 'unknown' when fewer than 3 series are available.",
    ]
    return "\n".join(lines)


def _fallback_regime(indicators: dict, reason: str) -> MacroRegimeOutput:
    series_used = [sid for sid in _REGIME_SERIES if indicators.get(sid, {}).get("current") is not None]
    return MacroRegimeOutput(
        rationale=f"Fallback: {reason}"[:300],
        regime="unknown",
        conviction=0.0,
        conviction_multiplier=_DEFAULT_MULTIPLIERS["unknown"],
        sector_hints=SectorWeights(),
        series_used=series_used,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


async def compute_macro_regime(use_cache: bool = True) -> MacroRegimeOutput:
    """Compute today's macro regime tag. Caches for 24 hours.

    Returns a `MacroRegimeOutput`. On any failure (no FRED key, no Anthropic key,
    LLM error, schema parse error) returns a `regime='unknown'` fallback so the
    screener stays operational.
    """
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            logger.info("Macro regime cache hit: %s (mult=%s)", cached.regime, cached.conviction_multiplier)
            return cached

    settings = get_settings()
    fred_key = getattr(settings, "fred_api_key", "") or ""
    if not fred_key:
        return _fallback_regime({}, "FRED_API_KEY not configured")

    indicators_payload = await get_macro_indicators(fred_key)
    if not indicators_payload.get("available"):
        return _fallback_regime({}, indicators_payload.get("summary", "FRED unavailable"))
    indicators = indicators_payload.get("indicators", {})

    available = [sid for sid in _REGIME_SERIES if indicators.get(sid, {}).get("current") is not None]
    if len(available) < 3:
        return _fallback_regime(indicators, f"only {len(available)} regime series available")

    anthropic_key = getattr(settings, "anthropic_api_key", "") or ""
    if not anthropic_key:
        return _fallback_regime(indicators, "ANTHROPIC_API_KEY not configured")

    from backend.agents.llm_client import ClaudeClient
    client = ClaudeClient(
        model_name=getattr(settings, "macro_regime_model", "claude-haiku-4-5"),
        api_key=anthropic_key,
        enable_prompt_caching=False,
    )

    prompt = _build_prompt(indicators)
    cleaned_schema = _strip_unsupported_schema_keys(MacroRegimeOutput.model_json_schema())
    try:
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            {
                "response_schema": cleaned_schema,
                "response_mime_type": "application/json",
                "max_output_tokens": 512,
                "temperature": 0.0,
            },
        )
    except Exception as e:
        logger.warning("Macro regime LLM call failed: %s", e)
        return _fallback_regime(indicators, f"LLM error: {type(e).__name__}")

    # Clamp / coerce raw fields BEFORE Pydantic validation, since Anthropic
    # structured outputs cannot enforce numerical or string-length constraints.
    try:
        raw = json.loads(response.text)
        if isinstance(raw.get("rationale"), str):
            raw["rationale"] = raw["rationale"][:300]
        if isinstance(raw.get("conviction"), (int, float)):
            raw["conviction"] = max(0.0, min(1.0, float(raw["conviction"])))
        if isinstance(raw.get("conviction_multiplier"), (int, float)):
            raw["conviction_multiplier"] = max(0.5, min(1.5, float(raw["conviction_multiplier"])))
        parsed = MacroRegimeOutput.model_validate(raw)
    except Exception as e:
        logger.warning("Macro regime LLM output unparsable: %s | raw=%s", e, response.text[:200])
        return _fallback_regime(indicators, "LLM output parse error")

    if not parsed.series_used:
        parsed = parsed.model_copy(update={"series_used": available})

    _save_cache(parsed)
    logger.info(
        "Macro regime computed: %s conviction=%.2f mult=%.2f series=%d",
        parsed.regime, parsed.conviction, parsed.conviction_multiplier, len(parsed.series_used),
    )
    return parsed


def apply_regime_to_score(
    base_score: float,
    sector: Optional[str],
    sector_etf_for: dict[str, str],
    regime: Optional[MacroRegimeOutput],
) -> float:
    """Apply regime multiplier and sector tilt to a screener composite score.

    Args:
        base_score: Original screener score.
        sector: GICS-style sector name for the candidate ticker (may be None).
        sector_etf_for: Mapping of sector name -> SPDR ETF ticker (e.g. screener.SECTOR_ETFS).
        regime: MacroRegimeOutput or None (no regime applied if None).
    """
    if regime is None:
        return base_score
    score = base_score * regime.conviction_multiplier
    if sector and sector_etf_for:
        etf = sector_etf_for.get(sector)
        if etf:
            if etf in regime.sector_hints.overweight:
                score *= 1.05
            elif etf in regime.sector_hints.underweight:
                score *= 0.95
    return score
