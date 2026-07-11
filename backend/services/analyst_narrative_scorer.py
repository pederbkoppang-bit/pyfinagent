"""
LLM analyst-narrative signal — phase-28.11.

**HONESTY NOTE:** The canonical 68bps/month signal (arXiv 2502.20489v1 "Do Sell-side
Analyst Reports Have Investment Value?") requires Thomson Reuters Investext, a paid
commercial feed ($10K-$100K/yr) — not viable for this local-only deployment
(see project_local_only_deployment auto-memory).

This module is a **MVP PROXY**: it scores MANAGEMENT FORWARD-LOOKING TONE from 8-K
Exhibit 99 press releases via Claude Haiku 4.5. Different scoring lens from
`pead_signal.py`:
    - pead_signal:     sentiment_score vs trailing 12Q mean (surprise-vs-baseline)
    - analyst_narrative: outlook_score from forward-looking language (guidance, strategy)

The two signals are likely correlated (same 8-K Exhibit 99 source) — correlation
analysis required before joint deployment. Boost magnitude conservatively 50% of
PEAD scale pending live A/B validation.

When/if paid analyst-report data becomes available, this interface can be
repointed without changing downstream callers.

Cost: ~$0.001 per LLM call (Claude Haiku); per-cycle target <$0.10 (10 recent
reporters analyzed).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.services.macro_regime import _strip_unsupported_schema_keys
from backend.services.pead_signal import _fetch_exhibit_99_text, _fetch_recent_8k
from backend.tools.sec_insider import SEC_HEADERS, _resolve_cik

logger = logging.getLogger(__name__)

_CONCURRENCY = 3


class AnalystNarrativeSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    outlook_score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 (strongly bearish forward outlook) to 1.0 (strongly bullish).",
    )
    outlook_tag: Literal["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"]
    rationale: str = Field(description="Brief reasoning (<=200 chars).")
    boost_multiplier: float = Field(description="Final multiplier on composite_score. 1.0 = no change.")
    source_note: str = Field(
        default="management_8k_proxy: NOT canonical analyst_strategic_outlook",
        description="Honesty marker — data source is 8-K Exhibit 99 management text, not paid analyst reports.",
    )


def _classify_boost(outlook_score: float, strong_thr: float, weak_thr: float,
                    strong_boost: float, moderate_boost: float) -> tuple[float, str]:
    if outlook_score >= strong_thr:
        return 1.0 + strong_boost, "strongly_bullish"
    if outlook_score >= 0.55:
        return 1.0 + moderate_boost, "bullish"
    if outlook_score <= 1.0 - strong_thr:
        return max(0.85, 1.0 - strong_boost), "strongly_bearish"
    if outlook_score <= weak_thr + (1.0 - strong_thr) - weak_thr + 0.10:
        return max(0.90, 1.0 - moderate_boost), "bearish"
    return 1.0, "neutral"


def _build_prompt(ticker: str, press_text: str) -> str:
    return (
        f"You are scoring MANAGEMENT FORWARD-LOOKING TONE in the latest 8-K press "
        f"release for {ticker}. This is a PROXY for the canonical analyst Strategic "
        f"Outlook signal (which needs paid data); focus on management's own forward "
        f"language: guidance raises/cuts, strategic commentary, capex outlook, "
        f"product/market expansion claims.\n\n"
        f"Press release text (first 4000 chars from SEC EDGAR Exhibit 99):\n\n"
        f"---\n{press_text}\n---\n\n"
        f"Score outlook_score in [0.0, 1.0]:\n"
        f"  >=0.70 = strongly_bullish forward language (raised guidance, expansion, accelerating capex)\n"
        f"  0.55-0.69 = bullish (modest raise / positive forward color)\n"
        f"  0.40-0.55 = neutral (steady / no notable forward signal)\n"
        f"  0.31-0.39 = bearish (cautious / softer forward language)\n"
        f"  <=0.30 = strongly_bearish (guidance cut, headwinds, restructuring)\n\n"
        f"Generate rationale FIRST (<=200 chars), then commit to numeric. Return JSON only."
    )


async def _fetch_one_narrative(
    ticker: str,
    model: str,
    strong_thr: float,
    weak_thr: float,
    strong_boost: float,
    moderate_boost: float,
    sem: asyncio.Semaphore,
) -> Optional[AnalystNarrativeSignal]:
    async with sem:
        settings = get_settings()
        anthropic_key = getattr(settings, "anthropic_api_key", "") or ""
        if not anthropic_key:
            logger.debug("analyst_narrative_scorer: no Anthropic key; skipping %s", ticker)
            return None
        if hasattr(anthropic_key, "get_secret_value"):
            try:
                anthropic_key = anthropic_key.get_secret_value()
            except Exception:
                pass
        if not anthropic_key:
            return None

        try:
            async with httpx.AsyncClient(headers=SEC_HEADERS) as http:
                cik = await _resolve_cik(http, ticker)
                if not cik:
                    return None
                latest_8k = await _fetch_recent_8k(http, cik)
                if not latest_8k:
                    return None
                press_text = await _fetch_exhibit_99_text(http, cik, latest_8k["accession"])
                if not press_text:
                    return None
        except Exception as e:
            logger.debug("analyst_narrative_scorer: %s EDGAR fetch failed: %s", ticker, e)
            return None

        try:
            from backend.agents.llm_client import ClaudeClient
            client = ClaudeClient(
                model_name=model,
                api_key=anthropic_key,
                enable_prompt_caching=False,
            )
        except Exception as e:
            logger.debug("analyst_narrative_scorer: ClaudeClient init failed: %s", e)
            return None

        prompt = _build_prompt(ticker, press_text)
        schema = {
            "type": "object",
            "properties": {
                "outlook_score": {"type": "number"},
                "outlook_tag": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["outlook_score", "outlook_tag", "rationale"],
            "additionalProperties": False,
        }
        cleaned = _strip_unsupported_schema_keys(schema)
        try:
            response = await asyncio.to_thread(
                client.generate_content,
                prompt,
                {
                    "response_schema": cleaned,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 256,
                    "temperature": 0.0,
                },
            )
        except Exception as e:
            logger.debug("analyst_narrative_scorer: %s LLM call failed: %s", ticker, e)
            return None

        try:
            raw = json.loads(response.text)
            score = float(raw.get("outlook_score", 0.5))
            score = max(0.0, min(1.0, score))
            rationale = str(raw.get("rationale", ""))[:200]
        except Exception as e:
            logger.debug("analyst_narrative_scorer: %s parse failed: %s | raw=%s", ticker, e, getattr(response, "text", "")[:200])
            return None

        boost, tag = _classify_boost(score, strong_thr, weak_thr, strong_boost, moderate_boost)
        return AnalystNarrativeSignal(
            ticker=ticker.upper(),
            outlook_score=round(score, 3),
            outlook_tag=tag,  # type: ignore[arg-type]
            rationale=rationale,
            boost_multiplier=round(boost, 4),
        )


async def fetch_narrative_signals(
    tickers: list[str],
    model: str = "claude-haiku-4-5",
    strong_threshold: float = 0.70,
    weak_threshold: float = 0.30,
    strong_boost: float = 0.05,
    moderate_boost: float = 0.025,
) -> dict[str, AnalystNarrativeSignal]:
    """Per-ticker management-outlook scoring via Claude Haiku on 8-K Exhibit 99 text.

    Returns one entry per ticker with a recent 8-K + successfully scored. Empty dict
    if no tickers qualify, no Anthropic key, or all fetches fail.

    HONESTY: this is a PROXY for the canonical analyst Strategic Outlook signal
    (which requires paid data) — see module docstring.
    """
    if not tickers:
        return {}
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _fetch_one_narrative(
                t, model, strong_threshold, weak_threshold,
                strong_boost, moderate_boost, sem,
            )
            for t in tickers
        ),
        return_exceptions=False,
    )
    out: dict[str, AnalystNarrativeSignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "analyst_narrative_scorer: %d/%d tickers scored (MVP proxy via 8-K, not canonical analyst reports)",
        len(out), len(tickers),
    )
    return out


def apply_narrative_signal_to_score(
    base_score: float,
    ticker: Optional[str],
    signals: Optional[dict[str, AnalystNarrativeSignal]],
) -> float:
    """Multiply score by signals[ticker].boost_multiplier. Identity if missing."""
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, sig.boost_multiplier)
