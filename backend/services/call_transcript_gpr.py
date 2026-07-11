"""
Earnings-call NLP for firm-level GPR exposure — phase-28.13.

**HONESTY:** The Federal Reserve's 2025 firm-level GPR study (R²=0.23 across 240K+
earnings call transcripts; "Measuring Geopolitical Risk Exposure Across Industries:
A Firm-Centered Approach") demonstrated CONTEMPORANEOUS relationship only — **NO
forward return predictability**. This signal is therefore a defensive RISK FILTER
on the candidate picker, NOT an alpha source.

The signal classifies a firm's earnings-call language into 4 tiers
(HIGH/MEDIUM/LOW/NONE) of GPR exposure. Firms in defense-benefiting sectors
(Industrials → defense contractors, Energy → oil majors) are EXEMPT from the
penalty because they BENEFIT from elevated GPR (per Caldara-Iacoviello
US-as-net-exporter asymmetry, used in phase-28.3).

Complements phase-28.3 (sector-level GPR tilt) by adding the firm-level dimension:
a Health Care firm explicitly discussing supply-chain disruption merits a small
defensive haircut even when sector GPR is low.

Cost: ~$0.001 per LLM call (Claude Haiku). Per-cycle target <$0.10 (~10 candidates).

Data source: `backend/tools/earnings_tone.py::get_earnings_tone` (Yahoo Finance
transcripts with GCS caching). No new API key required (the `api_ninjas_key`
setting exists but the active source is Yahoo).

Graceful degradation: transcript fetch or LLM fail → empty dict → identity.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.services.macro_regime import _strip_unsupported_schema_keys

logger = logging.getLogger(__name__)

_CONCURRENCY = 3


class GprExposureSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    exposure_tier: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    key_phrases: list[str] = Field(
        default_factory=list,
        description="Up to 3 short GPR-relevant phrases extracted from the transcript.",
    )
    rationale: str = Field(description="<=200 chars reasoning.")
    source_note: str = Field(
        default="defensive_filter_only_per_Fed_2025_R2_0.23_contemporaneous_no_forward_alpha",
        description="Honesty marker — Fed found no forward predictability; this is a risk filter, not alpha.",
    )


def _build_prompt(ticker: str, transcript_excerpt: str) -> str:
    return (
        f"You are classifying the GEOPOLITICAL RISK EXPOSURE of {ticker} based on "
        f"language in their recent earnings call. Use the Caldara-Iacoviello GPR "
        f"vocabulary: war/conflict, sanctions, tariff, supply-chain disruption, "
        f"geopolitical tension, embargo, trade dispute, sovereign risk.\n\n"
        f"Earnings call excerpt (~8000 chars):\n\n"
        f"---\n{transcript_excerpt[:8000]}\n---\n\n"
        f"Classify exposure_tier:\n"
        f"  HIGH   = management explicitly cites multiple GPR threats as material to results/outlook\n"
        f"  MEDIUM = single GPR mention with notable concern\n"
        f"  LOW    = passing reference; not material\n"
        f"  NONE   = no GPR language at all\n\n"
        f"Extract up to 3 short key_phrases that drove the classification. "
        f"Write rationale (<=200 chars) FIRST, then commit to the tier. Return JSON only.\n\n"
        f"NOTE: this signal is used as a DEFENSIVE FILTER on candidate stocks. The Fed (2025) "
        f"found contemporaneous relationship only, no forward predictability — be accurate, not generous."
    )


async def _fetch_one_exposure(
    ticker: str,
    model: str,
    sem: asyncio.Semaphore,
    bucket_name: str = "",
) -> Optional[GprExposureSignal]:
    async with sem:
        settings = get_settings()
        anthropic_key = getattr(settings, "anthropic_api_key", "") or ""
        if hasattr(anthropic_key, "get_secret_value"):
            try:
                anthropic_key = anthropic_key.get_secret_value()
            except Exception:
                pass
        if not anthropic_key:
            return None

        try:
            from backend.tools.earnings_tone import get_earnings_tone
            tone_result = await get_earnings_tone(
                ticker, api_key="", max_transcripts=1, bucket_name=bucket_name,
            )
        except Exception as e:
            logger.debug("call_transcript_gpr: %s transcript fetch failed: %s", ticker, e)
            return None
        transcript_excerpt = (tone_result or {}).get("transcript_excerpt") or ""
        if not transcript_excerpt or len(transcript_excerpt) < 200:
            return None

        try:
            from backend.agents.llm_client import ClaudeClient
            client = ClaudeClient(
                model_name=model,
                api_key=anthropic_key,
                enable_prompt_caching=False,
            )
        except Exception as e:
            logger.debug("call_transcript_gpr: ClaudeClient init failed: %s", e)
            return None

        prompt = _build_prompt(ticker, transcript_excerpt)
        schema = {
            "type": "object",
            "properties": {
                "exposure_tier": {"type": "string"},
                "key_phrases": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["exposure_tier", "key_phrases", "rationale"],
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
                    "max_output_tokens": 384,
                    "temperature": 0.0,
                },
            )
        except Exception as e:
            logger.debug("call_transcript_gpr: %s LLM call failed: %s", ticker, e)
            return None

        try:
            raw = json.loads(response.text)
            tier = str(raw.get("exposure_tier", "NONE")).upper()
            if tier not in ("HIGH", "MEDIUM", "LOW", "NONE"):
                tier = "NONE"
            phrases = raw.get("key_phrases") or []
            if not isinstance(phrases, list):
                phrases = []
            rationale = str(raw.get("rationale", ""))[:200]
        except Exception as e:
            logger.debug("call_transcript_gpr: %s parse failed: %s", ticker, e)
            return None

        return GprExposureSignal(
            ticker=ticker.upper(),
            exposure_tier=tier,  # type: ignore[arg-type]
            key_phrases=[str(p)[:80] for p in phrases[:3]],
            rationale=rationale,
        )


async def fetch_gpr_exposure_signals(
    tickers: list[str],
    model: str = "claude-haiku-4-5",
    bucket_name: str = "",
) -> dict[str, GprExposureSignal]:
    """Per-ticker GPR exposure classification from earnings-call transcripts.

    Returns one entry per ticker with a fetchable transcript + successful LLM classify.
    Empty dict if no Anthropic key, no transcripts, or all fetches fail.

    HONESTY: defensive risk filter only (Fed 2025: no forward predictability).
    """
    if not tickers:
        return {}
    sem = asyncio.Semaphore(_CONCURRENCY)
    results = await asyncio.gather(
        *(_fetch_one_exposure(t, model, sem, bucket_name=bucket_name) for t in tickers),
        return_exceptions=False,
    )
    out: dict[str, GprExposureSignal] = {}
    for sig in results:
        if sig is not None:
            out[sig.ticker] = sig
    logger.info(
        "call_transcript_gpr: %d/%d tickers classified (DEFENSIVE FILTER per Fed 2025 -- no forward alpha)",
        len(out), len(tickers),
    )
    return out


def apply_gpr_exposure_to_score(
    base_score: float,
    ticker: Optional[str],
    sector: Optional[str],
    signals: Optional[dict[str, GprExposureSignal]],
    exempt_sectors_csv: str = "Industrials,Energy",
    high_penalty: float = 0.97,
) -> float:
    """Apply defensive haircut to candidates with HIGH GPR exposure UNLESS sector is exempt.

    Identity for MEDIUM/LOW/NONE tiers, missing signal, or exempted sectors.
    Exempted sectors BENEFIT from elevated GPR (defense in Industrials, oil in Energy).
    """
    if not signals or not ticker:
        return base_score
    sig = signals.get(ticker.upper())
    if sig is None:
        return base_score
    if sig.exposure_tier != "HIGH":
        return base_score
    exempt = {s.strip() for s in exempt_sectors_csv.split(",") if s.strip()}
    if sector and sector.strip() in exempt:
        return base_score
    from backend.services.overlay_math import sign_safe_mult  # phase-69.3 sign-safe (default-OFF byte-identical)
    return sign_safe_mult(base_score, high_penalty)
