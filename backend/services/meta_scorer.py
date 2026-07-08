"""
LLM-as-judge meta-scorer — single batched Claude call combines all sub-signals
(momentum + macro + PEAD + news + sector) into conviction 1-10 per candidate.

This is the alpha-combination layer: instead of multiplicatively cascading
overlays (which can't capture "high momentum + risk_off = warning"), one Claude
Haiku 4.5 call genuinely re-weighs the signal stack with explicit anti-rubber-
stamp prompt design (counterargument-first, regime-momentum interaction rule,
randomized order).

Cost target: <$0.025/cycle (~15K input + 3K output tokens at Haiku pricing).
On any failure, falls back to `conviction_score = round(composite_score)`
clamped to [1, 10] so the cycle still runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.services.macro_regime import _strip_unsupported_schema_keys

logger = logging.getLogger(__name__)

_MAX_BATCH = 30


class MetaScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Uppercase equity ticker.")
    conviction_score: int = Field(
        ge=1, le=10,
        description="1-10 conviction. 10=highest. Considers all signals.",
    )
    conviction_reason: str = Field(
        description="Single sentence (<200 chars) stating the primary driver AND the primary risk.",
    )


class MetaScorerBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MetaScoredCandidate]


def _format_candidate_block(c: dict) -> str:
    """Render one candidate dict into the prompt block format."""
    momentum_parts = []
    if c.get("momentum_1m") is not None:
        momentum_parts.append(f"1m={c['momentum_1m']:+.1f}%")
    if c.get("momentum_3m") is not None:
        momentum_parts.append(f"3m={c['momentum_3m']:+.1f}%")
    if c.get("momentum_6m") is not None:
        momentum_parts.append(f"6m={c['momentum_6m']:+.1f}%")
    if c.get("rsi_14") is not None:
        momentum_parts.append(f"rsi={c['rsi_14']}")
    if c.get("volatility_ann") is not None:
        momentum_parts.append(f"vol_ann={c['volatility_ann']}")
    momentum_line = ", ".join(momentum_parts) if momentum_parts else "n/a"

    pead = c.get("pead_signal")
    pead_line = "none"
    if pead:
        pead_line = (
            f"{pead.get('sentiment_tag','?')} sentiment={pead.get('sentiment_score','?')} "
            f"surprise={pead.get('surprise_score','?')} hold={pead.get('holding_window_days','?')}d"
        )

    news = c.get("news_signal")
    news_line = "none"
    if news:
        news_line = (
            f"{news.get('impact_polarity','?')} ({news.get('event_type','?')}) "
            f"confidence={news.get('confidence','?')}"
        )

    sector_event = c.get("sector_event")
    sector_event_line = "none"
    if sector_event:
        sector_event_line = (
            f"{sector_event.get('event_type','?')} "
            f"({sector_event.get('signal_direction','?')}, "
            f"{sector_event.get('days_to_event','?')}d)"
        )

    return (
        "---\n"
        f"ticker: {c.get('ticker','?')}\n"
        f"sector: {c.get('sector','Unknown')}\n"
        f"momentum: {momentum_line}\n"
        f"pead_signal: {pead_line}\n"
        f"news_signal: {news_line}\n"
        f"sector_event: {sector_event_line}\n"
        f"composite_score_pre_meta: {c.get('composite_score','?')}"
    )


def _build_meta_prompt(candidates: list[dict], regime: Optional[Any]) -> str:
    n = len(candidates)
    if regime is not None:
        regime_line = (
            f'The current macro regime is: {regime.regime} '
            f'(multiplier={regime.conviction_multiplier:.2f}, conviction={regime.conviction:.2f}). '
            'A regime of "risk_off" means positive momentum may be a fade signal, not a buy signal.'
        )
    else:
        regime_line = "Macro regime: not available."

    block = "\n".join(_format_candidate_block(c) for c in candidates)

    return (
        f"You are evaluating {n} stock candidates for a long-only US equity paper portfolio.\n"
        f"{regime_line}\n\n"
        "For each candidate below, assign conviction 1-10 and one reason sentence.\n\n"
        "IMPORTANT rules:\n"
        "1. Score each candidate INDEPENDENTLY. Do not let one ticker's data influence another's.\n"
        "2. First state what could go WRONG with this pick (one clause), then state why you are still\n"
        "   bullish or bearish. This forces genuine re-weighting, not rubber-stamping.\n"
        "3. If momentum is strong but regime is risk_off: this is a warning sign, not a green light.\n"
        '4. A PEAD sentiment_tag of "positive_surprise" combined with high momentum = high conviction.\n'
        '   A "negative_surprise" PEAD should reduce conviction even if momentum looks good.\n'
        '5. "news_signal: none" means no catalyst -- score accordingly. Do not invent catalysts.\n'
        "6. Score 9-10 only when momentum, PEAD, regime, AND news all align positively.\n"
        "   Score 1-2 only when multiple signals conflict negatively.\n\n"
        "Candidates (ordered randomly -- do not assume position implies quality):\n\n"
        f"{block}\n\n"
        f"Return JSON matching MetaScorerBatch with EXACTLY {n} candidates in INPUT ORDER."
    )


def _fallback_conviction(c: dict) -> int:
    cs = c.get("composite_score")
    if isinstance(cs, (int, float)):
        return max(1, min(10, int(round(cs))))
    return 5


def _rank_normalized_convictions(cands: list[dict]) -> list[int]:
    """phase-61.2 (criterion 4): percentile-rank composite scores into 1-10.

    The legacy per-candidate clamp saturates every composite >= 9.5 to a
    constant 10 (live composites run 78-163 -> every fallback cycle emitted
    'conviction 10.00' for all candidates, erasing the ranking the overlay
    exists to provide). Midpoint tie ranks; single candidate -> 5.
    Returned list is aligned with the input order."""
    import bisect

    vals = sorted(float(c.get("composite_score") or 0.0) for c in cands)
    n = len(vals)
    out: list[int] = []
    for c in cands:
        if n <= 1:
            out.append(5)
            continue
        v = float(c.get("composite_score") or 0.0)
        lo = bisect.bisect_left(vals, v)
        hi = bisect.bisect_right(vals, v) - 1
        pct = ((lo + hi) / 2.0) / (n - 1)
        out.append(1 + int(round(9 * pct)))
    return out


def _fallback_convictions(cands: list[dict]) -> list[int]:
    """Set-aware fallback dispatcher: rank-normalized under the phase-61.2
    integrity flag, legacy per-candidate clamp otherwise (byte-identical OFF)."""
    from backend.config.settings import get_settings

    if getattr(get_settings(), "paper_synthesis_integrity_enabled", False):
        return _rank_normalized_convictions(cands)
    return [_fallback_conviction(c) for c in cands]


async def meta_score_candidates(
    candidates: list[dict],
    regime: Optional[Any] = None,
) -> list[dict]:
    """Meta-score candidates with a single Claude call. Returns sorted desc by conviction.

    On any failure, returns each candidate with `conviction_score` derived from
    `composite_score` (clamped to [1, 10]) so the cycle keeps running.
    """
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c.get("composite_score", 0) or 0,
        reverse=True,
    )
    head = candidates[:_MAX_BATCH]
    tail = candidates[_MAX_BATCH:]

    settings = get_settings()
    # phase-51.1: unwrap SecretStr (truthy wrapper bypassed `or ""` -> SDK header error).
    from backend.agents.llm_client import unwrap_secret
    anthropic_key = unwrap_secret(getattr(settings, "anthropic_api_key", ""))
    if not anthropic_key:
        logger.warning("meta_scorer: no ANTHROPIC_API_KEY -- using fallback")
        out = []
        for c, _cv in zip(candidates, _fallback_convictions(candidates)):
            c2 = dict(c)
            c2["conviction_score"] = _cv
            c2["conviction_reason"] = "fallback (no API key)"
            out.append(c2)
        return sorted(out, key=lambda c: c["conviction_score"], reverse=True)

    rng = random.Random(0xC0FFEE)
    shuffled = list(head)
    rng.shuffle(shuffled)
    prompt = _build_meta_prompt(shuffled, regime)
    cleaned_schema = _strip_unsupported_schema_keys(MetaScorerBatch.model_json_schema())

    from backend.agents.llm_client import ClaudeClient
    client = ClaudeClient(
        model_name=getattr(settings, "meta_scorer_model", "claude-haiku-4-5"),
        api_key=anthropic_key,
        enable_prompt_caching=False,
    )

    try:
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            {
                "response_schema": cleaned_schema,
                "response_mime_type": "application/json",
                "max_output_tokens": min(8192, 250 * len(shuffled)),
                "temperature": 0.0,
            },
        )
    except Exception as e:
        logger.warning("meta_scorer LLM call failed: %s", e)
        return _fallback_all(head + tail)

    try:
        raw = json.loads(response.text)
        for c in raw.get("candidates", []):
            if isinstance(c.get("conviction_score"), (int, float)):
                c["conviction_score"] = max(1, min(10, int(round(c["conviction_score"]))))
            if isinstance(c.get("conviction_reason"), str):
                c["conviction_reason"] = c["conviction_reason"][:200]
        batch = MetaScorerBatch.model_validate(raw)
    except Exception as e:
        logger.warning("meta_scorer parse failed: %s | raw=%s", e, response.text[:200])
        return _fallback_all(head + tail)

    by_ticker = {sc.ticker.upper(): sc for sc in batch.candidates}

    out: list[dict] = []
    for c in head:
        ticker = (c.get("ticker") or "").upper()
        c2 = dict(c)
        scored = by_ticker.get(ticker)
        if scored is None:
            c2["conviction_score"] = _fallback_conviction(c)
            c2["conviction_reason"] = "fallback (ticker missing in batch response)"
        else:
            c2["conviction_score"] = scored.conviction_score
            c2["conviction_reason"] = scored.conviction_reason
        out.append(c2)

    # phase-61.2 (criterion 4): under the integrity flag, tail convictions are
    # percentile-ranked over the FULL candidate set so head (LLM-scored) and
    # tail (fallback) stay on comparable 1-10 scales -- the legacy clamp put
    # saturated-10 tail entries ABOVE honestly-scored head entries.
    if getattr(settings, "paper_synthesis_integrity_enabled", False):
        _tail_convs = _rank_normalized_convictions(head + tail)[len(head):]
    else:
        _tail_convs = [_fallback_conviction(c) for c in tail]
    for c, _cv in zip(tail, _tail_convs):
        c2 = dict(c)
        c2["conviction_score"] = _cv
        c2["conviction_reason"] = "below batch cap (composite-score fallback)"
        out.append(c2)

    out.sort(key=lambda c: c["conviction_score"], reverse=True)
    logger.info(
        "meta_scorer scored %d candidates (top=%s/%d, bottom=%s/%d)",
        len(out), out[0]["ticker"], out[0]["conviction_score"],
        out[-1]["ticker"], out[-1]["conviction_score"],
    )
    return out


def _fallback_all(candidates: list[dict]) -> list[dict]:
    out = []
    for c, _cv in zip(candidates, _fallback_convictions(candidates)):
        c2 = dict(c)
        c2["conviction_score"] = _cv
        c2["conviction_reason"] = "fallback (LLM unavailable)"
        out.append(c2)
    return sorted(out, key=lambda c: c["conviction_score"], reverse=True)
