"""
PEAD signal — Earnings Post-Announcement Drift overlay.

For tickers that filed an 8-K (item 2.02) in the recent window, fetches the
press-release exhibit from SEC EDGAR (free, no API key) and asks Claude
Haiku 4.5 to score sentiment + compute sentiment surprise vs trailing 8-quarter
mean (read from local cache files; BQ persistence deferred to Phase 2).

Cost target: <$0.05/cycle. ~2-5 tickers/day on average for S&P 500.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.tools.sec_insider import SEC_HEADERS, _resolve_cik
from backend.services.macro_regime import _strip_unsupported_schema_keys

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "pead"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_VALID_HOLDING_WINDOWS = {14, 28, 42, 60}
_DEFAULT_HOLDING_WINDOW = 28
_VALID_TAGS = ("positive_surprise", "negative_surprise", "neutral", "insufficient_history")
_LOOKBACK_QUARTERS = 8

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"
_ARCHIVE_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


class PeadSignalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(description="Free-text explanation max 300 chars.")
    sentiment_score: float = Field(
        ge=0.0, le=1.0,
        description="Sentiment of THIS press release on 0.0-1.0 scale (1.0 = maximally positive).",
    )
    surprise_score: float = Field(
        description="sentiment_score - rolling-8Q mean. Positive = above-trend tone; negative = below-trend.",
    )
    sentiment_tag: Literal["positive_surprise", "negative_surprise", "neutral", "insufficient_history"]
    holding_window_days: int = Field(
        description="Recommended hold window in days. Must be 14, 28, 42, or 60.",
    )
    skip_reason: str = Field(
        default="",
        description='Empty on success. "no_8k_found" | "no_exhibit_99" | "llm_error" | "parse_error" | "edgar_error".',
    )


def _ticker_cache_path(ticker: str, quarter_end: str) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "", ticker.upper())
    safe_q = re.sub(r"[^0-9-]", "", quarter_end)
    return _CACHE_DIR / f"pead_{safe_ticker}_{safe_q}.json"


def _load_pead_cache(ticker: str, quarter_end: str) -> Optional[PeadSignalOutput]:
    path = _ticker_cache_path(ticker, quarter_end)
    if not path.exists():
        return None
    try:
        return PeadSignalOutput.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("PEAD cache unreadable for %s/%s: %s", ticker, quarter_end, e)
        return None


def _save_pead_cache(ticker: str, quarter_end: str, signal: PeadSignalOutput) -> None:
    path = _ticker_cache_path(ticker, quarter_end)
    try:
        path.write_text(signal.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("PEAD cache write failed for %s/%s: %s", ticker, quarter_end, e)


def _trailing_mean_from_cache(ticker: str, exclude_quarter: str) -> tuple[Optional[float], int]:
    """Read prior PEAD cache files for this ticker; return (mean_sentiment, n_quarters)."""
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "", ticker.upper())
    pattern = f"pead_{safe_ticker}_*.json"
    scores: list[tuple[str, float]] = []
    for f in _CACHE_DIR.glob(pattern):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            q = f.stem.split("_", 2)[-1]
            if q == exclude_quarter:
                continue
            score = payload.get("sentiment_score")
            if isinstance(score, (int, float)):
                scores.append((q, float(score)))
        except Exception:
            continue
    scores.sort(key=lambda t: t[0], reverse=True)
    use = scores[:_LOOKBACK_QUARTERS]
    if not use:
        return None, 0
    return sum(s for _, s in use) / len(use), len(use)


def _fallback(reason: str, sentiment: float = 0.0, surprise: float = 0.0,
              tag: str = "neutral") -> PeadSignalOutput:
    return PeadSignalOutput(
        rationale=f"Fallback: {reason}"[:300],
        sentiment_score=max(0.0, min(1.0, sentiment)),
        surprise_score=surprise,
        sentiment_tag=tag if tag in _VALID_TAGS else "neutral",
        holding_window_days=_DEFAULT_HOLDING_WINDOW,
        skip_reason=reason,
    )


async def _fetch_recent_8k(client: httpx.AsyncClient, cik: str) -> Optional[dict]:
    """Return the most recent 8-K with item 2.02. dict keys: accession, filing_date, primary_document."""
    url = _SUBMISSIONS_URL.format(cik=cik)
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=SEC_HEADERS, timeout=30)
            if resp.status_code == 429:
                await asyncio.sleep(2 ** attempt + 1)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 2:
                logger.warning("EDGAR submissions fetch failed for CIK %s: %s", cik, e)
                return None
            await asyncio.sleep(2 ** attempt)
    else:
        return None

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    items_arr = recent.get("items", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary = recent.get("primaryDocument", [])
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = items_arr[i] if i < len(items_arr) else ""
        if "2.02" not in items:
            continue
        return {
            "accession": accs[i] if i < len(accs) else "",
            "filing_date": dates[i] if i < len(dates) else "",
            "primary_document": primary[i] if i < len(primary) else "",
        }
    return None


async def _fetch_exhibit_99_text(client: httpx.AsyncClient, cik: str, accession: str) -> Optional[str]:
    """Return cleaned text from Exhibit 99.x of the 8-K (max 4000 chars)."""
    cik_int = str(int(cik))
    acc_nodash = accession.replace("-", "")
    index_url = _FILING_INDEX_URL.format(cik=cik_int, acc_nodash=acc_nodash)
    try:
        resp = await client.get(index_url, headers=SEC_HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        index = resp.json()
    except Exception as e:
        logger.warning("EDGAR index fetch failed for %s: %s", accession, e)
        return None

    # The `type` field in EDGAR's index.json is the icon (text.gif / compressed.gif),
    # not the document type. Identify Exhibit 99 by filename pattern.
    items = index.get("directory", {}).get("item", [])
    exhibit_doc = None
    for item in items:
        name = (item.get("name") or "").lower()
        if not name.endswith((".htm", ".html", ".txt")):
            continue
        if "ex99" in name or "ex-99" in name or "exhibit99" in name:
            exhibit_doc = item["name"]
            break
    if not exhibit_doc:
        return None

    doc_url = _ARCHIVE_DOC_URL.format(cik=cik_int, acc_nodash=acc_nodash, doc=exhibit_doc)
    try:
        resp = await client.get(doc_url, headers=SEC_HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        raw = resp.text
    except Exception as e:
        logger.warning("EDGAR exhibit fetch failed for %s: %s", exhibit_doc, e)
        return None

    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def _build_pead_prompt(ticker: str, press_release_text: str, prior_mean: Optional[float], n_prior: int) -> str:
    history_line = (
        f"trailing-{n_prior}Q mean sentiment for this ticker: {prior_mean:.3f}"
        if prior_mean is not None else
        "trailing sentiment history: NOT AVAILABLE (insufficient_history)"
    )
    return (
        f"Score the earnings press release for {ticker}.\n\n"
        f"Press release text (first 4000 chars from SEC EDGAR Exhibit 99):\n\n"
        f"---\n{press_release_text}\n---\n\n"
        f"Compute sentiment_score on 0.0-1.0 scale where 0.5 = neutral, 1.0 = maximally bullish "
        f"(strong beat, strong guidance raise, positive forward language).\n\n"
        f"{history_line}\n\n"
        f"surprise_score = current sentiment_score minus the trailing mean (or 0 if insufficient_history).\n\n"
        f"sentiment_tag rules:\n"
        f"- 'positive_surprise' if surprise_score > +0.10 AND sentiment_score >= 0.6\n"
        f"- 'negative_surprise' if surprise_score < -0.10 AND sentiment_score <= 0.4\n"
        f"- 'insufficient_history' if no trailing data\n"
        f"- 'neutral' otherwise\n\n"
        f"holding_window_days: 14 (mild surprise), 28 (default), 42 (strong surprise), or 60 (very strong + clean guidance).\n\n"
        f"Generate the rationale FIRST, then commit to numeric fields. Return JSON only."
    )


async def compute_pead_signal_for_ticker(
    ticker: str,
    quarter_end: Optional[str] = None,
    use_cache: bool = True,
) -> PeadSignalOutput:
    """Compute PEAD signal for a single ticker. Returns a PeadSignalOutput; never raises."""
    settings = get_settings()
    anthropic_key = getattr(settings, "anthropic_api_key", "") or ""

    async with httpx.AsyncClient(headers=SEC_HEADERS) as http:
        cik = await _resolve_cik(http, ticker)
        if not cik:
            return _fallback("cik_not_found")

        latest_8k = await _fetch_recent_8k(http, cik)
        if not latest_8k:
            return _fallback("no_8k_found")

        if quarter_end is None:
            quarter_end = latest_8k.get("filing_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if use_cache:
            cached = _load_pead_cache(ticker, quarter_end)
            if cached is not None:
                logger.info("PEAD cache hit %s/%s tag=%s", ticker, quarter_end, cached.sentiment_tag)
                return cached

        press_text = await _fetch_exhibit_99_text(http, cik, latest_8k["accession"])
        if not press_text:
            return _fallback("no_exhibit_99")

    if not anthropic_key:
        return _fallback("no_anthropic_key")

    prior_mean, n_prior = _trailing_mean_from_cache(ticker, quarter_end)

    from backend.agents.llm_client import ClaudeClient
    client = ClaudeClient(
        model_name=getattr(settings, "pead_signal_model", "claude-haiku-4-5"),
        api_key=anthropic_key,
        enable_prompt_caching=False,
    )

    prompt = _build_pead_prompt(ticker, press_text, prior_mean, n_prior)
    cleaned_schema = _strip_unsupported_schema_keys(PeadSignalOutput.model_json_schema())
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
        logger.warning("PEAD LLM call failed for %s: %s", ticker, e)
        return _fallback(f"llm_error:{type(e).__name__}")

    try:
        raw = json.loads(response.text)
        if isinstance(raw.get("rationale"), str):
            raw["rationale"] = raw["rationale"][:300]
        if isinstance(raw.get("sentiment_score"), (int, float)):
            raw["sentiment_score"] = max(0.0, min(1.0, float(raw["sentiment_score"])))
        if raw.get("holding_window_days") not in _VALID_HOLDING_WINDOWS:
            raw["holding_window_days"] = _DEFAULT_HOLDING_WINDOW
        if prior_mean is None:
            raw["sentiment_tag"] = "insufficient_history"
            raw["surprise_score"] = 0.0
        parsed = PeadSignalOutput.model_validate(raw)
    except Exception as e:
        logger.warning("PEAD output parse failed for %s: %s | raw=%s", ticker, e, response.text[:200])
        return _fallback("parse_error")

    _save_pead_cache(ticker, quarter_end, parsed)
    logger.info(
        "PEAD computed %s/%s tag=%s sentiment=%.2f surprise=%.2f",
        ticker, quarter_end, parsed.sentiment_tag, parsed.sentiment_score, parsed.surprise_score,
    )
    return parsed


async def fetch_pead_signals_for_recent_reporters() -> dict[str, PeadSignalOutput]:
    """Fetch PEAD signals for tickers that reported earnings in the last 7 days.

    Reads tickers from `pyfinagent_data.calendar_events` BQ. Falls back to empty
    dict on any BQ error (default-OFF behavior preserves cycle).
    """
    try:
        from backend.db.bigquery_client import BigQueryClient
        bq = BigQueryClient()
        query = (
            "SELECT ticker FROM `pyfinagent_data.calendar_events` "
            "WHERE event_type = 'earnings' "
            "AND scheduled_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) "
            "AND scheduled_at <= CURRENT_TIMESTAMP() "
            "GROUP BY ticker"
        )
        rows = list(bq.client.query(query).result())
        tickers = [r["ticker"] for r in rows if r.get("ticker")]
    except Exception as e:
        logger.warning("PEAD recent-reporters BQ query failed: %s", e)
        return {}

    if not tickers:
        return {}

    sem = asyncio.Semaphore(3)

    async def _one(t: str) -> tuple[str, PeadSignalOutput]:
        async with sem:
            sig = await compute_pead_signal_for_ticker(t)
            await asyncio.sleep(0.15)
            return t, sig

    results = await asyncio.gather(*(_one(t) for t in tickers), return_exceptions=True)
    out: dict[str, PeadSignalOutput] = {}
    for r in results:
        if isinstance(r, tuple):
            out[r[0]] = r[1]
    return out


def apply_pead_to_score(
    base_score: float,
    ticker: str,
    pead_signals: Optional[dict[str, PeadSignalOutput]],
) -> Optional[float]:
    """Apply PEAD signal to a screener composite score.

    Returns None if the candidate should be FILTERED OUT entirely (strong negative surprise).
    Returns the adjusted score otherwise.
    """
    if not pead_signals or ticker not in pead_signals:
        return base_score
    sig = pead_signals[ticker]
    tag = sig.sentiment_tag
    surprise = sig.surprise_score

    if tag == "negative_surprise" and surprise < -0.3:
        return None
    if tag == "positive_surprise":
        return base_score * (1.0 + min(max(surprise, 0.0) * 0.5, 0.3))
    if tag == "negative_surprise":
        return base_score * max(1.0 + surprise * 0.5, 0.6)
    return base_score
