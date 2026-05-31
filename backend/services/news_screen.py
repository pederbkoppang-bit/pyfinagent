"""
Worldwide news idea generator — no-API-key RSS sources + Claude batch event extractor.

Pulls business-news headlines from worldwide free RSS feeds (Google News with
US/UK/DE/JP editions, Reuters Business, BBC Business, FT Business), dedupes via
word-3-gram Jaccard, and sends the consolidated stream to Claude Haiku 4.5 in
ONE batched call to extract `(ticker, event_type, polarity, confidence)`.

Tickers with positive polarity get added to the candidate pool BEFORE ranking,
so high-conviction news can surface tickers that pure quant momentum misses.

Cost target: <$0.05/cycle (one Claude call). Zero data-vendor cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.services.macro_regime import _strip_unsupported_schema_keys

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "news"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_HOURS = 4

# 7 no-key RSS endpoints. Worldwide coverage by design.
_REGISTERED_FEEDS: list[tuple[str, str]] = [
    ("google_news_business_us", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"),
    ("google_news_business_uk", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-GB&gl=GB&ceid=GB:en"),
    ("google_news_business_de", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=de-DE&gl=DE&ceid=DE:de"),
    ("google_news_business_jp", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=ja-JP&gl=JP&ceid=JP:ja"),
    ("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("cnbc_top", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("yahoo_finance", "https://finance.yahoo.com/news/rssindex"),
    ("ft_world", "https://www.ft.com/world?format=rss"),
]

_USER_AGENT = "PyFinAgent/2.0 NewsScreen (peder.bkoppang@hotmail.no)"
_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,3})?$")
_NS_ATOM = "{http://www.w3.org/2005/Atom}"


EventType = Literal[
    "earnings_beat", "earnings_miss", "merger_acquisition", "leadership_change",
    "product_launch", "regulatory_action", "legal_action", "macro_indicator",
    "analyst_upgrade", "analyst_downgrade", "other", "no_event",
]
ImpactPolarity = Literal["positive", "negative", "neutral", "ambiguous"]
ConfidenceLevel = Literal["high", "medium", "low"]


class NewsHeadlineSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker_mentioned: Optional[str] = Field(
        default=None,
        description="Primary ticker symbol explicitly mentioned or strongly implied. Use exchange ticker (AAPL, 005930.KS, 7203.T). Null if no specific equity is the subject.",
    )
    event_type: EventType = Field(
        description="Category of financial event. Use 'no_event' if headline is general market commentary with no specific company event.",
    )
    impact_polarity: ImpactPolarity = Field(
        description="Expected directional impact on the named equity (or broad market if no equity). 'ambiguous' when signals conflict.",
    )
    confidence: ConfidenceLevel = Field(
        description="'high' = ticker + event explicit. 'medium' = either implied. 'low' = speculative.",
    )
    rationale: str = Field(description="Free-text max 200 chars. State the key phrase.")
    skip_reason: str = Field(default="")


class NewsSignalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signals: list[NewsHeadlineSignal]


def _parse_rss(xml_text: str, source_label: str) -> list[dict]:
    """Parse RSS 2.0 or Atom into a list of {title, link, source}."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    items: list[dict] = []
    # RSS 2.0
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        if title:
            items.append({"title": title, "url": link, "source": source_label})
    # Atom
    for entry in root.iter(f"{_NS_ATOM}entry"):
        title_el = entry.find(f"{_NS_ATOM}title")
        link_el = entry.find(f"{_NS_ATOM}link")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        if title:
            items.append({"title": title, "url": link, "source": source_label})
    return items


async def _fetch_one_feed(client: httpx.AsyncClient, label: str, url: str) -> list[dict]:
    try:
        resp = await client.get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning("News feed %s returned HTTP %s", label, resp.status_code)
            return []
        return _parse_rss(resp.text, label)
    except Exception as e:
        logger.warning("News feed %s failed: %s", label, e)
        return []


async def _fetch_all_feeds() -> list[dict]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(_fetch_one_feed(client, label, url) for label, url in _REGISTERED_FEEDS),
            return_exceptions=False,
        )
    flat: list[dict] = []
    for batch in results:
        flat.extend(batch)
    return flat


def _word_3grams(text: str) -> set[tuple[str, str, str]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < 3:
        return set()
    return set(zip(words, words[1:], words[2:]))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup_jaccard(items: list[dict], threshold: float = 0.4) -> list[dict]:
    """Greedy near-duplicate elimination on titles. Keeps the FIRST occurrence."""
    kept: list[dict] = []
    kept_grams: list[set] = []
    for item in items:
        title = item.get("title") or ""
        grams = _word_3grams(title)
        is_dup = False
        for prior in kept_grams:
            if _jaccard(grams, prior) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
            kept_grams.append(grams)
    return kept


def _normalize_ticker(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    clean = t.upper().strip().lstrip("$")
    if _TICKER_RE.match(clean):
        return clean
    return None


def _build_batch_prompt(headlines: list[dict]) -> str:
    lines = [
        "You receive a numbered list of business-news headlines from worldwide sources.",
        "For EACH numbered headline, return one NewsHeadlineSignal in the same order.",
        "",
        "Rules:",
        "- ticker_mentioned: ONLY if a specific public equity is the subject. Use the exchange ticker. Null otherwise.",
        "- event_type: pick the closest match; use 'no_event' for general market commentary.",
        "- impact_polarity: directional expectation on the equity (or broad market).",
        "- confidence: 'high' if ticker+event explicit; 'medium' if implied; 'low' if speculative.",
        "- rationale: max 200 chars, name the key phrase.",
        "",
        "Headlines:",
    ]
    for i, h in enumerate(headlines, start=1):
        lines.append(f"{i}. [{h.get('source','?')}] {h.get('title','')}")
    lines += [
        "",
        f"Return JSON only matching NewsSignalBatch with EXACTLY {len(headlines)} signals in input order.",
    ]
    return "\n".join(lines)


def _cache_path() -> Path:
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    return _CACHE_DIR / f"news_screen_{bucket}.json"


def _load_cache() -> Optional[dict[str, NewsHeadlineSignal]]:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        if age > timedelta(hours=_CACHE_TTL_HOURS):
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {t: NewsHeadlineSignal.model_validate(v) for t, v in raw.items()}
    except Exception as e:
        logger.warning("News cache unreadable: %s", e)
        return None


def _save_cache(signals: dict[str, NewsHeadlineSignal]) -> None:
    try:
        p = _cache_path()
        payload = {t: json.loads(s.model_dump_json()) for t, s in signals.items()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("News cache write failed: %s", e)


async def fetch_news_signals(
    use_cache: bool = True,
    max_headlines: int = 100,
) -> dict[str, NewsHeadlineSignal]:
    """Fetch worldwide news, dedup, classify, return {ticker: NewsHeadlineSignal}.

    Returns an empty dict on any unrecoverable error (default-OFF safety).
    """
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            logger.info("News screen cache hit: %d tickers", len(cached))
            return cached

    raw = await _fetch_all_feeds()
    if not raw:
        logger.warning("News screen: zero raw items from all feeds")
        return {}

    # Stage A: cap at 4x max_headlines to bound LLM cost on bursty days
    raw = raw[: max_headlines * 4]
    deduped = _dedup_jaccard(raw, threshold=0.4)[:max_headlines]
    if not deduped:
        return {}
    logger.info("News screen: %d raw -> %d deduped headlines", len(raw), len(deduped))

    settings = get_settings()
    # phase-51.1: unwrap SecretStr (a non-empty SecretStr is truthy -> `or ""`
    # returned the wrapper -> SDK "Header value must be str or bytes"). Never str().
    from backend.agents.llm_client import unwrap_secret
    anthropic_key = unwrap_secret(getattr(settings, "anthropic_api_key", ""))
    if not anthropic_key:
        logger.warning("News screen: ANTHROPIC_API_KEY missing, skipping LLM extract")
        return {}

    from backend.agents.llm_client import ClaudeClient
    client = ClaudeClient(
        model_name=getattr(settings, "news_screen_model", "claude-haiku-4-5"),
        api_key=anthropic_key,
        enable_prompt_caching=False,
    )

    prompt = _build_batch_prompt(deduped)
    cleaned_schema = _strip_unsupported_schema_keys(NewsSignalBatch.model_json_schema())
    try:
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            {
                "response_schema": cleaned_schema,
                "response_mime_type": "application/json",
                "max_output_tokens": min(8192, 250 * len(deduped)),
                "temperature": 0.0,
            },
        )
    except Exception as e:
        logger.warning("News screen LLM call failed: %s", e)
        return {}

    try:
        raw_payload = json.loads(response.text)
        # Trim rationale to 200 chars per signal before validation
        for s in raw_payload.get("signals", []):
            if isinstance(s.get("rationale"), str):
                s["rationale"] = s["rationale"][:200]
        batch = NewsSignalBatch.model_validate(raw_payload)
    except Exception as e:
        logger.warning("News screen parse failed: %s | raw=%s", e, response.text[:200])
        return {}

    out: dict[str, NewsHeadlineSignal] = {}
    for sig in batch.signals:
        ticker = _normalize_ticker(sig.ticker_mentioned)
        if not ticker:
            continue
        if sig.confidence == "low":
            continue
        # First-seen wins (preserves news ordering); positive polarity preferred
        if ticker not in out or (sig.impact_polarity == "positive" and out[ticker].impact_polarity != "positive"):
            sig = sig.model_copy(update={"ticker_mentioned": ticker})
            out[ticker] = sig

    _save_cache(out)
    logger.info("News screen produced %d ticker signals", len(out))
    return out


def apply_news_to_score(
    base_score: float,
    ticker: Optional[str],
    news_signals: Optional[dict[str, NewsHeadlineSignal]],
) -> float:
    """Apply news boost to a screener composite score. Identity if no signal."""
    if not news_signals or not ticker or ticker not in news_signals:
        return base_score
    sig = news_signals[ticker]
    if sig.confidence == "low":
        return base_score
    if sig.impact_polarity == "positive":
        return base_score * 1.10
    if sig.impact_polarity == "negative":
        return base_score * 0.90
    return base_score
