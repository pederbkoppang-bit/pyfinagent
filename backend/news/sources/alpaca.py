"""phase-6.3 Alpaca news adapter.

Uses `GET https://data.alpaca.markets/v1beta1/news` with
`Apca-Api-Key-Id` + `Apca-Api-Secret-Key` headers. Returns empty
iterable when either key is empty or the API returns a non-200 --
graceful degrade.

Alpaca response shape:
    {"news": [ {...}, ... ], "next_page_token": str | None}
Per article: {"id": int, "headline": str, "author": str,
              "created_at": str (ISO), "updated_at": str (ISO),
              "summary": str, "content": str, "url": str,
              "images": [...], "symbols": [str, ...],
              "source": str}

Pagination via `next_page_token` is NOT followed in phase-6.3 --
single page per `.fetch()`. The cron runs this often enough that
we do not need backfill here.

Rate limit: 200 req/min free, 10k/min Algo Trader Plus.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable

import httpx

from backend.config.settings import get_settings
from backend.news.fetcher import RawArticle
from backend.news.registry import register
from backend.services.observability import (
    get_rate_limiter,
    log_api_call,
    raise_cron_alert,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"
_TIMEOUT_SEC = 30.0


@register("alpaca")
class AlpacaSource:
    name = "alpaca"

    def fetch(self) -> Iterable[RawArticle]:
        settings = get_settings()
        key_id = settings.alpaca_api_key_id or ""
        secret = settings.alpaca_api_secret_key or ""
        if not key_id or not secret:
            logger.debug(
                "alpaca: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY missing, returning []"
            )
            return
        limiter = get_rate_limiter("alpaca")
        limiter.acquire_sync()
        t0 = time.perf_counter()
        articles: list = []
        status: int | None = None
        error_kind: str | None = None
        resp_bytes = 0
        try:
            def _do() -> httpx.Response:
                with httpx.Client(timeout=_TIMEOUT_SEC) as client:
                    return client.get(
                        _ENDPOINT,
                        headers={
                            "Apca-Api-Key-Id": key_id,
                            "Apca-Api-Secret-Key": secret,
                            "Accept": "application/json",
                        },
                        params={"limit": 50},
                    )

            resp = retry_with_backoff(
                _do,
                max_attempts=3,
                base=1.0,
                multiplier=2.0,
                cap=30.0,
                jitter="full",
                retry_on=(429, 502, 503, 504),
                honor_retry_after=True,
            )
            status = resp.status_code
            resp_bytes = len(resp.content or b"")
            if resp.status_code != 200:
                error_kind = "RateLimited" if resp.status_code == 429 else "HTTPError"
                logger.warning(
                    "alpaca: non-200 status=%d body=%s",
                    resp.status_code, resp.text[:200],
                )
                raise_cron_alert(
                    source="alpaca",
                    error_type=error_kind,
                    severity="P2",
                    title=f"Alpaca news {error_kind}",
                    details=f"status={resp.status_code} body={resp.text[:200]}",
                )
                return
            payload = resp.json() or {}
            articles = payload.get("news") or []
        except Exception as e:
            error_kind = type(e).__name__
            logger.warning("alpaca fetch failed: %s: %s", type(e).__name__, e)
            raise_cron_alert(
                source="alpaca",
                error_type=error_kind,
                severity="P2",
                title=f"Alpaca fetch failed: {error_kind}",
                details=str(e)[:500],
            )
            return
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            log_api_call(
                source="alpaca",
                endpoint=_ENDPOINT,
                http_status=status,
                latency_ms=latency_ms,
                response_bytes=resp_bytes,
                cost_usd_est=0.0,
                ok=(status == 200),
                error_kind=error_kind,
            )

        for row in articles:
            symbols = row.get("symbols") or []
            ticker = None
            if isinstance(symbols, list) and symbols:
                ticker = str(symbols[0])

            author = row.get("author")
            authors = [str(author)] if author else []

            provider_source = str(row.get("source") or "").strip()
            categories = [provider_source] if provider_source else []

            body = str(row.get("content") or row.get("summary") or "")
            published_at = (
                str(row.get("created_at") or row.get("updated_at") or "")
                or datetime.now(timezone.utc).isoformat()
            )

            yield RawArticle(
                source="alpaca",
                title=str(row.get("headline") or ""),
                body=body,
                url=str(row.get("url") or ""),
                published_at=published_at,
                ticker=ticker,
                authors=authors,
                categories=categories,
                raw_payload={
                    "provider_id": row.get("id"),
                    "symbols": list(symbols),
                    "images": row.get("images") or [],
                    "updated_at": row.get("updated_at"),
                },
            )
