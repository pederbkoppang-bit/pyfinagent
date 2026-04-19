"""phase-6.3 Finnhub news adapter (phase-6.7 hardened).

Uses `GET https://finnhub.io/api/v1/news?category=general&token=<key>`
(market news). Returns empty iterable when `settings.finnhub_api_key`
is empty or the API returns a non-200 -- graceful degrade so the
cron does not hard-crash when one provider is down.

phase-6.7 hardening:
 - Client-side leaky-bucket rate limiter (aiolimiter via
   `backend.services.observability.get_rate_limiter('finnhub')`)
 - Retry with jitter + Retry-After on 429/5xx/network errors
 - api_call_log BQ telemetry per call (latency, http_status, bytes, error_kind)
 - raise_cron_alert on consecutive failures (dedup-aware)

Finnhub response shape (per article):
    {"category": str, "datetime": int (unix),
     "headline": str, "id": int, "image": str,
     "related": str (ticker), "source": str,
     "summary": str, "url": str}

Rate limit: 30 req/sec free tier. We cap at 25 via settings to keep headroom.
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

_ENDPOINT = "https://finnhub.io/api/v1/news"
_CATEGORY = "general"
_TIMEOUT_SEC = 30.0


@register("finnhub")
class FinnhubSource:
    name = "finnhub"

    def fetch(self) -> Iterable[RawArticle]:
        settings = get_settings()
        key = settings.finnhub_api_key or ""
        if not key:
            logger.debug("finnhub: no FINNHUB_API_KEY set, returning []")
            return
        limiter = get_rate_limiter("finnhub")
        limiter.acquire_sync()
        t0 = time.perf_counter()
        payload: list = []
        status: int | None = None
        error_kind: str | None = None
        resp_bytes = 0
        try:
            def _do() -> httpx.Response:
                with httpx.Client(timeout=_TIMEOUT_SEC) as client:
                    return client.get(
                        _ENDPOINT,
                        params={"category": _CATEGORY, "token": key},
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
                error_kind = "HTTPError" if resp.status_code < 500 else "ServerError"
                if resp.status_code == 429:
                    error_kind = "RateLimited"
                logger.warning(
                    "finnhub: non-200 status=%d body=%s",
                    resp.status_code, resp.text[:200],
                )
                raise_cron_alert(
                    source="finnhub",
                    error_type=error_kind,
                    severity="P2",
                    title=f"Finnhub news {error_kind}",
                    details=f"status={resp.status_code} body={resp.text[:200]}",
                )
                return
            payload = resp.json() or []
        except Exception as e:
            error_kind = type(e).__name__
            logger.warning("finnhub fetch failed: %s: %s", type(e).__name__, e)
            raise_cron_alert(
                source="finnhub",
                error_type=error_kind,
                severity="P2",
                title=f"Finnhub news fetch failed: {error_kind}",
                details=str(e)[:500],
            )
            return
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            log_api_call(
                source="finnhub",
                endpoint=_ENDPOINT,
                http_status=status,
                latency_ms=latency_ms,
                response_bytes=resp_bytes,
                cost_usd_est=0.0,
                ok=(status == 200),
                error_kind=error_kind,
            )

        for row in payload:
            ts = row.get("datetime")
            if isinstance(ts, (int, float)) and ts > 0:
                published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            else:
                published_at = datetime.now(timezone.utc).isoformat()
            yield RawArticle(
                source="finnhub",
                title=str(row.get("headline") or ""),
                body=str(row.get("summary") or ""),
                url=str(row.get("url") or ""),
                published_at=published_at,
                ticker=str(row.get("related") or "") or None,
                authors=[],
                categories=[str(row.get("category") or "").strip()] if row.get("category") else [],
                raw_payload={
                    "provider_source": row.get("source"),
                    "provider_id": row.get("id"),
                    "image": row.get("image"),
                },
            )
