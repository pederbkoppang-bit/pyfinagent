"""phase-6.3 Benzinga news adapter.

Uses `GET https://api.benzinga.com/api/v2/news` with
`Authorization: token <key>` header. Returns empty iterable when
`settings.benzinga_api_key` is empty or the API returns a non-200
-- graceful degrade.

Benzinga response shape (per article):
    {"id": int, "title": str, "created": str (ISO),
     "updated": str (ISO), "teaser": str, "body": str, "url": str,
     "stocks": [{"name": str, "ticker": str}, ...],
     "channels": [{"name": str}, ...],
     "tags": [str, ...], "author": str}

`stocks` and `channels` are lists of dicts (not flat strings) --
take `stocks[0].ticker` for `ticker` and `channels[].name` for
`categories`.
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

_ENDPOINT = "https://api.benzinga.com/api/v2/news"
_TIMEOUT_SEC = 30.0


@register("benzinga")
class BenzingaSource:
    name = "benzinga"

    def fetch(self) -> Iterable[RawArticle]:
        settings = get_settings()
        key = settings.benzinga_api_key or ""
        if not key:
            logger.debug("benzinga: no BENZINGA_API_KEY set, returning []")
            return
        limiter = get_rate_limiter("benzinga")
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
                        headers={
                            "Authorization": f"token {key}",
                            "Accept": "application/json",
                        },
                        params={"pageSize": 100, "displayOutput": "full"},
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
                    "benzinga: non-200 status=%d body=%s",
                    resp.status_code, resp.text[:200],
                )
                raise_cron_alert(
                    source="benzinga",
                    error_type=error_kind,
                    severity="P2",
                    title=f"Benzinga news {error_kind}",
                    details=f"status={resp.status_code} body={resp.text[:200]}",
                )
                return
            payload = resp.json() or []
        except Exception as e:
            error_kind = type(e).__name__
            logger.warning("benzinga fetch failed: %s: %s", type(e).__name__, e)
            raise_cron_alert(
                source="benzinga",
                error_type=error_kind,
                severity="P2",
                title=f"Benzinga fetch failed: {error_kind}",
                details=str(e)[:500],
            )
            return
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            log_api_call(
                source="benzinga",
                endpoint=_ENDPOINT,
                http_status=status,
                latency_ms=latency_ms,
                response_bytes=resp_bytes,
                cost_usd_est=0.0,
                ok=(status == 200),
                error_kind=error_kind,
            )

        for row in payload:
            stocks = row.get("stocks") or []
            ticker = None
            if isinstance(stocks, list) and stocks:
                first = stocks[0]
                if isinstance(first, dict):
                    ticker = first.get("ticker") or first.get("symbol")
                elif isinstance(first, str):
                    ticker = first

            channels = row.get("channels") or []
            categories: list[str] = []
            for ch in channels:
                if isinstance(ch, dict) and ch.get("name"):
                    categories.append(str(ch["name"]))

            authors: list[str] = []
            author = row.get("author")
            if author:
                authors.append(str(author))

            body = str(row.get("body") or row.get("teaser") or "")
            published_at = (
                str(row.get("created") or row.get("updated") or "")
                or datetime.now(timezone.utc).isoformat()
            )

            yield RawArticle(
                source="benzinga",
                title=str(row.get("title") or ""),
                body=body,
                url=str(row.get("url") or ""),
                published_at=published_at,
                ticker=ticker,
                authors=authors,
                categories=categories,
                raw_payload={
                    "provider_id": row.get("id"),
                    "tags": row.get("tags") or [],
                    "updated": row.get("updated"),
                },
            )
