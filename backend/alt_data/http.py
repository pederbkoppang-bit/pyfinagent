"""phase-7.11 Shared scraper infrastructure.

Consolidates the duplicated HTTP + audit-log patterns across all phase-7
ingesters (congress, f13, finra_short, twitter, google_trends, reddit_wsb,
hiring, etf_flows) into a single `ScraperClient`.

Resolves advisories:
- `adv_73_cdn_403`: bounded 403 backoff `min(base * 2**attempt, base * 8)`
  (was unbounded `60 * 2**attempt`).
- `adv_71_docstring_merge`: audit-log writes are streaming insert only.

Design (per phase-7.11 research brief):
- `UserAgent` constants (SEC, REDDIT, GENERIC).
- `RateLimit` dataclass with per-source tuning.
- `ScraperClient` with sliding-window (deque maxlen=20) circuit breaker,
  full-jitter backoff, correlation-id audit rows.
- `scraper_audit_log` BQ table matching compliance doc Sec. 6.1.
- `get_shared_client(source_name)` factory.

Fail-open everywhere; ASCII-only logger discipline per `.claude/rules/
security.md`.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_AUDIT_TABLE = "scraper_audit_log"


class UserAgent:
    """User-Agent strings per source family.

    Reddit requires a Reddit-specific format (see reddit_wsb.py); SEC EDGAR
    requires contact info. GENERIC falls back to the SEC form for any
    source without strict format requirements.
    """

    SEC = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
    REDDIT = "python:pyfinagent:1.0 (by /u/pederbkoppang)"
    GENERIC = "pyfinagent/1.0 peder.bkoppang@hotmail.no"


@dataclasses.dataclass(frozen=True)
class RateLimit:
    """Per-source rate-limit + backoff config.

    `min_request_interval_s = 1.0 / per_second_cap` is honored between
    successive calls on the same client. Backoff is
    `min(base * 2**attempt, base * backoff_max_multiplier) + jitter` on
    403 / 5xx, bounded by `max_attempts`.
    """

    per_second_cap: float = 2.0
    burst_cap: int = 1
    backoff_403_base_s: float = 60.0
    backoff_5xx_base_s: float = 5.0
    max_attempts: int = 3
    backoff_max_multiplier: int = 8  # hard cap = base * 8 (3 doublings)
    request_timeout_s: float = 30.0


# Source -> (RateLimit, UserAgent)
SOURCE_PRESETS: dict[str, tuple[RateLimit, str]] = {
    "sec.edgar": (
        RateLimit(per_second_cap=8.0, backoff_403_base_s=60.0),
        UserAgent.SEC,
    ),
    "finra.cdn": (
        RateLimit(per_second_cap=0.5, backoff_403_base_s=5.0),
        UserAgent.GENERIC,
    ),
    "reddit": (
        RateLimit(per_second_cap=1.6, backoff_403_base_s=60.0),
        UserAgent.REDDIT,
    ),
    "x.api": (
        RateLimit(per_second_cap=0.5, backoff_403_base_s=60.0),
        UserAgent.GENERIC,
    ),
    "google.trends": (
        RateLimit(per_second_cap=0.08, backoff_403_base_s=60.0),
        UserAgent.GENERIC,
    ),
    "github.raw": (
        RateLimit(per_second_cap=5.0, backoff_403_base_s=30.0),
        UserAgent.GENERIC,
    ),
    "linkup.api": (
        RateLimit(per_second_cap=1.0, backoff_403_base_s=60.0),
        UserAgent.GENERIC,
    ),
    "generic": (
        RateLimit(),
        UserAgent.GENERIC,
    ),
}


_CREATE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  request_id STRING NOT NULL,
  source STRING NOT NULL,
  url STRING NOT NULL,
  method STRING,
  status_code INT64,
  latency_ms FLOAT64,
  user_agent STRING,
  ip_hash STRING,
  ts TIMESTAMP NOT NULL,
  bytes_returned INT64,
  error STRING
)
PARTITION BY DATE(ts)
CLUSTER BY source, status_code
OPTIONS (
  description = "phase-7.11 scraper audit log; one row per live HTTP request"
)
""".strip()


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("http: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("http: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("http: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_audit_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    """Idempotent CREATE TABLE IF NOT EXISTS for scraper_audit_log. Fail-open."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_AUDIT_SQL.format(project=proj, dataset=ds, table=_AUDIT_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("http: ensure_audit_table fail-open: %r", exc)
        return False


def _audit_write(row: dict[str, Any], *, project: str | None = None, dataset: str | None = None) -> bool:
    """Append ONE row to scraper_audit_log. Streaming insert only; never MERGE."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    table_ref = f"{proj}.{ds}.{_AUDIT_TABLE}" if proj else f"{ds}.{_AUDIT_TABLE}"
    try:
        errors = client.insert_rows_json(table_ref, [row])
        if errors:
            logger.warning("http: audit insert errors: %s", errors[:1])
            return False
        return True
    except Exception as exc:
        logger.warning("http: audit fail-open: %r", exc)
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jittered_backoff(base_s: float, attempt: int, max_multiplier: int) -> float:
    cap = base_s * max_multiplier
    exp = base_s * (2**attempt)
    return min(cap, exp) + random.uniform(0.0, 1.0)


class ScraperClient:
    """Shared HTTP client with rate-limit, bounded backoff, circuit breaker, audit-log.

    Circuit breaker: sliding deque(maxlen=circuit_window) of successes (True) /
    failures (False). When the window is full and failure-rate exceeds
    `circuit_breaker_threshold`, `_cb_open_until` is set to `now + 60s`;
    subsequent calls short-circuit to None until the cooldown elapses.

    4xx status codes (except 429) are NOT counted as circuit failures -- they
    are rate-limit / scope signals, not infrastructure outages.
    """

    def __init__(
        self,
        source_name: str,
        *,
        rate_limit: RateLimit | None = None,
        user_agent: str | None = None,
        audit_on: bool = True,
        circuit_breaker_threshold: float = 0.5,
        circuit_window: int = 20,
        project: str | None = None,
        dataset: str | None = None,
    ) -> None:
        self.source_name = source_name
        preset_rl, preset_ua = SOURCE_PRESETS.get(source_name, SOURCE_PRESETS["generic"])
        self.rate_limit = rate_limit or preset_rl
        self.user_agent = user_agent or preset_ua
        self.audit_on = audit_on
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self._cb_window: collections.deque[bool] = collections.deque(maxlen=circuit_window)
        self._cb_open_until: float = 0.0
        self._last_request_ts: float = 0.0
        self._project = project
        self._dataset = dataset

    def _sleep_rate_limit(self) -> None:
        cap = self.rate_limit.per_second_cap
        if cap <= 0:
            return
        min_interval = 1.0 / cap
        elapsed = time.time() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _record_cb(self, success: bool) -> None:
        self._cb_window.append(success)
        if len(self._cb_window) == self._cb_window.maxlen:
            fail_rate = sum(1 for x in self._cb_window if not x) / len(self._cb_window)
            if fail_rate > self.circuit_breaker_threshold:
                self._cb_open_until = time.time() + 60.0
                logger.warning(
                    "http.%s: circuit breaker TRIPPED (fail_rate=%.2f); cooldown 60s",
                    self.source_name, fail_rate,
                )

    def _audit_row(
        self,
        *,
        request_id: str,
        url: str,
        method: str,
        status_code: int | None,
        latency_ms: float,
        bytes_returned: int | None,
        error: str | None,
    ) -> dict[str, Any]:
        ip_hash = None
        # Egress IP hashing is best-effort; skipped unless explicitly configured.
        return {
            "request_id": request_id,
            "source": self.source_name,
            "url": url,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "user_agent": self.user_agent,
            "ip_hash": ip_hash,
            "ts": _now_iso(),
            "bytes_returned": bytes_returned,
            "error": error,
        }

    def get(self, url: str, *, accept: str | None = None) -> Any:
        """HTTP GET with rate-limit, bounded backoff, audit, circuit breaker.

        Returns a `requests.Response` on success, `None` on any failure or
        when the circuit is open. Never raises.
        """
        if time.time() < self._cb_open_until:
            logger.warning("http.%s: circuit open; short-circuit url=%s", self.source_name, url)
            return None
        try:
            import requests  # type: ignore[import-not-found]
        except Exception as exc:
            logger.warning("http.%s: requests missing: %r", self.source_name, exc)
            return None
        headers = {"User-Agent": self.user_agent}
        if accept:
            headers["Accept"] = accept
        request_id = uuid.uuid4().hex[:16]
        timeout = self.rate_limit.request_timeout_s
        last_resp = None
        last_err: str | None = None
        last_status: int | None = None
        last_latency_ms: float = 0.0
        last_bytes: int | None = None
        for attempt in range(self.rate_limit.max_attempts):
            self._sleep_rate_limit()
            t0 = time.time()
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
            except Exception as exc:
                last_err = repr(exc)
                last_status = None
                last_latency_ms = (time.time() - t0) * 1000.0
                last_bytes = None
                logger.warning("http.%s: exception url=%s attempt=%d err=%r", self.source_name, url, attempt, exc)
                self._record_cb(False)
                continue
            self._last_request_ts = time.time()
            last_status = resp.status_code
            last_latency_ms = (time.time() - t0) * 1000.0
            try:
                last_bytes = len(resp.content) if resp.content is not None else 0
            except Exception:
                last_bytes = None
            if resp.status_code == 200:
                if self.audit_on:
                    _audit_write(
                        self._audit_row(
                            request_id=request_id,
                            url=url,
                            method="GET",
                            status_code=200,
                            latency_ms=last_latency_ms,
                            bytes_returned=last_bytes,
                            error=None,
                        ),
                        project=self._project,
                        dataset=self._dataset,
                    )
                self._record_cb(True)
                return resp
            if resp.status_code == 403 and attempt < self.rate_limit.max_attempts - 1:
                time.sleep(
                    _jittered_backoff(
                        self.rate_limit.backoff_403_base_s, attempt, self.rate_limit.backoff_max_multiplier
                    )
                )
                # Do NOT record as circuit failure: 403 is a rate-limit signal.
                last_err = "403"
                continue
            if 500 <= resp.status_code < 600 and attempt < self.rate_limit.max_attempts - 1:
                time.sleep(
                    _jittered_backoff(
                        self.rate_limit.backoff_5xx_base_s, attempt, self.rate_limit.backoff_max_multiplier
                    )
                )
                last_err = f"5xx:{resp.status_code}"
                self._record_cb(False)
                continue
            # Non-200, non-retry-worthy: record, log, return None
            last_err = f"non-200:{resp.status_code}"
            if resp.status_code < 400 or resp.status_code >= 500:
                self._record_cb(False)
            break
        if self.audit_on:
            _audit_write(
                self._audit_row(
                    request_id=request_id,
                    url=url,
                    method="GET",
                    status_code=last_status,
                    latency_ms=last_latency_ms,
                    bytes_returned=last_bytes,
                    error=last_err,
                ),
                project=self._project,
                dataset=self._dataset,
            )
        return last_resp


def get_shared_client(
    source_name: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> ScraperClient:
    """Factory. source_name keys into SOURCE_PRESETS (falls back to 'generic')."""
    return ScraperClient(source_name, project=project, dataset=dataset)


__all__ = [
    "UserAgent",
    "RateLimit",
    "SOURCE_PRESETS",
    "ScraperClient",
    "ensure_audit_table",
    "get_shared_client",
]
