"""phase-7.9 Google Trends ingestion -- scaffold.

Will persist Google Trends 0-100 indices for a starter keyword set to
`pyfinagent_data.alt_google_trends`. This cycle ships the scaffold; live
`pytrends-modern` integration + rolling-z normalization deferred to phase-7.12.

Compliance (docs/compliance/alt-data.md row 7.9): pytrends + undocumented
Google endpoints; minimum viable <= 5 req/min; weekly pulls only. Library
note: pytrends (GeneralMills) was archived 2025-04-17; phase-7.12 should
depend on `pytrends-modern>=0.2.5` (drop-in, released 2026-03-05).

Privacy: Google aggregates + anonymizes upstream; no user-level PII.

CLI:
    python -m backend.alt_data.google_trends [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_google_trends"
_RATE_INTERVAL_S = 12.0  # 12s per keyword, well under 5 req/min ceiling

_STARTER_KEYWORDS: tuple[str, ...] = (
    "buy stocks",
    "sell stocks",
    "recession",
    "inflation",
    "bull market",
    "bear market",
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  trend_id STRING NOT NULL,
  keyword STRING NOT NULL,
  as_of_date DATE NOT NULL,
  timeframe STRING NOT NULL,
  date_point DATE NOT NULL,
  value INT64,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY keyword
OPTIONS (
  description = "phase-7.9 Google Trends 0-100 index; rolling-z in phase-7.12"
)
""".strip()


def _trend_id(keyword: str, as_of: str, timeframe: str, date_point: str) -> str:
    return hashlib.sha256(
        f"{keyword}|{as_of}|{timeframe}|{date_point}".encode("utf-8")
    ).hexdigest()[:24]


def _rate_limit() -> None:
    time.sleep(_RATE_INTERVAL_S)


def fetch_trend(keyword: str, timeframe: str = "now 7-d") -> list[dict[str, Any]]:
    """Scaffold -- returns [] until phase-7.12 wires pytrends-modern.

    # TODO phase-7.12: use pytrends-modern >=0.2.5
    #   from pytrends.request import TrendReq
    #   TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.3)
    #   .build_payload([keyword], timeframe=timeframe)
    #   .interest_over_time() -> DataFrame rows = [{date_point, value}]
    # TODO phase-7.12: call _rate_limit() BEFORE each request.
    """
    logger.debug("google_trends: fetch_trend scaffold keyword=%r timeframe=%r", keyword, timeframe)
    return []


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
            logger.warning("google_trends: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("google_trends: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("google_trends: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_TABLE_SQL.format(project=proj, dataset=ds, table=_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("google_trends: ensure_table fail-open: %r", exc)
        return False


def upsert(
    rows: list[dict[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return 0
    table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
    try:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("google_trends: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("google_trends: upsert fail-open: %r", exc)
        return 0


def ingest_keywords(
    keywords: Iterable[str] = _STARTER_KEYWORDS,
    *,
    timeframe: str = "now 7-d",
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    today_iso = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for kw in keywords:
        series = fetch_trend(kw, timeframe=timeframe)
        for p in series:
            dp = str(p.get("date_point") or "")
            rows.append(
                {
                    "trend_id": _trend_id(kw, today_iso, timeframe, dp),
                    "keyword": kw,
                    "as_of_date": today_iso,
                    "timeframe": timeframe,
                    "date_point": dp,
                    "value": p.get("value"),
                    "source": "google_trends/pytrends-modern",
                    "raw_payload": json.dumps(p, default=str, ensure_ascii=True),
                }
            )
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.9 Google Trends ingester (scaffold)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    count = ingest_keywords(dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "ingested": count,
                "scaffold_only": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
