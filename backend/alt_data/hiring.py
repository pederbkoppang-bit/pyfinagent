"""phase-7.10 Hiring signals ingestion -- scaffold (LinkUp).

Persists `(ticker, company_name, title, department, location, posted_at,
is_active, ...)` rows to `pyfinagent_data.alt_hiring_signals`. Live LinkUp
REST call + MSA-gated auth deferred to phase-7.12.

Compliance: `docs/compliance/alt-data.md` row 7.10 -- LinkUp (licensed feed,
not LinkedIn scrape). MSA-based contract; no OAuth click-through.

Research anchor: job openings / employment level is the strongest single
equity-premium predictor among 24+ tested variables (+2.91% annualised CEQ,
+0.20 Sharpe over historical mean) -- J. Financial Markets 2023.

Dedup key: `posting_id = sha256(ticker|title|posted_at)[:24]` surrogate over
vendor-assigned ids (guards against vendor re-id on history refresh).

CLI:
    python -m backend.alt_data.hiring [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_hiring_signals"

_STARTER_COMPANIES: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  posting_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  ticker STRING,
  company_name STRING,
  title STRING,
  department STRING,
  location STRING,
  posted_at TIMESTAMP,
  last_seen_at TIMESTAMP,
  is_active BOOL,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY ticker, department
OPTIONS (
  description = "phase-7.10 LinkUp hiring signals; live MSA-gated fetch deferred to phase-7.12"
)
""".strip()


def _posting_id(ticker: str | None, title: str | None, posted_at: str | None) -> str:
    key = f"{ticker or ''}|{title or ''}|{posted_at or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def fetch_postings(ticker: str) -> list[dict[str, Any]]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will hit the LinkUp REST API with env var `LINKUP_API_KEY`
    (read INSIDE this function, never at import). Returns [] until
    implemented.

    # TODO phase-7.12: LinkUp Raw GET /postings?ticker={ticker}
    # TODO phase-7.12: rate-limit per MSA tier
    # TODO phase-7.12: BQ-native delivery alternative via bucket load
    """
    logger.debug("hiring: fetch_postings scaffold ticker=%s", ticker)
    return []


def normalize(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map LinkUp raw posting rows to the alt_hiring_signals DDL shape."""
    today_iso = date.today().isoformat()
    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        posted_at = r.get("posted_at") or r.get("date_posted") or None
        last_seen = r.get("last_seen_at") or r.get("last_seen") or None
        ticker = (r.get("ticker") or "").strip().upper() or None
        # is_active is derived from last_seen_at vs snapshot (today):
        # active if last_seen_at >= today - 7 days. Phase-7.12 will refine.
        is_active = r.get("is_active")
        if is_active is None and last_seen:
            try:
                ls = datetime.fromisoformat(str(last_seen).replace("Z", "+00:00"))
                today_dt = datetime.now(timezone.utc)
                is_active = (today_dt - ls).days <= 7
            except Exception:
                is_active = None
        out.append(
            {
                "posting_id": _posting_id(ticker, r.get("title"), str(posted_at) if posted_at else None),
                "as_of_date": today_iso,
                "ticker": ticker,
                "company_name": r.get("company_name") or r.get("company") or None,
                "title": r.get("title"),
                "department": r.get("department") or r.get("occupation_family") or None,
                "location": r.get("location") or r.get("city") or None,
                "posted_at": posted_at,
                "last_seen_at": last_seen,
                "is_active": is_active,
                "source": "linkup.com/raw",
                "raw_payload": json.dumps(r, default=str, ensure_ascii=True),
            }
        )
    return out


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
            logger.warning("hiring: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("hiring: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("hiring: bigquery.Client() init failed (%r)", exc)
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
        logger.warning("hiring: ensure_table fail-open: %r", exc)
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
            logger.warning("hiring: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("hiring: upsert fail-open: %r", exc)
        return 0


def ingest_companies(
    tickers: Iterable[str] = _STARTER_COMPANIES,
    *,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    rows: list[dict[str, Any]] = []
    for t in tickers:
        raw = fetch_postings(t)
        rows.extend(normalize(raw))
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.10 LinkUp hiring signals ingester (scaffold)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    count = ingest_companies(dry_run=args.dry_run)
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
