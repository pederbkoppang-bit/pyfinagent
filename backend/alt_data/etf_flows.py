"""phase-7.4 ETF flows ingestion -- scaffold.

Derives daily net fund flows from issuer-page shares-outstanding + NAV data
and will persist to `pyfinagent_data.alt_etf_flows`. This cycle ships the
scaffold; live implementation lives in phase-7.12 feature integration.

Flow formula:  flow_usd = (shares_out_today - shares_out_prev) * nav_today

Starter ticker universe (top-20 by AUM, 2026):
    SPY QQQ IWM DIA VTI VOO VEA VWO EFA EEM AGG TLT IEF HYG
    LQD JNK GLD SLV USO XLK XLF XLE XLV XLP

Compliance: `docs/compliance/alt-data.md` row 7.4 -- HTTP scraper on issuer
pages, 1 req/2s cap. Public pages post-hiQ/X Corp framework; no login.

CLI:
    python -m backend.alt_data.etf_flows [--dry-run]
"""
from __future__ import annotations

import argparse
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
_TABLE = "alt_etf_flows"
_RATE_INTERVAL_S = 2.0  # 1 req per 2s per compliance row 7.4

_STARTER_TICKERS: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
    "EFA", "EEM", "AGG", "TLT", "IEF", "HYG", "LQD", "JNK",
    "GLD", "SLV", "USO", "XLK", "XLF", "XLE", "XLV", "XLP",
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  flow_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  ticker STRING NOT NULL,
  issuer STRING,
  nav FLOAT64,
  shares_out INT64,
  shares_out_prev INT64,
  flow_usd FLOAT64,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY ticker, issuer
OPTIONS (
  description = "phase-7.4 ETF daily net flows; flow_usd = (shares_out - shares_out_prev) * nav"
)
""".strip()


def _rate_limit() -> None:
    time.sleep(_RATE_INTERVAL_S)


def fetch_issuer_page(ticker: str) -> dict[str, Any]:
    """Fetch the issuer's product page for `ticker` and return parsed fields.

    Scaffold -- live implementation deferred to phase-7.12. Must return a dict
    shaped like `{"nav": float|None, "shares_out": int|None, "issuer": str|None,
    "raw": dict}`. For now returns an empty dict so callers fail-open cleanly.
    """
    _rate_limit()
    logger.debug("etf_flows: fetch_issuer_page scaffold for ticker=%s", ticker)
    return {}


def derive_flow(
    shares_out_t: int | None,
    shares_out_prev: int | None,
    nav: float | None,
) -> float | None:
    """Return (shares_out_t - shares_out_prev) * nav, or None if any input missing."""
    if shares_out_t is None or shares_out_prev is None or nav is None:
        return None
    try:
        return float(shares_out_t - shares_out_prev) * float(nav)
    except (TypeError, ValueError):
        return None


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
            logger.warning("etf_flows: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("etf_flows: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("etf_flows: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    """Idempotent CREATE TABLE IF NOT EXISTS. Fail-open."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_TABLE_SQL.format(project=proj, dataset=ds, table=_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("etf_flows: ensure_table fail-open: %r", exc)
        return False


def upsert(
    rows: list[dict[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Stream-insert rows into alt_etf_flows. Fail-open; returns count inserted."""
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
            logger.warning("etf_flows: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("etf_flows: upsert fail-open: %r", exc)
        return 0


def ingest_tickers(
    tickers: Iterable[str] = _STARTER_TICKERS,
    *,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Scaffold orchestrator.

    Live implementation (phase-7.12) will:
    1. Look up prior-day shares_out for each ticker.
    2. fetch_issuer_page(ticker) -> current shares_out + nav.
    3. derive_flow(...) -> flow_usd.
    4. upsert rows.

    This scaffold walks the tickers, calls the stub fetcher, and returns 0
    because no rows are produced yet. Callers can monkeypatch
    `fetch_issuer_page` in tests to smoke the full pipeline.
    """
    today_iso = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for t in tickers:
        info = fetch_issuer_page(t)
        if not info:
            continue
        nav = info.get("nav")
        shares_out = info.get("shares_out")
        shares_out_prev = info.get("shares_out_prev")
        flow_usd = derive_flow(shares_out, shares_out_prev, nav)
        rows.append(
            {
                "flow_id": f"{t}-{today_iso}",
                "as_of_date": today_iso,
                "ticker": t,
                "issuer": info.get("issuer"),
                "nav": nav,
                "shares_out": shares_out,
                "shares_out_prev": shares_out_prev,
                "flow_usd": flow_usd,
                "source": info.get("source"),
                "raw_payload": json.dumps(info.get("raw") or {}, default=str, ensure_ascii=True),
            }
        )
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.4 ETF flows ingester (scaffold)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    count = ingest_tickers(dry_run=args.dry_run)
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
