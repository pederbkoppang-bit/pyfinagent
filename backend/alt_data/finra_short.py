"""phase-7.3 FINRA daily short-sale volume ingestion.

Pulls pipe-delimited CDN TXT files per market (FNRAshvol, CNMSshvol, OTCshvol),
parses, normalizes, and persists to `pyfinagent_data.alt_finra_short_volume`.

Compliance:
- `docs/compliance/alt-data.md` row 7.3 specifies the developer API; this cycle
  uses the CDN TXT files pending developer-key provisioning (see phase-7.3
  contract deviation note).
- User-Agent: `pyfinagent/1.0 peder.bkoppang@hotmail.no`
- Rate limit: 8 req/s (shared with 7.2).

Fail-open. ASCII-only.

CLI:
    python -m backend.alt_data.finra_short                        # latest day, all 3 markets
    python -m backend.alt_data.finra_short --market FNRAshvol
    python -m backend.alt_data.finra_short --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_finra_short_volume"
_RATE_INTERVAL_S = 1.0 / 8.0

_CDN_URL_TMPL = "https://cdn.finra.org/equity/regsho/daily/{market}{yyyymmdd}.txt"
_MARKETS: tuple[str, ...] = ("FNRAshvol", "CNMSshvol", "OTCshvol")
_DEFAULT_WALKBACK_DAYS = 5  # search yesterday + 4 prior days for a posted file

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  trade_date DATE NOT NULL,
  symbol STRING NOT NULL,
  market STRING NOT NULL,
  short_volume INT64,
  short_exempt_volume INT64,
  total_volume INT64,
  as_of_date DATE NOT NULL,
  source STRING,
  raw_row STRING
)
PARTITION BY trade_date
CLUSTER BY market, symbol
OPTIONS (
  description = "phase-7.3 FINRA daily short-sale volume from cdn.finra.org/equity/regsho TXT files"
)
""".strip()


def _rate_limit() -> None:
    time.sleep(_RATE_INTERVAL_S)


def _http_get(url: str, *, attempt: int = 0, max_attempts: int = 2) -> bytes | None:
    try:
        import requests  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("finra: requests missing: %r", exc)
        return None
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/plain"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except Exception as exc:
        logger.warning("finra: get fail-open url=%s err=%r", url, exc)
        return None
    if resp.status_code == 404:
        return None  # file for that date not yet posted; callers walk back
    if resp.status_code == 403 and attempt < max_attempts:
        time.sleep(60 * (2**attempt))
        return _http_get(url, attempt=attempt + 1, max_attempts=max_attempts)
    if 500 <= resp.status_code < 600 and attempt < max_attempts:
        time.sleep(5 * (2**attempt))
        return _http_get(url, attempt=attempt + 1, max_attempts=max_attempts)
    if resp.status_code != 200:
        logger.warning("finra: non-200 status=%s url=%s", resp.status_code, url)
        return None
    return resp.content


def fetch_daily(trade_date: date, market: str, *, timeout: int = 30) -> str | None:
    """Fetch one FINRA CDN TXT file for (trade_date, market). Returns text or None."""
    url = _CDN_URL_TMPL.format(market=market, yyyymmdd=trade_date.strftime("%Y%m%d"))
    _rate_limit()
    raw = _http_get(url)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("finra: decode fail-open: %r", exc)
        return None


def parse(text: str) -> list[dict[str, Any]]:
    """Parse pipe-delimited FINRA CDN TXT content to list of row dicts.

    Format:
        Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market   <- header
        20260418|AAPL|12345|0|67890|Q                                  <- data rows
        ...
        20260418|TOTAL RECORDS|N                                       <- trailer (stripped)
    """
    if not text:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        # Filter header + trailer: data rows begin with an 8-digit YYYYMMDD
        # AND have a non-empty ticker symbol in position 1.
        dt_raw = parts[0].strip()
        if not (dt_raw.isdigit() and len(dt_raw) == 8 and dt_raw.startswith("20")):
            continue
        symbol = parts[1].strip().upper()
        if not symbol or symbol in ("TOTAL", "TOTAL RECORDS"):
            continue
        try:
            short_vol = int(parts[2]) if parts[2].strip() else None
        except ValueError:
            short_vol = None
        try:
            short_exempt = int(parts[3]) if parts[3].strip() else None
        except ValueError:
            short_exempt = None
        try:
            total_vol = int(parts[4]) if parts[4].strip() else None
        except ValueError:
            total_vol = None
        row_market = parts[5].strip() if len(parts) >= 6 else ""
        out.append(
            {
                "trade_date_raw": dt_raw,
                "symbol": symbol,
                "short_volume": short_vol,
                "short_exempt_volume": short_exempt,
                "total_volume": total_vol,
                "row_market": row_market,
                "raw_row": line,
            }
        )
    return out


def _fmt_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def normalize(
    rows: Iterable[dict[str, Any]], market: str, *, as_of_date: date | None = None
) -> list[dict[str, Any]]:
    asd = (as_of_date or date.today()).isoformat()
    out: list[dict[str, Any]] = []
    for r in rows or []:
        try:
            td_iso = _fmt_date(r["trade_date_raw"])
        except Exception:
            continue
        out.append(
            {
                "trade_date": td_iso,
                "symbol": r.get("symbol") or "",
                "market": market,
                "short_volume": r.get("short_volume"),
                "short_exempt_volume": r.get("short_exempt_volume"),
                "total_volume": r.get("total_volume"),
                "as_of_date": asd,
                "source": "cdn.finra.org/equity/regsho",
                "raw_row": r.get("raw_row", "")[:1000],
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
            logger.warning("finra: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("finra: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("finra: bigquery.Client() init failed (%r)", exc)
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
        logger.warning("finra: ensure_table fail-open: %r", exc)
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
            logger.warning("finra: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("finra: upsert fail-open: %r", exc)
        return 0


def _latest_available_date(market: str, *, walkback: int = _DEFAULT_WALKBACK_DAYS) -> tuple[date, str] | None:
    """Walk back up to `walkback` days (skipping weekends cheaply) until a file is found."""
    today = date.today()
    tried: list[date] = []
    d = today - timedelta(days=1)
    while len(tried) < walkback:
        # skip Sat (5) / Sun (6)
        if d.weekday() < 5:
            tried.append(d)
            text = fetch_daily(d, market)
            if text:
                return d, text
        d -= timedelta(days=1)
    logger.warning("finra: no file found for market=%s in last %s business days", market, walkback)
    return None


def ingest_recent(
    *,
    markets: Iterable[str] = _MARKETS,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    total = 0
    for market in markets:
        found = _latest_available_date(market)
        if found is None:
            continue
        trade_date, text = found
        parsed = parse(text)
        rows = normalize(parsed, market)
        logger.info("finra: market=%s trade_date=%s rows=%s", market, trade_date.isoformat(), len(rows))
        if dry_run:
            total += len(rows)
            continue
        total += upsert(rows, project=project, dataset=dataset)
    return total


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.3 FINRA short-volume ingester")
    ap.add_argument("--market", default=None, help="restrict to a single market (FNRAshvol|CNMSshvol|OTCshvol)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    table_ready = ensure_table()
    logger.info("finra: ensure_table -> %s", table_ready)
    markets = (args.market,) if args.market else _MARKETS
    count = ingest_recent(markets=markets, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "markets": list(markets),
                "dry_run": args.dry_run,
                "table_ready": table_ready,
                "ingested": count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
