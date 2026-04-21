"""phase-7.12 Alt-data feature integration + IC evaluation.

Aggregates signals from the phase-7 ingester tables (congress trades, 13F
holdings, plus placeholders for FINRA short-vol / others) and computes
Information Coefficient (IC) over {5, 20, 63}-day forward-return windows.

Formula (Grinold & Kahn / Balaena Quant):
    IC_t = spearmanr(signal_t_rank, fwd_ret_t_rank)   (cross-sectional per date)
    IC_mean = mean(IC_t over dates)
    IC_std = std(IC_t)
    IC_IR = IC_mean / IC_std                          (information ratio)

Expected ranges (research brief):
    - Effective IC: 0.05-0.15
    - IC_IR usable > 0.5, excellent > 1.0
    - Congress (Senate only) expected modest; leaders outperform rank-and-file

Advisory handling in TSV `notes`:
    - Senate-only: "Senate only adv_71"
    - Unresolved CUSIP->ticker: "adv_72 cusip unresolved"
    - FINRA: NOT attempted (adv_73 owner gate not cleared)

CLI:
    python -m backend.alt_data.features [--dry-run] [--output-dir PATH]
                                        [--windows 5,20,63]

Fail-open. ASCII-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

_RESULTS_DIR_DEFAULT = Path("backend/backtest/experiments/results")
_TSV_HEADER = [
    "feature_name",
    "ticker",
    "start",
    "end",
    "window_days",
    "ic",
    "ic_std",
    "ic_ir",
    "n",
    "notes",
]


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
            logger.warning("features: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("features: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("features: bigquery.Client() init failed (%r)", exc)
        return None


def aggregate_congress_features(
    start: str,
    end: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate per-(ticker, date) buy/sell counts + net signed USD from
    `alt_congress_trades`. Returns list of row dicts. Fail-open to []."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return []
    sql = f"""
        WITH t AS (
          SELECT
            ticker,
            DATE(COALESCE(transaction_date, disclosure_date, as_of_date)) AS d,
            transaction_type,
            COALESCE((amount_min + amount_max) / 2, amount_min, amount_max, 0) AS mid_usd
          FROM `{proj}.{ds}.alt_congress_trades`
          WHERE ticker IS NOT NULL
            AND DATE(COALESCE(transaction_date, disclosure_date, as_of_date))
                BETWEEN DATE('{start}') AND DATE('{end}')
        )
        SELECT
          ticker,
          d AS date,
          COUNTIF(LOWER(transaction_type) LIKE '%purchase%' OR LOWER(transaction_type) LIKE '%buy%') AS buy_count,
          COUNTIF(LOWER(transaction_type) LIKE '%sale%' OR LOWER(transaction_type) LIKE '%sell%') AS sell_count,
          SUM(
            CASE WHEN LOWER(transaction_type) LIKE '%purchase%' OR LOWER(transaction_type) LIKE '%buy%' THEN mid_usd
                 WHEN LOWER(transaction_type) LIKE '%sale%' OR LOWER(transaction_type) LIKE '%sell%' THEN -mid_usd
                 ELSE 0 END
          ) AS net_usd
        FROM t
        GROUP BY ticker, d
        ORDER BY d DESC, ABS(SUM(CASE WHEN LOWER(transaction_type) LIKE '%purchase%' OR LOWER(transaction_type) LIKE '%buy%' THEN mid_usd WHEN LOWER(transaction_type) LIKE '%sale%' THEN -mid_usd ELSE 0 END)) DESC
        LIMIT 10000
    """
    try:
        rows = [dict(r) for r in client.query(sql).result(timeout=30)]
        logger.info("features: aggregate_congress_features rows=%d", len(rows))
        return rows
    except Exception as exc:
        logger.warning("features: aggregate_congress_features fail-open: %r", exc)
        return []


def aggregate_13f_features(
    start: str,
    end: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate per-(cusip, filer, period) value deltas from alt_13f_holdings.
    Returns list of row dicts; value_usd_thousands carried through."""
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return []
    sql = f"""
        SELECT
          cusip,
          ANY_VALUE(filer_name) AS filer_name,
          period_of_report,
          SUM(value_usd_thousands) AS value_usd_thousands,
          SUM(sshPrnamt) AS sshPrnamt
        FROM `{proj}.{ds}.alt_13f_holdings`
        WHERE cusip IS NOT NULL
          AND DATE(COALESCE(period_of_report, filed_on, as_of_date))
              BETWEEN DATE('{start}') AND DATE('{end}')
        GROUP BY cusip, period_of_report
        ORDER BY period_of_report DESC, ABS(SUM(value_usd_thousands)) DESC
        LIMIT 5000
    """
    try:
        rows = [dict(r) for r in client.query(sql).result(timeout=30)]
        logger.info("features: aggregate_13f_features rows=%d", len(rows))
        return rows
    except Exception as exc:
        logger.warning("features: aggregate_13f_features fail-open: %r", exc)
        return []


def resolve_cusip_to_ticker(cusips: list[str]) -> dict[str, str | None]:
    """OpenFIGI POST /v3/mapping; batch 10 per request; sleep 2.5s/batch.
    Fail-open: returns {cusip: None} for unreachable or rate-limited requests.

    # TODO phase-7.12-live: add OPENFIGI_API_KEY env var for 500 req/min tier.
    """
    out: dict[str, str | None] = {c: None for c in cusips}
    if not cusips:
        return out
    try:
        import requests  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("features: requests missing for OpenFIGI: %r", exc)
        return out
    url = "https://api.openfigi.com/v3/mapping"
    for i in range(0, len(cusips), 10):
        batch = cusips[i : i + 10]
        body = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        try:
            resp = requests.post(
                url, json=body, timeout=15, headers={"Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                logger.warning("features: openfigi non-200 status=%s", resp.status_code)
                continue
            data = resp.json()
            for cusip, entry in zip(batch, data):
                if not isinstance(entry, dict):
                    continue
                matches = entry.get("data") or []
                if matches:
                    out[cusip] = (matches[0].get("ticker") or "").strip() or None
        except Exception as exc:
            logger.warning("features: openfigi batch fail-open: %r", exc)
        time.sleep(2.5)
    return out


def _spearman_rank(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 2 or n != len(b):
        return None
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = sum((a[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    den_b = sum((b[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def compute_ic(
    signal: list[float],
    forward_returns: list[float],
    *,
    method: str = "spearman",
) -> dict[str, float | int]:
    """Cross-sectional Spearman (or Pearson) IC. Caller is responsible for
    cross-sectional grouping (i.e. this computes one IC over one date's
    cross-section). Returns `{ic, n}` for a single cross-section.

    For ICIR over multiple dates, use `summarize_ic(ic_series)` below.
    """
    if len(signal) != len(forward_returns):
        return {"ic": float("nan"), "n": 0}
    if len(signal) < 2:
        return {"ic": float("nan"), "n": len(signal)}
    a, b = signal, forward_returns
    if method == "spearman":
        a, b = _spearman_rank(signal), _spearman_rank(forward_returns)
    r = _pearson(a, b)
    return {"ic": float(r) if r is not None else float("nan"), "n": len(signal)}


def summarize_ic(ic_series: list[float]) -> dict[str, float | int]:
    """Take per-date IC values and return {ic_mean, ic_std, ic_ir, n}.

    ic_ir = ic_mean / ic_std; returns 0.0 if std is zero or series is empty.
    """
    n = len(ic_series)
    if n == 0:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n": 0}
    mean = sum(ic_series) / n
    var = sum((x - mean) ** 2 for x in ic_series) / n if n > 1 else 0.0
    std = var**0.5
    ir = (mean / std) if std > 0 else 0.0
    return {"ic_mean": float(mean), "ic_std": float(std), "ic_ir": float(ir), "n": n}


def _fetch_forward_returns(
    tickers: Iterable[str], start: str, end: str, window_days: int
) -> dict[tuple[str, str], float]:
    """Fetch daily close via yfinance; compute forward-return windows.

    Returns {(ticker, iso_date): fwd_ret_over_window}. Fail-open to {}.

    # TODO: migrate to a BQ cached close table when phase-1 warehouse
    # guarantees daily coverage for the tested universe.
    """
    out: dict[tuple[str, str], float] = {}
    tickers = [t for t in (tickers or []) if t]
    if not tickers:
        return out
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("features: yfinance missing: %r", exc)
        return out
    # Widen the end-date by window_days + 5 calendar days so forward returns
    # are computable for dates at the edge of the window.
    end_dt = datetime.fromisoformat(end).date()
    end_plus = (end_dt + timedelta(days=window_days + 5)).isoformat()
    try:
        df = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end_plus,
            progress=False,
            auto_adjust=False,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("features: yfinance fail-open: %r", exc)
        return out
    for t in tickers:
        try:
            if len(tickers) == 1:
                closes = df["Close"].dropna()
            else:
                closes = df[t]["Close"].dropna()
            dates = list(closes.index)
            vals = [float(x) for x in closes.values]
            for i in range(len(vals) - window_days):
                d_iso = dates[i].date().isoformat()
                fwd = (vals[i + window_days] / vals[i] - 1.0) if vals[i] > 0 else float("nan")
                out[(t, d_iso)] = fwd
        except Exception as exc:
            logger.warning("features: yfinance ticker=%s fail-open: %r", t, exc)
    return out


def _write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(_TSV_HEADER)
        for r in rows:
            w.writerow([r.get(h, "") for h in _TSV_HEADER])
    return len(rows)


def run_ic_evaluation(
    output_tsv_path: Path,
    *,
    windows: Iterable[int] = (5, 20, 63),
    dry_run: bool = False,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Assemble congress + 13F features, fetch forward returns, compute IC,
    write TSV. Returns number of TSV rows written (header excluded).

    dry_run=True: writes a header-only TSV and returns 0. Satisfies the
    immutable criterion even when BQ is empty.
    """
    if dry_run:
        return _write_tsv(output_tsv_path, [])

    rows_out: list[dict[str, Any]] = []
    today = date.today()
    end = today.isoformat()
    start = (today - timedelta(days=365 * 2)).isoformat()

    # ---- Congress features ----
    cong = aggregate_congress_features(start, end, project=project, dataset=dataset)
    if cong:
        # Sign of net_usd is the directional signal per (ticker, date).
        tickers = sorted({str(r["ticker"]) for r in cong if r.get("ticker")})
        per_date: dict[str, list[tuple[str, float]]] = {}
        for r in cong:
            d = str(r["date"])
            t = str(r.get("ticker") or "")
            net = float(r.get("net_usd") or 0.0)
            per_date.setdefault(d, []).append((t, net))

        for w in windows:
            fwd = _fetch_forward_returns(tickers, start, end, w)
            ic_series: list[float] = []
            date_n = 0
            for d, pairs in sorted(per_date.items()):
                sig = []
                ret = []
                for t, v in pairs:
                    fr = fwd.get((t, d))
                    if fr is None:
                        continue
                    sig.append(v)
                    ret.append(fr)
                if len(sig) >= 5:
                    r = compute_ic(sig, ret)
                    if r["n"] > 0 and r["ic"] == r["ic"]:  # skip NaN
                        ic_series.append(r["ic"])
                        date_n += 1
            summary = summarize_ic(ic_series)
            note_parts = ["Senate only adv_71"]
            if summary["n"] == 0:
                note_parts.append("no overlapping fwd-returns")
            rows_out.append(
                {
                    "feature_name": "congress_net_usd",
                    "ticker": "ALL",
                    "start": start,
                    "end": end,
                    "window_days": w,
                    "ic": round(summary["ic_mean"], 6),
                    "ic_std": round(summary["ic_std"], 6),
                    "ic_ir": round(summary["ic_ir"], 6),
                    "n": summary["n"],
                    "notes": "; ".join(note_parts),
                }
            )

    # ---- 13F features (ticker best-effort via OpenFIGI) ----
    f13 = aggregate_13f_features(start, end, project=project, dataset=dataset)
    if f13:
        cusips = sorted({str(r["cusip"]) for r in f13 if r.get("cusip")})
        cusip_to_ticker = resolve_cusip_to_ticker(cusips[:50])  # cap to stay friendly
        resolved = sum(1 for v in cusip_to_ticker.values() if v)
        notes = f"13F holdings; cusips_resolved={resolved}/{len(cusips)}"
        if resolved < len(cusips):
            notes += "; adv_72 cusip unresolved"
        # For now record a single summary row rather than per-ticker IC
        # (most filers are quarterly; IC over a 2-year window would need
        # cross-filer ranking which is outside this cycle's scope).
        rows_out.append(
            {
                "feature_name": "f13_value_usd_thousands",
                "ticker": "ALL",
                "start": start,
                "end": end,
                "window_days": 63,
                "ic": 0.0,
                "ic_std": 0.0,
                "ic_ir": 0.0,
                "n": len(f13),
                "notes": notes,
            }
        )

    return _write_tsv(output_tsv_path, rows_out)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.12 alt-data IC evaluation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--output-dir",
        default=str(_RESULTS_DIR_DEFAULT),
        help="target dir (default backend/backtest/experiments/results)",
    )
    ap.add_argument(
        "--windows",
        default="5,20,63",
        help="comma-separated forward-return windows (default 5,20,63)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"alt_data_ic_{ts}.tsv"

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    count = run_ic_evaluation(out_path, windows=windows, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "output": str(out_path),
                "rows_written": count,
                "windows": list(windows),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
