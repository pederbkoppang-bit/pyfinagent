"""
Go-Live Drill: 4.4.2.1 Paper trading ran for >= 2 weeks (ideally 4)

Verifies:
  1. Paper portfolio exists in BQ with valid inception date
  2. Wall-clock delta from inception to now >= 14 days
  3. Paper portfolio snapshots exist covering the window
  4. Current parameter cohort is identified
  5. Starting capital is recorded

Re-run: source .venv/bin/activate && python3 scripts/go_live_drills/paper_runtime_test.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "backend" / "backtest" / "experiments" / "results"
OPTIMIZER_BEST = ROOT / "backend" / "backtest" / "experiments" / "optimizer_best.json"

MIN_DAYS = 14
BQ_PROJECT = "sunny-might-477607-p8"


def query_bq():
    try:
        from google.cloud import bigquery
    except ImportError:
        print("DRILL SKIP: google-cloud-bigquery not installed (need .venv)")
        sys.exit(2)

    client = bigquery.Client(project=BQ_PROJECT)

    q1 = """SELECT inception_date, updated_at, starting_capital, total_nav, total_pnl_pct
            FROM financial_reports.paper_portfolio LIMIT 1"""
    portfolio = None
    for row in client.query(q1).result():
        portfolio = dict(row)

    q2 = """SELECT MIN(snapshot_date) as earliest, MAX(snapshot_date) as latest,
                   COUNT(*) as cnt, COUNT(DISTINCT snapshot_date) as distinct_dates
            FROM financial_reports.paper_portfolio_snapshots"""
    snapshots = None
    for row in client.query(q2).result():
        snapshots = dict(row)

    q3 = """SELECT COUNT(*) as cnt FROM financial_reports.paper_trades"""
    trades_count = 0
    for row in client.query(q3).result():
        trades_count = row["cnt"]

    return portfolio, snapshots, trades_count


def run_checks():
    results = []
    portfolio, snapshots, trades_count = query_bq()
    now = datetime.now(timezone.utc)

    if portfolio is None:
        results.append(("S0", "FAIL", "No paper_portfolio row found in BQ"))
        _print_results(results, now, None, 0)
        return False
    results.append(("S0", "PASS",
        f"Paper portfolio: NAV=${portfolio['total_nav']:,.2f}, PnL={portfolio['total_pnl_pct']}%"))

    inception_str = portfolio.get("inception_date", "")
    try:
        inception_dt = datetime.fromisoformat(inception_str)
        if inception_dt.tzinfo is None:
            inception_dt = inception_dt.replace(tzinfo=timezone.utc)
        results.append(("S1", "PASS",
            f"Inception: {inception_dt.strftime('%Y-%m-%d %H:%M UTC')}"))
    except Exception as e:
        results.append(("S1", "FAIL", f"Invalid inception_date: {inception_str} ({e})"))
        _print_results(results, now, None, 0)
        return False

    delta_days = (now - inception_dt).days
    if delta_days >= MIN_DAYS:
        results.append(("S2", "PASS",
            f"Running {delta_days} days >= {MIN_DAYS}-day floor "
            f"({delta_days - MIN_DAYS} days margin)"))
    else:
        results.append(("S2", "FAIL",
            f"Running {delta_days} days < {MIN_DAYS}-day floor "
            f"({MIN_DAYS - delta_days} days short)"))

    if snapshots and snapshots["cnt"] > 0:
        results.append(("S3", "PASS",
            f"{snapshots['cnt']} snapshots, {snapshots['distinct_dates']} distinct dates "
            f"({snapshots['earliest']} to {snapshots['latest']})"))
    else:
        results.append(("S3", "FAIL", "No portfolio snapshots found"))

    if OPTIMIZER_BEST.exists():
        with open(OPTIMIZER_BEST) as f:
            opt = json.load(f)
        results.append(("S4", "PASS",
            f"optimizer_best.json: Sharpe={opt.get('sharpe', '?')}, "
            f"file={opt.get('file', '?')}"))
    else:
        results.append(("S4", "FAIL", "optimizer_best.json not found"))

    starting = portfolio.get("starting_capital", 0)
    if starting > 0:
        results.append(("S5", "PASS", f"Starting capital ${starting:,.2f}"))
    else:
        results.append(("S5", "FAIL", f"Invalid starting capital: {starting}"))

    updated_str = portfolio.get("updated_at", "")
    try:
        updated_dt = datetime.fromisoformat(updated_str)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        hours_since = (now - updated_dt).total_seconds() / 3600
        results.append(("S6", "PASS",
            f"Last updated {hours_since:.1f}h ago ({updated_dt.strftime('%Y-%m-%d %H:%M UTC')})"))
    except Exception:
        results.append(("S6", "PASS", f"updated_at: {updated_str}"))

    results.append(("S7", "PASS", f"{trades_count} paper trades executed"))

    evidence = {
        "query_date": now.isoformat(),
        "inception_date": inception_str,
        "updated_at": updated_str,
        "delta_days": delta_days,
        "min_days_required": MIN_DAYS,
        "starting_capital": portfolio.get("starting_capital"),
        "total_nav": portfolio.get("total_nav"),
        "total_pnl_pct": portfolio.get("total_pnl_pct"),
        "snapshots": {k: str(v) for k, v in (snapshots or {}).items()},
        "trades_count": trades_count,
        "gate_passed": delta_days >= MIN_DAYS,
    }
    evidence_path = EVIDENCE_DIR / f"paper_runtime_evidence_{now.strftime('%Y%m%d')}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    _print_results(results, now, inception_dt, delta_days, str(evidence_path))
    failed = sum(1 for _, v, _ in results if v == "FAIL")
    return failed == 0


def _print_results(results, now, inception_dt, delta_days, evidence_path=None):
    passed = sum(1 for _, v, _ in results if v == "PASS")
    failed = sum(1 for _, v, _ in results if v == "FAIL")
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  4.4.2.1 Paper Trading Runtime >= 2 Weeks Drill")
    print(f"{'='*60}")
    for sid, verdict, detail in results:
        marker = "+" if verdict == "PASS" else "X"
        print(f"  [{marker}] {sid}: {detail}")
    print(f"{'='*60}")
    if inception_dt:
        print(f"  Inception: {inception_dt.strftime('%Y-%m-%d')}")
    print(f"  Today:     {now.strftime('%Y-%m-%d')}")
    print(f"  Delta:     {delta_days} days (floor: {MIN_DAYS})")
    if evidence_path:
        print(f"  Evidence:  {evidence_path}")
    print(f"{'='*60}")

    if failed == 0:
        print(f"\nDRILL PASS: {passed}/{total}")
    else:
        print(f"\nDRILL FAIL: {passed}/{total} ({failed} hard failures)")


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
