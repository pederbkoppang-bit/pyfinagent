"""phase-66.2: read-only per-stage BUY-funnel report (criterion-b tooling).

Assembles candidate counts at every countable gate of the daily trading cycle
from EXISTING sources only (no new instrumentation, no writes):

  stage sources
  - cycle spine + rail health : handoff/cycle_history.jsonl
      (cycle_id, status, n_trades, meta_scorer_degraded, rail_skipped,
       breaker_tripped -- the last two exist from phase-66.1 onward)
  - analysis outcomes         : financial_reports.analysis_results
      (per-day recommendation mix + degraded count -- stage 6)
  - rail call health          : pyfinagent_data.llm_call_log
      (agent LIKE 'cc_rail%' ok true/false per day -- stage 6/7)
  - signal emissions          : financial_reports.signals_log
      (BUY/SELL signals; '$CYCLE' HOLD heartbeat = explicit no-trade marker)
  - executed trades           : financial_reports.paper_trades
      (grouped by action+reason -- the ONLY complete trade count;
       cycle_history.n_trades EXCLUDES stop-loss sells and scale-outs)

Stages with NO durable counter (log-only; reported explicitly, never silently
omitted): universe size, per-reason screener drops, analyze-cap truncation,
decide_trades per-gate rejections (research_brief_66.2.md section 1 gap list).

Verdict heuristic per day:
  - "ALL-HOLD COLLAPSE (pipeline defect)": analyses ran but zero non-HOLD recs
    AND rail failures present -> gates never evaluated.
  - "GATES EVALUATED": non-HOLD recs exist; zero trades then means the gates
    (sector cap / risk judge / sizing) rejected -- check decide_trades logs.
  - "NO ANALYSES": cycle didn't reach the analysis stage (weekend/timeout).

Usage: python scripts/diagnostics/funnel_report.py [--start YYYY-MM-DD]
       [--end YYYY-MM-DD]   (defaults: last 7 days, UTC)
Read-only; exits 0.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = "sunny-might-477607-p8"

EU_SUFFIXES = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".LS", ".HE", ".ST", ".OL", ".CO", ".L")


def market_for(ticker: str) -> str:
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "KR"
    if any(ticker.endswith(s) for s in EU_SUFFIXES):
        return "EU"
    return "US"


def load_cycles(start: str, end: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = REPO / "handoff" / "cycle_history.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = (r.get("started_at") or "")[:10]
        if not (start <= d <= end) or r.get("status") == "started":
            continue
        out[d] = {
            "cycle_id": (r.get("cycle_id") or "")[:8],
            "status": r.get("status"),
            "n_trades": r.get("n_trades"),
            "scorer_degraded": r.get("meta_scorer_degraded"),
            "rail_skipped": r.get("rail_skipped"),
            "breaker_tripped": r.get("breaker_tripped"),
            # phase-66.2: persisted per-stage counts (cycles from 07-07 on)
            "funnel": r.get("funnel") or {},
        }
    return out


def bq_maps(start: str, end: str):
    from google.cloud import bigquery

    c = bigquery.Client(project=PROJECT)
    cfg = lambda: bigquery.QueryJobConfig(  # noqa: E731
        query_parameters=[
            bigquery.ScalarQueryParameter("s", "DATE", start),
            bigquery.ScalarQueryParameter("e", "DATE", end),
        ]
    )
    rail = {
        str(r.d): {"rail_ok": r.ok_n, "rail_fail": r.fail_n}
        for r in c.query(
            """SELECT DATE(ts) d, COUNTIF(ok) ok_n, COUNTIF(NOT ok) fail_n
               FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
               WHERE agent LIKE 'cc_rail%' AND DATE(ts) BETWEEN @s AND @e GROUP BY d""",
            job_config=cfg(),
        ).result(timeout=30)
    }
    analyses: dict[str, dict] = defaultdict(lambda: {"n": 0, "non_hold": 0, "degraded": 0, "recs": defaultdict(int), "mkts": defaultdict(int)})
    for r in c.query(
        """SELECT DATE(analysis_date) d, ticker, recommendation,
                  COALESCE(final_score,0) fs, COALESCE(recommendation_confidence,0) rc
           FROM `sunny-might-477607-p8.financial_reports.analysis_results`
           WHERE DATE(analysis_date) BETWEEN @s AND @e""",
        job_config=cfg(),
    ).result(timeout=30):
        a = analyses[str(r.d)]
        a["n"] += 1
        rec = (r.recommendation or "HOLD").upper()
        a["recs"][rec] += 1
        if rec not in ("HOLD", "NEUTRAL"):
            a["non_hold"] += 1
        if r.fs == 0 or r.rc == 0:
            a["degraded"] += 1
        a["mkts"][market_for(r.ticker or "")] += 1
    signals: dict[str, dict] = defaultdict(lambda: {"buy": 0, "sell": 0, "hold_heartbeat": 0})
    for r in c.query(
        """SELECT DATE(signal_date) d, ticker, signal_type, COUNT(*) n
           FROM `sunny-might-477607-p8.financial_reports.signals_log`
           WHERE DATE(signal_date) BETWEEN @s AND @e GROUP BY d, ticker, signal_type""",
        job_config=cfg(),
    ).result(timeout=30):
        s = signals[str(r.d)]
        if (r.ticker or "") == "$CYCLE":
            s["hold_heartbeat"] += r.n
        elif (r.signal_type or "").upper() == "BUY":
            s["buy"] += r.n
        elif (r.signal_type or "").upper() == "SELL":
            s["sell"] += r.n
    trades: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in c.query(
        """SELECT DATE(created_at) d, action, reason, ticker, COUNT(*) n
           FROM `sunny-might-477607-p8.financial_reports.paper_trades`
           WHERE DATE(created_at) BETWEEN @s AND @e GROUP BY d, action, reason, ticker""",
        job_config=cfg(),
    ).result(timeout=30):
        trades[str(r.d)][f"{r.action}:{r.reason}"] += r.n
        trades[str(r.d)][f"mkt:{market_for(r.ticker or '')}"] += r.n
    return rail, analyses, signals, trades


def verdict(cy: dict | None, an: dict | None, rl: dict | None, tr: dict | None) -> str:
    if not an or an["n"] == 0:
        return "NO ANALYSES (cycle absent/timeout/weekend)"
    if an["non_hold"] == 0 and (rl or {}).get("rail_fail", 0) > 0:
        return "ALL-HOLD COLLAPSE (pipeline defect: rail down; gates never evaluated)"
    if an["non_hold"] == 0 and an["degraded"] >= max(1, an["n"] // 2):
        return "ALL-HOLD COLLAPSE (pipeline defect: degraded scoring)"
    if an["non_hold"] > 0 and not any(k.startswith("BUY") for k in (tr or {})):
        return "GATES EVALUATED, zero BUYs (check decide_trades per-gate logs)"
    if an["non_hold"] > 0:
        return "GATES EVALUATED, trades executed"
    return "ALL HOLD (no rail failures observed -- genuine no-signal day)"


def main() -> int:
    ap = argparse.ArgumentParser()
    today = dt.date.today()
    ap.add_argument("--start", default=str(today - dt.timedelta(days=7)))
    ap.add_argument("--end", default=str(today))
    a = ap.parse_args()

    cycles = load_cycles(a.start, a.end)
    rail, analyses, signals, trades = bq_maps(a.start, a.end)
    days = sorted(set(cycles) | set(rail) | set(analyses) | set(signals) | set(trades))

    print(f"# BUY-funnel report {a.start}..{a.end} (read-only; phase-66.2)\n")
    print("| day | cycle | funnel u/s/c/a | rail ok/fail | rail_skip | breaker | analyses (deg) | rec mix | non-HOLD | signals B/S/hb | trades (by reason) | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in days:
        cy, an, rl, sg, tr = cycles.get(d), analyses.get(d), rail.get(d), signals.get(d), trades.get(d)
        recs = " ".join(f"{k}:{v}" for k, v in sorted((an or {}).get("recs", {}).items())) or "-"
        tr_s = " ".join(f"{k}={v}" for k, v in sorted((tr or {}).items()) if not k.startswith("mkt:")) or "-"
        fn = (cy or {}).get("funnel") or {}
        fn_s = (
            f"{fn.get('universe_size', '?')}/{fn.get('screened', '?')}/"
            f"{fn.get('candidates', '?')}/{fn.get('new_to_analyze', '?')}"
            if fn else "-"
        )
        print(
            f"| {d} | {cy['cycle_id'] if cy else '-'} | {fn_s} "
            f"| {rl['rail_ok'] if rl else 0}/{rl['rail_fail'] if rl else 0} "
            f"| {(cy or {}).get('rail_skipped', '-')} | {(cy or {}).get('breaker_tripped', '-')} "
            f"| {an['n'] if an else 0} ({an['degraded'] if an else 0}) | {recs} "
            f"| {an['non_hold'] if an else 0} "
            f"| {sg['buy'] if sg else 0}/{sg['sell'] if sg else 0}/{sg['hold_heartbeat'] if sg else 0} "
            f"| {tr_s} | {verdict(cy, an, rl, tr)} |"
        )
    print(
        "\nStages with NO durable counter (log-only; per research_brief_66.2.md "
        "section 1): universe size, per-reason screener drops, analyze-cap "
        "truncation, decide_trades per-gate rejections. These require "
        "backend.log parsing and are NOT included above -- absence here is a "
        "known instrumentation gap, not evidence of absence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
