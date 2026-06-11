"""phase-60.2 (AW-5) criterion-4: $0 decision-replay event study, ON vs OFF.

Replays every recorded swap decision in the window through the REAL
production function (`portfolio_manager._compute_swap_candidates`) twice --
flag OFF (fidelity arm: must reproduce the recorded order) and flag ON
(counterfactual arm: suppressed or surviving, with the reason). Then builds
a counterfactual ledger for the suppressed pairs and reports
Sharpe/return/turnover/maxDD for both arms.

Method (57.1 event-study precedent; researcher brief section 7):
- Decision-level replay, NOT a full multi-cycle simulation: each recorded
  swap pair (SELL reason='swap_for_higher_conviction' + same-cycle BUY
  reason='swap_buy') is re-decided from its reconstructed inputs (the sold
  holding + the bought candidate + that day's analyses from
  financial_reports.analysis_results).
- Counterfactual NAV (one-step, disclosed): for each ON-suppressed pair,
  remove the realized post-swap path of the bought ticker (its in-window
  realized P&L; open remainder marked at the window-end close) and add the
  hold-through P&L of the kept holding (sell price -> window-end close).
  Cascade effects beyond the pair (the bought ticker later displacing
  something else) are NOT re-simulated -- disclosed limitation per Balch
  et al. (replay diverges from interactive simulation).
- Sharpe delta via backend.backtest.analytics.sharpe_diff_test on the daily
  NAV-return series of both arms; T is tiny (~9 trading days) -- the test is
  UNDERPOWERED and reported for completeness, not as a promotion gate
  (criterion 4 requires measurement + operator decision, not an LW pass).

Usage:
  source .venv/bin/activate
  python scripts/replay/replay_60_2_swap_fix.py [--start 2026-05-29] [--end 2026-06-10]

Cost: $0 LLM (BQ reads + yfinance closes only). Output: stdout + markdown at
handoff/current/replay_60_2_results.md.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from backend.config.settings import get_settings  # noqa: E402
from backend.services.portfolio_manager import _compute_swap_candidates  # noqa: E402


def _bq_client():
    from google.cloud import bigquery

    s = get_settings()
    return bigquery.Client(project=s.gcp_project_id), s


def fetch_window(client, settings, start: str, end: str):
    from google.cloud import bigquery

    pt = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}.paper_trades"
    ar = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}.analysis_results"
    snap = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}.paper_portfolio_snapshots"
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("start", "STRING", start),
        bigquery.ScalarQueryParameter("end", "STRING", end),
    ])
    trades = [dict(r) for r in client.query(
        f"""SELECT ticker, action, quantity, price, total_value, reason,
                   created_at, round_trip_id, holding_days, realized_pnl_pct
            FROM `{pt}`
            WHERE SUBSTR(created_at, 1, 10) BETWEEN @start AND @end
            ORDER BY created_at""", job_config=cfg).result()]
    analyses = [dict(r) for r in client.query(
        f"""SELECT ticker, DATE(analysis_date) AS day, final_score, sector,
                   analysis_date
            FROM `{ar}`
            WHERE DATE(analysis_date) BETWEEN @start AND @end
            ORDER BY analysis_date""", job_config=cfg).result()]
    snaps = [dict(r) for r in client.query(
        f"""SELECT snapshot_date, total_nav FROM `{snap}`
            WHERE SUBSTR(CAST(snapshot_date AS STRING), 1, 10) BETWEEN @start AND @end
            ORDER BY snapshot_date""", job_config=cfg).result()]
    return trades, analyses, snaps


_YF_CACHE: dict[str, object] = {}


def yf_pct_move(ticker: str, from_day: str, to_day: str) -> float | None:
    """Close-to-close percentage move (local currency -- currency-neutral, so
    KRW closes never mix with USD trade notionals; FX drift over a days-scale
    window is ignored and disclosed)."""
    try:
        import yfinance as yf

        if ticker not in _YF_CACHE:
            _YF_CACHE[ticker] = yf.Ticker(ticker).history(period="3mo")
        h = _YF_CACHE[ticker]
        if h is None or not len(h):
            return None
        closes = h["Close"]
        idx = closes.index.strftime("%Y-%m-%d")
        c_from = c_to = None
        for d, v in zip(idx, closes.values):
            if d <= from_day:
                c_from = float(v)
            if d <= to_day:
                c_to = float(v)
        if c_from and c_to:
            return c_to / c_from - 1.0
    except Exception:
        pass
    return None


def replay(start: str, end: str) -> str:
    client, settings = _bq_client()
    trades, analyses, snaps = fetch_window(client, settings, start, end)

    # Index analyses: (day, ticker) -> latest row that day
    by_day_ticker: dict[tuple[str, str], dict] = {}
    for a in analyses:
        by_day_ticker[(str(a["day"]), a["ticker"])] = a

    # Pair recorded swaps per day: SELL swap_for_higher_conviction + BUY swap_buy
    swaps: list[dict] = []
    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_day[str(t["created_at"])[:10]].append(t)
    for day, day_trades in sorted(by_day.items()):
        sells = [t for t in day_trades if t["action"] == "SELL" and t["reason"] == "swap_for_higher_conviction"]
        buys = [t for t in day_trades if t["action"] == "BUY" and t["reason"] == "swap_buy"]
        for i, s in enumerate(sells):
            b = buys[i] if i < len(buys) else None
            swaps.append({"day": day, "sell": s, "buy": b})

    settings_off = get_settings().model_copy(update={"paper_swap_churn_fix_enabled": False})
    settings_on = get_settings().model_copy(update={"paper_swap_churn_fix_enabled": True})

    rows = []
    suppressed_pairs = []
    fidelity_fail = 0
    for sw in swaps:
        s, b, day = sw["sell"], sw["buy"], sw["day"]
        sold, bought = s["ticker"], (b or {}).get("ticker")
        sold_a = by_day_ticker.get((day, sold))
        bought_a = by_day_ticker.get((day, bought)) if bought else None
        sector = (bought_a or {}).get("sector") or (sold_a or {}).get("sector") or "Technology"
        holding_mv = float(s["quantity"] or 0) * float(s["price"] or 0)
        pos = [{"ticker": sold, "sector": sector, "market_value": holding_mv,
                "current_price": float(s["price"] or 0)}]
        lookup = {sold: {"final_score": float(sold_a["final_score"])}} if sold_a else {}
        cand = [{"ticker": bought or "?", "sector": sector,
                 "final_score": float((bought_a or {}).get("final_score") or 0.0)}]

        def _arm(st):
            return _compute_swap_candidates(
                sector_blocked=cand, current_positions=pos, holding_lookup=lookup,
                sector_counts={sector: 1}, sector_market_values={sector: holding_mv},
                selling_tickers=set(), settings=st, nav=holding_mv * 10,
            )

        off_orders = _arm(settings_off)
        on_orders = _arm(settings_on)
        off_fired = any(o.action == "SELL" and o.ticker == sold for o in off_orders)
        on_fired = any(o.action == "SELL" and o.ticker == sold for o in on_orders)
        if not off_fired:
            fidelity_fail += 1
        if sold_a is None:
            why = "sentinel: sold ticker had NO same-day analysis (fabricated 0.0)"
        else:
            hs, cs = float(sold_a["final_score"]), float((bought_a or {}).get("final_score") or 0)
            why = f"true scores {cs:.1f} vs {hs:.1f} -> delta {((cs - hs) / max(abs(hs), 1.0)) * 100:.0f}%"
        rows.append({
            "day": day, "sold": sold, "bought": bought, "off": off_fired,
            "on": on_fired, "why": why,
            "sold_analyzed_same_day": sold_a is not None,
        })
        if off_fired and not on_fired:
            suppressed_pairs.append(sw)

    # Counterfactual ledger (one-step, disclosed)
    adjustments = []  # (day, usd_delta description)
    cf_delta_usd = 0.0
    suppressed_turnover = 0.0
    for sw in suppressed_pairs:
        s, b = sw["sell"], sw["buy"]
        sold = s["ticker"]
        sold_usd_notional = abs(float(s["total_value"] or 0))
        # Currency-neutral: pct move (local closes) x USD notional at the
        # suppressed sell -- KRW closes never mix with USD trade prices
        # (the away week's own AW-9 lesson).
        move = yf_pct_move(sold, sw["day"], end)
        hold_through = (move or 0.0) * sold_usd_notional
        suppressed_turnover += sold_usd_notional
        bought_pnl = 0.0
        if b:
            b_usd_notional = abs(float(b["total_value"] or 0))
            suppressed_turnover += b_usd_notional
            bt = b["ticker"]
            later_sells = [t for t in trades if t["ticker"] == bt and t["action"] == "SELL"
                           and str(t["created_at"]) > str(b["created_at"])]
            if later_sells:
                ls = later_sells[0]
                # realized_pnl_pct is recorded on the SELL row (post-56.1 USD-true)
                rp = ls.get("realized_pnl_pct")
                if rp is not None:
                    bought_pnl = (float(rp) / 100.0) * b_usd_notional
                else:
                    bmove = yf_pct_move(bt, sw["day"], str(ls["created_at"])[:10])
                    bought_pnl = (bmove or 0.0) * b_usd_notional
                suppressed_turnover += abs(float(ls["total_value"] or 0))
            else:
                bmove = yf_pct_move(bt, sw["day"], end)
                bought_pnl = (bmove or 0.0) * b_usd_notional
        delta = hold_through - bought_pnl
        cf_delta_usd += delta
        adjustments.append(
            f"{sw['day']} SELL {sold} suppressed: hold-through {hold_through:+.2f} "
            f"({(move or 0) * 100:+.2f}% on {sold_usd_notional:,.0f} USD) minus "
            f"bought-leg({(b or {}).get('ticker')}) {bought_pnl:+.2f} = {delta:+.2f} USD"
        )

    # NAV series + metrics
    nav = [(str(r["snapshot_date"])[:10], float(r["total_nav"])) for r in snaps]
    nav_off = [v for _, v in nav]
    # ON arm: apply the total counterfactual delta linearly from the first
    # suppression day (one-step approximation, disclosed).
    first_supp = min((sw["day"] for sw in suppressed_pairs), default=None)
    nav_on = []
    for d, v in nav:
        nav_on.append(v + (cf_delta_usd if (first_supp and d >= first_supp) else 0.0))

    def _metrics(series):
        if len(series) < 3:
            return {}
        r = np.diff(series) / np.array(series[:-1])
        sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else 0.0
        ret = (series[-1] / series[0] - 1) * 100
        peak = np.maximum.accumulate(series)
        maxdd = float(np.max((peak - series) / peak)) * 100
        return {"sharpe_ann": round(sharpe, 2), "return_pct": round(ret, 2), "maxdd_pct": round(maxdd, 2)}

    m_off, m_on = _metrics(nav_off), _metrics(nav_on)
    total_turnover = sum(abs(float(t["total_value"] or 0)) for t in trades)
    rt_window = [t for t in trades if t["action"] == "SELL" and (t["holding_days"] or 99) <= 1]

    lw_note = "sharpe_diff_test not run: T<10 daily points makes any p-value decorative; deltas reported raw (criterion 4 demands measurement + operator decision, not an LW gate)."
    try:
        from backend.backtest.analytics import sharpe_diff_test
        r_off = np.diff(nav_off) / np.array(nav_off[:-1])
        r_on = np.diff(nav_on) / np.array(nav_on[:-1])
        if len(r_off) >= 5:
            lw = sharpe_diff_test(r_on, r_off)
            lw_note = f"sharpe_diff_test (UNDERPOWERED, T={len(r_off)}): {lw}"
    except Exception as exc:
        lw_note = f"sharpe_diff_test unavailable/failed ({exc}); raw deltas reported."

    named = {"MU": "06-08->06-09", "SNDK": "06-08->06-09", "DELL": "06-05->06-08->06-09"}
    named_rows = []
    for tkr, span in named.items():
        verdicts = [r for r in rows if r["sold"] == tkr]
        if verdicts:
            for v in verdicts:
                named_rows.append(f"| {tkr} {span} | {'fired' if v['off'] else 'NOT REPRODUCED'} | "
                                  f"{'SURVIVES' if v['on'] else 'SUPPRESSED'} | {v['why']} |")
        else:
            named_rows.append(f"| {tkr} {span} | no swap-SELL recorded in window for this ticker | n/a | "
                              f"(round trip may have closed via stop/signal, not swap) |")

    out = []
    out.append(f"# replay_60_2 results -- window {start}..{end} (generated {datetime.now(timezone.utc).isoformat()[:19]}Z)\n")
    out.append(f"Recorded trades: {len(trades)}; recorded swap pairs: {len(swaps)}; "
               f"window round-trip SELLs (holding_days<=1): {len(rt_window)}\n")
    out.append(f"## ARM A fidelity (flag OFF must reproduce recorded swaps): "
               f"{len(swaps) - fidelity_fail}/{len(swaps)} reproduced"
               + (" -- FAILURES present, see table" if fidelity_fail else " -- PASS") + "\n")
    out.append("## Per-swap replay\n")
    out.append("| day | sold | bought | OFF fired | ON verdict | basis |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        out.append(f"| {r['day']} | {r['sold']} | {r['bought']} | {r['off']} | "
                   f"{'SURVIVES' if r['on'] else 'SUPPRESSED'} | {r['why']} |")
    out.append("\n## The 3 named away-week round trips (criterion 4)\n")
    out.append("| round trip | OFF (recorded) | ON | basis |")
    out.append("|---|---|---|---|")
    out.extend(named_rows)
    out.append("\n## Counterfactual ledger (one-step, suppressed pairs)\n")
    out.extend(f"- {a}" for a in adjustments) if adjustments else out.append("- (none suppressed)")
    out.append(f"\nNet counterfactual P&L delta (ON minus OFF): {cf_delta_usd:+.2f} USD")
    out.append(f"Suppressed turnover: {suppressed_turnover:,.2f} USD of {total_turnover:,.2f} USD recorded "
               f"({(suppressed_turnover / total_turnover * 100) if total_turnover else 0:.1f}%)\n")
    out.append("## Metrics (window, daily snapshots)\n")
    out.append(f"| arm | Sharpe(ann) | return % | maxDD % |")
    out.append(f"|---|---|---|---|")
    out.append(f"| OFF (recorded) | {m_off.get('sharpe_ann')} | {m_off.get('return_pct')} | {m_off.get('maxdd_pct')} |")
    out.append(f"| ON (counterfactual) | {m_on.get('sharpe_ann')} | {m_on.get('return_pct')} | {m_on.get('maxdd_pct')} |")
    out.append(f"\n{lw_note}\n")
    out.append("## Disclosed limitations\n"
               "- Decision-level replay through the production `_compute_swap_candidates`; "
               "full multi-cycle path dependence (a suppressed swap changing later candidate streams) "
               "is NOT re-simulated (Balch et al.).\n"
               "- Counterfactual NAV applies the net pair delta from the first suppression day; "
               "intraday path within the window is approximate.\n"
               "- T is ~9 trading days: all risk-adjusted numbers are descriptive, not inferential.")
    report = "\n".join(out)
    with open("handoff/current/replay_60_2_results.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-05-29")
    p.add_argument("--end", default="2026-06-10")
    a = p.parse_args()
    print(replay(a.start, a.end))
