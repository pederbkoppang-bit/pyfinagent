"""phase-53.1: no-trade / rebalance buffer band -- ON vs OFF, $0 replay.

Measures the transaction-cost-aware no-trade band (backend/backtest/rebalance_band.py)
against the live momentum baseline on the production S&P-500 universe via the SAME $0
replay machinery as 51.2/52.x (free yfinance prices + Wikipedia GICS; no LLM, no BQ).

Reports per arm: ann Sharpe / avg monthly return / turnover / maxDD, GROSS and
NET-of-cost. Subjects band-vs-baseline to the 52.3 Ledoit-Wolf SR-difference gate
(backend/backtest/analytics.sharpe_diff_test) on TWO pre-registered legs:
  - GROSS  = do-no-harm leg  : require ci_low > -0.05 (band must not significantly HURT gross)
  - NET    = promote   leg  : a-priori rule p_one_sided<0.05 AND delta>=+0.05 AND ci_low>0
A 'not robust' REJECT is a VALID, honestly-reported outcome (phase-53.1 criterion 3).
"""
from __future__ import annotations

import json

import numpy as np

from backend.tools.screener import rank_candidates
from backend.backtest.analytics import sharpe_diff_test
from backend.backtest.rebalance_band import apply_no_trade_band, max_drawdown
# reuse the 51.2/52.x loaders verbatim (single source of truth for the universe + prices)
from scripts.ablation.sector_neutral_replay import (
    load_universe_sectors,
    build_screen_row,
    basket_fwd_return,
    ann_sharpe,
    TOP_N,
    START,
    END,
    log,
)

import pandas as pd
import yfinance as yf

BAND_PCT = 0.2                 # hysteresis width (researcher default; exit rank = top_n*1.2)
ROUND_TRIP_COST = 0.002        # backtest_engine: 2 x transaction_cost_pct(0.1%) = 0.2% round-trip
N_BOOT = 5000                  # match the 52.3 verdict (dsr_52wh_verdict.py)
SEED = 42


def _net(monthly, turnover):
    """Net-of-cost monthly returns: gross - turnover*round_trip. turnover[i] None
    (cold start / no prev) -> no cost that month (both arms identical there)."""
    out = []
    for g, t in zip(monthly, turnover):
        if g is None:
            out.append(None)
        elif t is None:
            out.append(g)
        else:
            out.append(g - t * ROUND_TRIP_COST)
    return out


def main():
    log("[1/4] loading S&P 500 tickers + GICS sectors ...")
    sec_map = load_universe_sectors()
    tickers = list(sec_map.keys())
    log(f"      {len(tickers)} tickers")

    log(f"[2/4] batch-downloading prices {START}..{END} ...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    closes = {}
    for tk in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if tk in raw.columns.get_level_values(0):
                    closes[tk] = raw[tk]["Close"]
            else:
                closes[tk] = raw["Close"]
        except Exception:
            continue
    closes = pd.DataFrame(closes).dropna(how="all")
    log(f"      usable: {closes.shape[1]} tickers x {closes.shape[0]} days")

    idx = closes.index
    rebal = []
    for ym, grp in closes.groupby([idx.year, idx.month]):
        d = grp.index[0]
        if d.year >= 2022:
            rebal.append(d)
    log(f"[3/4] replaying {len(rebal)} monthly rebalances, top_n={TOP_N}, band={BAND_PCT} ...")

    pos = {d: i for i, d in enumerate(closes.index)}
    arms = ["baseline", "band"]
    monthly = {a: [] for a in arms}
    turnover = {a: [] for a in arms}
    prev = {a: set() for a in arms}
    prev_band_ordered: list[str] = []

    for d in rebal:
        t_idx = pos[d]
        win_lo = max(0, t_idx - 260)
        rows = []
        for tk in closes.columns:
            row = build_screen_row(tk, sec_map.get(tk, ""), closes[tk].iloc[win_lo:t_idx + 1])
            if row:
                rows.append(row)
        if len(rows) < 50:
            continue
        ranked_all = rank_candidates(rows, top_n=len(rows), strategy="momentum")
        ranked_tickers = [r["ticker"] for r in ranked_all]

        baseline_basket = ranked_tickers[:TOP_N]
        band_basket = apply_no_trade_band(prev_band_ordered, ranked_tickers, TOP_N, BAND_PCT, enabled=True)
        prev_band_ordered = band_basket

        for name, basket in (("baseline", baseline_basket), ("band", band_basket)):
            fwd = basket_fwd_return(basket, closes, t_idx)
            monthly[name].append(fwd)
            bs = set(basket)
            if prev[name]:
                turnover[name].append(1 - len(bs & prev[name]) / max(len(bs), 1))
            else:
                turnover[name].append(None)  # cold start
            prev[name] = bs

    log("[4/4] results\n")
    net = {a: _net(monthly[a], turnover[a]) for a in arms}
    print(f"{'arm':<10}{'grossSharpe':>13}{'netSharpe':>11}{'avgRet%':>10}{'turnover':>10}{'grossMaxDD':>12}{'netMaxDD':>10}")
    print("-" * 76)
    for a in arms:
        ms = [m for m in monthly[a] if m is not None]
        to = [t for t in turnover[a] if t is not None]
        print(f"{a:<10}{ann_sharpe(monthly[a]):>13.3f}{ann_sharpe(net[a]):>11.3f}"
              f"{(np.mean(ms) * 100 if ms else 0):>10.3f}{(np.mean(to) if to else 0):>10.3f}"
              f"{max_drawdown(monthly[a]):>12.3f}{max_drawdown(net[a]):>10.3f}")

    gross = sharpe_diff_test(monthly["band"], monthly["baseline"], periods_per_year=12, n_boot=N_BOOT, seed=SEED)
    netd = sharpe_diff_test(net["band"], net["baseline"], periods_per_year=12, n_boot=N_BOOT, seed=SEED)

    print("\n--- 53.1 ROBUSTNESS GATE (Ledoit-Wolf SR-difference, n_boot=%d) ---" % N_BOOT)
    print(f"GROSS (do-no-harm leg): dSharpe={gross['delta']:+.3f}  p={gross['p_one_sided']:.3f}  "
          f"CI90=[{gross['ci_low']:+.3f},{gross['ci_high']:+.3f}]")
    do_no_harm_ok = gross["ci_low"] > -0.05
    print(f"   do-no-harm (gross CI_low > -0.05)? {do_no_harm_ok}")
    print(f"NET   (promote   leg): dSharpe={netd['delta']:+.3f}  p={netd['p_one_sided']:.3f}  "
          f"CI90=[{netd['ci_low']:+.3f},{netd['ci_high']:+.3f}]")
    promote = (netd["p_one_sided"] < 0.05) and (netd["delta"] >= 0.05) and (netd["ci_low"] > 0)
    print(f"   promote (a-priori: p<0.05 AND delta>=+0.05 AND CI_low>0)? {promote}")

    verdict = ("PROMOTE (escalate to a live operator-gated enable)" if (promote and do_no_harm_ok)
               else "REJECT (not robust on the net-of-cost a-priori gate) -- honest negative result")
    print(f"\n53.1 RECOMMENDATION: {verdict}")

    dump = {
        "baseline_gross": monthly["baseline"], "band_gross": monthly["band"],
        "baseline_net": net["baseline"], "band_net": net["band"],
        "turnover_baseline": turnover["baseline"], "turnover_band": turnover["band"],
        "gross_srdiff": gross, "net_srdiff": netd,
        "do_no_harm_ok": do_no_harm_ok, "promote": promote, "verdict": verdict,
        "params": {"top_n": TOP_N, "band_pct": BAND_PCT, "round_trip_cost": ROUND_TRIP_COST,
                   "n_boot": N_BOOT, "seed": SEED, "start": START, "end": END},
        "n_rebalances": len([m for m in monthly["baseline"] if m is not None]),
    }
    out = "handoff/current/_53_1_band_paired_returns.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    print(f"\nphase-53.1: paired returns + verdict dumped -> {out}")


if __name__ == "__main__":
    main()
