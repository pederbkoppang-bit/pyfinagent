"""phase-52.4: residual / idiosyncratic momentum (Blitz-Huij-Martens 2011) -- $0 replay + robustness gate.

Strips market beta (single-factor OLS over a W=504d window) and ranks by the 12-1 std-normalized
residual sum -- a structurally-DIFFERENT signal from the (rejected) total-return tweaks. Compares
baseline momentum vs resid_mom over ~monthly rebalances on the S&P 500, then applies the SAME
Ledoit-Wolf SR-difference gate as 52.3 (a-priori rule). NO live change. $0 (free yfinance, no LLM).

Honest prior (per the 52.4 research): modern-regime decay + long-only (no short leg) + large-cap
(low idiosyncratic content) all haircut the documented ~2x edge -> likely REJECT on this book.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf

from backend.tools.screener import rank_candidates
from backend.backtest.analytics import sharpe_diff_test

# reuse the 51.2 replay helpers (scripts/ not a package -> load by path)
_RP = pathlib.Path(__file__).resolve().parent / "sector_neutral_replay.py"
_spec = importlib.util.spec_from_file_location("snr_for_residmom", _RP)
snr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snr)

TOP_N = 10
START = "2019-01-01"   # 504d lookback satisfied from the first 2021 rebalance
END = "2025-12-31"
W = 504                # OLS beta-estimation window (~2yr; 36mo not load-bearing per the gate)
FORM = 252             # formation = trailing 12mo of residuals
SKIP = 21              # skip the most recent ~1mo (12-1)


def log(m):
    print(m, flush=True)


def resid_mom_signal(s_ret: np.ndarray, m_ret: np.ndarray, form: int = FORM, skip: int = SKIP):
    """Single-factor residual momentum. s_ret/m_ret = ALIGNED daily returns over the regression
    window (most recent last). OLS r=alpha+beta*m -> residuals; iMOM = sum(formation residuals) /
    std(formation residuals), formation = the 12-1 window (residuals from t-(form+skip) to t-skip)."""
    n = len(s_ret)
    if n < form + skip + 10:
        return None
    m_mean = m_ret.mean()
    var_m = float(((m_ret - m_mean) ** 2).mean())
    if var_m <= 0:
        return None
    beta = float(((s_ret - s_ret.mean()) * (m_ret - m_mean)).mean() / var_m)
    alpha = float(s_ret.mean() - beta * m_mean)
    eps = s_ret - alpha - beta * m_ret
    form_eps = eps[-(form + skip):-skip] if skip > 0 else eps[-form:]
    if len(form_eps) < 60:
        return None
    sd = float(form_eps.std(ddof=0))
    return float(form_eps.sum() / sd) if sd > 0 else None


def main():
    log(f"[1/4] universe + sectors ...")
    sec = snr.load_universe_sectors()
    tickers = list(sec.keys())
    log(f"      {len(tickers)} tickers")
    log(f"[2/4] batch-downloading {START}..{END} (one yfinance call) ...")
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
    mkt = closes.pct_change().mean(axis=1)   # equal-weight market daily return (the single factor)

    idx = closes.index
    pos = {d: i for i, d in enumerate(idx)}
    rebal = [grp.index[0] for _, grp in closes.groupby([idx.year, idx.month]) if grp.index[0].year >= 2021]
    log(f"[3/4] replaying {len(rebal)} monthly rebalances (W={W}d, 12-1 formation), top_n={TOP_N} ...")

    monthly = {"baseline": [], "resid_mom": []}
    prev = {"baseline": set(), "resid_mom": set()}
    turnover = {"baseline": [], "resid_mom": []}
    n_scored = 0
    for d in rebal:
        t = pos[d]
        if t < W + 10:
            continue
        # baseline: production momentum composite (260d feature window)
        rows = []
        for tk in closes.columns:
            r = snr.build_screen_row(tk, sec.get(tk, ""), closes[tk].iloc[max(0, t - 260):t + 1])
            if r:
                rows.append(r)
        if len(rows) < 50:
            continue
        b_basket = [x["ticker"] for x in rank_candidates(rows, top_n=TOP_N, strategy="momentum")]
        # resid_mom: single-factor residual momentum
        win_lo = max(0, t - W)
        m_w = mkt.iloc[win_lo:t + 1]
        sigs = {}
        for tk in closes.columns:
            s_w = closes[tk].iloc[win_lo:t + 1].pct_change()
            df = pd.concat([s_w, m_w], axis=1).dropna()
            if len(df) < int(W * 0.8):
                continue
            sig = resid_mom_signal(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
            if sig is not None and np.isfinite(sig):
                sigs[tk] = sig
        if len(sigs) < 50:
            continue
        rm_basket = sorted(sigs, key=lambda k: sigs[k], reverse=True)[:TOP_N]

        for name, basket in (("baseline", b_basket), ("resid_mom", rm_basket)):
            fwd = snr.basket_fwd_return(basket, closes, t)
            monthly[name].append(fwd)
            bs = set(basket)
            if prev[name]:
                turnover[name].append(1 - len(bs & prev[name]) / max(len(bs), 1))
            prev[name] = bs
        n_scored += 1

    log("[4/4] results\n")
    print(f"{'config':<14}{'ann_Sharpe':>12}{'avg_fwd_mo%':>14}{'avg_turnover':>14}")
    print("-" * 54)
    for name in ["baseline", "resid_mom"]:
        ms = [m for m in monthly[name] if m is not None]
        sh = snr.ann_sharpe(monthly[name])
        ar = float(np.mean(ms) * 100) if ms else 0.0
        to = float(np.mean(turnover[name])) if turnover[name] else 0.0
        print(f"{name:<14}{sh:>12.3f}{ar:>14.3f}{to:>14.3f}")

    # robustness gate (SAME as 52.3): Ledoit-Wolf SR-difference + the a-priori rule
    r = sharpe_diff_test(monthly["resid_mom"], monthly["baseline"], periods_per_year=12, n_boot=5000, block=4, seed=42, ci=0.90)
    R1 = r["p_one_sided"] < 0.05
    R2 = (r["delta"] >= 0.05) and (r["ci_low"] > 0)
    promote = R1 and R2
    print("\n--- ROBUSTNESS GATE (52.3 Ledoit-Wolf SR-difference; a-priori rule) ---")
    print(f"SR_resid_mom={r['sr_a']:.3f}  SR_base={r['sr_b']:.3f}  delta={r['delta']:+.3f}  (n={r['n']}, n_boot={r['n_boot']})")
    print(f"one-sided p (H0 SR_resid<=SR_base) = {r['p_one_sided']:.4f}  -> R1 (p<0.05): {R1}")
    print(f"90% CI for delta = [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  (se={r['se']:.3f})  -> R2 (delta>=+0.05 AND CI_low>0): {R2}")
    print(f"\nVERDICT: {'PROMOTE -- residual momentum survives the gate (new highest earner)' if promote else 'REJECT -- not statistically distinguishable from baseline; cited-alpha-signal search EXHAUSTED. The +20% momentum engine stands.'}")

    with open("handoff/current/_residmom_paired_returns.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": monthly["baseline"], "resid_mom": monthly["resid_mom"],
                   "n_rebalances": n_scored, "verdict": "PROMOTE" if promote else "REJECT",
                   "p_one_sided": r["p_one_sided"], "delta": r["delta"],
                   "ci": [r["ci_low"], r["ci_high"]]}, f, indent=2)
    print("phase-52.4: paired returns dumped -> handoff/current/_residmom_paired_returns.json")


if __name__ == "__main__":
    main()
