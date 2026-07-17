"""phase-51.2: screener-level replay -- sector-neutral (and vol-scaled) ON vs OFF.

WHY a new harness: the ML backtest engine ranks via candidate_selector (a DIFFERENT
formula, no sector field), so it cannot measure the LIVE screener's sector-neutral
lever. This replays the PRODUCTION `screener.rank_candidates` over historical monthly
dates on the S&P 500, isolating the sector-neutral delta (both configs share identical
screen_data -> the only difference is the within-sector percentile regroup).

$0: free yfinance prices (one batch download) + Wikipedia GICS sectors (one read_html).
No LLM, no BQ writes, no live change. Measures the SIGN of the tradeoff on OUR universe
before any live enable (the Harvey et al. long-only caveat: neutralizing may HURT).

Output: per-config annualized Sharpe of the monthly equal-weight top-N basket, average
sector spread (# distinct GICS), average turnover. Verdict per masterplan 51.2.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import yfinance as yf

from backend.tools.screener import rank_candidates

TOP_N = 10
START = "2021-06-01"   # leaves 6mo lookback before the first 2022 rebalance
END = "2025-12-31"
TARGET_ANN_VOL = 0.15  # vol-scaling target (Barroso-Santa-Clara)
VOL_CAP = 2.0          # max leverage


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
def load_universe_sectors():
    """S&P 500 tickers + GICS sector from the Wikipedia table (same source +
    User-Agent as the production `get_sp500_tickers`; the default urllib UA gets
    a 403). Returns {yf_symbol: sector}."""
    import io
    import urllib.request
    from backend.tools.screener import SP500_URL
    req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0 pyfinagent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
    df = pd.read_html(io.StringIO(html), header=0)[0]
    out = {}
    for _, r in df.iterrows():
        sym = str(r["Symbol"]).strip().replace(".", "-")  # BRK.B -> BRK-B for yfinance
        out[sym] = str(r["GICS Sector"]).strip()
    return out


def rsi_14(close: pd.Series) -> float:
    d = close.diff().dropna()
    if len(d) < 15:
        return 50.0
    up = d.clip(lower=0).rolling(14).mean().iloc[-1]
    dn = (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
    if dn == 0:
        return 100.0
    rs = up / dn
    return float(100 - 100 / (1 + rs))


def build_screen_row(ticker, sector, win: pd.Series):
    """win = close prices up to AND INCLUDING the rebalance date (causal)."""
    if win is None or len(win) < 130:
        return None
    c = win.dropna()
    if len(c) < 130:
        return None
    last = c.iloc[-1]
    if not np.isfinite(last) or last <= 0:
        return None
    def mom(n):
        if len(c) <= n:
            return None
        base = c.iloc[-1 - n]
        return float(last / base - 1) if base > 0 else None
    daily = c.pct_change().dropna()
    vol_ann = float(daily.iloc[-63:].std() * np.sqrt(252)) if len(daily) >= 63 else 0.3
    sma50 = c.iloc[-50:].mean()
    # phase-52.1: 52-week-high proximity (George-Hwang 2004) -- price-only, in (0,1].
    high_52w = float(c.rolling(252, min_periods=20).max().iloc[-1])
    pct_to_52w = float(last / high_52w) if high_52w > 0 else None
    return {
        "ticker": ticker,
        "sector": sector,
        "momentum_1m": mom(21),
        "momentum_3m": mom(63),
        "momentum_6m": mom(126),
        "rsi_14": rsi_14(c),
        "volatility_ann": vol_ann,
        "sma_50_distance_pct": float(last / sma50 - 1) if sma50 > 0 else 0.0,
        "pct_to_52w_high": pct_to_52w,
    }


def basket_fwd_return(basket, closes: pd.DataFrame, t_idx: int, horizon: int = 21):
    """Equal-weight realized forward return of the basket over `horizon` trading days."""
    rets = []
    for tk in basket:
        if tk not in closes.columns:
            continue
        s = closes[tk]
        if t_idx + horizon >= len(s):
            continue
        p0, p1 = s.iloc[t_idx], s.iloc[t_idx + horizon]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            rets.append(p1 / p0 - 1)
    return float(np.mean(rets)) if rets else None


def ann_sharpe(monthly: list[float]) -> float:
    a = np.array([m for m in monthly if m is not None])
    if len(a) < 3 or a.std() == 0:
        return 0.0
    return float(a.mean() / a.std() * np.sqrt(12))


def hi52_tilt_basket(ranked_all, k, top_n, mean_pct=None):
    """phase-52.1: re-rank rows (already scored by the PRODUCTION rank_candidates
    composite) by a CENTERED 52-week-high multiplicative tilt -- tilt UP names nearer
    their 52w high, DOWN names far below it, centered on the universe mean so the
    average tilt ~= 1.0 (turnover-neutral on average). Reuses the production composite
    verbatim (this is replay-side post-processing -> zero live-engine change).
    Returns the top_n tickers."""
    pcts = [r.get("pct_to_52w_high") for r in ranked_all if r.get("pct_to_52w_high") is not None]
    mp = mean_pct if mean_pct is not None else (float(np.mean(pcts)) if pcts else 1.0)
    scored = []
    for r in ranked_all:
        p = r.get("pct_to_52w_high")
        tilt = (1 + k * (p - mp)) if p is not None else 1.0
        scored.append((r["ticker"], (r.get("composite_score") or 0.0) * tilt))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def main():
    log("[1/4] loading S&P 500 tickers + GICS sectors from Wikipedia ...")
    sec_map = load_universe_sectors()
    tickers = list(sec_map.keys())
    log(f"      {len(tickers)} tickers; sectors: {len(set(sec_map.values()))} distinct")

    log(f"[2/4] batch-downloading prices {START}..{END} (one yfinance call) ...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    # flatten to a close-only frame {ticker: close series}
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
    log(f"      usable price series: {closes.shape[1]} tickers x {closes.shape[0]} days")

    # monthly rebalance dates = first trading day of each month in 2022-2025
    idx = closes.index
    rebal = []
    for ym, grp in closes.groupby([idx.year, idx.month]):
        d = grp.index[0]
        if d.year >= 2022:
            rebal.append(d)
    log(f"[3/4] replaying {len(rebal)} monthly rebalances, top_n={TOP_N} ...")

    configs = {"baseline": False, "sector_neutral": True}
    # phase-52.1: 52-week-high multiplicative tilt configs (k = tilt strength).
    tilt_configs = {"hi52_k0.5": 0.5, "hi52_k1.0": 1.0}
    # phase-70.2: SOFT cross-sector diversity penalty configs (w_d rank-decay strength).
    # Same PRODUCTION rank_candidates path, soft_sector_diversity=True. w=0 would be the
    # baseline, so the grid probes the OOS Sharpe / breadth trade for the activation gate.
    soft_configs = {"soft_w0.10": 0.10, "soft_w0.20": 0.20, "soft_w0.30": 0.30}
    _all = list(configs) + ["vol_scaled"] + list(tilt_configs) + list(soft_configs)
    monthly = {k: [] for k in _all}
    spread = {k: [] for k in _all}
    prev_basket = {k: set() for k in _all}
    turnover = {k: [] for k in _all}

    pos = {d: i for i, d in enumerate(closes.index)}
    for d in rebal:
        t_idx = pos[d]
        win_lo = max(0, t_idx - 260)   # phase-52.1: >=252 days for the 52-week high
        rows = []
        for tk in closes.columns:
            row = build_screen_row(tk, sec_map.get(tk, ""), closes[tk].iloc[win_lo:t_idx + 1])
            if row:
                rows.append(row)
        if len(rows) < 50:
            continue
        for name, sn in configs.items():
            ranked = rank_candidates(rows, top_n=TOP_N, strategy="momentum", sector_neutral=sn)
            basket = [r["ticker"] for r in ranked]
            fwd = basket_fwd_return(basket, closes, t_idx)
            monthly[name].append(fwd)
            spread[name].append(len({sec_map.get(t, "") for t in basket}))
            bs = set(basket)
            if prev_basket[name]:
                turnover[name].append(1 - len(bs & prev_basket[name]) / max(len(bs), 1))
            prev_basket[name] = bs
            if name == "baseline" and fwd is not None:
                # vol-target the baseline basket return by trailing realized vol
                daily_basket = []
                for tk in basket:
                    s = closes[tk].iloc[max(0, t_idx - 63):t_idx + 1].pct_change().dropna()
                    if len(s) >= 20:
                        daily_basket.append(s)
                if daily_basket:
                    rv = float(pd.concat(daily_basket, axis=1).mean(axis=1).std() * np.sqrt(252))
                    lev = min(VOL_CAP, TARGET_ANN_VOL / rv) if rv > 0 else 1.0
                    monthly["vol_scaled"].append(fwd * lev)
                else:
                    monthly["vol_scaled"].append(fwd)

        # phase-52.1: 52-week-high multiplicative tilt -- post-process the PRODUCTION
        # composite (centered so avg tilt ~= 1.0; tilt UP names nearer their 52w high).
        ranked_all = rank_candidates(rows, top_n=len(rows), strategy="momentum")
        for tname, k in tilt_configs.items():
            basket = hi52_tilt_basket(ranked_all, k, TOP_N)
            fwd = basket_fwd_return(basket, closes, t_idx)
            monthly[tname].append(fwd)
            spread[tname].append(len({sec_map.get(t, "") for t in basket}))
            bs = set(basket)
            if prev_basket[tname]:
                turnover[tname].append(1 - len(bs & prev_basket[tname]) / max(len(bs), 1))
            prev_basket[tname] = bs

        # phase-70.2: SOFT cross-sector diversity -- PRODUCTION rank_candidates with the
        # soft multiplicative rank-decay penalty (rows carry sector via sec_map). This is
        # the rank-time lever whose OOS impact IS measurable in the top-N basket.
        for sname, w in soft_configs.items():
            ranked = rank_candidates(rows, top_n=TOP_N, strategy="momentum",
                                     soft_sector_diversity=True, soft_sector_diversity_w=w)
            basket = [r["ticker"] for r in ranked]
            fwd = basket_fwd_return(basket, closes, t_idx)
            monthly[sname].append(fwd)
            spread[sname].append(len({sec_map.get(t, "") for t in basket}))
            bs = set(basket)
            if prev_basket[sname]:
                turnover[sname].append(1 - len(bs & prev_basket[sname]) / max(len(bs), 1))
            prev_basket[sname] = bs

    log("[4/4] results\n")
    print(f"{'config':<16}{'ann_Sharpe':>12}{'avg_fwd_mo%':>14}{'avg_sectors':>13}{'avg_turnover':>14}")
    print("-" * 69)
    base_sharpe = ann_sharpe(monthly["baseline"])
    base_to = float(np.mean(turnover["baseline"])) if turnover["baseline"] else 0.0
    for name in ["baseline", "sector_neutral", "vol_scaled", "hi52_k0.5", "hi52_k1.0"]:
        ms = [m for m in monthly[name] if m is not None]
        sh = ann_sharpe(monthly[name])
        avg_ret = float(np.mean(ms) * 100) if ms else 0.0
        avg_sec = float(np.mean(spread[name])) if name in spread and spread[name] else (float(np.mean(spread["baseline"])) if name == "vol_scaled" else 0.0)
        avg_to = float(np.mean(turnover[name])) if name in turnover and turnover[name] else (float(np.mean(turnover["baseline"])) if name == "vol_scaled" else 0.0)
        print(f"{name:<16}{sh:>12.3f}{avg_ret:>14.3f}{avg_sec:>13.2f}{avg_to:>14.3f}")

    sn_sharpe = ann_sharpe(monthly["sector_neutral"])
    sn_spread = float(np.mean(spread["sector_neutral"])) if spread["sector_neutral"] else 0
    base_spread = float(np.mean(spread["baseline"])) if spread["baseline"] else 0
    vs_sharpe = ann_sharpe(monthly["vol_scaled"])
    print("\n--- VERDICT (51.2 sector-neutral gate) ---")
    print(f"sector_neutral vs baseline: dSharpe={sn_sharpe - base_sharpe:+.3f}, "
          f"dSectors={sn_spread - base_spread:+.2f}")
    keep_sn = (sn_spread - base_spread >= 2.0) and (sn_sharpe - base_sharpe >= -0.05)
    print(f"KEEP sector_neutral? {keep_sn}  (gate: breadth +>=2 sectors AND dSharpe >= -0.05)")
    print(f"vol_scaled vs baseline: dSharpe={vs_sharpe - base_sharpe:+.3f} "
          f"({'BETTER' if vs_sharpe > base_sharpe else 'worse'})")

    # phase-52.1: 52-week-high tilt verdict (keep if dSharpe >= +0.05 AND dTurnover <= +10%)
    print("\n--- VERDICT (52.1 52-week-high tilt) ---")
    keep_any = False
    for tname in ["hi52_k0.5", "hi52_k1.0"]:
        t_sharpe = ann_sharpe(monthly[tname])
        t_to = float(np.mean(turnover[tname])) if turnover[tname] else 0.0
        d_sharpe = t_sharpe - base_sharpe
        d_to = t_to - base_to
        keep = (d_sharpe >= 0.05) and (d_to <= 0.10)
        keep_any = keep_any or keep
        print(f"{tname} vs baseline: dSharpe={d_sharpe:+.3f}, dTurnover={d_to:+.3f} -> KEEP? {keep}")
    print(f"52wh-tilt recommendation: {'ESCALATE to a live operator gate' if keep_any else 'REJECT (large-cap mute, per Barroso-Wang) -> pivot to residual momentum (52.2)'}")

    # phase-70.2: SOFT cross-sector diversity verdict. Gate: breadth up >=1 sector AND
    # OOS Sharpe not materially worse than baseline (dSharpe >= -0.05). The FULL activation
    # gate additionally requires DSR>=0.95 + PBO<=0.5 on the paired monthly returns (dumped).
    print("\n--- VERDICT (70.2 soft cross-sector diversity) ---")
    print(f"baseline: ann_Sharpe={base_sharpe:+.3f}, avg_sectors={base_spread:.2f}")
    soft_keep = False
    soft_dump = {}
    for sname in soft_configs:
        s_sharpe = ann_sharpe(monthly[sname])
        s_spread = float(np.mean(spread[sname])) if spread[sname] else 0.0
        s_to = float(np.mean(turnover[sname])) if turnover[sname] else 0.0
        d_sharpe = s_sharpe - base_sharpe
        d_sec = s_spread - base_spread
        keep = (d_sec >= 1.0) and (d_sharpe >= -0.05)
        soft_keep = soft_keep or keep
        soft_dump[sname] = {"ann_sharpe": s_sharpe, "d_sharpe": d_sharpe,
                            "avg_sectors": s_spread, "d_sectors": d_sec,
                            "avg_turnover": s_to, "keep": keep, "monthly": monthly[sname]}
        print(f"{sname} vs baseline: dSharpe={d_sharpe:+.3f}, dSectors={d_sec:+.2f}, avgTurnover={s_to:.3f} -> KEEP? {keep}")
    print(f"soft-diversity recommendation: {'ESCALATE to operator activation gate (then DSR>=0.95 + PBO<=0.5 on the dumped paired returns)' if soft_keep else 'HOLD DARK -- no w cleared the breadth+Sharpe gate'}")

    import json as _json
    _soft_out = "handoff/current/_70_2_soft_diversity_replay.json"
    with open(_soft_out, "w", encoding="utf-8") as f:
        _json.dump({"baseline_sharpe": base_sharpe, "baseline_avg_sectors": base_spread,
                    "baseline_monthly": monthly["baseline"], "soft": soft_dump,
                    "n_rebalances": len([m for m in monthly["baseline"] if m is not None])},
                   f, indent=2)
    print(f"phase-70.2: soft-diversity paired returns dumped -> {_soft_out}")

    print(f"\nN rebalances scored: {len([m for m in monthly['baseline'] if m is not None])}")

    # phase-52.3: dump the paired monthly arrays + the 5 config Sharpes -> the reproducibility
    # PIN for the Ledoit-Wolf SR-difference test (so the 52.3 verdict is deterministic vs live drift).
    import json
    dump = {
        "baseline": monthly["baseline"],
        "hi52_k0.5": monthly["hi52_k0.5"],
        "config_sharpes": {c: ann_sharpe(monthly[c]) for c in ["baseline", "sector_neutral", "vol_scaled", "hi52_k0.5", "hi52_k1.0"]},
        "n_rebalances": len([m for m in monthly["baseline"] if m is not None]),
    }
    out = "handoff/current/_52wh_paired_returns.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)
    print(f"phase-52.3: paired returns dumped -> {out}")


if __name__ == "__main__":
    main()
