"""phase-53.1: transaction-cost-aware no-trade / rebalance buffer band.

Quant-elevation lever (PORTFOLIO CONSTRUCTION, not signal -- the cheap-price signal
search was exhausted in phase-52). On each monthly top-N reconstitution, apply a
hysteresis buffer: RETAIN a currently-held name unless its rank drops below
`top_n * (1 + band_pct)`; ADD a new name only when it clears the entry rank (top_n).
This is the discrete-portfolio analogue of Garleanu-Pedersen "trade partially toward
the aim" + the Kitces/Daryanani tolerance band, specialized to the long-only
equal-weight US momentum basket. Its value is NET-OF-COST via reduced turnover
(momentum's gross alpha is intact net of costs -- arXiv:2412.11575, Alpha Architect 2025).

DO-NO-HARM: `enabled=False` OR `band_pct<=0` OR no `prev_holdings` -> returns
`ranked_tickers[:top_n]` (byte-identical to full reconstitution). This is a pure,
stateless-per-call function (the caller threads `prev_holdings`); it is the SINGLE
SOURCE OF TRUTH used by both the $0 replay (scripts/ablation/no_trade_band_replay.py)
and -- once an operator enables the gated setting in a SEPARATE step -- the live
rebalance. phase-53.1 is measure-first: NO live wiring / NO live flag flip here.
"""
from __future__ import annotations


def apply_no_trade_band(
    prev_holdings,
    ranked_tickers,
    top_n: int,
    band_pct: float = 0.2,
    enabled: bool = True,
) -> list[str]:
    """Return the next basket (<= top_n names) under a no-trade band.

    Args:
        prev_holdings: the basket held going into this rebalance (iterable of tickers).
        ranked_tickers: ALL candidates ranked best-first (rank 0 = strongest).
        top_n: target basket size + the ADD (entry) rank threshold.
        band_pct: hysteresis width; a held name is retained while its rank is
            < top_n * (1 + band_pct) (the EXIT threshold).
        enabled: master gate. False -> full reconstitution (byte-identical).

    OFF / cold-start / degenerate -> ranked_tickers[:top_n] (full reconstitution).
    """
    base = list(ranked_tickers[:top_n])
    if not enabled or band_pct <= 0 or not prev_holdings:
        return base

    rank = {t: i for i, t in enumerate(ranked_tickers)}
    exit_thresh = top_n * (1.0 + band_pct)

    # 1) RETAIN held names still within the exit band (preserve prior order).
    retained = [t for t in prev_holdings if t in rank and rank[t] < exit_thresh]
    retained = retained[:top_n]  # never exceed the target size
    retained_set = set(retained)

    # 2) FILL the freed slots from names that CLEAR the entry rank (top_n),
    #    best-first, excluding already-retained names.
    out = list(retained)
    for t in ranked_tickers[:top_n]:
        if len(out) >= top_n:
            break
        if t not in retained_set:
            out.append(t)

    # 3) If still short (small universe), top up from the next-best remaining.
    if len(out) < top_n:
        seen = set(out)
        for t in ranked_tickers:
            if len(out) >= top_n:
                break
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out[:top_n]


def max_drawdown(monthly_returns) -> float:
    """Max drawdown (a NEGATIVE fraction, e.g. -0.23) of a monthly-return series.

    Equity = cumulative product of (1+r); drawdown = equity/running_peak - 1;
    maxDD = the minimum (most negative) drawdown. Empty/all-None -> 0.0.
    """
    rets = [r for r in monthly_returns if r is not None]
    if not rets:
        return 0.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in rets:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = equity / peak - 1.0 if peak > 0 else 0.0
        if dd < worst:
            worst = dd
    return float(worst)
