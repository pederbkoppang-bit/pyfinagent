"""phase-82.2 deterministic label fixture.

A committed, seeded, BQ-free synthetic market used to MEASURE the label
distribution of the candidate strategies. Deterministic by construction: the
same seed yields the same prices on every machine, so "no class exceeds 95%" is
a reproducible claim rather than a lucky sample.

The fixture deliberately spans a WIDE cross-section -- low- and high-volatility
names, up/down/flat drifts, and names sitting above, below and near their
50-day average -- because a label set can look healthy on a narrow slice and
collapse on a real universe. It also includes SPY, which
`_compute_stretch_regime_label` needs for its market-turbulence proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 82_2
N_DAYS = 900
START = "2022-01-03"

# (ticker, annualised vol %, annual drift, sma-offset seed)
_SPEC = [
    ("LOWVOL_UP",    12.0,  0.10,  0.02),
    ("LOWVOL_DOWN",  12.0, -0.10, -0.02),
    ("MIDVOL_UP",    28.0,  0.15,  0.05),
    ("MIDVOL_DOWN",  28.0, -0.15, -0.05),
    ("MIDVOL_FLAT",  28.0,  0.00,  0.00),
    ("HIVOL_UP",     55.0,  0.25,  0.09),
    ("HIVOL_DOWN",   55.0, -0.25, -0.09),
    ("HIVOL_FLAT",   55.0,  0.00,  0.01),
    ("MEANREV_A",    35.0,  0.00,  0.00),
    ("MEANREV_B",    45.0,  0.00,  0.00),
]

SPY_VOL = 16.0


def _path(rng: np.random.Generator, n: int, vol_ann: float, drift_ann: float,
          oscillate: bool = False) -> np.ndarray:
    sig_d = (vol_ann / 100.0) / np.sqrt(252.0)
    mu_d = drift_ann / 252.0
    shocks = rng.normal(mu_d, sig_d, n)
    if oscillate:
        # Superimpose a slow cycle so the name repeatedly stretches away from
        # and reverts to its 50-day mean -- otherwise a pure random walk almost
        # never produces the overextension the reversion label looks for.
        cycle = 0.9 * sig_d * np.sin(np.linspace(0, 14 * np.pi, n))
        shocks = shocks + cycle
    return 100.0 * np.exp(np.cumsum(shocks))


def build_prices() -> dict[str, pd.DataFrame]:
    """ticker -> DataFrame indexed by date with a `close` column."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, periods=N_DAYS)
    out: dict[str, pd.DataFrame] = {}
    for name, vol, drift, _ in _SPEC:
        osc = name.startswith("MEANREV")
        out[name] = pd.DataFrame({"close": _path(rng, N_DAYS, vol, drift, osc)}, index=dates)
    # SPY: a calm regime followed by a turbulent one, so `_market_stretch`
    # actually varies across the sample instead of sitting at ~1.0.
    calm = _path(rng, N_DAYS // 2, SPY_VOL, 0.08)
    turb = _path(rng, N_DAYS - N_DAYS // 2, SPY_VOL * 2.6, -0.05) * (calm[-1] / 100.0)
    out["SPY"] = pd.DataFrame({"close": np.concatenate([calm, turb])}, index=dates)
    return out


PRICES = build_prices()
TICKERS = [t for t, *_ in _SPEC]


def entry_dates(step: int = 5, warmup: int = 260, tail: int = 200) -> list[str]:
    """Dates with enough history for a 50-day SMA and enough future for the
    forward walk."""
    idx = PRICES["SPY"].index
    return [d.strftime("%Y-%m-%d") for d in idx[warmup:len(idx) - tail:step]]


def cached_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Drop-in for `cache.cached_prices`, inclusive on both ends."""
    df = PRICES.get(ticker)
    if df is None:
        return pd.DataFrame()
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


# Fundamentals: a spread that puts roughly half the names inside the QARP gate.
_FUNDAMENTALS = {
    "LOWVOL_UP":   dict(pe_ratio=14.0, roe=0.18, debt_equity=0.4, profit_margin=0.15),
    "LOWVOL_DOWN": dict(pe_ratio=11.0, roe=0.14, debt_equity=0.9, profit_margin=0.09),
    "MIDVOL_UP":   dict(pe_ratio=19.0, roe=0.12, debt_equity=1.2, profit_margin=0.07),
    "MIDVOL_DOWN": dict(pe_ratio=48.0, roe=0.05, debt_equity=2.6, profit_margin=0.02),
    "MIDVOL_FLAT": dict(pe_ratio=22.0, roe=0.11, debt_equity=1.0, profit_margin=0.06),
    "HIVOL_UP":    dict(pe_ratio=70.0, roe=0.03, debt_equity=3.1, profit_margin=-0.04),
    "HIVOL_DOWN":  dict(pe_ratio=-8.0, roe=-0.12, debt_equity=4.0, profit_margin=-0.20),
    "HIVOL_FLAT":  dict(pe_ratio=24.0, roe=0.10, debt_equity=1.4, profit_margin=0.05),
    "MEANREV_A":   dict(pe_ratio=17.0, roe=0.13, debt_equity=0.8, profit_margin=0.10),
    "MEANREV_B":   dict(pe_ratio=31.0, roe=0.08, debt_equity=1.9, profit_margin=0.04),
}

_VOL = {t: v for t, v, *_ in _SPEC}


def build_feature_vector(ticker: str, entry_date: str) -> dict | None:
    """Drop-in for `HistoricalDataProvider.build_feature_vector`.

    Computes `sma_50_distance` and `rsi_14` from the SAME synthetic prices the
    label walks, so the fixture is internally consistent -- a fixture whose
    features disagree with its prices cannot represent the real failure.
    """
    df = PRICES.get(ticker)
    if df is None:
        return None
    hist = df.loc[df.index <= pd.Timestamp(entry_date)]
    if len(hist) < 60:
        return None
    close = hist["close"].astype(float)
    price = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean())

    delta = close.diff().dropna().tail(14)
    gain = float(delta[delta > 0].sum())
    loss = float(-delta[delta < 0].sum())
    rsi = 100.0 if loss == 0 else 100.0 - (100.0 / (1.0 + gain / loss))

    fv = {
        "price_at_analysis": price,
        "sma_50_distance": (price - sma50) / sma50,
        "rsi_14": rsi,
        "annualized_volatility": _VOL[ticker],
        "momentum_6m": float(close.iloc[-1] / close.iloc[-126] - 1.0) * 100 if len(close) > 126 else 0.0,
        "quality_score": 0.5,
    }
    fv.update(_FUNDAMENTALS[ticker])
    return fv
