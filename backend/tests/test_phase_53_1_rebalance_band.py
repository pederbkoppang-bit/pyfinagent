"""phase-53.1: no-trade rebalance band helper -- logic + OFF byte-identity + maxDD.

The band is the measured quant-elevation lever. These tests pin the hysteresis logic
and the DO-NO-HARM contract (OFF / band_pct<=0 / cold-start -> full reconstitution,
byte-identical to the live momentum core). Pure-function, no network.
"""
from __future__ import annotations

from backend.backtest.rebalance_band import apply_no_trade_band, max_drawdown

# 15 ranked names, best-first
RANKED = [f"T{i:02d}" for i in range(15)]  # T00 (best) .. T14
TOP_N = 10


def test_off_is_full_reconstitution():
    prev = ["T11", "T12", "T13"]  # held names now ranked outside top_n
    assert apply_no_trade_band(prev, RANKED, TOP_N, band_pct=0.2, enabled=False) == RANKED[:TOP_N]


def test_band_pct_zero_is_full_reconstitution():
    prev = ["T11", "T12"]
    assert apply_no_trade_band(prev, RANKED, TOP_N, band_pct=0.0, enabled=True) == RANKED[:TOP_N]


def test_cold_start_is_full_reconstitution():
    assert apply_no_trade_band([], RANKED, TOP_N, band_pct=0.2, enabled=True) == RANKED[:TOP_N]
    assert apply_no_trade_band(None, RANKED, TOP_N, band_pct=0.2, enabled=True) == RANKED[:TOP_N]


def test_held_name_inside_exit_band_is_retained():
    # exit threshold = 10*(1.2) = 12.0 -> ranks 0..11 retained. T10 (rank 10) + T11 (rank 11)
    # are OUTSIDE top_n (10) but INSIDE the band -> a held T11 must be RETAINED (not churned).
    prev = ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T11"]
    out = apply_no_trade_band(prev, RANKED, TOP_N, band_pct=0.2, enabled=True)
    assert "T11" in out                      # retained despite being rank 11 (outside top_n)
    assert "T09" not in out                  # the top_n name it displaced is NOT force-added
    assert len(out) == TOP_N


def test_held_name_beyond_exit_band_is_dropped():
    # T13 (rank 13) is BEYOND exit threshold 12.0 -> dropped; slot filled from top_n.
    prev = ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T13"]
    out = apply_no_trade_band(prev, RANKED, TOP_N, band_pct=0.2, enabled=True)
    assert "T13" not in out
    assert "T09" in out                      # the freed slot is filled by the best non-held in top_n
    assert len(out) == TOP_N


def test_band_reduces_churn_vs_full_reconstitution():
    # held basket = top_n; next month ranks shift so T08/T09 slip to 10/11 (inside band).
    prev = RANKED[:TOP_N]
    shifted = ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T10", "T11", "T08", "T09"] + ["T12", "T13", "T14"]
    band = apply_no_trade_band(prev, shifted, TOP_N, band_pct=0.2, enabled=True)
    full = shifted[:TOP_N]
    churn_band = len(set(prev) - set(band))
    churn_full = len(set(prev) - set(full))
    assert churn_band <= churn_full          # band churns no more than full reconstitution
    assert "T08" in band and "T09" in band    # retained (slipped only into the band)


def test_never_exceeds_top_n():
    prev = RANKED[:TOP_N]
    out = apply_no_trade_band(prev, RANKED, TOP_N, band_pct=0.5, enabled=True)
    assert len(out) == TOP_N
    assert len(set(out)) == len(out)          # no duplicates


def test_max_drawdown():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([0.1, 0.2, 0.05]) == 0.0          # monotonically up -> no DD
    # +10% then -20% then +5%: peak 1.1, trough 0.88 -> DD = 0.88/1.1 - 1 = -0.20
    dd = max_drawdown([0.10, -0.20, 0.05])
    assert abs(dd - (-0.20)) < 1e-9
