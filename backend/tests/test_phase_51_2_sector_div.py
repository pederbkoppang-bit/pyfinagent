"""phase-51.2: sector diversification -- byte-identity (flag OFF) + no-op-fixed (flag ON).

Pins, on the PRODUCTION `screener.rank_candidates`:
- with sector_neutral=False, attaching a `sector` field does NOT change the ranked order
  (the wiring is byte-identical live while the flag stays default-OFF).
- with sector_neutral=True + a multi-sector screen, the top-N basket SPREADS across sectors
  (the silent no-op -- caused by candidates having sector=None at rank time -- is fixed once
  candidates carry a sector). This is the criterion-1 behavior the wiring enables.
"""
import pandas as pd  # noqa: F401  (rank_candidates imports pandas internally)

from backend.tools.screener import rank_candidates


def _row(ticker, sector, m1, m3, m6):
    # neutral rsi/vol -> no penalty; composite = m1*0.4 + m3*0.35 + m6*0.25
    return {
        "ticker": ticker, "sector": sector,
        "momentum_1m": m1, "momentum_3m": m3, "momentum_6m": m6,
        "rsi_14": 50, "volatility_ann": 0.3, "sma_50_distance_pct": 0.0,
    }


def _tech_heavy_screen():
    # 4 Tech (higher momentum) + 4 Health (lower) -- so raw momentum favors Tech
    tech = [_row(f"T{i}", "Information Technology", 0.30 - 0.02 * i, 0.30 - 0.02 * i, 0.30 - 0.02 * i)
            for i in range(4)]
    health = [_row(f"H{i}", "Health Care", 0.20 - 0.02 * i, 0.20 - 0.02 * i, 0.20 - 0.02 * i)
              for i in range(4)]
    return tech + health


def test_flag_off_is_byte_identical_with_or_without_sector():
    rows = _tech_heavy_screen()
    rows_nosec = [{k: v for k, v in r.items() if k != "sector"} for r in rows]
    a = [r["ticker"] for r in rank_candidates(rows, top_n=4, strategy="momentum", sector_neutral=False)]
    b = [r["ticker"] for r in rank_candidates(rows_nosec, top_n=4, strategy="momentum", sector_neutral=False)]
    assert a == b, "passing a sector field changed the OFF-path ranking (NOT byte-identical)"


def test_flag_off_basket_is_tech_concentrated():
    rows = _tech_heavy_screen()
    basket = rank_candidates(rows, top_n=4, strategy="momentum", sector_neutral=False)
    sectors = {r["sector"] for r in basket}
    assert sectors == {"Information Technology"}, "baseline should concentrate in the top-momentum sector"


def test_flag_on_spreads_across_sectors():
    rows = _tech_heavy_screen()
    basket = rank_candidates(rows, top_n=4, strategy="momentum", sector_neutral=True)
    sectors = {r["sector"] for r in basket}
    assert len(sectors) >= 2, "sector_neutral should spread the basket across >1 sector (no-op fixed)"
    assert "Health Care" in sectors, "the best Health name should surface via within-sector percentile"


def test_flag_on_requires_sectors_to_work():
    # candidates with NO sector field -> all _UNKNOWN_ -> global pool -> monotone -> same order
    # as baseline (documents WHY the live no-op happened before the wiring).
    rows = [{k: v for k, v in r.items() if k != "sector"} for r in _tech_heavy_screen()]
    on = [r["ticker"] for r in rank_candidates(rows, top_n=4, strategy="momentum", sector_neutral=True)]
    off = [r["ticker"] for r in rank_candidates(rows, top_n=4, strategy="momentum", sector_neutral=False)]
    assert on == off, "without sectors, sector_neutral is a no-op (the bug the wiring fixes)"
