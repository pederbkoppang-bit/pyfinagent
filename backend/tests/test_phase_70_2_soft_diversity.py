"""phase-70.2 (P1, S2) verification: SOFT, profit-aware cross-sector diversification.

Deterministic (network-free) proofs for the four immutable criteria:
  1. min-K round-robin makes the analyze slice span >=2 sectors (vs the plain
     top-N slice that reproduces today's monosector funnel).
  3. the "Unknown" sector bucket can be exempted from the count cap so an
     enrichment failure no longer freezes the funnel.
  4. every lever is default-OFF -> byte-identical (soft penalty w=0 / flag OFF;
     min-K K=0; Unknown-exempt OFF).
Plus: the soft penalty is SOFT (shades, never hard-neutralizes) and SIGN-SAFE
(a penalty lowers rank even for a negative composite score).

The OOS-P&L check (criterion 2) is the extended $0/macro-free ablation replay
(scripts/ablation/sector_neutral_replay.py) -- see experiment_results.md; not
unit-testable here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- fixtures
def _row(ticker: str, sector: str, m1: float, m3: float, m6: float) -> dict:
    return {"ticker": ticker, "sector": sector, "momentum_1m": m1,
            "momentum_3m": m3, "momentum_6m": m6, "rsi_14": 55, "volatility_ann": 0.3}


def _scored(ticker: str, sector: str, score: float) -> dict:
    return {"ticker": ticker, "sector": sector, "composite_score": score}


def _settings(**kw):
    base = dict(
        paper_max_per_sector=2,
        paper_max_per_sector_nav_pct=0.0,
        paper_max_positions=10,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_target_position_size_pct=10.0,
        paper_default_stop_loss_pct=8.0,
        paper_min_cash_buffer=100.0,
        paper_trade_min_confidence=0.0,
        paper_max_position_value_pct=100.0,
        paper_unknown_sector_cap_exempt=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _position(ticker: str, sector: str) -> dict:
    return {"ticker": ticker, "sector": sector, "qty": 10, "current_price": 100.0,
            "entry_price": 100.0, "market_value": 1000.0, "stop_loss_price": 92.0,
            "recommendation": "BUY"}


def _candidate(ticker: str, sector: str) -> dict:
    return {"ticker": ticker, "sector": sector, "recommendation": "BUY",
            "confidence": 0.7, "risk_assessment": {"risk_level": "moderate"},
            "current_price": 100.0, "analysis_date": "2026-07-17T00:00:00Z"}


# ---------------------------------------------------------------- criterion 4 (OFF byte-identical)
def test_soft_off_and_w0_byte_identical():
    from backend.tools.screener import rank_candidates
    rows = [_row("A", "Technology", 20, 18, 15), _row("B", "Technology", 18, 16, 12),
            _row("C", "Energy", 10, 9, 8), _row("D", "Health Care", 6, 5, 4)]
    base = rank_candidates([dict(r) for r in rows], top_n=10, strategy="momentum")
    off = rank_candidates([dict(r) for r in rows], top_n=10, strategy="momentum",
                          soft_sector_diversity=False, soft_sector_diversity_w=0.30)
    w0 = rank_candidates([dict(r) for r in rows], top_n=10, strategy="momentum",
                         soft_sector_diversity=True, soft_sector_diversity_w=0.0)
    assert [c["ticker"] for c in base] == [c["ticker"] for c in off] == [c["ticker"] for c in w0]
    assert [c["composite_score"] for c in base] == [c["composite_score"] for c in off] == [c["composite_score"] for c in w0]
    # w=0 skips the block entirely -> no composite_score_raw side-channel written
    assert all("composite_score_raw" not in c for c in w0)


# ---------------------------------------------------------------- soft penalty: SOFT + sign-safe
def test_apply_soft_sector_diversity_shades_sign_safe():
    from backend.tools.screener import _apply_soft_sector_diversity
    from backend.services.overlay_math import sign_safe_mult
    scored = [_scored("T1", "Technology", 10.0), _scored("T2", "Technology", 8.0),
              _scored("E1", "Energy", 5.0), _scored("T3", "Technology", -4.0)]
    _apply_soft_sector_diversity(scored, 0.5)
    by = {s["ticker"]: s for s in scored}
    # leader of each sector untouched (j=0 -> mult=1)
    assert by["T1"]["composite_score"] == 10.0
    assert by["E1"]["composite_score"] == 5.0
    # 2nd Tech (j=1) shaded by sign_safe_mult(8, 0.5) = 8 + 8*(0.5-1) = 4.0
    assert by["T2"]["composite_score"] == round(sign_safe_mult(8.0, 0.5, enabled=True), 4) == 4.0
    # 3rd Tech (j=2) has a NEGATIVE base -> penalty makes it MORE negative (demoted),
    # NOT raised toward zero (the sign-inversion a raw base*mult would cause).
    assert by["T3"]["composite_score"] < -4.0
    # raw preserved
    assert by["T2"]["composite_score_raw"] == 8.0


# ---------------------------------------------------------------- criterion 1 (min-K spread)
def test_min_k_slice_reproduces_and_diversifies():
    from backend.services.autonomous_loop import _min_k_sector_slice
    cands = [_scored("T1", "Technology", 10), _scored("T2", "Technology", 9),
             _scored("T3", "Technology", 8), _scored("T4", "Technology", 7),
             _scored("T5", "Technology", 6), _scored("E1", "Energy", 4),
             _scored("H1", "Health Care", 3)]
    # plain top-5 slice reproduces today's monosector funnel (1 sector)
    plain = cands[:5]
    assert len({c["sector"] for c in plain}) == 1
    # min-K=3 spans >=2 (here 3) distinct sectors, best names still first
    picked = _min_k_sector_slice(cands, 5, 3)
    assert len(picked) == 5
    assert len({c["sector"] for c in picked}) >= 2
    assert {"Technology", "Energy", "Health Care"} <= {c["sector"] for c in picked}
    # re-sorted by score desc for the analyzer
    scores = [c["composite_score"] for c in picked]
    assert scores == sorted(scores, reverse=True)


def test_min_k_graceful_single_sector():
    from backend.services.autonomous_loop import _min_k_sector_slice
    cands = [_scored("T1", "Technology", 10), _scored("T2", "Technology", 9),
             _scored("T3", "Technology", 8)]
    picked = _min_k_sector_slice(cands, 5, 3)  # fewer sectors than k, fewer cands than n
    assert [c["ticker"] for c in picked] == ["T1", "T2", "T3"]  # no crash, best-effort


# ---------------------------------------------------------------- criterion 3 (Unknown exemption)
def _decide(exempt: bool, caplog):
    from backend.services.portfolio_manager import decide_trades
    caplog.set_level(logging.INFO, logger="backend.services.portfolio_manager")
    settings = _settings(paper_max_per_sector=2, paper_unknown_sector_cap_exempt=exempt)
    # two held positions with a MISSING/Unknown sector (enrichment failed)
    current_positions = [_position("XXA", ""), _position("XXB", "")]
    candidates = [_candidate("XXC", "")]  # also Unknown sector
    portfolio_state = {"nav": 10000.0, "cash": 8000.0, "positions_value": 2000.0, "position_count": 2}
    orders = decide_trades(current_positions, candidates, [], portfolio_state, settings)
    return {o.ticker for o in orders if o.action == "BUY"}


def test_unknown_exempt_off_blocks(caplog):
    """OFF (default) -> Unknown bucket at cap blocks the new Unknown BUY (byte-identical to today)."""
    assert "XXC" not in _decide(exempt=False, caplog=caplog)


def test_unknown_exempt_on_allows(caplog):
    """ON -> the Unknown bucket is exempt so the enrichment-failed candidate is not frozen out."""
    assert "XXC" in _decide(exempt=True, caplog=caplog)


# ---------------------------------------------------------------- mutation guard: flags present
def test_flags_present_in_settings():
    from backend.config.settings import Settings
    for f in ("paper_soft_sector_diversity_enabled", "paper_soft_sector_diversity_w",
              "paper_min_k_sectors_analyzed", "paper_unknown_sector_cap_exempt"):
        assert f in Settings.model_fields, f"settings must define {f}"
    # defaults are OFF/identity
    d = Settings.model_fields
    assert d["paper_soft_sector_diversity_enabled"].default is False
    assert d["paper_soft_sector_diversity_w"].default == 0.0
    assert d["paper_min_k_sectors_analyzed"].default == 0
    assert d["paper_unknown_sector_cap_exempt"].default is False
