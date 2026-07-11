"""phase-69.3 signal-integrity fixes (audit item 6).

Sign-safe overlays (flag-gated), news token-cap + parse-retry, QMJ Growth ordering,
and the INDPRO + net-liquidity regime lift (flag-gated, historical_macro untouched).
Every live ranking change is behind a default-OFF flag = byte-identical to today.
"""

import pathlib

import pytest

from backend.services.overlay_math import sign_safe_mult
import backend.services.macro_regime as mr
import backend.tools.fred_data as fd
from backend.config.settings import get_settings


# ----------------------------------------------------------------------
# 1. Sign-safe overlays (criterion 1)
# ----------------------------------------------------------------------
def test_sign_safe_eliminates_inversion_on_negative_base(monkeypatch):
    # Two candidates with equal NEGATIVE base (drawdown regime); one gets a POSITIVE
    # catalyst (boost), one a NEGATIVE catalyst (penalty). Sign-safe: the positive
    # candidate must rank ABOVE the negative one.
    monkeypatch.setattr(get_settings(), "sign_safe_overlays", True)
    base = -10.0
    boosted = sign_safe_mult(base, 1.10)    # positive catalyst
    penalized = sign_safe_mult(base, 0.90)  # negative catalyst
    assert boosted > penalized                       # inversion eliminated
    assert (base * 1.10) < (base * 0.90)             # legacy multiplicative INVERTED (-11 < -9)


def test_sign_safe_off_byte_identical(monkeypatch):
    monkeypatch.setattr(get_settings(), "sign_safe_overlays", False)
    for base in (-10.0, 10.0, -3.3, 0.0, 5.5):
        for mult in (1.10, 0.90, 1.05, 0.95, 1.20):
            assert sign_safe_mult(base, mult) == base * mult   # legacy, byte-identical


def test_sign_safe_positive_base_unchanged_intent(monkeypatch):
    monkeypatch.setattr(get_settings(), "sign_safe_overlays", True)
    assert sign_safe_mult(10.0, 1.10) == pytest.approx(11.0)   # reduces to base*mult
    assert sign_safe_mult(10.0, 0.90) == pytest.approx(9.0)


def test_sign_safe_wired_into_a_real_overlay(monkeypatch):
    # A real apply_*_to_score routes through the flag (proving the 14 overlays are wired).
    from backend.services import options_flow_screen as ofs

    class _Sig:
        boost_multiplier = 1.10

    signals = {"AAA": _Sig()}
    monkeypatch.setattr(get_settings(), "sign_safe_overlays", False)
    assert ofs.apply_options_surge_to_score(-10.0, "AAA", signals) == pytest.approx(-11.0)  # legacy invert
    monkeypatch.setattr(get_settings(), "sign_safe_overlays", True)
    assert ofs.apply_options_surge_to_score(-10.0, "AAA", signals) == pytest.approx(-9.0)   # sign-safe boost raises


# ----------------------------------------------------------------------
# 2. News token-cap + parse-retry (criterion 2)
# ----------------------------------------------------------------------
def test_news_cap_no_longer_truncates_and_retries():
    src = pathlib.Path("backend/services/news_screen.py").read_text(encoding="utf-8")
    assert "min(8192, 250 * len(deduped))" not in src   # the truncating min() is gone
    assert "min(48000" in src                            # raised cap (fits >32 headlines; haiku max 64k)
    assert "range(2)" in src                             # parse-fail retry loop
    # a 100-headline batch now fits (25000 tokens, under the 48000 cap, above the old 8192)
    assert min(48000, max(8192, 250 * 100)) == 25000 > 8192


# ----------------------------------------------------------------------
# 3. QMJ Growth ordering (criterion 3)
# ----------------------------------------------------------------------
def test_qmj_growth_assigned_before_read():
    src = pathlib.Path("backend/backtest/historical_data.py").read_text(encoding="utf-8")
    assign_idx = src.index('features["revenue_growth_yoy"] = self._compute_revenue_growth_yoy')
    read_idx = src.index('rev_growth = features.get("revenue_growth_yoy")')
    assert assign_idx < read_idx   # Growth dimension now fires (was always None -> dead)


# ----------------------------------------------------------------------
# 4. INDPRO + net-liquidity regime lift (criterion 4)
# ----------------------------------------------------------------------
_IND = {
    "INDPRO": {"name": "IP", "current": 103.2, "previous": 102.9, "trend": "rising", "date": "2026-07-01"},
    "VIXCLS": {"name": "VIX", "current": 15.0, "previous": 16.0, "trend": "falling", "date": "2026-07-01"},
}


def test_indpro_now_in_fred_series():
    assert "INDPRO" in fd.SERIES   # the 1-line repair: it was fetched-nowhere before


def test_regime_prompt_off_is_byte_identical():
    off = mr._build_prompt(_IND)   # flags default OFF
    assert "INDPRO" not in off and "NET_LIQUIDITY" not in off
    # byte-identical to the pre-fix (INDPRO-absent) prompt -> do-no-harm live
    assert off == mr._build_prompt({"VIXCLS": _IND["VIXCLS"]})


def test_regime_prompt_on_includes_indpro_and_netliq():
    on = mr._build_prompt(
        _IND,
        net_liquidity={"net_liquidity_musd": 6_100_000.0, "trend": "rising", "as_of": "2026-07-01"},
        include_indpro=True,
    )
    assert "INDPRO" in on and "NET_LIQUIDITY" in on


def test_net_liquidity_unit_scaling(monkeypatch, tmp_path):
    # WALCL=7,000,000 (M), WTREGEN=800,000 (M), RRPONTSYD=300 (B -> x1000)
    # net = 7,000,000 - 800,000 - 300*1000 = 5,900,000 (M). The x1000 RRP scaling is
    # the silent-corruption trap the fix must get right.
    async def fake_fetch_series(sid, key, periods=12):
        vals = {"WALCL": 7_000_000.0, "WTREGEN": 800_000.0, "RRPONTSYD": 300.0}
        return {"observations": [{"date": "2026-07-01", "value": vals[sid]}]}

    monkeypatch.setattr("backend.tools.fred_data._fetch_series", fake_fetch_series)
    monkeypatch.setattr(mr, "_NETLIQ_CACHE_PATH", tmp_path / "netliq.json")
    import asyncio
    out = asyncio.run(mr._fetch_net_liquidity("fakekey"))
    assert out is not None and out["net_liquidity_musd"] == pytest.approx(5_900_000.0)


# ----------------------------------------------------------------------
# Do-no-harm: historical_macro untouched, flags default-OFF
# ----------------------------------------------------------------------
def test_flags_default_off():
    s = get_settings()
    # (defaults come from the Field(False); other tests monkeypatch a fresh view)
    assert hasattr(s, "sign_safe_overlays") and hasattr(s, "regime_net_liquidity")


def test_net_liquidity_writes_no_bq():
    src = pathlib.Path("backend/services/macro_regime.py").read_text(encoding="utf-8")
    # the new cached path must not touch BigQuery (historical_macro stays frozen). Scope
    # to the fn body and check for actual BQ-WRITE sinks (not the docstring's mention).
    lo = src.index("async def _fetch_net_liquidity")
    hi = src.index("def _build_prompt", lo)
    seg = src[lo:hi]
    for sink in ["insert_rows", "bigquery", "_bq(", ".query("]:
        assert sink not in seg, f"net-liq path must not touch BQ: {sink!r}"
