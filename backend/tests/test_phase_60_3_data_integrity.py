"""phase-60.3 (AW-9) tests: decision-input integrity for non-USD markets.

Covers: KRW prompt rendering (no '$'-labeled KRW magnitude), deterministic
integrity pre-check enforcement IN CODE (the 06-09 066570.KS regression),
staleness as-of labeling, and US byte-identity in BOTH flag states.

File name carries `60_3`; the immutable selector is
`-k 'prompt_fx or lite_prompt or 60_3'`.
"""
from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.settings import Settings
from backend.services.data_integrity import (
    MARKET_CAP_CEILING_USD,
    check_data_integrity,
    normalize_market_values,
    render_market_lines,
)

# A '$' immediately preceding a KRW-scale magnitude (>= 7 digits incl.
# thousands separators, or scientific-scale xB values above any real cap).
_DOLLAR_KRW_RE = re.compile(r"\$\s?\d{7,}|\$\s?\d{1,3}(,\d{3}){2,}|\$\s?\d{5,}\.?\d*B")

# 2026-06-09 066570.KS persisted row (BQ, researcher brief section 5.4):
# market_cap=44540606021632.0, pe_ratio=0.0 -- rendered as "$44540.6B".
_KS_INFO = {
    "currentPrice": 64500.0,
    "marketCap": 44540606021632.0,
    "trailingPE": 0.0,
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "shortName": "LG Electronics",
    "currency": "KRW",
    "regularMarketTime": int(datetime(2026, 6, 9, 6, 30, tzinfo=timezone.utc).timestamp()),
}

_US_INFO = {
    "currentPrice": 123.45,
    "marketCap": 2.5e11,
    "trailingPE": 21.7,
    "sector": "Technology",
    "industry": "Semiconductors",
    "shortName": "Micron",
    "currency": "USD",
    "regularMarketTime": int(datetime(2026, 6, 10, 20, 0, tzinfo=timezone.utc).timestamp()),
}


def _patch_fx(monkeypatch, rate):
    import backend.services.fx_rates as fx

    monkeypatch.setattr(fx, "get_fx_rate", lambda f, t, date=None: rate)


# ── normalize + check (criterion 2) ──────────────────────────────────────


def test_60_3_normalize_krw_with_fx(monkeypatch):
    _patch_fx(monkeypatch, 0.00072)
    n = normalize_market_values("066570.KS", _KS_INFO)
    assert n["currency"] == "KRW" and n["fx_available"] is True
    assert n["market_cap_usd"] == pytest.approx(44540606021632.0 * 0.00072)
    assert n["as_of"].startswith("2026-06-09T06:30")


def test_60_3_regression_066570_away_week_state_blocks_in_code(monkeypatch):
    # The away-week state: no FX normalization existed -- the raw KRW
    # magnitude was presented as USD. With FX UNAVAILABLE the value cannot
    # be unit-verified AND the raw magnitude breaches the $10T ceiling:
    # both flags are BLOCKING -> the candidate is excluded IN CODE, no LLM
    # call, no prose-only flagging.
    _patch_fx(monkeypatch, None)
    n = normalize_market_values("066570.KS", _KS_INFO)
    flags = check_data_integrity("066570.KS", _KS_INFO, n)
    kinds = {f["flag"] for f in flags}
    assert "implausible_market_cap" in kinds
    assert "currency_unverified" in kinds
    assert any(f["blocking"] for f in flags)


def test_60_3_krw_with_fx_is_sane_and_unblocked(monkeypatch):
    # With FX available, 44.54T KRW ~= $32B -- plausible; only the P/E==0
    # mega-cap missing-data tag remains (non-blocking).
    _patch_fx(monkeypatch, 0.00072)
    n = normalize_market_values("066570.KS", _KS_INFO)
    flags = check_data_integrity("066570.KS", _KS_INFO, n)
    assert not any(f["blocking"] for f in flags), flags
    assert {f["flag"] for f in flags} == {"missing_pe_large_cap"}


def test_60_3_currency_mismatch_blocks(monkeypatch):
    _patch_fx(monkeypatch, 0.00072)
    bad = dict(_KS_INFO, currency="USD")  # suffix says KR -> KRW expected
    n = normalize_market_values("066570.KS", bad)
    flags = check_data_integrity("066570.KS", bad, n)
    assert any(f["flag"] == "currency_mismatch" and f["blocking"] for f in flags)


def test_60_3_us_mega_cap_over_ceiling_blocks():
    corrupt = dict(_US_INFO, marketCap=MARKET_CAP_CEILING_USD * 4.5)
    n = normalize_market_values("MU", corrupt)
    flags = check_data_integrity("MU", corrupt, n)
    assert any(f["flag"] == "implausible_market_cap" and f["blocking"] for f in flags)


def test_60_3_clean_us_ticker_no_flags():
    n = normalize_market_values("MU", _US_INFO)
    assert n["is_us"] and n["fx_available"]
    assert check_data_integrity("MU", _US_INFO, n) == []


# ── prompt rendering (criteria 1 + 3) ────────────────────────────────────


def test_60_3_lite_prompt_krw_no_dollar_labeled_magnitude(monkeypatch):
    _patch_fx(monkeypatch, 0.00072)
    n = normalize_market_values("066570.KS", _KS_INFO)
    lines = render_market_lines("066570.KS", 64500.0, 44540606021632.0, 0.0, n, True)
    assert not _DOLLAR_KRW_RE.search(lines), lines
    assert "converted from KRW" in lines
    # criterion 3: as-of staleness labeled, not presented as live
    assert "Data as-of: 2026-06-09T06:30" in lines and "NOT live" in lines


def test_60_3_lite_prompt_krw_label_native_fallback(monkeypatch):
    # Defense-in-depth branch (a blocking flag normally excludes first):
    # FX unavailable -> native-labeled, never '$'-labeled.
    _patch_fx(monkeypatch, None)
    n = normalize_market_values("066570.KS", _KS_INFO)
    lines = render_market_lines("066570.KS", 64500.0, 44540606021632.0, 0.0, n, True)
    assert not _DOLLAR_KRW_RE.search(lines), lines
    assert "KRW" in lines and "NOT USD" in lines


def test_60_3_prompt_fx_us_byte_identity_both_flag_states():
    n = normalize_market_values("MU", _US_INFO)
    legacy = f"Price: ${123.45:.2f} | Market Cap: ${2.5e11/1e9:.1f}B | P/E: {21.7:.1f}"
    off = render_market_lines("MU", 123.45, 2.5e11, 21.7, n, False)
    on = render_market_lines("MU", 123.45, 2.5e11, 21.7, n, True)
    assert off == legacy == on  # byte-identical, no as-of line for US


def test_60_3_prompt_fx_flag_off_krw_renders_legacy():
    # Flag OFF: even KRW renders the historical (defective) line --
    # byte-identical do-no-harm; the fix only acts when the operator
    # promotes the flag.
    n = normalize_market_values("066570.KS", _KS_INFO)
    off = render_market_lines("066570.KS", 64500.0, 44540606021632.0, 0.0, n, False)
    assert off.startswith("Price: $64500.00 | Market Cap: $44540.6B")


# ── analyzer wiring: pre-LLM block (criterion 2 IN CODE) ────────────────


class _FakeTicker:
    def __init__(self, info):
        self.info = info

    def history(self, period="3mo"):
        idx = pd.date_range(end="2026-06-09", periods=70)
        return pd.DataFrame({"Close": [60000.0] * 70}, index=idx)


def test_60_3_claude_analyzer_blocks_pre_llm(monkeypatch):
    import backend.services.autonomous_loop as al
    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", lambda t: _FakeTicker(dict(_KS_INFO)))
    import backend.services.fx_rates as fx
    monkeypatch.setattr(fx, "get_fx_rate", lambda f, t, date=None: None)  # away-week state

    # Any LLM touch fails the test: the block must fire BEFORE the rails.
    import backend.agents.claude_code_client as ccc
    def _boom(*a, **k):
        raise AssertionError("LLM rail was called despite a blocking integrity flag")
    monkeypatch.setattr(ccc, "claude_code_invoke", _boom)

    settings = Settings(paper_data_integrity_enabled=True, paper_use_claude_code_route=True)
    result = asyncio.run(al._run_claude_analysis("066570.KS", settings))
    assert result["_data_integrity_blocked"] is True
    assert result["recommendation"] == "HOLD"
    assert result["risk_assessment"]["decision"] == "REJECT"
    assert result["risk_assessment"]["recommended_position_pct"] == 0.0
    assert "implausible_market_cap" in result["full_report"]["market_data"]["integrity_flags"]
    assert result["total_cost_usd"] == 0.0


def test_60_3_provenance_fields_ungated(monkeypatch):
    # Flag OFF: no block, no prompt change -- but the additive provenance
    # fields still land in market_data (criterion 4's BQ-auditable values).
    from backend.services.autonomous_loop import _integrity_market_data

    _patch_fx(monkeypatch, 0.00072)
    n = normalize_market_values("005930.KS", dict(_KS_INFO, shortName="Samsung"))
    flags = check_data_integrity("005930.KS", _KS_INFO, n)
    md = _integrity_market_data("Samsung", 64500.0, 4.45e13, 0.0,
                                "Technology", "Semis", 1.0, 2.0, n, flags)
    assert md["currency"] == "KRW"
    assert md["price_usd"] == pytest.approx(64500.0 * 0.00072)
    assert md["as_of"].startswith("2026-06-09T06:30")
    assert "integrity_flags" in md
    # legacy keys intact (additive only)
    assert md["price"] == 64500.0 and md["market_cap"] == 4.45e13


def test_60_3_flag_defaults_off():
    assert Settings().paper_data_integrity_enabled is False
