"""phase-86.116 -- duplicate price rows must not reach the backtest stack.

`financial_reports.historical_prices` carries duplicate `(ticker, date)` rows:
706,875 duplicated keys of 1,152,607 (61.33% of keys), 706,875 excess rows
(38.01% of the table), 336 of 513 tickers, ~63% of keys in every year 2017-2025
and 0.1% in 2026. Nothing under `backend/` de-duplicated them, so every
POSITIONAL lookback in the backtest stack silently spanned about half the
sessions it named.

These tests pin the fix AND the two things that make it easy to get wrong:

* **the method** -- `drop_duplicates()` ignores the index, so it is not a
  substitute (`test_value_keyed_dedup_is_insufficient`);
* **the shape of the harm** -- duplicating only the RECENT bars corrupts
  positional factors while leaving volatility almost untouched, because `iloc`
  lookbacks respond to recent duplicates and `std()` to the global rate
  (`test_recent_only_duplication_corrupts_momentum_but_not_vol`). A fixture that
  only duplicates uniformly would miss half the defect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.backtest.cache import _dedupe_index
from backend.tools.screener import _compute_rsi, _pct_change

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

N = 260


def _clean_frame(n: int = N) -> pd.DataFrame:
    """A unique-index OHLCV frame with a deterministic non-flat price path."""
    idx = pd.bdate_range("2025-01-01", periods=n)
    # Deterministic, non-monotone, and never zero -- a flat series would make
    # every momentum assertion below vacuously equal.
    close = [100.0 + 12.0 * ((i % 17) / 17.0) + 0.05 * i for i in range(n)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": [1_000_000 + i for i in range(n)],
        },
        index=idx,
    )


def _duplicate_all(df: pd.DataFrame) -> pd.DataFrame:
    """Every bar appears twice -- the 2017-2025 shape, multiplicity exactly 2."""
    return pd.concat([df, df]).sort_index()


def _duplicate_tail(df: pd.DataFrame, k: int = 40) -> pd.DataFrame:
    """Only the last k bars appear twice -- the asymmetric shape."""
    return pd.concat([df, df.tail(k)]).sort_index()


# --------------------------------------------------------------------------
# the fix itself
# --------------------------------------------------------------------------

def test_dedupe_is_inert_on_a_unique_index():
    """Criterion 5: with nothing to fix, the frame comes back byte-identical."""
    clean = _clean_frame()
    out = _dedupe_index(clean, "T")
    assert out is clean, "an already-unique frame must not be copied or altered"
    pd.testing.assert_frame_equal(out, clean)


def test_dedupe_removes_duplicate_index_entries():
    dup = _duplicate_all(_clean_frame())
    assert len(dup) == 2 * N and not dup.index.is_unique
    out = _dedupe_index(dup, "T")
    assert out.index.is_unique
    assert len(out) == N


def test_dedupe_restores_the_original_frame_exactly():
    """De-duplicating a doubled frame must reproduce the original, not approximate it.

    `check_freq=False` because `concat` drops the DatetimeIndex `freq`
    attribute, so the rebuilt index carries `freq=None` where the fixture
    carried `<BusinessDay>`. That is index METADATA, not data: the assertion
    below still compares every label and every value. Narrowed deliberately
    rather than by loosening the comparison wholesale -- `check_like`,
    `check_dtype` and `check_exact` all stay at their strict defaults.
    """
    clean = _clean_frame()
    out = _dedupe_index(_duplicate_all(clean), "T")
    pd.testing.assert_frame_equal(out, clean, check_freq=False)
    assert list(out.index) == list(clean.index)


def test_dedupe_handles_empty_and_none():
    assert _dedupe_index(None, "T") is None
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert _dedupe_index(empty, "T") is empty


# --------------------------------------------------------------------------
# the method: drop_duplicates() is NOT a substitute
# --------------------------------------------------------------------------

def test_value_keyed_dedup_is_insufficient():
    """`drop_duplicates()` ignores the index, so it keeps rows the fix removes.

    This is not hypothetical: 394,719 duplicated keys in the real table carry a
    differing `close`. Measured on AVB 2026 through the real loader,
    `drop_duplicates()` returned 159 rows for 155 distinct dates.
    """
    clean = _clean_frame(10)
    noisy = clean.copy()
    # Float-noise disagreement of the kind the real table carries.
    noisy["close"] = noisy["close"] + 1e-9
    dup = pd.concat([clean, noisy]).sort_index()

    assert len(dup) == 20 and dup.index.nunique() == 10

    value_keyed = dup.drop_duplicates()
    assert len(value_keyed) == 20, (
        "precondition: rows differing by float noise are NOT value-duplicates"
    )
    assert not value_keyed.index.is_unique, (
        "drop_duplicates() must be shown to LEAVE duplicate dates behind -- "
        "that is the whole reason the fix keys on the index"
    )

    index_keyed = _dedupe_index(dup, "T")
    assert index_keyed.index.is_unique and len(index_keyed) == 10


# --------------------------------------------------------------------------
# the harm: positional lookbacks
# --------------------------------------------------------------------------

def test_duplication_compresses_a_positional_lookback():
    """A '21-period' lookback must span 21 sessions, not ~11."""
    clean = _clean_frame()
    dup = _duplicate_all(clean)

    span_clean = (clean.index[-1] - clean.index[-22]).days
    span_dup = (dup.index[-1] - dup.index[-22]).days
    assert span_dup < span_clean, (
        f"duplication must SHORTEN the calendar span of a fixed positional "
        f"lookback (dup={span_dup}d vs clean={span_clean}d)"
    )
    assert dup.index[-22:].nunique() < 22


def test_duplication_changes_momentum_materially():
    clean = _clean_frame()["close"]
    dup = _duplicate_all(_clean_frame())["close"]

    m1_clean, m1_dup = _pct_change(clean, 21), _pct_change(dup, 21)
    m3_clean, m3_dup = _pct_change(clean, 63), _pct_change(dup, 63)
    assert m1_clean is not None and m1_dup is not None
    assert m1_clean != pytest.approx(m1_dup, abs=1e-6), (
        "momentum_1m must differ between the doubled and clean series"
    )
    assert m3_clean != pytest.approx(m3_dup, abs=1e-6)

    after = _pct_change(_dedupe_index(dup.to_frame("close"), "T")["close"], 21)
    assert after == pytest.approx(m1_clean), "the fix must restore the clean value"


def test_full_duplication_understates_annualized_volatility_toward_1_over_sqrt2():
    """Closed form: returns become [r, 0, r, 0, ...], so std falls by 1/sqrt(2)."""
    clean = _clean_frame()["close"]
    dup = _duplicate_all(_clean_frame())["close"]

    v_clean = float(clean.pct_change().dropna().std() * (252 ** 0.5))
    v_dup = float(dup.pct_change().dropna().std() * (252 ** 0.5))
    ratio = v_dup / v_clean

    assert 0.60 < ratio < 0.80, (
        f"annualized vol must be understated toward 1/sqrt(2)=0.7071 under full "
        f"duplication; measured ratio {ratio:.4f}"
    )
    v_fixed = float(
        _dedupe_index(dup.to_frame("close"), "T")["close"]
        .pct_change().dropna().std() * (252 ** 0.5)
    )
    assert v_fixed == pytest.approx(v_clean)


def test_recent_only_duplication_corrupts_momentum_but_not_vol():
    """The asymmetry the research gate surfaced, pinned as its own case.

    `iloc` lookbacks break on RECENT duplicates; `std()` responds to the GLOBAL
    rate. So duplicating only the tail corrupts momentum and RSI while leaving
    volatility nearly intact -- a uniform-duplication fixture cannot see this.
    """
    clean = _clean_frame()
    tail = _duplicate_tail(clean, 40)

    m1_clean = _pct_change(clean["close"], 21)
    m1_tail = _pct_change(tail["close"], 21)
    assert m1_clean != pytest.approx(m1_tail, abs=1e-6), (
        "tail-only duplication must still corrupt a 21-period lookback"
    )

    rsi_clean = _compute_rsi(clean["close"], 14)
    rsi_tail = _compute_rsi(tail["close"], 14)
    assert rsi_clean != pytest.approx(rsi_tail, abs=1e-6)

    v_clean = float(clean["close"].pct_change().dropna().std() * (252 ** 0.5))
    v_tail = float(tail["close"].pct_change().dropna().std() * (252 ** 0.5))
    assert 0.80 < (v_tail / v_clean) < 1.05, (
        f"tail-only duplication must leave volatility RELATIVELY intact "
        f"(ratio {v_tail / v_clean:.4f}) -- that asymmetry is the point"
    )

    fixed = _dedupe_index(tail, "T")
    assert _pct_change(fixed["close"], 21) == pytest.approx(m1_clean)
    assert _compute_rsi(fixed["close"], 14) == pytest.approx(rsi_clean)


# --------------------------------------------------------------------------
# the read paths -- both of them
# --------------------------------------------------------------------------

def _fake_rows(tickers, n=6, duplicate=True):
    rows = []
    for tk in tickers:
        for i in range(n):
            r = {
                "ticker": tk, "date": f"2026-01-0{i + 1}", "open": 10.0 + i,
                "high": 11.0 + i, "low": 9.0 + i, "close": 10.5 + i,
                "volume": 1000 + i, "market": "US",
            }
            rows.append(r)
            if duplicate:
                rows.append(dict(r))
    return rows


class _FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self, timeout=None):
        return self._rows


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _FakeJob(self._rows)


def test_preload_prices_returns_a_unique_index(monkeypatch):
    import backend.backtest.cache as cache

    monkeypatch.setattr(cache, "_bq_client", _FakeClient(_fake_rows(["AAA"])))
    monkeypatch.setattr(cache, "_project", "p")
    monkeypatch.setattr(cache, "_dataset", "d")
    monkeypatch.setattr(cache, "_prices_full", {})

    cache.preload_prices(["AAA"], "2026-01-01", "2026-01-31")
    df = cache._prices_full["AAA"]
    assert df.index.is_unique, "preload_prices must not hand out a doubled index"
    assert len(df) == 6


def test_cached_prices_returns_a_unique_index(monkeypatch):
    import backend.backtest.cache as cache

    rows = [
        {k: v for k, v in r.items() if k != "ticker"}
        for r in _fake_rows(["BBB"])
    ]
    monkeypatch.setattr(cache, "_bq_client", _FakeClient(rows))
    monkeypatch.setattr(cache, "_project", "p")
    monkeypatch.setattr(cache, "_dataset", "d")
    monkeypatch.setattr(cache, "_prices_full", {})
    monkeypatch.setattr(cache, "_prices_cache", {})

    df = cache.cached_prices("BBB", "2026-01-01", "2026-01-31")
    assert df.index.is_unique, "cached_prices must not hand out a doubled index"
    assert len(df) == 6


def test_both_read_paths_are_covered():
    """Guarding one of two structurally identical call sites is not guarding.

    Step 86.59 paid three evaluation cycles for exactly this, so it is asserted
    rather than assumed: `_dedupe_index` must be called from BOTH read paths.
    """
    import inspect

    import backend.backtest.cache as cache

    for fn in (cache.preload_prices, cache.cached_prices):
        src = inspect.getsource(fn)
        assert "_dedupe_index" in src, (
            f"{fn.__name__} does not de-duplicate; both read paths must"
        )
