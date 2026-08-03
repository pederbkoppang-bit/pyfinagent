"""phase-82.15 -- point-in-time macro reads.

82.0 added a `realtime_start` vintage column and populated it, but nothing read
it, so every macro-conditioned backtest still consumed values that did not exist
at the cutoff. These tests guard the read side on BOTH paths.
"""
from __future__ import annotations

from datetime import date

import pytest

from backend.backtest import cache


# ── fixtures ─────────────────────────────────────────────────────────

def _rows():
    """A series whose NEWEST dated observation was not yet published.

    GDP dated 2023-01-01 with a true vintage of 2023-04-28: dated before a
    2023-02-01 cutoff, but unknown then. The correct answer at that cutoff is
    the OLDER 2022-07-01 row.
    """
    return {
        "GDP": [
            {"value": 27.0, "date": "2023-01-01", "realtime_start": date(2023, 4, 28)},
            {"value": 26.0, "date": "2022-07-01", "realtime_start": date(2022, 10, 27)},
            {"value": 25.0, "date": "2022-01-01", "realtime_start": date(2022, 4, 28)},
        ]
    }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(cache, "_macro_cache", {}, raising=False)
    monkeypatch.setattr(cache, "_cache_stats", {"hits": 0, "misses": 0}, raising=False)


def _fast_path(monkeypatch, rows, cutoff, pit=True):
    monkeypatch.setattr(cache, "_macro_full", rows, raising=False)
    monkeypatch.setattr(cache, "_pit_enabled", lambda: pit, raising=False)
    cache._macro_cache.clear()
    return cache.cached_macro(cutoff)


# ── the helper ───────────────────────────────────────────────────────

def test_effective_vintage_takes_the_earlier_of_stamp_and_derived_lag():
    """MIN(realtime_start, date + lag) selects the EARLIER availability date.

    NOTE, because an earlier version of this docstring claimed the opposite:
    earlier availability makes a row visible SOONER, which is ANTI-conservative
    with respect to look-ahead. The rule is still correct -- a true vintage
    should govern, and discarding a far-future backfill stamp is what keeps the
    sample from being blanked -- but it is conservative only where the lag
    upper-bounds the real release delay. See the module header and step 82.18.
    """
    # true stamp earlier than the derived date -> stamp wins
    assert cache._effective_vintage("GDP", "2026-04-01", date(2026, 3, 22)) == date(2026, 3, 22)
    # write-date stamp far in the future for a 2019 row -> the lag governs
    assert cache._effective_vintage("GDP", "2019-01-01", date(2026, 3, 22)) == date(2019, 5, 6)


def test_effective_vintage_falls_back_to_the_lag_when_the_stamp_is_absent():
    """DOCUMENTED DECISION for the pre-migration / NULL population: a missing
    vintage resolves to `date + per-series publication lag`. It is neither
    silently dropped (which would blank the whole 2018-2025 sample -- measured
    0/4729 rows survive a strict filter at 2020-01-01) nor silently included
    (which is the look-ahead this step exists to remove)."""
    assert cache._effective_vintage("DGS10", "2019-01-02", None) == date(2019, 1, 3)
    assert cache._effective_vintage("GDP", "2019-01-01", None) == date(2019, 5, 6)


def test_effective_vintage_fails_closed_on_an_unparseable_date():
    assert cache._effective_vintage("GDP", "not-a-date", None) is None


def test_cached_macro_omits_a_series_whose_date_cannot_be_parsed(monkeypatch):
    """Q/A cycle-2: the HELPER's fail-closed return was covered, but the
    CALLER's handling of it was not -- flipping `vintage is None or ...` to
    `vintage is not None and ...` (fail-OPEN) left the suite green. That would
    admit a row whose availability cannot be established at all."""
    # The date string must sort <= cutoff (so it survives the lexicographic
    # date check) yet be unparseable (so it reaches the vintage branch).
    # "not-a-date" does NOT work: it sorts ABOVE "2023-02-01", so the row is
    # dropped by the date check and the test would pass for the wrong reason --
    # which is exactly what a mutation run caught here.
    rows = {"GDP": [{"value": 1.0, "date": "2022-99-99", "realtime_start": None}]}
    assert "2022-99-99" <= "2023-02-01", "fixture must clear the date check"
    got = _fast_path(monkeypatch, rows, "2023-02-01")
    assert "GDP" not in got, (
        "a row with an unparseable observation date was admitted; availability "
        "cannot be established for it, so it must be excluded"
    )


def test_production_flag_read_is_exercised_and_defaults_on():
    """Q/A cycle-2: every other test monkeypatches `_pit_enabled`, so the real
    settings read ran in ZERO tests -- replacing its body with `return False`
    left 13/13 green while the entire step was inert. This is the one assertion
    that touches the unpatched production path.

    Mutate the FLAG READ, not just the guard it gates.
    """
    from backend.config.settings import get_settings

    assert get_settings().macro_point_in_time_enabled is True, (
        "macro_point_in_time_enabled is not defaulting True; macro-conditioned "
        "backtests would silently regain publication-lag look-ahead"
    )
    assert cache._pit_enabled() is True, (
        "_pit_enabled() does not read through to the setting -- the flag is "
        "disconnected and the filter would be inert in production"
    )


def test_every_ingested_series_has_a_publication_lag():
    from backend.backtest.data_ingestion import FRED_SERIES
    missing = set(FRED_SERIES) - set(cache.MACRO_PUBLICATION_LAG_DAYS)
    assert not missing, f"series ingested but absent from the lag table: {sorted(missing)}"


# ── criterion 1 + 2a: the PRELOADED FAST PATH excludes unknown rows ──

def test_fast_path_excludes_an_observation_not_yet_published(monkeypatch):
    got = _fast_path(monkeypatch, _rows(), "2023-02-01")
    assert got["GDP"]["date"] == "2022-07-01", (
        "returned the 2023-01-01 observation, which was not published until "
        "2023-04-28 -- that is look-ahead"
    )


def test_fast_path_does_not_break_on_a_date_only_match(monkeypatch):
    """The pre-fix loop `break`s on the first row dated <= cutoff. A vintage
    predicate bolted onto that shape would drop the series ENTIRELY rather than
    falling through to the older row that WAS known."""
    got = _fast_path(monkeypatch, _rows(), "2023-02-01")
    assert "GDP" in got, "the series vanished instead of falling back to an older row"


def test_fast_path_returns_the_newest_row_that_was_actually_known(monkeypatch):
    got = _fast_path(monkeypatch, _rows(), "2023-05-01")
    assert got["GDP"]["date"] == "2023-01-01", (
        "after its publication date the newest row should be visible again"
    )


# ── criterion 4: the pre-fix behaviour DIFFERED ──────────────────────

def test_regression_pin_the_filter_changes_the_answer(monkeypatch):
    """Guard against a vacuous fix: with the filter OFF the same query must
    return the un-published row, and with it ON it must not."""
    off = _fast_path(monkeypatch, _rows(), "2023-02-01", pit=False)
    on = _fast_path(monkeypatch, _rows(), "2023-02-01", pit=True)
    assert off["GDP"]["date"] == "2023-01-01", (
        "with the filter OFF the pre-fix look-ahead should reproduce"
    )
    assert on["GDP"]["date"] != off["GDP"]["date"], (
        "the filter did not change the answer -- it is inert"
    )


# ── criterion 2b: the BQ FALLBACK carries the same predicate ─────────

def test_bq_fallback_sql_includes_the_vintage_predicate(monkeypatch):
    """The fallback runs when nothing is preloaded. Fixing only the fast path
    leaves every ad-hoc lookup reading the future."""
    captured = {}

    class _Job:
        def result(self, *a, **k):
            return []

    class _Client:
        def query(self, sql, job_config=None):
            captured["sql"] = sql
            captured["params"] = job_config.query_parameters if job_config else []
            return _Job()

    monkeypatch.setattr(cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(cache, "_bq_client", _Client(), raising=False)
    monkeypatch.setattr(cache, "_pit_enabled", lambda: True, raising=False)
    cache._macro_cache.clear()
    cache.cached_macro("2023-02-01")

    sql = captured["sql"]
    assert "realtime_start" in sql, "the fallback query ignores the vintage column"
    assert "DATE(@cutoff)" in sql, (
        "`date` is a STRING column but `realtime_start` is a DATE; comparing a "
        "STRING-bound @cutoff against a DATE without DATE() is a type error"
    )

    # Q/A cycle-1 BLOCKER: the two substring assertions above were the SOLE
    # coverage of this criterion's fallback half, and they are satisfied by SQL
    # that reads the future. Three mutants survived them -- inverting the
    # comparison (`<=` -> `>=`), swapping LEAST for GREATEST, and zeroing the
    # lag CASE -- and the inverted one was executed against live BigQuery,
    # returning exactly the 184-day look-ahead rows this step exists to remove.
    # Criterion 2 demands the two halves be "asserted separately so fixing one
    # and not the other cannot pass", so the SHAPE of the predicate is pinned.
    assert ") <= DATE(@cutoff)" in sql, (
        "the vintage comparison is not `) <= DATE(@cutoff)` -- an inverted or "
        "rewritten predicate would admit exactly the rows this filter removes"
    )
    assert "LEAST(" in sql and "GREATEST(" not in sql, (
        "availability must be LEAST(stamp, date+lag); GREATEST would pick the "
        "later date and blank the historical sample"
    )

    # Pin the lag table to ONE source of truth: the SQL must carry every series'
    # lag from MACRO_PUBLICATION_LAG_DAYS. A zeroed or drifted CASE cannot pass.
    for sid, lag in cache.MACRO_PUBLICATION_LAG_DAYS.items():
        assert f"WHEN '{sid}' THEN {lag}" in sql, (
            f"the fallback SQL does not carry the publication lag for {sid} "
            f"({lag}d) -- the two read paths would disagree"
        )


# ── criterion 1: the preload SITES must actually carry the vintage ───

def test_preload_select_requests_the_vintage_column(monkeypatch):
    """Q/A cycle-1: dropping `realtime_start` from the preload SELECT left every
    test green, because the fixture's row is also excluded by the lag. That makes
    criterion 1's exclusion attributable to the lag rather than to the stamp, and
    a regression here would silently degrade every series to a lag-only estimate.
    """
    captured = {}

    class _Job:
        def result(self, *a, **k):
            return []

    class _Client:
        def query(self, sql, *a, **k):
            captured["sql"] = sql
            return _Job()

    monkeypatch.setattr(cache, "_bq_client", _Client(), raising=False)
    monkeypatch.setattr(cache, "_macro_full", {}, raising=False)
    cache.preload_macro()
    assert "realtime_start" in captured["sql"], (
        "preload_macro does not SELECT realtime_start, so the fast path can "
        "never filter on a true vintage"
    )


def test_preloaded_rows_carry_the_vintage(monkeypatch):
    """The SELECT fetching the column is not enough -- the row builder must keep
    it. Dropping the key survived the cycle-1 suite."""
    from datetime import date as _d

    rows = [
        {"series_id": "GDP", "value": 27.0, "date": "2023-01-01",
         "realtime_start": _d(2023, 4, 28)},
        {"series_id": "GDP", "value": 26.0, "date": "2022-07-01",
         "realtime_start": _d(2022, 10, 27)},
    ]

    class _Job:
        def result(self, *a, **k):
            return rows

    class _Client:
        def query(self, *a, **k):
            return _Job()

    monkeypatch.setattr(cache, "_bq_client", _Client(), raising=False)
    monkeypatch.setattr(cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(cache, "MACRO_SERIES_MAX_AGE_DAYS", {"GDP": 10**6}, raising=False)
    cache.preload_macro()

    entries = cache._macro_full.get("GDP") or []
    assert entries, "preload stored nothing"
    assert all("realtime_start" in e for e in entries), (
        "stored rows dropped realtime_start; the fast path would silently fall "
        "back to lag-only estimates with a green suite"
    )
    assert entries[0]["realtime_start"] == _d(2023, 4, 28)


def test_bq_fallback_omits_the_predicate_when_disabled(monkeypatch):
    captured = {}

    class _Job:
        def result(self, *a, **k):
            return []

    class _Client:
        def query(self, sql, job_config=None):
            captured["sql"] = sql
            return _Job()

    monkeypatch.setattr(cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(cache, "_bq_client", _Client(), raising=False)
    monkeypatch.setattr(cache, "_pit_enabled", lambda: False, raising=False)
    cache._macro_cache.clear()
    cache.cached_macro("2023-02-01")
    assert "realtime_start" not in captured["sql"], (
        "the flag is inert on the fallback path"
    )


# ── the sample must SURVIVE (the naive strict filter destroys it) ────

def test_the_filter_does_not_blank_the_historical_sample(monkeypatch):
    """A strict `realtime_start <= cutoff` returns 0/4729 rows at every cutoff
    before 2026-03-22, because 82.0 backfilled our WRITE date. That would blank
    all six macro features across the whole 2018-2025 backtest window -- and do
    it silently, since historical_data.py guards on `if macro:`."""
    rows = {
        "GDP": [{"value": 20.0, "date": "2019-01-01", "realtime_start": date(2026, 3, 22)}],
        "DGS10": [{"value": 2.5, "date": "2019-06-27", "realtime_start": date(2026, 3, 22)}],
    }
    got = _fast_path(monkeypatch, rows, "2019-06-28")
    assert set(got) == {"GDP", "DGS10"}, (
        f"the historical sample was blanked: {sorted(got)}. The write-date stamp "
        "carries no information about 2019; the per-series lag must govern"
    )
