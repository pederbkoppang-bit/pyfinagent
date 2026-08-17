"""phase-86.109 -- the freshness alarm's weekend false positive and page storm.

The step's own criteria demand REPRODUCTION BY EXECUTION, so nothing here
asserts from source text. Three properties get the most attention:

  - **detection must stay calendar-blind.** Criterion 3 requires a genuinely
    stale WEEKDAY source to still classify red and still page. The calendar
    gates NOTIFICATION only; if it ever reaches `_band()` these tests break.
  - **a withheld page must be DEFERRED, not dropped.** The first draft of the
    gate updated the baseline before the gate, which would absorb a weekend
    red into "already known" and never page it. That is a weekend mute that
    silently becomes permanent, and it is what `test_..._deferred_not_dropped`
    exists to prevent.
  - **a read path must not page.** The three HTTP handlers are asserted to pass
    `emit_alarm=False` by DRIVING them against a fake, not by grepping.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from backend.services import cycle_health as ch


# ── 1. criterion 1 -- reproduce the false positive on the CURRENT code ──


def test_band_is_a_pure_ratio_and_reds_a_healthy_weekend_gap():
    """A Friday-15:00 write read on Monday-08:00 is ~64h. With the 24h
    interval the ratio is 2.67, so `_band` returns red purely from elapsed
    calendar time -- correct arithmetic, wrong question."""
    friday_close_to_monday_open = 64 * 3600.0
    interval = 86400.0
    assert friday_close_to_monday_open / interval > ch.CRITICAL_RATIO
    assert ch._band(friday_close_to_monday_open, interval) == "red"

    # And the same code is green on a normal weekday cadence, so the red above
    # is about the WEEKEND, not about the function being broken.
    assert ch._band(20 * 3600.0, interval) == "green"


def test_band_has_no_day_of_week_term_after_the_fix():
    """Criterion 3's core property: DETECTION must be calendar-blind.

    An earlier version of this test asserted `first == second` over two
    identical `_band` calls made in the same instant. That is a tautology: it
    is true of every possible implementation, including a calendar-aware one,
    and a Q/A proved it by injecting a `_band` that returns "green" on
    weekends -- the mutant SURVIVED all 17 tests, on a Monday, invisibly.

    Two guards replace it, neither of which depends on the day this suite
    happens to run:

    1. **Structural.** `_band`'s own source must contain no calendar or
       day-of-week reference. This is a source scan and observes no behaviour,
       but it is the only check that fires on a Tuesday for a bug that only
       manifests on a Saturday.
    2. **Behavioural, over the whole week.** `_band` is a pure function of
       (age, interval), so calling it cannot depend on today -- but a
       calendar-aware version would read the clock. Freezing the clock to each
       day of the week and asserting the answer never moves catches that.
    """
    import inspect
    import re

    src = inspect.getsource(ch._band)
    banned = re.compile(r"weekday|is_trading_day|is_us_trading_day|_NYSE_TZ|"
                        r"datetime|date\(|calendar", re.I)
    hit = banned.search(src)
    assert hit is None, (
        f"_band references {hit.group(0)!r} -- the calendar has reached "
        f"DETECTION. It belongs on the notification leg only; a calendar-aware "
        f"band makes a Friday-dead writer indistinguishable from an idle "
        f"weekend.\n{src}"
    )

    # Behavioural: the answer must be identical on every day of the week.
    import datetime as _dt

    interval = 86400.0
    # 10h=0.42 green, 40h=1.67 amber (>=WARN 1.5, <CRITICAL 2.0), 64h=2.67 red,
    # 200h=8.33 red. All three bands are covered so a mutation that collapses
    # the band vocabulary is visible here too. (An earlier draft used 30h and
    # expected amber; 30/24 = 1.25 is green -- the expectation was wrong, not
    # the code, and the assertion caught it.)
    ages = (10 * 3600.0, 40 * 3600.0, 64 * 3600.0, 200 * 3600.0)
    baseline = [ch._band(a, interval) for a in ages]
    real_dt = ch.datetime if hasattr(ch, "datetime") else None
    for offset in range(7):  # Mon..Sun
        day = _dt.date(2026, 8, 17) + _dt.timedelta(days=offset)

        class _FrozenDT(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime(day.year, day.month, day.day, 12, 0, tzinfo=tz)

        if real_dt is not None:
            ch.datetime = _FrozenDT
        try:
            assert [ch._band(a, interval) for a in ages] == baseline, (
                f"_band's answer changed on {day} ({day.strftime('%A')}) -- "
                "detection is day-dependent"
            )
        finally:
            if real_dt is not None:
                ch.datetime = real_dt

    assert baseline == ["green", "amber", "red", "red"], baseline
    # A stale WEEKDAY source is still red -- detection is untouched.
    assert ch._band(50 * 3600.0, interval) == "red"


# ── 2. the trading-day helper is ONE definition, reused ────────────


def test_trading_day_helper_is_shared_not_duplicated():
    """Criterion 2: the fix must reuse the mechanism 51.3 already applied to
    the digests, not grow a second definition. The digest wrapper must now
    resolve to the same callable body."""
    from backend.backtest.markets import is_us_trading_day_now
    import backend.slack_bot.scheduler as sched

    calls = []
    import backend.backtest.markets as markets

    real = markets.is_us_trading_day_now
    try:
        markets.is_us_trading_day_now = lambda m="US": calls.append(m) or True
        assert sched._is_us_trading_day_now() is True
    finally:
        markets.is_us_trading_day_now = real
    assert calls == ["US"], "the digest wrapper no longer delegates to the shared helper"
    assert callable(is_us_trading_day_now)


@pytest.mark.parametrize(
    "day, expected",
    [
        ("2026-08-15", False),  # Saturday
        ("2026-08-16", False),  # Sunday
        ("2026-08-18", True),   # Tuesday
    ],
)
def test_is_trading_day_weekend_and_weekday(day, expected):
    from backend.backtest.markets import is_trading_day

    assert is_trading_day(day, "US") is expected


def test_trading_day_helper_fails_OPEN(monkeypatch):
    """A calendar-library failure must never SUPPRESS a page."""
    import backend.backtest.markets as markets

    monkeypatch.setattr(markets, "get_trading_calendar", lambda *_a, **_k: None)
    assert markets.is_us_trading_day_now("US") is True


# ── 3. criterion 4 -- the read paths do not page (DRIVEN) ──────────


def _drive_freshness_handler(monkeypatch, module, attr):
    """Call a real HTTP handler and capture the emit_alarm it forwards."""
    seen = {}

    def _fake_compute(bq, cycle_interval_sec, *, emit_alarm=True):
        seen["emit_alarm"] = emit_alarm
        return {"sources": {}, "overall_band": "green"}

    monkeypatch.setattr(module, attr, _fake_compute, raising=False)
    return seen


def test_paper_trading_freshness_route_does_not_page(monkeypatch):
    import backend.api.paper_trading as pt

    seen = _drive_freshness_handler(monkeypatch, pt, "compute_freshness")
    monkeypatch.setattr(pt, "get_bq_client", lambda: object())
    asyncio.run(pt.get_freshness())
    assert seen["emit_alarm"] is False, "the canonical read path still pages"


@pytest.mark.parametrize(
    "handler_name",
    ["get_observability_freshness", "get_observability_data_freshness"],
)
def test_observability_freshness_aliases_do_not_page(monkeypatch, handler_name):
    import backend.api.observability_api as obs
    import backend.services.cycle_health as real_ch
    import backend.db.bigquery_client as bqmod

    seen = {}

    def _fake_compute(bq, cycle_interval_sec, *, emit_alarm=True):
        seen["emit_alarm"] = emit_alarm
        return {"sources": {}, "overall_band": "green"}

    # The aliases import inside the function body, so patch at the source.
    monkeypatch.setattr(real_ch, "compute_freshness", _fake_compute)
    monkeypatch.setattr(bqmod, "BigQueryClient", lambda *_a, **_k: object())
    asyncio.run(getattr(obs, handler_name)())
    assert seen["emit_alarm"] is False, f"{handler_name} still pages"


def test_compute_freshness_still_pages_when_asked(monkeypatch):
    """Anti-vacuity control -- and this time it actually drives the thing.

    An earlier version of this test never called `compute_freshness` at all:
    it built a `_BQ` class it did not use, saved and restored
    `_fire_freshness_alarm` without ever triggering it, and asserted only that
    the kwarg default was still True. A Q/A made the alarm branch inert
    (`if False and emit_alarm ...`) and this test SURVIVED -- so the claim that
    it guarded the alarm path was false.

    Now it forces an all-red payload through the REAL `compute_freshness` and
    counts REAL `_fire_freshness_alarm` invocations: 1 when asked to emit, 0
    when not. If the alarm ever goes inert, the first assertion fails.
    """
    import backend.services.cycle_health as c

    fired = []
    monkeypatch.setattr(c, "_fire_freshness_alarm", lambda sources: fired.append(sources))
    # Force every source red regardless of BQ.
    monkeypatch.setattr(c, "_bq_max_event_age", lambda *a, **k: 10_000_000.0, raising=False)
    monkeypatch.setattr(c, "_read_heartbeat", lambda *a, **k: {}, raising=False)

    class _BQ:
        def _pt_table(self, name):
            return f"p.d.{name}"

        class client:
            @staticmethod
            def query(*a, **k):
                raise RuntimeError("no BQ in this test")

    out = c.compute_freshness(_BQ(), 86400.0, emit_alarm=True)
    assert out["overall_band"] == "red", out["overall_band"]
    assert len(fired) == 1, f"the alarm did not fire when asked: {fired!r}"

    fired.clear()
    out2 = c.compute_freshness(_BQ(), 86400.0, emit_alarm=False)
    assert out2["overall_band"] == "red", "the payload must be IDENTICAL -- only the side effect goes"
    assert fired == [], f"emit_alarm=False still fired: {fired!r}"


# ── 4. criterion 2 + 3 -- the notifier gate, driven both ways ──────


def _run_evaluator(monkeypatch, *, trading_day, red):
    """Drive the REAL `run_freshness_check` with its own injection points.

    `bq`/`settings`/`notify` are injectable by design, so this reaches the real
    control flow -- the gate, the baseline update, the deferral -- with only
    BigQuery and Slack replaced.
    """
    import backend.services.freshness_cron as fc
    import backend.services.cycle_health as c
    import backend.backtest.markets as markets

    pages = []
    monkeypatch.setattr(markets, "is_us_trading_day_now", lambda m="US": trading_day)
    monkeypatch.setattr(
        c, "compute_freshness",
        lambda bq, interval, *, emit_alarm=True: {
            "sources": {t: {"band": "red", "ratio": 3.0, "last_tick_age_sec": 1,
                            "interval_sec": 86400} for t in red},
            "overall_band": "red" if red else "green",
        },
    )
    return fc, pages, (lambda: fc.run_freshness_check(
        bq=object(), settings=types.SimpleNamespace(),
        notify=lambda **kw: pages.append(kw["error_type"]),
    ))


def test_weekend_newly_red_does_not_page(monkeypatch):
    fc, pages, run = _run_evaluator(monkeypatch, trading_day=False, red={"paper_trades"})
    fc.reset_transition_state()
    run()
    assert pages == [], "a non-trading day still paged"


def test_weekday_newly_red_STILL_pages(monkeypatch):
    """Criterion 3's control: the fix is not a blanket suppression."""
    fc, pages, run = _run_evaluator(monkeypatch, trading_day=True, red={"paper_trades"})
    fc.reset_transition_state()
    run()
    assert pages == ["freshness_critical_paper_trades"], pages


def test_weekend_suppression_is_DEFERRED_not_dropped(monkeypatch):
    """The bug the first draft of this gate had: committing the baseline before
    the gate absorbs the withheld source, so it never pages at all -- a weekend
    mute that silently becomes permanent."""
    fc, pages, run = _run_evaluator(monkeypatch, trading_day=False, red={"paper_trades"})
    fc.reset_transition_state()
    run()
    assert pages == []

    fc2, pages2, run2 = _run_evaluator(monkeypatch, trading_day=True, red={"paper_trades"})
    run2()  # NOTE: no reset -- this is the same process, the next tick
    assert pages2 == ["freshness_critical_paper_trades"], (
        "the weekend-withheld source was absorbed into the baseline and never paged"
    )


def test_a_source_that_recovers_over_the_weekend_never_pages(monkeypatch):
    """The converse: deferral must not resurrect a page for a source that
    healed while it was withheld."""
    fc, pages, run = _run_evaluator(monkeypatch, trading_day=False, red={"paper_trades"})
    fc.reset_transition_state()
    run()
    assert pages == []

    fc2, pages2, run2 = _run_evaluator(monkeypatch, trading_day=True, red=set())
    run2()
    assert pages2 == []


def test_steady_state_red_on_a_weekday_pages_only_once(monkeypatch):
    """The transition gate must still work -- the calendar change must not
    convert an edge-triggered alarm into a level-triggered one."""
    fc, pages, run = _run_evaluator(monkeypatch, trading_day=True, red={"paper_trades"})
    fc.reset_transition_state()
    run()
    run()
    run()
    assert pages == ["freshness_critical_paper_trades"], pages


def test_calendar_failure_fails_open_and_still_pages(monkeypatch):
    """A calendar-library error must never suppress a page."""
    import backend.services.freshness_cron as fc
    import backend.services.cycle_health as c
    import backend.backtest.markets as markets

    pages = []

    def _explode(*_a, **_k):
        raise RuntimeError("exchange_calendars is gone")

    monkeypatch.setattr(markets, "is_us_trading_day_now", _explode)
    monkeypatch.setattr(
        c, "compute_freshness",
        lambda bq, interval, *, emit_alarm=True: {
            "sources": {"paper_trades": {"band": "red", "ratio": 3.0,
                                         "last_tick_age_sec": 1, "interval_sec": 86400}},
            "overall_band": "red",
        },
    )
    fc.reset_transition_state()
    fc.run_freshness_check(
        bq=object(), settings=types.SimpleNamespace(),
        notify=lambda **kw: pages.append(kw["error_type"]),
    )
    assert pages == ["freshness_critical_paper_trades"], (
        "a calendar failure SUPPRESSED a page -- the polarity is inverted"
    )
