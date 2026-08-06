"""phase-82.51 -- publication-lag embargo on every fundamentals read.

`report_date` is the PERIOD END, not the date the figure became public. A
company's Q2 ending 2025-06-30 is not filed for weeks, so the pre-82.51 filter
`report_date <= cutoff` let a backtest at cutoff 2025-07-05 read a number no
market participant could have seen. Every fundamentals-derived feature carried
forward-looking information.

WHY A FIXED EMBARGO AND NOT A REAL FILING DATE. The table HAS a `filing_date`
column, it is NOT NULL on 100% of 4798 rows -- and it is a verbatim COPY of
`report_date` (`data_ingestion.py:278` sets it that way because yfinance has no
true filing date). Switching the filter to it would produce a byte-identical
result set and a byte-identical backtest while looking like the correct fix.
That decoy is why `test_filing_date_is_still_a_decoy_and_must_not_be_filtered_on`
exists below. A real per-row filing date is phase-82.50's job.

THE SEAM. The two read paths cannot share a filter directly -- one is a Python
list comprehension, the other a BigQuery predicate -- but they share
`_embargoed_cutoff` because of an exact identity:

    report_date + N <= cutoff   <=>   report_date <= cutoff - N

so the CUTOFF is shifted once instead of every row. Criterion 3 demands both
paths be driven independently, which is what
`test_both_read_paths_agree_on_the_same_fixture` does.
"""

from __future__ import annotations

import ast
import pathlib
import re
from unittest import mock

import pytest

from backend.backtest import cache
from backend.backtest import fundamentals_coverage as fc

REPO = pathlib.Path(__file__).resolve().parents[2]

TICKER = "TESTCO"

#: A quarter that ENDED 2025-06-30. Under a 60-day embargo it becomes visible
#: on 2025-08-29 -- so at a cutoff of 2025-07-05 it must NOT be readable, which
#: is precisely what the pre-82.51 filter got wrong.
ROW_NOT_YET_PUBLIC = {"ticker": TICKER, "report_date": "2025-06-30", "total_revenue": 999.0}

#: A quarter that ended 2025-03-31, hence visible from 2025-05-30. At a cutoff
#: of 2025-07-05 it is demonstrably public and MUST still be returned.
ROW_ALREADY_PUBLIC = {"ticker": TICKER, "report_date": "2025-03-31", "total_revenue": 111.0}

CUTOFF = "2025-07-05"


@pytest.fixture(autouse=True)
def _clean_cache():
    """`cached_fundamentals` has three branches and two of them are caches.
    Leaking state between cases would let one branch silently answer for
    another -- the exact confound criterion 3 is written to prevent."""
    cache.clear_cache()
    yield
    cache.clear_cache()


# ---------------------------------------------------------------------------
# criterion 1 -- a not-yet-published row is EXCLUDED
# ---------------------------------------------------------------------------


def test_a_row_whose_period_ended_but_was_not_yet_filed_is_excluded():
    """FAILS against the pre-82.51 `report_date <= cutoff` filter, which would
    return both rows because 2025-06-30 <= 2025-07-05."""
    cache._fundamentals_full[TICKER] = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]

    got = cache.cached_fundamentals(TICKER, CUTOFF)
    dates = [r["report_date"] for r in got]

    assert "2025-06-30" not in dates, (
        f"a quarter ending 2025-06-30 was readable at cutoff {CUTOFF}, only "
        f"{(fc._date.fromisoformat('2025-06-30') - fc._date.fromisoformat(CUTOFF)).days * -1} "
        "days later -- no filer could have published it that fast. Got "
        f"{dates}. This is the look-ahead 82.51 exists to remove."
    )


def test_the_excluded_row_becomes_visible_once_the_embargo_has_elapsed():
    """The row is not banished, only delayed. Without this, 'exclude always'
    would satisfy the test above."""
    cache._fundamentals_full[TICKER] = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]

    # 2025-06-30 + 60 days = 2025-08-29
    before = cache.cached_fundamentals(TICKER, "2025-08-28")
    cache.clear_cache()
    cache._fundamentals_full[TICKER] = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]
    on_the_day = cache.cached_fundamentals(TICKER, "2025-08-29")

    assert "2025-06-30" not in [r["report_date"] for r in before]
    assert "2025-06-30" in [r["report_date"] for r in on_the_day], (
        "the row never becomes visible -- the embargo is a deletion, not a delay"
    )


# ---------------------------------------------------------------------------
# criterion 2 -- an already-public row is still INCLUDED
# ---------------------------------------------------------------------------


def test_a_row_that_was_demonstrably_public_is_still_included():
    """The fix must not pass by excluding everything."""
    cache._fundamentals_full[TICKER] = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]

    got = cache.cached_fundamentals(TICKER, CUTOFF)

    assert got, "the embargo returned NOTHING -- it is excluding everything"
    assert [r["report_date"] for r in got] == ["2025-03-31"]
    assert got[0]["total_revenue"] == 111.0


# ---------------------------------------------------------------------------
# criterion 3 -- BOTH read paths, driven independently, from one rule
# ---------------------------------------------------------------------------


def _drive_branch_1(rows):
    cache.clear_cache()
    cache._fundamentals_full[TICKER] = list(rows)
    return cache.cached_fundamentals(TICKER, CUTOFF)


def _drive_branch_3(rows):
    """Drive the BQ fallback with a stub that returns rows UNFILTERED.

    The stub deliberately does NOT apply the cutoff. A stub that pre-filtered
    would make this test pass no matter what the production predicate does --
    it would be testing the stub. Because the real filter lives in the SQL
    text, the value actually under test is the `@cutoff` parameter, which
    `test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter` pins
    separately.
    """
    cache.clear_cache()  # NOT just _fundamentals_full: branch 2's memo would
    # otherwise serve branch 1's result and the "independent" drive would lie.
    captured = {}

    def fake_query(query, job_config=None):
        captured["query"] = query
        captured["params"] = {p.name: p.value for p in (job_config.query_parameters or [])}
        cutoff = captured["params"]["cutoff"]
        result = mock.MagicMock()
        # Emulate BigQuery applying the WHERE clause to the parameter it was
        # GIVEN -- so the production cutoff value is what decides the outcome.
        result.result.return_value = [
            r for r in rows if str(r["report_date"]) <= cutoff
        ]
        return result

    with mock.patch.object(cache, "_bq_client", mock.MagicMock(query=fake_query)):
        out = cache.cached_fundamentals(TICKER, CUTOFF)
    return out, captured


def test_both_read_paths_agree_on_the_same_fixture():
    """criterion 3. Each path is driven on its own and the outputs compared."""
    rows = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]

    b1 = _drive_branch_1(rows)
    b3, _ = _drive_branch_3(rows)

    assert b1 == b3, (
        f"the preload path and the BQ fallback path disagree:\n"
        f"  preload:  {[r['report_date'] for r in b1]}\n"
        f"  fallback: {[r['report_date'] for r in b3]}\n"
        "one of them is not applying the embargo"
    )
    for name, out in (("preload", b1), ("fallback", b3)):
        assert "2025-06-30" not in [r["report_date"] for r in out], (
            f"the {name} path still serves the not-yet-published row"
        )


def test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter():
    """The fallback's filter is SQL text, so the checkable artifact is the
    parameter VALUE it binds -- proving the embargo reached the query and not
    merely the Python side."""
    _, captured = _drive_branch_3([ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC])

    assert captured["params"]["cutoff"] == "2025-05-06", (
        f"expected the embargoed cutoff 2025-05-06 (= {CUTOFF} - 60d), got "
        f"{captured['params']['cutoff']!r} -- the SQL path is querying with the "
        "raw cutoff"
    )
    assert "report_date <= @cutoff" in captured["query"]


def test_mutating_the_shared_seam_moves_BOTH_paths():
    """The seam is only real if one change flips both. If a mutation of
    `_embargoed_cutoff` moved only one branch, this suite would be stopping one
    seam short of the defect."""
    rows = [ROW_NOT_YET_PUBLIC, ROW_ALREADY_PUBLIC]

    with mock.patch.object(fc, "FUNDAMENTALS_EMBARGO_DAYS", 0):
        b1 = _drive_branch_1(rows)
        b3, captured = _drive_branch_3(rows)

    assert "2025-06-30" in [r["report_date"] for r in b1], (
        "branch 1 did not react to the embargo constant"
    )
    assert "2025-06-30" in [r["report_date"] for r in b3], (
        "branch 3 did not react to the embargo constant"
    )
    assert captured["params"]["cutoff"] == CUTOFF


# ---------------------------------------------------------------------------
# the decoy tripwire -- `filing_date` looks like the fix and is not
# ---------------------------------------------------------------------------


def test_filing_date_is_still_a_decoy_and_must_not_be_filtered_on():
    """`filing_date` is NOT NULL on 100% of rows and is a verbatim copy of
    `report_date`, so filtering on it is a silent no-op that would LOOK like
    the correct fix and would report a Sharpe delta of exactly 0.0000.

    This converts that latent trap into a tripwire. It fails if production
    starts filtering on `filing_date` while the producer still copies it.
    When 82.50 lands a real filing date, the producer line goes away and this
    guard should be revisited deliberately -- not deleted quietly.
    """
    ingestion = (REPO / "backend/backtest/data_ingestion.py").read_text(encoding="utf-8")
    still_a_copy = re.search(r'"filing_date":\s*report_date', ingestion)

    cache_src = (REPO / "backend/backtest/cache.py").read_text(encoding="utf-8")
    filters_on_it = re.search(r"filing_date\s*<=|<=\s*@?\w*filing", cache_src)

    if still_a_copy:
        assert not filters_on_it, (
            "cache.py filters on `filing_date` while data_ingestion.py still "
            "sets it to `report_date` verbatim -- that filter is a NO-OP that "
            "cannot exclude anything. Either land a real filing date (82.50) "
            "or keep the fixed embargo."
        )
    else:
        pytest.fail(
            "data_ingestion.py no longer copies report_date into filing_date. "
            "If 82.50 landed a real filing date, 82.51's fixed embargo should "
            "be replaced by a filing-date filter -- revisit this deliberately."
        )


# ---------------------------------------------------------------------------
# criterion 5's record, and the false-pass this fix would otherwise create
# ---------------------------------------------------------------------------


def test_effective_coverage_start_is_derived_not_a_second_literal():
    """A hardcoded '2024-08-29' would drift the moment either input changes."""
    assert fc.effective_coverage_start() == "2024-08-29"

    with mock.patch.object(fc, "FUNDAMENTALS_EMBARGO_DAYS", 90):
        assert fc.effective_coverage_start() == "2024-09-28"
    with mock.patch.object(fc, "FUNDAMENTALS_EMBARGO_DAYS", 0):
        assert fc.effective_coverage_start() == fc.FUNDAMENTALS_COVERAGE_START

    src = (REPO / "backend/backtest/fundamentals_coverage.py").read_text(encoding="utf-8")
    assert '"2024-08-29"' not in src, (
        "the effective start is hardcoded somewhere -- it must be derived"
    )


def test_the_raw_measured_coverage_start_is_left_alone():
    """82.21's snapshot is a RAW measurement. The embargo must extend the
    record, not corrupt the measurement it extends."""
    assert fc.FUNDAMENTALS_COVERAGE_START == "2024-06-30"
    assert fc.load_snapshot()["min_report_date"] == "2024-06-30"


def _engine(strategy: str, start_date: str):
    """Same shape 82.21 uses -- drive the engine's real method without BQ."""
    import datetime as dt

    from backend.backtest.backtest_engine import BacktestEngine

    e = BacktestEngine.__new__(BacktestEngine)
    e.strategy = strategy
    e.scheduler = mock.MagicMock()
    e.scheduler.start_date = dt.date.fromisoformat(start_date)
    return e


def test_a_window_inside_the_embargo_gap_is_REFUSED_not_silently_uncovered():
    """BEHAVIOURAL, because the source-scan version of this test was illusory.

    2024-07-01 is AFTER the raw coverage start (2024-06-30) but BEFORE the
    effective one (2024-08-29) -- the 60-day gap where rows exist but none is
    visible yet. Pre-82.51 semantics would call this window covered, record
    `fundamentals: True`, and hand back [] on every read: the exact false pass
    82.21 closed.

    An earlier version asserted only that the `start=` KWARG was present, which
    the 82.51 Q/A defeated by substituting `start=FUNDAMENTALS_COVERAGE_START`
    -- semantically the bug, syntactically a pass. Driving the behaviour cannot
    be fooled that way.
    """
    with pytest.raises(ValueError, match="REFUSED"):
        _engine("qarp", "2024-07-01")._preload_fundamentals_and_record()


def test_the_refusal_message_names_the_embargo_so_an_operator_can_act():
    with pytest.raises(ValueError) as exc:
        _engine("qarp", "2024-07-01")._preload_fundamentals_and_record()
    msg = str(exc.value)
    assert fc.effective_coverage_start() in msg
    assert str(fc.FUNDAMENTALS_EMBARGO_DAYS) in msg


def test_a_window_after_the_effective_start_still_runs():
    """The refusal must be caused by the GAP, not by the embargo existing."""
    rec = _engine("qarp", "2025-01-01")._preload_fundamentals_and_record()
    assert rec["fundamentals"] is True
    assert rec["fundamentals_effective_coverage_start"] == fc.effective_coverage_start()


def test_the_optimizer_drops_a_dependent_strategy_inside_the_embargo_gap():
    """BEHAVIOURAL coverage for the second consumer.

    `selectable_strategies_for_window` must agree with the engine: for a window
    inside the 60-day gap the engine raises `REFUSED`, so the optimizer must not
    offer `qarp` as selectable. Cycle 1 of this step left this consumer on the
    RAW start, so the optimizer said COVERED for windows the engine refused --
    and a source scan was its only coverage. This drives the real function.
    """
    from backend.backtest.quant_optimizer import selectable_strategies_for_window

    inside_gap = selectable_strategies_for_window("2024-07-01")
    after_start = selectable_strategies_for_window("2025-01-01")

    dependent = fc.label_fundamentals_dependent_strategies()
    assert dependent, "no fundamentals-dependent strategy derived -- check vacuous"

    for strat in dependent:
        assert strat not in inside_gap, (
            f"{strat!r} is offered as selectable for a window starting "
            "2024-07-01, but the engine REFUSES that window -- the optimizer "
            "and the engine disagree"
        )
        assert strat in after_start, (
            f"{strat!r} is missing from a window the engine accepts -- the "
            "exclusion is firing too broadly"
        )


def test_every_window_is_covered_consumer_judges_against_the_effective_start():
    """CONSUMER SWEEP -- defence in depth behind the two behavioural tests.

    Hardened against two blinding shapes the 82.51 Q/A demonstrated against the
    first version of this sweep:

      * a variable merely NAMED `effective_start` holding the RAW constant
        passed a substring check -- so the bound value is now RESOLVED through
        local assignments instead of pattern-matched;
      * aliasing the import (`window_is_covered as _wic`) removed the literal
        from the file and dropped it out of a text-derived file set -- so the
        file set is now derived from IMPORTS of the symbol under any alias.
    """
    files, aliases = [], {}
    for path in sorted((REPO / "backend").rglob("*.py")):
        if path.name.startswith("test_") or path.name == "fundamentals_coverage.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        local = {
            a.asname or a.name
            for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            and (n.module or "").endswith("fundamentals_coverage")
            for a in n.names if a.name == "window_is_covered"
        }
        if local:
            files.append(path)
            aliases[path] = local

    assert files, "no consumer of window_is_covered found -- the sweep is broken"
    assert len(files) >= 2, (
        f"only {len(files)} consumer file(s) found; the engine and the optimizer "
        "are both known consumers -- the sweep is under-deriving"
    )

    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # local name -> the expression it was last assigned, so a raw value
        # hiding behind an `effective`-sounding name is caught
        assigned: dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                assigned[n.targets[0].id] = ast.unparse(n.value)

        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = getattr(n.func, "id", getattr(n.func, "attr", None))
            if name not in aliases[path]:
                continue
            start_kw = next((k for k in n.keywords if k.arg == "start"), None)
            where = f"{path.relative_to(REPO)}:{n.lineno}"
            if start_kw is None:
                offenders.append(f"{where} no start= (judges against the RAW start)")
                continue
            expr = ast.unparse(start_kw.value)
            resolved = assigned.get(expr, expr)  # follow one hop
            if "effective_coverage_start" not in resolved:
                offenders.append(
                    f"{where} start={expr} resolves to {resolved!r}, which is "
                    "not the effective coverage start"
                )

    assert not offenders, (
        "window_is_covered consumers judging against the RAW coverage start, "
        "which disagrees with the engine's refusal and re-creates the 82.21 "
        "false pass:\n  " + "\n  ".join(offenders)
    )


def test_the_live_data_server_does_NOT_embargo():
    """The embargo answers 'what could I have known at a past cutoff'. The MCP
    data server asks 'what is true now' at `date.today()`, where every row in
    the table has already been published -- embargoing there would hide the
    most recent reported quarter from the LIVE agent pipeline for 60 days.

    Found by the 82.51 Q/A: the first cut of this step silently changed that
    live path while the artifact claimed no live behaviour changed.
    """
    src = (REPO / "backend/agents/mcp_servers/data_server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", getattr(n.func, "id", None)) == "cached_fundamentals"
    ]
    assert calls, "no cached_fundamentals call found in data_server.py -- sweep broken"

    for c in calls:
        kw = next((k for k in c.keywords if k.arg == "apply_embargo"), None)
        assert kw is not None and kw.value.value is False, (
            f"data_server.py:{c.lineno} calls cached_fundamentals without "
            "apply_embargo=False -- the live as-of-today query would hide the "
            "most recent published quarter for 60 days"
        )


def test_the_embargo_default_is_on_so_a_forgetful_caller_gets_the_safe_behaviour():
    """`apply_embargo` defaults True: a new backtest-side caller that does not
    think about leakage must get the protected path, and only a caller that
    deliberately opts out gets raw reads."""
    import inspect

    sig = inspect.signature(cache.cached_fundamentals)
    assert sig.parameters["apply_embargo"].default is True


def test_availability_record_carries_the_embargo():
    """criterion 5 + the 82.21 cross-reference: the record must say what was
    actually applied, so a stored result can be interpreted later."""
    src = (REPO / "backend/backtest/backtest_engine.py").read_text(encoding="utf-8")
    for key in ("fundamentals_embargo_days", "fundamentals_effective_coverage_start"):
        assert f'"{key}"' in src, f"availability dict does not record {key}"
    # 82.21's keys must survive -- this extends the record, not replaces it
    for key in ("fundamentals", "fundamentals_coverage_start",
                "fundamentals_window_start", "fundamentals_label_dependent"):
        assert f'"{key}"' in src, f"82.21's availability key {key} was dropped"


def test_the_embargo_is_sixty_days_and_the_reason_is_recorded():
    """criterion 5. 45 would cover the 10-Q deadline but under-cover the 10-K
    by 15 days -- and the fiscal-year-end cohort is the largest in the table."""
    assert fc.FUNDAMENTALS_EMBARGO_DAYS == 60

    src = (REPO / "backend/backtest/fundamentals_coverage.py").read_text(encoding="utf-8")
    for token in ("10-K", "10-Q", "82.50", "APPROXIMATION"):
        assert token in src, (
            f"the embargo constant's rationale does not mention {token!r} -- "
            "criterion 5 requires the reason be recorded with the choice"
        )
