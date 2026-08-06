"""phase-82.13 -- a macro REFUSAL must not degrade the engine silently.

WHY THE OBVIOUS GUARD IS WRONG. `preload_macro` returns an int, and `0` is AMBIGUOUS:
it means "the table was empty" on one path and "I REFUSED to cache this" on two others,
while the already-warm path returns a POSITIVE cached total having loaded nothing. So
`if not cache.preload_macro(): abort` cannot tell a refusal from an empty table and
misfires on every warm optimizer iteration. These tests pin the ambiguity itself, not
just the happy path.

WHY IT ONLY MATTERS NOW. Before 82.0 armed the staleness gate, the refusal branches were
unreachable -- `_macro_full` was always populated -- so the discarded return genuinely
never mattered. 82.0 made them reachable, which is what turned a dormant code smell into
a live degradation path.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.backtest import cache as bq_cache

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_macro_state():
    bq_cache.reset_macro_status()
    yield
    bq_cache.reset_macro_status()
    bq_cache._macro_full.clear()
    bq_cache._macro_cache.clear()


def _rows(day: str, series: str = "DGS10"):
    return [{"series_id": series, "value": 1.0, "date": day, "realtime_start": None}]


class _RS:
    def __init__(self, rows): self._rows = rows
    def result(self, timeout=None): return iter(self._rows)


class _Client:
    def __init__(self, rows): self._rows = rows
    def query(self, *a, **k): return _RS(self._rows)


def _run_preload(rows, monkeypatch):
    monkeypatch.setattr(bq_cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client(rows), raising=False)
    monkeypatch.setattr(bq_cache, "_table", lambda n: f"proj.ds.{n}", raising=False)
    return bq_cache.preload_macro()


# ---------------------------------------------------------------------------
# The ambiguity that makes the naive guard wrong
# ---------------------------------------------------------------------------

def test_zero_return_is_ambiguous_between_empty_and_refused(monkeypatch):
    """Both return 0. Only the STATUS distinguishes them.

    This is the whole reason the fix is a status accessor rather than a check on
    the return value: a guard keyed on `== 0` would treat an empty table as a
    refusal and put a perfectly ordinary run into macro-free mode.
    """
    import datetime as dt

    empty_rc = _run_preload([], monkeypatch)
    empty_status = bq_cache.macro_load_status()
    empty_refused = bq_cache.macro_was_refused()

    bq_cache.reset_macro_status()
    old = (dt.date.today() - dt.timedelta(days=400)).isoformat()
    stale_rc = _run_preload(_rows(old), monkeypatch)
    stale_status = bq_cache.macro_load_status()
    stale_refused = bq_cache.macro_was_refused()

    assert empty_rc == 0 and stale_rc == 0, (
        f"precondition: both paths must return 0 (got {empty_rc}, {stale_rc}); "
        "if they no longer do, the ambiguity this step exists for has changed"
    )
    assert empty_status["outcome"] == "empty"
    assert stale_status["outcome"] == "refused_stale"
    assert empty_refused is False, "an empty table is NOT a refusal"
    assert stale_refused is True


def test_warm_path_returns_positive_and_is_not_a_refusal(monkeypatch):
    """The already-warm early return reports a POSITIVE total having loaded nothing.

    It fires on every optimizer iteration under skip_cache_clear=True, so a guard
    that mistook it for anything else would fire constantly.
    """
    # The init assert (cache.py) runs BEFORE the warm early-return, so a client is
    # required even on the path that issues no query -- reflect production, don't
    # bypass it.
    monkeypatch.setattr(bq_cache, "_macro_full", {"DGS10": [{"date": "2026-01-01"}]},
                        raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client([]), raising=False)
    rc = bq_cache.preload_macro()
    assert rc == 1
    assert bq_cache.macro_load_status()["outcome"] == "warm"
    assert bq_cache.macro_was_refused() is False


# ---------------------------------------------------------------------------
# CRITERION 1 -- a refusal is surfaced explicitly, not proceeded past
# ---------------------------------------------------------------------------

def test_engine_branches_on_the_status_accessor_not_the_return_value():
    """The engine must not decide on the int. Asserted structurally on the source,
    because the decision itself is what the criterion is about."""
    src = (_REPO / "backend/backtest/backtest_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "macro_was_refused"]
    assert calls, (
        "backtest_engine.py never consults cache.macro_was_refused(); a refusal "
        "would be indistinguishable from an empty table and the engine would "
        "proceed silently -- the exact discard-the-return behaviour of this defect"
    )
    tests = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             for c in ast.walk(n.test)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "macro_was_refused"]
    assert tests, "macro_was_refused() is called but never branched on"


class _CountingClient:
    """COUNTS queries instead of raising.

    The first version of this guard raised AssertionError from `query` -- but
    `cached_macro` wraps the call in `except Exception`, which swallows it. The
    guard could not observe its own defect and mutant M3 (ignore macro-free mode)
    SURVIVED it. Counting is visible through the except.
    """

    def __init__(self): self.n = 0

    def query(self, *a, **k):
        self.n += 1
        return _RS([])


def test_refusal_puts_the_cache_into_macro_free_mode_so_the_fallback_cannot_fire():
    """THE CONSEQUENCE. A 'degraded mode' that leaves the per-cutoff fallback armed
    does not fix anything -- cached_macro would still issue one BQ round-trip per
    distinct cutoff."""
    bq_cache._macro_full.clear()
    bq_cache._macro_cache.clear()
    bq_cache.set_macro_unavailable(True)
    client = _CountingClient()
    bq_cache._bq_client = client

    out = bq_cache.cached_macro("2024-01-02")
    assert client.n == 0, (
        f"cached_macro issued {client.n} BQ queries in macro-free mode -- the "
        "per-cutoff fallback was NOT suppressed, so the degraded run is still slow"
    )
    assert out == {}, f"macro-free mode must yield no macro features, got {out!r}"


def test_fallback_still_fires_when_macro_is_available():
    """Mirror of the above -- suppression must be conditional, not permanent."""
    bq_cache._macro_full.clear()
    bq_cache._macro_cache.clear()
    bq_cache.set_macro_unavailable(False)
    client = _CountingClient()
    bq_cache._bq_client = client
    bq_cache.cached_macro("2024-01-03")
    assert client.n == 1, (
        f"the per-cutoff fallback fired {client.n} times in normal mode (expected "
        "1) -- the suppression flag is stuck on and every run is silently macro-free"
    )


# ---------------------------------------------------------------------------
# CRITERION 2 -- the run result records macro availability
# ---------------------------------------------------------------------------

def test_result_carries_data_availability_by_default():
    """phase-82.13 guard; the assertion was widened by phase-82.21.

    ORIGINAL INTENT, unchanged and still enforced: a default result must claim
    availability, so ONLY an explicit refusal marks a run degraded.

    WHAT CHANGED: this used to assert `== {"macro": True}` by exact equality,
    which also pinned the dict's SIZE. That was incidental over-specification --
    82.13's own design note says the field is defaulted precisely so it can be
    extended without disturbing existing construction sites. phase-82.21 adds
    `"fundamentals"` on exactly that basis. The rewrite below is strictly
    STRONGER on the intent (it now requires that NO key defaults to False, for
    any key present now or later) while no longer breaking when the record
    legitimately grows.
    """
    from backend.backtest.backtest_engine import BacktestResult

    r = BacktestResult()
    assert r.data_availability.get("macro") is True, (
        "a default result must claim macro availability, so only an explicit "
        "refusal marks a run degraded"
    )
    assert r.data_availability, "the record must not default to empty"
    assert all(v is True for v in r.data_availability.values()), (
        f"no availability key may default to a degraded value: {r.data_availability}"
    )


def test_report_surfaces_availability_at_top_level_and_inside_analytics():
    """Both placements are required. strategy_backtest_adapter and
    strategy_candidate_producer read ONLY report["analytics"], so a top-level key
    alone would leave a macro-free run looking normal to exactly the code that
    compares candidates against one another."""
    from backend.backtest.analytics import generate_report
    from backend.backtest.backtest_engine import BacktestResult

    degraded = BacktestResult()
    degraded.data_availability = {
        "macro": False, "macro_outcome": "refused_stale",
        "macro_detail": "DGS10 past SLA", "macro_rows": 0,
    }
    rep = generate_report(degraded)

    assert rep["data_availability"]["macro"] is False
    assert rep["data_availability"]["macro_outcome"] == "refused_stale"
    assert rep["analytics"]["macro_available"] is False, (
        "report['analytics'] does not carry macro_available; the two candidate "
        "comparators read only that sub-dict and would pool a macro-free run with "
        "macro-complete ones"
    )

    normal = generate_report(BacktestResult())
    assert normal["data_availability"]["macro"] is True
    assert normal["analytics"]["macro_available"] is True


# ---------------------------------------------------------------------------
# CRITERION 3 -- every preload_* call site enumerated FROM THE SOURCE
# ---------------------------------------------------------------------------

def _preload_call_sites() -> list[dict]:
    """AST, not grep. A text search cannot tell a CALL from a mention in a comment,
    a docstring, or a local function of the same name -- all three exist in this
    repo. `discarded` iff the call's direct parent is an ast.Expr statement.
    """
    sites: list[dict] = []
    # Scoped to the source trees. Walking _REPO.rglob("*.py") also crawls .venv
    # (tens of thousands of files) and makes this test take minutes; preload_* is a
    # first-party symbol, so third-party packages cannot contain a call site.
    roots = [_REPO / d for d in ("backend", "scripts", "tests", "dev")]
    candidates = [q for root in roots if root.is_dir() for q in root.rglob("*.py")]
    assert candidates, "no source files found to scan -- the roots are wrong"
    for path in sorted(candidates):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        discarded_calls = {
            id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        }
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else getattr(n.func, "id", ""))
            if not name.startswith("preload_"):
                continue
            sites.append({
                "file": path.relative_to(_REPO).as_posix(),
                "line": n.lineno,
                "callee": name,
                "discarded": id(n) in discarded_calls,
            })
    return sites


def test_preload_call_sites_are_enumerated_and_classified():
    sites = _preload_call_sites()
    assert sites, (
        "the preload_* enumeration is EMPTY -- that is an instrument failure, not a "
        "clean codebase; a scanner that finds nothing looks identical to one with "
        "nothing to find"
    )
    for s in sites:
        assert isinstance(s["discarded"], bool)
        assert s["file"] and s["line"] > 0 and s["callee"].startswith("preload_")

    used = [s for s in sites if not s["discarded"]]
    assert used, "no call site uses a preload_* return value at all"


def test_the_engine_macro_call_site_now_uses_its_return_value():
    """The named instance. Before this step it was a bare expression statement."""
    sites = [s for s in _preload_call_sites()
             if s["file"] == "backend/backtest/backtest_engine.py"
             and s["callee"] == "preload_macro"]
    assert len(sites) == 1, f"expected exactly one engine preload_macro site, got {sites}"
    assert sites[0]["discarded"] is False, (
        "backend/backtest/backtest_engine.py still DISCARDS preload_macro's return "
        f"(line {sites[0]['line']})"
    )


def test_remaining_discarded_sites_are_recorded_not_silently_tolerated():
    """Criterion 3 asks for classification, not for fixing all of them. The other
    discards are out of this step's scope (82.40 owns the two sibling preloads), so
    they are enumerated here rather than quietly ignored."""
    discarded = sorted(
        (s["file"], s["callee"]) for s in _preload_call_sites() if s["discarded"]
    )
    for f, callee in discarded:
        assert callee in {"preload_prices", "preload_fundamentals", "preload_macro"}, (
            f"unexpected preload_* callee {callee} discarded at {f}"
        )
    assert not [
        (f, c) for f, c in discarded
        if f == "backend/backtest/backtest_engine.py" and c == "preload_macro"
    ], "the engine macro site must no longer be in the discarded set"


# ---------------------------------------------------------------------------
# CRITERION 4 -- success is unaffected; the guard cannot pass by always aborting
# ---------------------------------------------------------------------------

def test_successful_preload_reports_loaded_and_is_not_refused(monkeypatch):
    import datetime as dt

    fresh = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    rc = _run_preload(_rows(fresh), monkeypatch)

    assert rc == 1, f"fresh macro was not loaded (rc={rc})"
    assert bq_cache.macro_load_status()["outcome"] == "loaded"
    assert bq_cache.macro_was_refused() is False
    assert bq_cache.macro_is_loaded() is True
    assert bq_cache.macro_is_unavailable() is False, (
        "a successful load left the cache in macro-free mode -- the guard is "
        "passing by always degrading"
    )


def test_docstring_no_longer_claims_the_return_is_a_row_count():
    """The old docstring said 'Returns the total number of rows loaded', which is
    false on 3 of the 5 paths and is what made the discard look harmless."""
    doc = bq_cache.preload_macro.__doc__ or ""
    assert "NOT a simple row count" in doc, (
        "preload_macro's docstring still describes its return as a row count"
    )


# ---------------------------------------------------------------------------
# ENGINE-LEVEL guards. The cache-only tests above verify the primitives; these
# drive the engine's REAL code path (`_preload_macro_and_record`). Without them,
# mutants that broke the ENGINE's wiring -- not suppressing the fallback on a
# refusal, not recording availability, degrading on success -- all survived a
# fully green suite, because every guard stopped at the cache boundary.
# ---------------------------------------------------------------------------

def _engine():
    """Construct without running __init__ (which reaches for BigQuery)."""
    from backend.backtest.backtest_engine import BacktestEngine

    e = BacktestEngine.__new__(BacktestEngine)
    e._current_window_id = 0
    e.progress_callback = None
    return e


def test_engine_records_a_refusal_and_enters_macro_free_mode(monkeypatch):
    import datetime as dt

    old = (dt.date.today() - dt.timedelta(days=400)).isoformat()
    monkeypatch.setattr(bq_cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client(_rows(old)), raising=False)
    monkeypatch.setattr(bq_cache, "_table", lambda n: f"proj.ds.{n}", raising=False)

    avail = _engine()._preload_macro_and_record()

    assert avail["macro"] is False, "engine did not mark the run macro-free"
    assert avail["macro_outcome"] == "refused_stale"
    assert avail["macro_detail"], "the refusal reason was not recorded"
    assert bq_cache.macro_is_unavailable() is True, (
        "the engine recorded the refusal but did NOT suppress the per-cutoff "
        "fallback -- the run is labelled slow instead of being made fast"
    )


def test_engine_leaves_a_successful_run_untouched(monkeypatch):
    import datetime as dt

    fresh = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    monkeypatch.setattr(bq_cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client(_rows(fresh)), raising=False)
    monkeypatch.setattr(bq_cache, "_table", lambda n: f"proj.ds.{n}", raising=False)

    avail = _engine()._preload_macro_and_record()

    assert avail["macro"] is True
    assert avail["macro_outcome"] == "loaded"
    assert bq_cache.macro_is_unavailable() is False, (
        "a successful load left the cache macro-free -- the guard passes by "
        "always degrading"
    )


def test_engine_does_not_treat_an_empty_table_as_a_refusal(monkeypatch):
    """The ambiguity, at the engine boundary: empty is not refused."""
    monkeypatch.setattr(bq_cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client([]), raising=False)
    monkeypatch.setattr(bq_cache, "_table", lambda n: f"proj.ds.{n}", raising=False)

    avail = _engine()._preload_macro_and_record()

    assert avail["macro_outcome"] == "empty"
    assert bq_cache.macro_is_unavailable() is False, (
        "an empty table was treated as a refusal -- an ordinary run would be "
        "silently forced macro-free"
    )


def test_run_backtest_labels_the_result_from_that_record():
    """Criterion 2 at the construction site: the result must be built FROM the
    availability record, not from a hardcoded default."""
    import ast as _ast

    src = (_REPO / "backend/backtest/backtest_engine.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)
    ctor = [n for n in _ast.walk(tree) if isinstance(n, _ast.Call)
            and getattr(n.func, "id", "") == "BacktestResult"]
    assert ctor, "BacktestResult is never constructed in the engine"
    kw = {k.arg: k.value for c in ctor for k in c.keywords if k.arg}
    assert "data_availability" in kw, (
        "BacktestResult is constructed without data_availability -- a macro-free "
        "run would be indistinguishable from a normal one"
    )
    rendered = _ast.unparse(kw["data_availability"])
    assert "_availability" in rendered, (
        f"data_availability is not derived from the availability record "
        f"(got {rendered!r}) -- a literal would label every run identically"
    )
