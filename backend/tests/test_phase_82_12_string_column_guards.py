"""phase-82.12 -- the BigQuery schema oracle, and the guards it makes possible.

READ THIS BEFORE JUDGING THE RESULT AS "SUSPICIOUSLY CLEAN".

The step asked for a sweep of vacuous type guards. The research gate ran it and the
answer is ZERO remaining defects of that shape -- the 82.0 fix closed the only one. So a
"look how many bugs I found" outcome is not available here, and manufacturing one would
be worse than reporting none. What these tests do instead is make "zero" a MEASURED
result rather than an assertion:

  * the oracle is derived from LIVE table schemas (33 tables / 477 columns), never a
    hand-written list, and every stage asserts non-empty -- because a scanner that scans
    nothing produces output identical to a clean codebase;
  * the instrument is RECALL-TESTED against the known pre-fix defect, so "it found
    nothing" is distinguishable from "it cannot see anything";
  * the one fixed guard in the class is driven with the PRODUCTION column type read
    FROM THE ORACLE, not a hardcoded literal.

The sweep DID find a live defect of the same root cause with a different symptom -- a
query selecting columns that do not exist -- and that is pinned here too.
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import pytest

from backend.db import schema_oracle as so

ORACLE = so.load_snapshot()


# ---------------------------------------------------------------------------
# CRITERION 1 -- scope derived from live schemas, asserted non-empty
# ---------------------------------------------------------------------------

def test_oracle_is_non_empty_and_covers_the_real_datasets():
    assert ORACLE, "the schema oracle is EMPTY -- every consumer would look valid"
    n_cols = sum(len(c) for c in ORACLE.values())
    assert len(ORACLE) >= 30, f"only {len(ORACLE)} tables in the oracle"
    assert n_cols >= 400, f"only {n_cols} columns in the oracle"
    assert "financial_reports.paper_trades" in ORACLE
    assert "financial_reports.historical_macro" in ORACLE


def test_refresh_refuses_to_persist_an_empty_oracle(tmp_path, monkeypatch):
    """The failure mode that makes every other guard vacuous at once."""
    monkeypatch.setattr(so, "fetch_live_schema", lambda client=None: {})
    with pytest.raises(RuntimeError, match="EMPTY"):
        so.refresh_snapshot(client=object(), path=tmp_path / "snap.json")
    assert not (tmp_path / "snap.json").exists(), "an empty oracle was written to disk"


def test_string_column_set_is_derived_from_declared_types_not_names():
    """Criterion 1's 'not a hand-written list' clause, structurally.

    The candidate set is every column the ORACLE declares STRING. Proof that no name
    heuristic is in the gate path: columns whose names carry no date/number token at
    all are still in scope purely because of their declared type.
    """
    string_cols = so.columns_of_type(ORACLE, "STRING")
    assert string_cols, "no STRING columns derived -- the join would be vacuous"
    names = {c for _, c in string_cols}
    date_tokened = {n for n in names if any(t in n for t in ("date", "time", "_at", "ts"))}
    assert names - date_tokened, (
        "every STRING column happens to carry a date/time token, so this test cannot "
        "distinguish a type-derived scope from a name-derived one"
    )


def test_derived_scope_counters_are_all_non_empty():
    """A checker that scans nothing reports exactly what a clean codebase reports."""
    sql = so.derive_scope(ORACLE)
    py = so.derive_python_scope(ORACLE)

    assert sql["files_scanned"] > 0, "scanned zero files"
    assert sql["columns_in_oracle"] > 0, "zero columns in the oracle"
    assert sql["sql_literals"] > 0, "found zero SQL literals -- the extractor is broken"
    assert sql["tables_resolved"] > 0, (
        "resolved zero tables from SQL -- the fully-qualified-table regex is broken "
        "(this exact failure produced a false 'clean' result during research)"
    )
    assert py["files_scanned"] > 0
    assert py["string_columns_in_oracle"] > 0


def test_derived_scope_is_non_empty():
    """Criterion 1: 'the derived scope is asserted non-empty'."""
    py = so.derive_python_scope(ORACLE)
    assert py["hits"], (
        "the derived scope is EMPTY. That is not evidence of a clean codebase -- it is "
        "evidence the instrument sees nothing. Recall-test it before believing it."
    )


# ---------------------------------------------------------------------------
# RECALL TEST -- can the instrument see the defect it was built for?
# ---------------------------------------------------------------------------

# The phase-25.D7 guard exactly as it stood BEFORE 82.0 repaired it: a row read
# bound to a name, then isinstance-tested against `date`. `historical_macro.date`
# is declared STRING, so this branch could never execute.
_PRE_FIX_SOURCE = '''
from datetime import date
def preload_macro(rows):
    max_date = None
    for row in rows:
        rd = row.get("date")
        if isinstance(rd, date):
            if max_date is None or rd > max_date:
                max_date = rd
    return max_date
'''

# The shape 82.0 replaced it with: coerce FIRST, then compare. Correct, and must
# NOT be flagged -- a sweep that cries wolf on the fix is worse than no sweep.
_FIXED_SOURCE = '''
from datetime import date as _date, datetime as _dt
def _coerce_date(v):
    if isinstance(v, _dt):
        return v.date()
    if isinstance(v, _date):
        return v
    if isinstance(v, str):
        return _date.fromisoformat(v[:10])
    return None
def preload_macro(rows):
    for row in rows:
        # Testing the PRODUCTION type is the correct thing to do on a STRING
        # column, and must never be reported as a date-guard defect. It is also
        # RESOLVABLE by the instrument, which is what gives the `assert not
        # fixed_hits` half of the discrimination test a failing mutation:
        # drop the isinstance_other filter and this becomes a false positive.
        if isinstance(row.get("date"), str):
            rd = _coerce_date(row.get("date"))
    return rd
'''


def test_instrument_detects_the_known_pre_fix_defect(tmp_path):
    """Recall test. Without this, 'found nothing' is indistinguishable from blind."""
    f = tmp_path / "prefix_defect.py"
    f.write_text(_PRE_FIX_SOURCE, encoding="utf-8")
    reads = so._row_key_reads(ast.parse(f.read_text(encoding="utf-8")))
    contexts = {(k, ctx) for _, k, ctx in reads}
    assert ("date", "isinstance_date") in contexts, (
        f"the scanner did NOT flag the known pre-fix defect; it found {contexts!r}. "
        "Every 'clean' result from this instrument is worthless until this passes."
    )
    assert ORACLE["financial_reports.historical_macro"]["date"]["type"] == "STRING", (
        "precondition: the oracle must declare this column STRING for the pre-fix "
        "guard to be vacuous at all"
    )


# Shapes of the SAME defect class. The cycle-1 Q/A ran 8 of these and measured that
# the instrument saw only ONE -- which made "the sweep found nothing" unfalsifiable.
# The binder now resolves through attribute access, AnnAssign, tuple unpack, walrus
# and call arguments. This table IS the recall envelope, asserted rather than prosed.
_RECALL_SHAPES: dict[str, tuple[str, bool]] = {
    "subscript_then_isinstance": (
        'from datetime import date\ndef f(row):\n    rd = row["date"]\n'
        '    return isinstance(rd, date)\n', True),
    "get_then_isinstance": (
        'from datetime import date\ndef f(row):\n    rd = row.get("date")\n'
        '    return isinstance(rd, date)\n', True),
    "attribute_access": (
        'from datetime import date\ndef f(row):\n'
        '    return isinstance(row.date, date)\n', True),
    "annassign": (
        'from datetime import date\ndef f(row):\n    rd: date = row["date"]\n'
        '    return isinstance(rd, date)\n', True),
    "tuple_unpack": (
        'from datetime import date\ndef f(row):\n'
        '    a, b = row["date"], row["value"]\n    return isinstance(a, date)\n', True),
    "helper_wrapped_call": (
        'from datetime import date\ndef f(row, parse):\n'
        '    return isinstance(parse(row["date"]), date)\n', True),
    # The TWO-STATEMENT bind-through-call form. The cycle-2 Q/A measured that this
    # is the exact shape all four real production coercers use (cache.py:137/382,
    # reconciliation.py:41, paper_round_trips.py:39, wash_sale_filter.py:30-32), so
    # an envelope that omitted it was not characterising the instrument at all.
    "bind_through_call_then_isinstance": (
        'from datetime import date\ndef f(row, parse):\n    rd = parse(row["date"])\n'
        '    return isinstance(rd, date)\n', True),
    # A call between the row read and an ARITHMETIC use is the FIX shape, not the
    # defect shape -- a coercer's job is to change the type -- so it must NOT be
    # flagged. Treating it like the isinstance case took the scope 5 -> 18 sites,
    # all 13 new ones correctly-coerced values.
    "bind_through_call_then_arith": (
        'from datetime import datetime\ndef f(a, b):\n'
        '    d1 = datetime.fromisoformat(a.get("date", ""))\n'
        '    d2 = datetime.fromisoformat(b.get("date", ""))\n'
        '    return (d2 - d1).days\n', False),
    "walrus": (
        'from datetime import date\ndef f(row):\n'
        '    return (rd := row["date"]) and isinstance(rd, date)\n', True),
    # KNOWN AND ACCEPTED MISS, asserted so it cannot silently change: implicit date
    # semantics via attribute access on a bound row value, with no isinstance and no
    # comparison. Catching it needs type inference, not an AST walk.
    "attribute_on_bound_value": (
        'def f(row):\n    rd = row["date"]\n    return rd.year\n', False),
}


@pytest.mark.parametrize("shape", sorted(_RECALL_SHAPES))
def test_recall_envelope_is_measured_not_assumed(shape, tmp_path):
    """THE anti-Overgeneralization guard.

    "The sweep found nothing" is only meaningful alongside "and here is exactly what
    it can see". Each shape is asserted SEEN or MISSED; if the instrument's recall
    ever changes in either direction, this fails and forces the completeness claim in
    the handoff to be restated.
    """
    src, should_see = _RECALL_SHAPES[shape]
    reads = so._row_key_reads(ast.parse(src))
    # Cycle-2 Q/A: the previous form was `A or B` where A is a strict SUBSET of B,
    # so A was dead and the envelope asserted only "the key was seen in some
    # context" -- not that the isinstance_date CONTEXT is recognised, which is the
    # property section 11 claims it measures. Proven by killing the context regex:
    # `seen` stayed True. Now the context is asserted directly.
    seen = any(k == "date" and c == "isinstance_date" for _, k, c in reads)
    assert seen is should_see, (
        f"recall changed for {shape!r}: expected seen={should_see}, got {seen}. "
        "Update the envelope AND the completeness wording in experiment_results."
    )


def test_instrument_discriminates_coercing_from_vacuous_helper():
    """The cycle-1 Q/A proved the previous version of this test was ILLUSORY.

    It asserted the 82.0 coercing form is not flagged -- but the instrument could not
    resolve a row value through ANY call, so it returned [] for the correct coercing
    form AND for a genuinely vacuous helper-wrapped isinstance. It demonstrated
    blindness to both, not discrimination between them, and no mutation could turn it
    red for the property it claimed.

    Now that call arguments resolve, the two forms must come apart -- which is what
    makes the anti-cry-wolf claim mean something.
    """
    vacuous = (
        'from datetime import date\n'
        'def passthrough(v):\n    return v\n'
        'def f(rows):\n'
        '    for row in rows:\n'
        '        if isinstance(passthrough(row.get("date")), date):\n'
        '            pass\n'
    )
    fixed_hits = [r for r in so._row_key_reads(ast.parse(_FIXED_SOURCE))
                  if r[2] == "isinstance_date"]
    vacuous_hits = [r for r in so._row_key_reads(ast.parse(vacuous))
                    if r[2] == "isinstance_date"]

    assert not fixed_hits, f"the correct coercing form was flagged: {fixed_hits!r}"
    assert vacuous_hits, (
        "a helper-wrapped isinstance on a STRING column was NOT flagged -- the "
        "instrument is blind to both forms, so it is not discriminating, it is "
        "just silent"
    )


def test_string_equality_is_not_treated_as_date_or_number_semantics(tmp_path):
    """`status == "done"` on a STRING column is correct and must not be in scope.

    Measured: counting equality produced 175 hits across 61 sites, burying the two
    real ones. Criterion 1 scopes to columns consumed as DATES OR NUMBERS.
    """
    f = tmp_path / "equality.py"
    f.write_text('def f(row):\n    return row["status"] == "done"\n', encoding="utf-8")
    reads = so._row_key_reads(ast.parse(f.read_text(encoding="utf-8")))
    assert reads == [], f"string equality was scoped as date/number semantics: {reads!r}"


def test_ordering_comparison_and_numeric_arithmetic_are_in_scope(tmp_path):
    f = tmp_path / "ordering.py"
    f.write_text(
        'def f(row):\n'
        '    a = row["date"] >= "2026-01-01"\n'
        '    b = row["value"] * 2\n'
        '    c = row["ticker"] + "x"\n'
        '    return a, b, c\n',
        encoding="utf-8",
    )
    reads = so._row_key_reads(ast.parse(f.read_text(encoding="utf-8")))
    ctx = {(k, c) for _, k, c in reads}
    assert ("date", "order_compare") in ctx
    assert ("value", "arith") in ctx
    assert ("ticker", "arith") not in ctx, "string concatenation is not numeric semantics"


# ---------------------------------------------------------------------------
# CRITERION 2 -- every guard in scope classified, with file:line
# ---------------------------------------------------------------------------

# Classification of every member of the derived scope. VACUOUS / CORRECT /
# NEEDS-COERCION. Keyed by (file, column) so a NEW site cannot appear without
# this table being updated -- see the completeness test below.
CLASSIFICATION: dict[tuple[str, str], dict] = {
    ("backend/services/outcome_tracker.py", "analysis_date"): {
        "verdict": "CORRECT",
        "line": 100,
        "why": (
            "Reads `analysis_results.analysis_date`, which the oracle declares "
            "TIMESTAMP -- so the isinstance(datetime) branch DOES execute. The code "
            "handles both shapes explicitly (:101 datetime passthrough, :104 "
            "fromisoformat(str(...))). It surfaced in the scope only because the same "
            "column NAME is STRING on financial_reports.outcome_tracking; the Python "
            "side matches on column name across tables, which is a disclosed "
            "over-approximation resolved by reading, never by a name filter."
        ),
    },
    ("backend/backtest/cache.py", "report_date"): {
        "verdict": "CORRECT",
        "line": 530,
        "why": (
            "`str(r.get(\"report_date\", \"\")) <= cutoff_date` -- an EXPLICIT str() "
            "coercion followed by a lexical comparison against an ISO-8601 cutoff. "
            "historical_fundamentals.report_date is declared STRING, both operands "
            "are zero-padded YYYY-MM-DD, so lexical ordering is chronological "
            "ordering. Correct by construction."
        ),
    },
    ("backend/backtest/cache.py", "date"): {
        "verdict": "CORRECT",
        "line": 590,
        "why": (
            "`str(entry[\"date\"]) > cutoff_date` -- same explicit-str() + lexical-ISO "
            "pattern on historical_prices.date (declared STRING). Note this is the "
            "SAME MODULE as the 82.0 defect and is NOT the same shape: 82.0's bug was "
            "an isinstance test that could never be true, whereas this comparison "
            "operates on the production type directly."
        ),
    },
    ("backend/tools/sec_insider.py", "date"): {
        "verdict": "CORRECT",
        "line": 226,
        "why": (
            "`t[\"date\"] >= cutoff_date` where cutoff_date is "
            "strftime('%Y-%m-%d') (:160). Both sides are zero-padded ISO-8601, so "
            "lexical ordering equals chronological ordering -- correct BY "
            "CONSTRUCTION, the known-good pattern the sweep must not cry wolf on. "
            "Additionally these dicts come from SEC Form-4 parsing, not from a BQ "
            "row at all."
        ),
    },
}


def test_every_derived_scope_member_is_classified():
    """Criterion 2, and a completeness trap: a NEW site fails until classified."""
    py = so.derive_python_scope(ORACLE)
    sites = {(h["file"], h["column"]) for h in py["hits"]}
    assert sites, "empty scope -- see test_derived_scope_is_non_empty"
    unclassified = sites - set(CLASSIFICATION)
    assert not unclassified, (
        f"derived scope members with no classification: {sorted(unclassified)}. "
        "Criterion 2 requires every guard in scope be classified vacuous / correct / "
        "needs-coercion with file:line."
    )
    stale = set(CLASSIFICATION) - sites
    assert not stale, (
        f"classification entries for sites no longer in scope: {sorted(stale)} -- "
        "the table is describing code that has moved"
    )
    for key, entry in CLASSIFICATION.items():
        assert entry["verdict"] in {"VACUOUS", "CORRECT", "NEEDS-COERCION"}
        assert entry["why"].strip(), f"{key} has no rationale"


def test_classified_line_numbers_still_point_at_a_row_read():
    """file:line claims rot. Re-derive rather than trusting the table."""
    root = Path(__file__).resolve().parents[2]
    for (rel, column), entry in CLASSIFICATION.items():
        src = (root / rel).read_text(encoding="utf-8")
        reads = so._row_key_reads(ast.parse(src))
        lines = {ln for ln, k, _ in reads if k == column}
        assert lines, f"{rel}: no row read of {column!r} found at all"
        assert any(abs(ln - entry["line"]) <= 6 for ln in lines), (
            f"{rel}:{entry['line']} no longer points at a read of {column!r}; "
            f"actual lines {sorted(lines)}"
        )


# ---------------------------------------------------------------------------
# CRITERION 3 -- vacuous hits fixed or queued; the NAME-variant defect is queued
# ---------------------------------------------------------------------------

def test_no_vacuous_hit_is_left_unfixed_and_unqueued():
    vacuous = [k for k, v in CLASSIFICATION.items() if v["verdict"] == "VACUOUS"]
    if vacuous:
        import json
        mp = json.loads((Path(__file__).resolve().parents[2] / ".claude/masterplan.json")
                        .read_text(encoding="utf-8"))
        text = json.dumps(mp)
        for rel, column in vacuous:
            assert rel in text and column in text, (
                f"VACUOUS hit {rel}:{column} is neither fixed nor named in a queued step"
            )


def test_query_selecting_nonexistent_columns_is_detected():
    """The live defect the sweep found: the NAME variant of the same root cause.

    `nightly_outcome_rebuild`'s source selects paper_trades.timestamp and
    .realized_pnl. Neither exists, BigQuery 400s, and a fail-open `except` returns
    [] -- so the job has been running on zero rows, silently.
    """
    cols = ORACLE["financial_reports.paper_trades"]
    assert "timestamp" not in cols, "schema changed: `timestamp` now exists"
    assert "realized_pnl" not in cols, "schema changed: `realized_pnl` now exists"
    assert "created_at" in cols and "realized_pnl_pct" in cols, (
        "the real columns are gone -- re-derive before trusting this test"
    )

    found = so.derive_scope(ORACLE)["unknown_columns"]
    flagged = {(u["identifier"], u["file"]) for u in found}
    assert ("timestamp", "backend/slack_bot/jobs/_production_fns.py") in flagged, (
        f"the unknown-column checker missed the live defect; it found {flagged!r}"
    )
    assert ("realized_pnl", "backend/slack_bot/jobs/_production_fns.py") in flagged


def test_the_nonexistent_column_defect_is_queued_as_its_own_step():
    """It is a live data defect with its own consequence -- queued, not absorbed.

    STRUCTURAL, not a substring scan. The first version of this test asserted
    `"_production_fns" in masterplan_text`, which PASSED BEFORE THE STEP WAS QUEUED --
    both tokens already occur in unrelated steps (measured: `_production_fns` x1,
    `nightly_outcome_rebuild` x7). That is the same guard-that-cannot-fail class this
    whole step is about, so it is fixed rather than disclosed: walk the actual step
    objects and require ONE step that names every part of the defect signature.
    """
    import json
    mp = json.loads(
        (Path(__file__).resolve().parents[2] / ".claude/masterplan.json").read_text(
            encoding="utf-8")
    )
    steps = [s for ph in mp["phases"] for s in ph.get("steps", [])]
    assert steps, "no masterplan steps parsed -- the check would be vacuous"

    signature = ("_production_fns", "paper_trades", "timestamp", "realized_pnl")
    matches = [
        s for s in steps
        if s.get("status") not in {"done", "dropped", "superseded"}
        and all(tok in s.get("name", "") for tok in signature)
    ]
    assert matches, (
        "no OPEN masterplan step names the full defect signature "
        f"{signature}; a queued step must let an executor with no memory of this "
        "sweep find the exact query and columns"
    )
    assert any(s.get("verification", {}).get("criteria") for s in matches), (
        "the queued step has no verification criteria, so it could never be closed"
    )


def test_alias_is_not_reported_as_a_missing_column():
    """`SAFE_CAST(x AS FLOAT64) AS pnl` invents `pnl`; flagging it is a false positive."""
    found = so.derive_scope(ORACLE)["unknown_columns"]
    assert not [u for u in found if u["identifier"] == "pnl"], (
        "a SELECT alias was reported as a missing column"
    )


# ---------------------------------------------------------------------------
# CRITERION 4 -- the fixed guard, fed the PRODUCTION column type from the oracle
# ---------------------------------------------------------------------------

def _macro_row(series_id: str, day: dt.date) -> dict:
    """A row shaped like production. The date type is read FROM THE ORACLE, so if
    the column is ever migrated STRING -> DATE this fixture breaks loudly instead
    of quietly testing the wrong thing."""
    declared = ORACLE["financial_reports.historical_macro"]["date"]["type"]
    if declared == "STRING":
        value: object = day.isoformat()
    elif declared == "DATE":
        value = day
    else:  # pragma: no cover
        raise AssertionError(f"unhandled declared type {declared!r} for historical_macro.date")
    return {"series_id": series_id, "value": 1.0, "date": value, "realtime_start": None}


def test_fixture_emits_the_type_the_oracle_declares():
    """Precondition on the fixture ITSELF -- a fixture that cannot represent the
    production failure cannot prove anything about the guard in front of it."""
    declared = ORACLE["financial_reports.historical_macro"]["date"]["type"]
    assert declared == "STRING", (
        f"historical_macro.date is now {declared}; the 82.0 defect and this fixture "
        "both assumed STRING -- re-derive before editing"
    )
    row = _macro_row("DGS10", dt.date(2026, 1, 2))
    assert isinstance(row["date"], str), (
        "the fixture is emitting a Python date where production emits str -- this is "
        "exactly the fixture that CANNOT represent the production failure"
    )


def _run_preload(rows, monkeypatch):
    from backend.backtest import cache as bq_cache

    class _RS:
        def result(self, timeout=None):
            return iter(rows)

    class _Client:
        def query(self, *a, **k):
            return _RS()

    monkeypatch.setattr(bq_cache, "_macro_full", {}, raising=False)
    monkeypatch.setattr(bq_cache, "_bq_client", _Client(), raising=False)
    monkeypatch.setattr(bq_cache, "_table", lambda name: f"proj.ds.{name}", raising=False)
    return bq_cache.preload_macro()


def test_guard_fires_on_stale_data_of_the_production_type(monkeypatch):
    """POSITIVE: the repaired guard must REFUSE stale macro.

    Pre-82.0 this returned a non-zero count: `isinstance(rd, date)` was False for
    every production str row, so the staleness branch never ran and stale data was
    cached silently.
    """
    old = dt.date.today() - dt.timedelta(days=400)
    inserted = _run_preload([_macro_row("DGS10", old)], monkeypatch)
    assert inserted == 0, (
        f"preload_macro cached {inserted} rows of 400-day-old macro; the staleness "
        "gate did not fire on the PRODUCTION column type"
    )


def test_guard_does_not_fire_on_fresh_data_of_the_production_type(monkeypatch):
    """NEGATIVE: the guard must not pass by always refusing."""
    fresh = dt.date.today() - dt.timedelta(days=1)
    inserted = _run_preload([_macro_row("DGS10", fresh)], monkeypatch)
    assert inserted == 1, (
        f"preload_macro refused fresh macro (returned {inserted}); a guard that "
        "always refuses is not a guard"
    )


def test_guard_fails_closed_on_unparseable_dates(monkeypatch):
    """A date column it cannot evaluate must refuse, not silently disable the gate."""
    bad = {"series_id": "DGS10", "value": 1.0, "date": "not-a-date", "realtime_start": None}
    assert _run_preload([bad], monkeypatch) == 0


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_snapshot_carries_the_datasets_it_claims():
    import json
    payload = json.loads(so.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert payload["project"] == so.PROJECT
    assert set(payload["datasets"]) == set(so.DATASETS)
    assert payload["tables"], "snapshot has no tables"
