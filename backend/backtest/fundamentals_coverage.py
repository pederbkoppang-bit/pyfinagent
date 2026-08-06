"""phase-82.21: what `historical_fundamentals` actually covers, and which
strategies cannot be evaluated outside that coverage.

THE DEFECT. `financial_reports.historical_fundamentals` holds 4798 rows across
503 tickers and EVERY row is dated 2024-06-30 or later -- measured live
2026-08-06, `COUNTIF(report_date < '2024-06-30') = 0`. The walk-forward window
is 2018-01-01..2025-12-31, so roughly four-fifths of the sample has no
fundamentals at all. Nothing recorded that. The root cause is the SOURCE, not a
missed backfill: `data_ingestion.ingest_fundamentals` pulls yfinance
`quarterly_financials` / `quarterly_balance_sheet`, which serve only ~5-7 recent
quarters, so re-running the ingester CANNOT deepen the table.

WHY A STRING COMPARISON IS SAFE HERE, AND ONLY HERE. `report_date` is a
BigQuery **STRING** column, so `MIN(report_date)` is a lexicographic minimum.
That is correct ONLY because the producer writes zero-padded ISO
(`data_ingestion.py:257`, `pd.Timestamp(col_date).strftime("%Y-%m-%d")`).
`_ISO_DATE_RE` below pins that format, and the guard asserts it. Do NOT add an
`isinstance(value, datetime.date)` check over rows from this table -- it can
never fire on a STRING column, which is a documented dead-guard class in this
project.

WHAT THIS MODULE IS NOT. It does not fix coverage. The operator's decision
(recorded verbatim in `handoff/current/contract_82.21.md`) is to build a SEC
EDGAR XBRL ingester, queued as its own step. Until that lands, this module
makes the hole VISIBLE rather than silent.

ASCII-only logger messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
SNAPSHOT_PATH = _HERE / "_fundamentals_coverage.json"

#: The producer's format. A lexicographic MIN over `report_date` is only
#: meaningful while every value matches this.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: What production BELIEVES the coverage start to be. The guard compares this
#: against the checked-in snapshot, so editing one without re-measuring the
#: other fails the suite. Re-measure with `measure_live_coverage()`.
FUNDAMENTALS_COVERAGE_START = "2024-06-30"

_TABLE = "sunny-might-477607-p8.financial_reports.historical_fundamentals"


def load_snapshot(path: Path | None = None) -> dict:
    """Read the checked-in coverage snapshot. Raises if it is missing or
    malformed -- an absent snapshot must not degrade to an optimistic default,
    because the whole point is that nobody re-measured."""
    p = Path(path) if path is not None else SNAPSHOT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("min_report_date", "n_rows", "n_tickers", "measured_at", "date_format"):
        if key not in data:
            raise ValueError(f"coverage snapshot missing required key: {key}")
    if not _ISO_DATE_RE.match(str(data["min_report_date"])):
        raise ValueError(
            "coverage snapshot min_report_date is not zero-padded ISO "
            f"({data['min_report_date']!r}); a lexicographic MIN over the STRING "
            "column would be meaningless"
        )
    return data


def coverage_start(path: Path | None = None) -> str:
    """The measured earliest `report_date`, from the snapshot."""
    return str(load_snapshot(path)["min_report_date"])


def snapshot_drift(path: Path | None = None) -> Optional[str]:
    """Return a description of any disagreement between the snapshot and what
    production believes, else None."""
    snap = coverage_start(path)
    if snap != FUNDAMENTALS_COVERAGE_START:
        return (
            f"FUNDAMENTALS_COVERAGE_START={FUNDAMENTALS_COVERAGE_START!r} but the "
            f"measured snapshot says {snap!r}; re-measure with "
            "measure_live_coverage() and update BOTH, or the code's belief is stale"
        )
    return None


def measure_live_coverage(bq: Any = None) -> dict:
    """Re-measure coverage against live BigQuery. NOT called by the default test
    path -- the offline snapshot is what the guard reads, so the suite stays fast
    and free. Use this to refresh the snapshot deliberately."""
    if bq is None:
        from google.cloud import bigquery

        bq = bigquery.Client(project="sunny-might-477607-p8")
    sql = f"""
        SELECT COUNT(*) AS n_rows, COUNT(DISTINCT ticker) AS n_tickers,
               MIN(report_date) AS min_report_date, MAX(report_date) AS max_report_date
        FROM `{_TABLE}`
    """
    row = next(iter(bq.query(sql).result(timeout=30)))
    return {
        "min_report_date": row["min_report_date"],
        "max_report_date": row["max_report_date"],
        "n_rows": int(row["n_rows"]),
        "n_tickers": int(row["n_tickers"]),
    }


def window_is_covered(window_start: str, start: str | None = None) -> bool:
    """True iff `window_start` is at or after the coverage start.

    Both are zero-padded ISO strings, so the string comparison is the date
    comparison -- see the module docstring.
    """
    ref = start if start is not None else FUNDAMENTALS_COVERAGE_START
    if not _ISO_DATE_RE.match(str(window_start)):
        raise ValueError(
            f"window_start must be zero-padded ISO, got {window_start!r}"
        )
    return str(window_start) >= str(ref)


# ---------------------------------------------------------------------------
# Which strategies cannot be labelled without fundamentals -- DERIVED, never
# hardcoded. A literal `{"qarp"}` would go stale the next time a strategy is
# added, which is exactly the measure-don't-assert defect class this project
# keeps re-learning. The rule is written down so it can be argued with:
#
#   Let F = the feature keys assigned ONLY inside the `if fundamentals:` block
#   of `historical_data.build_feature_vector`. A strategy S is
#   LABEL-fundamentals-dependent iff the label function named by
#   STRATEGY_REGISTRY[S] reads at least one key in F off the feature-vector
#   dict, transitively through any helper it calls.
#
# Same AST shape as `schema_oracle._reads_in_scope`.
# ---------------------------------------------------------------------------

_HISTORICAL_DATA = _HERE / "historical_data.py"
_BACKTEST_ENGINE = _HERE / "backtest_engine.py"


def _features_assigned_under(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "features"
                    and isinstance(t.slice, ast.Constant)
                    and isinstance(t.slice.value, str)
                ):
                    out.add(t.slice.value)
    return out


def fundamentals_only_feature_keys(path: Path | None = None) -> set[str]:
    """F -- keys `build_feature_vector` assigns ONLY inside `if fundamentals:`.

    Measured 17 keys on 2026-08-06. Asserted non-empty: an empty F would make
    every downstream membership test vacuously false, which is the "found
    nothing vs sees nothing" failure this project has hit before.
    """
    src = Path(path or _HISTORICAL_DATA).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "build_feature_vector"
        ),
        None,
    )
    if fn is None:
        raise ValueError("build_feature_vector not found -- the rule cannot be applied")
    blocks = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "fundamentals"
    ]
    if not blocks:
        raise ValueError(
            "no `if fundamentals:` block in build_feature_vector -- the derivation "
            "rule no longer matches the code and must be revisited, not silently "
            "returned empty"
        )
    inside: set[str] = set()
    block_node_ids: set[int] = set()
    for b in blocks:
        inside |= _features_assigned_under(b)
        block_node_ids |= {id(n) for n in ast.walk(b)}

    # `outside` must be computed by EXCLUDING the block subtrees, not by
    # subtracting `inside`. Subtracting would classify a key assigned in BOTH
    # places as fundamentals-only, which is the opposite of the rule ("assigned
    # ONLY inside"). Caught by a mutation run, not by inspection.
    outside: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and id(n) not in block_node_ids:
            for t in n.targets:
                if (
                    isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "features"
                    and isinstance(t.slice, ast.Constant)
                    and isinstance(t.slice.value, str)
                ):
                    outside.add(t.slice.value)

    keys = inside - outside
    if not keys:
        raise ValueError("derived F is EMPTY -- refusing to return a vacuous set")
    return keys


def _string_keys_read(node: ast.AST) -> set[str]:
    """String constants used as `.get("k")` / `["k"]` inside `node`."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            out.add(n.slice.value)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        ):
            out.add(n.args[0].value)
    return out


def label_fundamentals_dependent_strategies(
    registry: dict[str, str] | None = None,
    engine_path: Path | None = None,
    historical_data_path: Path | None = None,
) -> set[str]:
    """Apply the rule above and return the dependent strategy names.

    TRANSITIVE, and the transitivity is real: a label function that delegates to
    a helper defined in the same module inherits that helper's reads, to
    arbitrary depth, with cycle protection.

    An earlier revision excluded every callee whose name started with
    `_compute`. That was unjustified and silently broke the documented
    behaviour, because EVERY label function in `STRATEGY_REGISTRY` is named
    `_compute_*` -- so the exclusion dropped exactly this module's dominant
    helper-naming convention and could only ever produce FALSE NEGATIVES. A
    false negative here means a fundamentals-dependent strategy is allowed to
    run on an uncovered window, which is the precise failure this gate exists to
    prevent. Found by the phase-82.21 Q/A, not by reading the code.

    Deliberately biased toward OVER-approximation: `_string_keys_read` collects
    every string-constant subscript / `.get()` key in scope, not only those
    taken off the feature-vector dict. So a helper that happens to read
    `something["pe_ratio"]` for an unrelated reason will mark its caller
    dependent. That direction is safe -- it over-REFUSES, which is loud, rather
    than under-refusing, which is silent.
    """
    if registry is None:
        from backend.backtest.backtest_engine import STRATEGY_REGISTRY

        registry = STRATEGY_REGISTRY
    F = fundamentals_only_feature_keys(historical_data_path)

    src = Path(engine_path or _BACKTEST_ENGINE).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def reads(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in fns:
            return set()
        seen.add(name)
        node = fns[name]
        out = _string_keys_read(node)
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                callee = None
                if isinstance(n.func, ast.Attribute):
                    callee = n.func.attr
                elif isinstance(n.func, ast.Name):
                    callee = n.func.id
                # Follow EVERY callee defined in this module. No name-based
                # exclusion -- see the docstring: the previous `_compute` filter
                # matched the convention every label function uses.
                if callee and callee in fns:
                    out |= reads(callee, seen)
        return out

    dependent = set()
    for strategy, label_fn in registry.items():
        if label_fn not in fns:
            logger.warning(
                "fundamentals_coverage: label fn %r for strategy %r not found -- "
                "treating as dependent (fail-closed)", label_fn, strategy,
            )
            dependent.add(strategy)
            continue
        if reads(label_fn) & F:
            dependent.add(strategy)
    return dependent


__all__ = [
    "FUNDAMENTALS_COVERAGE_START",
    "SNAPSHOT_PATH",
    "coverage_start",
    "fundamentals_only_feature_keys",
    "label_fundamentals_dependent_strategies",
    "load_snapshot",
    "measure_live_coverage",
    "snapshot_drift",
    "window_is_covered",
]
