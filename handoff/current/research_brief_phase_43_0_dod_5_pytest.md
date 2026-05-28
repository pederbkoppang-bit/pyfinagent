# Research Brief — phase-43 cycle-19 DoD-5 pytest follow-up

**Tier:** simple
**Date:** 2026-05-28
**Topic:** pytest coverage for cycle-14 `_STRING_DATE_TIMESTAMP_COLS` type-branch fix in `backend/services/cycle_health.py::_bq_max_event_age`
**Cycle:** 19 (follow-up to cycle 14)
**Status:** GATE PASSED

## Headline

Write `backend/tests/test_phase_43_dod5_freshness.py` with 4 MagicMock-based tests that exercise the `_STRING_DATE_TIMESTAMP_COLS` set-membership branch added in cycle 14 (`backend/services/cycle_health.py:420-462`). The canonical pattern for this project ALREADY exists at `backend/tests/test_dod4_tier1_coverage_investment.py:417-424`:

```python
bq = MagicMock()
row = MagicMock()
row.get.return_value = 1234.5  # age in seconds
bq.client.query.return_value.result.return_value = [row]
age = cycle_health._bq_max_event_age(bq, "paper_trades", "created_at")
assert age == 1234.5
```

The new tests extend that pattern by also inspecting `bq.client.query.call_args[0][0]` (the SQL string) for the substrings `"SAFE.TIMESTAMP(MAX("` (STRING/DATE branch) or bare `"MAX("` without `"SAFE.TIMESTAMP"` (TIMESTAMP branch). Both `call_args[0][0]` and `call_args.args[0]` are documented in the Python `unittest.mock` docs as equivalent ways to read positional args. The 4 test cases cover both members of the cycle-14 set (paper_trades.created_at, paper_portfolio_snapshots.snapshot_date) AND two non-members (historical_prices.ingested_at, signals_log.recorded_at) — proving both branches of `needs_coerce` are exercised, not aliases.

## Sources read in full

| # | URL | Kind | Fetched how | Key quote / finding |
|---|-----|------|-------------|---------------------|
| 1 | https://docs.python.org/3/library/unittest.mock.html | Official doc | WebFetch | "`call_args[0]` returns the positional arguments as a tuple. Alternatively, you can use the `.args` property." Confirms `mock.call_args[0][0]` is the documented way to read the first positional arg. Also confirms `mock.return_value.attribute = X` and `mock.method.return_value.other.return_value = Y` chain syntax. |
| 2 | https://github.com/pjlast/pytest-bigquery-mock | Library README | WebFetch | Library exists for declaratively configuring BQ mock responses via `@pytest.mark.bq_query_return_data([...])`. Decision: NOT introducing a new dep; the in-tree MagicMock pattern at `test_dod4_tier1_coverage_investment.py:422` already does the chain in one line. Documented for future reference. |
| 3 | https://elvanco.com/blog/how-to-mock-google-bigquery-client-for-unit-tests | Tutorial (Sep 13 2025) | WebFetch | Confirms canonical pattern: `mock_client = Mock(spec=bigquery.Client); mock_job = Mock(spec=bigquery.job.QueryJob); mock_client.query.return_value = mock_job`. Aligns with project's existing pattern. |
| 4 | https://pytest-with-eric.com/mocking/pytest-mock-assert-called/ | Tutorial (Mar 23 2026) | WebFetch | Documents `assert_called_with`, `assert_called_once_with`, `call_args`, `call_args_list`. Recency-passes the last-2-year window. |
| 5 | https://docs.pytest.org/en/stable/example/parametrize.html | Official doc | WebFetch | Canonical `@pytest.mark.parametrize` pattern with (in / not in) boundaries: `("input_value,expected", [(3, True), (5, False)])`. Endorses the 4-row parametrize shape proposed below. |
| 6 | https://pytest-mock.readthedocs.io/en/latest/usage.html | Official doc | WebFetch | Confirms `mocker` fixture has the same API as `mock.patch`. Spy example `spy.assert_called_once_with(21)`. (Less directly useful but confirms pytest-mock vs unittest.mock parity.) |

(6 sources read in full — exceeds the >=5 floor.)

## Sources snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://pypi.org/project/pytest-bigquery-mock/ | PyPI | Same content as the GitHub README in #2. |
| https://blog.flowpoint.ai/post/v2/solved-how-to-mock-google-bigquery-client-for-unittests-in-pytests/ | Blog | WebFetch timed out (60s exceeded); search snippet sufficient and content overlapped with elvanco #3. |
| https://github.com/tylertreat/BigQuery-Python/blob/master/bigquery/tests/test_client.py | GitHub | Third-party non-Google BQ client, less relevant to google-cloud-bigquery's `Client.query()` chain. |
| https://www.py4u.org/blog/python-unit-test-assert-called-with-partial/ | Blog | Covered by pytest-with-eric #4. |
| https://train.rse.ox.ac.uk/material/HPCu/technology_and_tooling/testing/mocking | University | General mock primer; not specific. |
| https://testdriven.io/tips/462a6d54-2eed-4a72-b733-de1b20b806e2/ | Tip | Short tip; overlap with #1. |
| https://alysivji.com/mocking-functions-inputs-args.html | Blog | Side-effect-based mock; not chain. |
| https://github.com/python/cpython/issues/95839 | GH issue | Issue about a feature request; not load-bearing. |
| https://articles.mergify.com/parameterized-tests-python/ | Blog | Overlaps with pytest #5. |
| https://adil.medium.com/how-to-fix-returning-magicmock-object-instead-of-return-value-... | Medium | Beginner-level; covered by #1. |
| https://www.hrekov.com/blog/pytest-mocking-objects-classes | Blog | Covered by #4. |
| https://oneuptime.com/blog/post/2026-02-02-pytest-mocking/view | Blog (Feb 2 2026) | Recent but overlaps with #1+#4; no new info. |
| https://blog.francium.tech/stubbing-deeper-methods-using-magicmock-in-python-... | Blog | Covered by chained-return guidance in #1. |

URLs collected total: 19 (well above the 10 floor).

## Recency scan (last 2 years, 2024-2026)

Performed. New findings in window:
- **elvanco (Sep 13 2025)** — confirms `Mock(spec=bigquery.Client)` pattern still canonical in 2025. No API churn since Python 3.10 unittest.mock.
- **pytest-with-eric (last edited Mar 23 2026)** — confirms all six `assert_called*` methods unchanged.
- **oneuptime (Feb 2 2026)** — overlaps; no new patterns.
- **No deprecation or replacement** has occurred for `MagicMock.call_args[0][0]` syntax. The canonical pattern from the existing test at `test_dod4_tier1_coverage_investment.py:422` (May 2026) is current and aligned with 2026 guidance.

## Search queries run (3-variant discipline)

1. **Current-year frontier (2026):** `pytest mock google-cloud-bigquery client query result chain 2026`
2. **Last-2-year window:** `pytest-mock call_args assert SQL string substring 2025` and `MagicMock return_value chained query result pytest 2024 2025`
3. **Year-less canonical:** `pytest parametrize boundary set membership branch` and `call_args python mock positional arguments docs`

Mix is verified in the read-in-full table: #1 (Python docs, year-less canonical), #3 (Sep 2025 — last-2-year), #4 (Mar 2026 — current frontier), #5 (year-less canonical), #6 (year-less canonical).

## Internal code audit

### Cycle-14 fix at `backend/services/cycle_health.py:414-470` (CONFIRMED)

```
414  # phase-43.0 cycle-14: STRING/DATE-typed timestamp columns that require
...
420  _STRING_DATE_TIMESTAMP_COLS = {
421      ("paper_trades", "created_at"),                  # STRING (RFC3339)
422      ("paper_portfolio_snapshots", "snapshot_date"),  # STRING (YYYY-MM-DD)
423  }
...
426  def _bq_max_event_age(bq: Any, table_logical: str, time_col: str) -> Optional[float]:
...
447      table = bq._pt_table(table_logical)
448      needs_coerce = (table_logical, time_col) in _STRING_DATE_TIMESTAMP_COLS
449      max_expr = (
450          f"SAFE.TIMESTAMP(MAX({time_col}))" if needs_coerce
451          else f"MAX({time_col})"
452      )
453      sql = (
454          f"SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), "
455          f"{max_expr}, SECOND) AS age "
456          f"FROM `{table}`"
457      )
458      rows = list(bq.client.query(sql).result())
459      if not rows:
460          return None
461      age = rows[0].get("age") if hasattr(rows[0], "get") else rows[0][0]
462      return float(age) if age is not None else None
463      except Exception as e:
464          logger.warning(
465              "bq_max_event_age(%s.%s) failed: %s", table_logical, time_col, e
466          )
467          return None
```

Verified: the cycle-14 fix is in place exactly as the prompt described. The `needs_coerce` branch is at line 448; SQL composition at line 449-457; query chain at line 458. The function defensively returns None on any exception (line 463-467).

### Existing test conventions

- **`backend/tests/test_dod4_tier1_coverage_investment.py:408-424`** — *Already covers* the success-case and the fail-open-on-exception case for `_bq_max_event_age` (using `paper_trades.created_at`). Mocks `bq.client.query` chain via:
  ```python
  bq = MagicMock()
  row = MagicMock()
  row.get.return_value = 1234.5
  bq.client.query.return_value.result.return_value = [row]
  ```
  The new file must use THIS exact pattern (no library imports, no `Mock(spec=bigquery.Client)`, no introduction of pytest-bigquery-mock).
- **`backend/tests/test_phase_43_dod2_window.py:1-80`** — Shows the cycle-17 pytest pattern: in-test `MagicMock()` construction, no shared conftest fixture, snake_case test names with `_returns_X_when_Y` shape. The new file should mirror this docstring + casefile style.
- **`backend/tests/test_phase_23_2_11_bq_table_freshness.py`** — Live-BQ probe (NOT mock-based). Different category; uses `pytest.skip()` when google-cloud-bigquery is absent. Useful as a contrast: our new tests must NOT depend on live BQ.
- **`bq.client.query` is a MagicMock attribute access**, so SQL inspection works via `bq.client.query.call_args[0][0]` (the first positional arg is `sql`).

### File existence check

`backend/tests/test_phase_43_dod5_freshness.py` does NOT exist. Recommendation: **CREATE NEW** (not extend an existing file). The existing cycle-14 coverage at `test_dod4_tier1_coverage_investment.py:408-424` exercises only ONE table (`paper_trades`) and tests success + exception, but does NOT verify the branch on the SQL-string level. The new file is genuinely additive: it asserts the SQL string shape for all 4 representative table+col combinations.

## Recommended test skeleton

```python
"""phase-43.0 cycle-19 -- DoD-5 freshness type-branch tests.

Covers the cycle-14 fix at backend/services/cycle_health.py:420-462:
the `_STRING_DATE_TIMESTAMP_COLS` set-membership branch inside
`_bq_max_event_age` that selects between `SAFE.TIMESTAMP(MAX(col))`
(STRING/DATE columns) and bare `MAX(col)` (native TIMESTAMP columns).

Pattern mirrors test_dod4_tier1_coverage_investment.py:417-424
(existing canonical chain mock) + test_phase_43_dod2_window.py
(in-test MagicMock construction, no shared conftest).

Bug context: applying SAFE.TIMESTAMP to a native-TIMESTAMP MAX result
raises BQ 400 `SAFE with function timestamp is not supported`; the
broad `except` swallowed it; 4 sources stayed band=unknown (DoD-5
cycle-12 FAIL). The cycle-14 fix branched on column type.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.services import cycle_health


def _fresh_bq(age_seconds: float = 12345.0) -> MagicMock:
    """Build a MagicMock BQ client that returns a single row with `age`."""
    bq = MagicMock()
    row = MagicMock()
    row.get.return_value = age_seconds
    bq.client.query.return_value.result.return_value = [row]
    return bq


# ---- STRING/DATE columns: must use SAFE.TIMESTAMP(MAX(col)) ----

@pytest.mark.parametrize(
    "table,col",
    [
        ("paper_trades", "created_at"),
        ("paper_portfolio_snapshots", "snapshot_date"),
    ],
)
def test_bq_max_event_age_string_columns_use_safe_timestamp(table, col):
    """STRING/DATE columns must be coerced via SAFE.TIMESTAMP(MAX(col))."""
    bq = _fresh_bq()
    age = cycle_health._bq_max_event_age(bq, table, col)
    assert age == 12345.0
    sql = bq.client.query.call_args[0][0]
    assert f"SAFE.TIMESTAMP(MAX({col}))" in sql, (
        f"expected SAFE.TIMESTAMP wrapper for {table}.{col}, got: {sql}"
    )


# ---- Native TIMESTAMP columns: must use bare MAX(col), no SAFE wrapper ----

@pytest.mark.parametrize(
    "table,col",
    [
        ("historical_prices", "ingested_at"),
        ("signals_log", "recorded_at"),
    ],
)
def test_bq_max_event_age_timestamp_columns_use_bare_max(table, col):
    """Native TIMESTAMP columns must use bare MAX(col); the SAFE.TIMESTAMP
    wrapper triggers BQ 400 on TIMESTAMP-typed inputs (cycle-12 root cause)."""
    bq = _fresh_bq()
    age = cycle_health._bq_max_event_age(bq, table, col)
    assert age == 12345.0
    sql = bq.client.query.call_args[0][0]
    assert f"MAX({col})" in sql
    assert "SAFE.TIMESTAMP" not in sql, (
        f"unexpected SAFE.TIMESTAMP wrapper for native TIMESTAMP "
        f"{table}.{col}, got: {sql}"
    )
```

Notes on the skeleton:
- **No conftest dependency** — in-test `MagicMock()` matches the cycle-17 style at `test_phase_43_dod2_window.py:46`.
- **`@pytest.mark.parametrize` with 2+2 split** — endorsed by pytest docs canonical example. Each parametrize covers one branch of `needs_coerce`. This gives 4 logical test cases for the cost of 2 function bodies.
- **SQL inspection via `bq.client.query.call_args[0][0]`** — documented in Python unittest.mock docs; `call_args[0]` is the positional-args tuple, `[0]` is the first positional arg (the SQL string).
- **`row.get.return_value = age_seconds`** — matches the existing `row.get.return_value = 1234.5` pattern at `test_dod4_tier1_coverage_investment.py:421`. The production code uses `rows[0].get("age")` (line 461) when `hasattr(rows[0], "get")` is true; MagicMock returns True for hasattr by default, so this branch is taken.
- **Boundary-condition test (defensive exception path)** — explicitly NOT included because `test_dod4_tier1_coverage_investment.py:408-414` already covers it (`bq.client.query.side_effect = Exception(...)` returns None). Re-asserting here would be redundant. Keep the file focused on the cycle-14 fix.

## Boundary discipline (4 cases minimum)

| Case | Table | Col | Branch | Expected SQL substring |
|------|-------|-----|--------|------------------------|
| 1 | paper_trades | created_at | IN set → coerce | `SAFE.TIMESTAMP(MAX(created_at))` |
| 2 | paper_portfolio_snapshots | snapshot_date | IN set → coerce | `SAFE.TIMESTAMP(MAX(snapshot_date))` |
| 3 | historical_prices | ingested_at | NOT in set → bare | `MAX(ingested_at)` only |
| 4 | signals_log | recorded_at | NOT in set → bare | `MAX(recorded_at)` only |

The 4-row split mirrors the union of (a) the prompt's request and (b) `compute_freshness` at `backend/services/cycle_health.py:486-489` (the actual production call sites). Adding historical_fundamentals/historical_macro would be repetition without coverage gain — they share the bare-MAX branch with historical_prices.

## Confidence

**HIGH** — the canonical pattern is already established in this project (test_dod4_tier1_coverage_investment.py:408-424). The new file extends it from "does the function return the right value" to "does the function generate the right SQL", which is the precise contract the cycle-14 fix introduced. SQL inspection via `call_args[0][0]` is in the Python stdlib docs and has been stable for years.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 13,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief_phase_43_0_dod_5_pytest.md",
  "gate_passed": true
}
```

### Hard-blocker checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read)
- [x] 10+ unique URLs collected (19 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] 3-variant query discipline (current-year, last-2-year, year-less)

### Soft checks

- [x] Internal exploration covered every relevant module (cycle_health.py + bigquery_client.py + 3 existing test files)
- [x] Consensus is overwhelming; no contradictions surfaced
- [x] All claims cited per-claim
