# Research Brief — Phase 43 DoD-2 Pytest Follow-up

**Tier:** simple
**Cycle:** 17
**Target:** `tests/test_phase_43_dod2_window.py` (recommended path:
`backend/tests/test_phase_43_dod2_window.py` — see Internal §1)
**Date:** 2026-05-28

---

## 1. Headline

The cycle-16 windowed-Sharpe helpers (`compute_paper_sharpe_window`,
`compute_sharpe_gap(window_days=…)`) follow the same `MagicMock`-based
BQ-mock + `return_value`/`side_effect` pattern that is already in use
in `backend/tests/test_dod4_tier1_coverage_investment.py:312-705` for
`compute_sharpe_from_snapshots` and `compute_sharpe_gap`. The
recommended file is **`backend/tests/test_phase_43_dod2_window.py`**
(co-located with existing phase-43 perf-metrics tests; pytest already
collects this dir, evidenced by `backend/tests/test_phase_41_*.py`
etc.). Boundary set per BVA = {below-min, at-min, just-over-min,
nominal} → directly maps to the four cases the cycle-16 Q/A flagged.

---

## 2. Sources read in full (5 required — gate floor)

| # | URL | Accessed | Kind | Fetched | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://docs.python.org/3/library/unittest.mock.html | 2026-05-28 | Official doc | WebFetch full | "`MagicMock` is a subclass of `Mock` with default implementations of most of the magic methods." Setting `side_effect = Exception('Boom!')` raises on call; setting `side_effect = [a, b, c]` yields next value per call. |
| 2 | https://oneuptime.com/blog/post/2026-02-02-pytest-mocking/view | 2026-05-28 | Blog (2026-02) | WebFetch full | 2026 best practice: mock at system boundaries; use descriptive names; for sequential behaviours pass a list to `side_effect`. Recommends fixture-based mocking for reusability. |
| 3 | https://pytest-with-eric.com/mocking/python-magicmock-raise-exception/ | 2026-05-28 | Blog | WebFetch full | Canonical pattern: `mocker.patch(target, side_effect=ExceptionClass)` + `pytest.raises(...)` OR — when the SUT swallows the exception (fail-open) — `side_effect=Exception("X")` and assert returned default. |
| 4 | https://edbennett.github.io/python-testing-ci/04-edges/index.html | 2026-05-28 | Tutorial | WebFetch full | Canonical categories: **bulk/normal**, **edge** (one parameter at boundary), **corner** (multiple at boundary). Stresses: "Corner cases are especially important to test, as it is very easy for two pieces of code … to conflict with one another." |
| 5 | https://en.wikipedia.org/wiki/Boundary_testing | 2026-05-28 | Reference | WebFetch full | "A series of edge cases around each 'boundary' can be used to give reasonable coverage" — establishes that the {N-1, N, N+1} pattern is the canonical BVA. (Article does not formalise BVA itself but corroborates the heuristic.) |

**Floor check:** 5/5 fetched via `WebFetch`, full HTML render, no
PDF skip. Gate satisfied on read-in-full count.

---

## 3. Snippet-only (context, not counted toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://docs.python.org/3/library/unittest.mock-examples.html | Official doc | Companion examples page; primary doc (row 1) covered the same API. |
| https://realpython.com/python-mock-library/ | Blog | Strong but stable content already covered by row 1 + 2. |
| https://thedatasavvycorner.com/blogs/15-pytest-mocking | Blog | Surface-level; superseded by row 2. |
| https://aaronlelevier.github.io/python-unit-testing-with-magicmock/ | Blog | Older intro, no boundary patterns. |
| https://pytest-mock.readthedocs.io/en/latest/usage.html | Official doc | `mocker` fixture sugar; pyfinagent's existing tests use plain `MagicMock`, not `pytest-mock`, so this is informational. |
| https://recca0120.github.io/en/2026/04/03/pytest-mock/ | Blog (2026-04) | Recency-scan hit; same content as pytest-mock readthedocs. |
| http://carpentries-incubator.github.io/python-testing/06-edges/index.html | Tutorial | Same content as row 4 (sibling lesson). |
| https://hackmd.io/@cs111/python_guide | Tutorial | General Python testing; no perf-metrics-specific content. |
| https://github.com/pytest-dev/pytest-mock/issues/175 | GitHub issue | Bug discussion (spy + side_effect interaction); not applicable to the four-test scope. |
| https://idlirshkurti.github.io/tutorials/mocking_pytest.html | Blog | Same patterns as row 2. |

URLs collected (read-in-full + snippet-only): **15** unique. Floor
of 10+ satisfied.

---

## 4. Recency scan (last 2 years)

Searched: `pytest unittest mock MagicMock return_value patterns 2026`,
`pytest mocker fixture vs patch decorator 2025`,
`pytest mock side_effect exception fail-open pattern` (year-less).

**Findings 2024–2026:** No API breaking changes in `unittest.mock`
since Python 3.12. Two 2026 blogs (rows 2 + recca0120 snippet) confirm
the same patterns. `pytest-mock` continues to receive interface
refinements (`mocker` fixture is the more recent ergonomic) but the
underlying `MagicMock` semantics are unchanged. The cycle-16 helper
mocks `bq.get_paper_snapshots(limit=N)` — a method call returning a
list — which is a stable use case.

**Verdict:** No 2024-2026 finding supersedes the canonical
`MagicMock().return_value = [...]` /
`MagicMock().side_effect = Exception(...)` patterns the existing
pyfinagent tests use. Continue with that style for the new file.

---

## 5. Search-query composition (3-variant discipline)

1. **Current-year frontier:** `pytest unittest mock MagicMock
   return_value patterns 2026`
2. **Last-2-year window:** `pytest mocker fixture vs patch decorator
   2025` (alt 2024)
3. **Year-less canonical:** `unittest.mock MagicMock side_effect
   best practices` + `pytest boundary condition testing edge cases
   trailing window function` + `pytest mock side_effect exception
   fail-open pattern`

The year-less queries hit the canonical Python.org docs page (row 1)
and the carpentries edge-cases lesson (row 4) — both pre-dated 2024
and would not have surfaced under year-locked queries alone. Mix is
verifiable in the source table: row 1 = year-less canonical, row 2 =
2026-02 (current-year), row 4 = 2019-era carpentries lesson
(year-less canonical), row 5 = year-less reference.

---

## 6. Existing pyfinagent test pattern findings

### 6.1 Where tests live

- Two test trees: `tests/` (top-level, integration-style — includes
  `test_mcp_*.py`, `test_end_to_end.py`) and **`backend/tests/`**
  (unit-test home — 100+ files including all `test_phase_*` and
  `test_dod4_tier1_coverage_investment.py`).
- **Recommendation:** put the new file under
  `backend/tests/test_phase_43_dod2_window.py`. This matches:
  - `backend/tests/test_phase_41_0_bundle_close.py`
  - `backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py`
  - `backend/tests/test_dod4_tier1_coverage_investment.py` (the
    direct precedent — tests `compute_sharpe_gap` already)
- Cycle-17 contract may say `tests/test_phase_43_dod2_window.py`;
  flag during PLAN if the path needs to be reconciled. Both pytest
  collection roots work; `backend/tests/` is the conventional unit-
  test home for module-under-test in `backend/services/`.

### 6.2 Import pattern

Verbatim from `backend/tests/test_dod4_tier1_coverage_investment.py`:

- Top of file:
  ```python
  from unittest.mock import MagicMock, patch
  import pytest
  ```
- Per-test (lazy import to keep collection cheap, and to avoid side-
  effects from import-time singletons):
  ```python
  def test_perf_metrics_sharpe_gap_no_snapshots_returns_no_data():
      from backend.services.perf_metrics import compute_sharpe_gap
      bq = MagicMock()
      bq.get_paper_snapshots.return_value = []
      out = compute_sharpe_gap(bq)
      assert out["live_sharpe"] is None
  ```
- For `compute_sharpe_from_snapshots`:
  ```python
  from backend.services.perf_metrics import compute_sharpe_from_snapshots
  assert compute_sharpe_from_snapshots([{"total_nav": 100.0}] * 3) == 0.0
  ```

### 6.3 Fixture / BQ-mock pattern

- **No `conftest.py` fixture** is used for the BQ mock in the perf-
  metrics tests — `bq = MagicMock()` is constructed in-test (line 676
  of test_dod4). This is the pyfinagent convention for
  `compute_sharpe_gap` tests.
- `return_value` is set per-method: `bq.get_paper_snapshots.return_value = [...]`
- `side_effect` is used for exceptions:
  `bq.get_paper_snapshots.side_effect = Exception("BQ down")` (test_dod4:700)
- For the snapshot dicts: the canonical key set is
  `{"total_nav": float, "snapshot_date": "YYYY-MM-DD"}` — the new
  windowed helper reads BOTH keys (NAV via `nav_key` kwarg default
  `"total_nav"`; date via `snapshot_date_key` default
  `"snapshot_date"` for the sort step at perf_metrics.py:157).
  **The existing test_dod4 tests omit `snapshot_date`** because
  `compute_sharpe_from_snapshots` doesn't sort. The new tests MUST
  add it because `compute_paper_sharpe_window` sorts.

### 6.4 Helper signature recap (internal audit)

From `backend/services/perf_metrics.py:118-169`:

```python
def compute_paper_sharpe_window(
    bq: Any,
    *,
    window_days: int = 30,
    risk_free_rate: float = 0.04,
    nav_key: str = "total_nav",
    snapshot_date_key: str = "snapshot_date",
) -> Optional[float]:
    if window_days < 6:          # line 145 — early-return guard #1
        return None
    try:
        snapshots = bq.get_paper_snapshots(
            limit=max(window_days * 2, 60)
        ) or []
    except Exception:             # line 150 — fail-open on BQ raise
        return None
    if not snapshots:             # line 152 — empty result
        return None
    try:
        snapshots_sorted = sorted(
            snapshots, key=lambda s: str(s.get(snapshot_date_key, ""))
        )
    except Exception:
        snapshots_sorted = snapshots
    window = snapshots_sorted[-window_days:]
    if len(window) < 6:           # line 161 — early-return guard #2
        return None
    sharpe = compute_sharpe_from_snapshots(
        window, nav_key=nav_key, risk_free_rate=risk_free_rate
    )
    if sharpe == 0.0:             # line 166 — 0.0 means "could not compute"
        return None
    return sharpe
```

And from `:240-349`:

```python
def compute_sharpe_gap(
    bq: Any,
    *,
    backtest_sharpe_source: str = "optimizer_best",
    risk_free_rate: float = 0.04,
    min_snapshots: int = 6,
    window_days: Optional[int] = None,
) -> dict:
    # ...
    if window_days is not None:
        live_sharpe = compute_paper_sharpe_window(bq, ...)
    else:
        # Pre-cycle (legacy) all-snapshot path. byte-identical.
        snapshots = bq.get_paper_snapshots(limit=365) or []
        if len(snapshots) >= min_snapshots:
            live_sharpe = compute_sharpe_from_snapshots(snapshots, ...)
```

The "legacy byte-identical" condition is the union of:
- `bq.get_paper_snapshots.return_value = [...30 monotonic dicts...]`
  must call `bq.get_paper_snapshots(limit=365)` (not `limit=60`).
- Output dict keys must be the same set the cycle-15 tests already
  asserted on (`live_sharpe`, `backtest_sharpe`, `gap_abs`, `gap_rel`,
  `threshold`, `gap_within_threshold`, `source`, `note`,
  `proxy_fallback`, `computed_at`).

---

## 7. Recommended test file content

**Path:** `backend/tests/test_phase_43_dod2_window.py`

```python
"""phase-43.0 cycle-17 — windowed-Sharpe helper (DoD-2) tests.

Covers the cycle-16 helpers in backend/services/perf_metrics.py:
  - compute_paper_sharpe_window (lines 118-169)
  - compute_sharpe_gap(window_days=...) (lines 240-349)

Q/A verdict a30ae6755518b9ced (cycle-16) flagged four NOTEs:
  1. window_days < 6 early-return guard (line 145)
  2. len(window) < 6 post-slice insufficiency (line 161-162)
  3. windowed value differs from legacy on a synthetic snapshot set
  4. compute_sharpe_gap(window_days=None) is byte-identical to legacy

Fixture pattern mirrors backend/tests/test_dod4_tier1_coverage_investment.py:
in-test MagicMock() construction, no shared conftest fixture.
"""

from __future__ import annotations

from unittest.mock import MagicMock


# ---------- helpers ----------

def _snap(day: int, nav: float) -> dict:
    """One mock paper-portfolio snapshot row.

    Uses the canonical key shape (total_nav + snapshot_date) that
    compute_paper_sharpe_window expects for both NAV access and the
    sort step at perf_metrics.py:157.
    """
    return {
        "total_nav": float(nav),
        "snapshot_date": f"2026-04-{day:02d}",
    }


# ---------- Case 1: window_days < 6 early-return guard ----------

def test_compute_paper_sharpe_window_returns_none_when_window_too_small():
    """window_days < 6 hits the early-return guard at perf_metrics.py:145.

    The helper must return None BEFORE calling bq.get_paper_snapshots,
    so we verify the mock was not called.
    """
    from backend.services.perf_metrics import compute_paper_sharpe_window

    bq = MagicMock()
    # Even with a healthy snapshot list, window_days=5 must short-circuit.
    bq.get_paper_snapshots.return_value = [_snap(i + 1, 100.0 + i)
                                            for i in range(30)]

    for n in (0, 1, 5):
        result = compute_paper_sharpe_window(bq, window_days=n)
        assert result is None, f"expected None for window_days={n}, got {result}"

    # Guard fires before any BQ call.
    bq.get_paper_snapshots.assert_not_called()


# ---------- Case 2: post-slice insufficiency ----------

def test_compute_paper_sharpe_window_returns_none_when_window_slice_too_short():
    """Even with window_days >= 6, if BQ returns < 6 rows the post-slice
    len(window) < 6 guard at perf_metrics.py:161-162 must fire."""
    from backend.services.perf_metrics import compute_paper_sharpe_window

    bq = MagicMock()
    # Only 5 snapshots available; window_days=30 -> window has 5 entries.
    bq.get_paper_snapshots.return_value = [_snap(i + 1, 100.0 + i)
                                            for i in range(5)]

    result = compute_paper_sharpe_window(bq, window_days=30)
    assert result is None
    bq.get_paper_snapshots.assert_called_once_with(limit=60)  # max(30*2, 60)


# ---------- Case 3: windowed differs from legacy on synthetic data ----------

def test_compute_paper_sharpe_window_differs_from_legacy_on_synthetic_set():
    """When the trailing window has a different return distribution from
    the all-time series, the windowed Sharpe must differ from the legacy
    all-snapshot Sharpe — proving both branches are exercised, not aliases.
    """
    from backend.services.perf_metrics import (
        compute_paper_sharpe_window,
        compute_sharpe_from_snapshots,
    )

    # Build a synthetic 60-snapshot series:
    #   - first 30 days: noisy / negative drift (NAV bounces 95-100)
    #   - last 30 days: clean monotone uptrend (NAV 100 -> 130)
    # The trailing 30 should yield a markedly higher Sharpe than all 60.
    legacy_snaps = []
    for i in range(30):
        # Alternating up/down around 97 -> high variance, low mean.
        nav = 97.0 + (1.5 if i % 2 == 0 else -1.5)
        legacy_snaps.append(_snap(i + 1, nav))
    for i in range(30):
        legacy_snaps.append(_snap(i + 1, 100.0 + i))  # snapshot_date prefix overlap is fine; window is index-based after sort
    # Re-stamp dates so sort is well-defined across the full 60.
    for idx, snap in enumerate(legacy_snaps):
        snap["snapshot_date"] = f"2026-{(idx // 30) + 3:02d}-{(idx % 30) + 1:02d}"

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = legacy_snaps

    windowed = compute_paper_sharpe_window(bq, window_days=30)
    legacy = compute_sharpe_from_snapshots(legacy_snaps)

    # Both must be well-defined.
    assert windowed is not None, "trailing window of 30 should compute"
    assert legacy != 0.0, "60-snapshot series should compute a Sharpe"

    # They must differ — proves the slice path is exercised, not a no-op.
    assert windowed != legacy, (
        f"windowed Sharpe ({windowed}) must differ from legacy ({legacy}) "
        "on this divergent synthetic set"
    )


# ---------- Case 4: compute_sharpe_gap(window_days=None) byte-identical ----------

def test_compute_sharpe_gap_window_none_byte_identical_to_legacy():
    """compute_sharpe_gap(window_days=None) must preserve the
    pre-cycle-16 behaviour byte-for-byte: same BQ call shape
    (limit=365), same output dict shape, same live_sharpe value
    given the same mock snapshots.
    """
    from backend.services.perf_metrics import (
        compute_sharpe_gap,
        compute_sharpe_from_snapshots,
    )

    # Deterministic monotonic uptrend, 30 snapshots.
    snaps = [_snap(i + 1, 100_000.0 + i * 50.0) for i in range(30)]

    bq = MagicMock()
    bq.get_paper_snapshots.return_value = snaps

    out = compute_sharpe_gap(bq)  # window_days defaults to None

    # 4a. BQ call shape: limit=365 (the legacy all-time pull).
    bq.get_paper_snapshots.assert_called_once_with(limit=365)

    # 4b. Live Sharpe matches the direct compute_sharpe_from_snapshots
    #     value — proves the window_days=None branch routes through the
    #     legacy primitive, not the new windowed helper.
    expected_live = compute_sharpe_from_snapshots(snaps)
    if expected_live == 0.0:
        # legacy "could not compute" -> compute_sharpe_gap remaps to None
        assert out["live_sharpe"] is None
    else:
        assert out["live_sharpe"] == expected_live

    # 4c. Output dict shape — full key set unchanged.
    expected_keys = {
        "live_sharpe", "backtest_sharpe", "gap_abs", "gap_rel",
        "threshold", "gap_within_threshold", "source", "note",
        "proxy_fallback", "computed_at",
    }
    assert set(out.keys()) == expected_keys

    # 4d. threshold is the SR_GAP_THRESHOLD constant (0.30), not a
    #     window-dependent override.
    assert out["threshold"] == 0.30
```

### 7.1 Run command (proposed for the contract's success criteria)

```bash
source .venv/bin/activate
python -m pytest backend/tests/test_phase_43_dod2_window.py -v
```

Expected: 4 passed.

### 7.2 Why this set covers the Q/A NOTEs

| Q/A NOTE (cycle-16) | Test fn | perf_metrics.py line covered |
|---|---|---|
| window_days < 6 guard | `test_..._window_too_small` | :145 |
| post-slice <6 guard | `test_..._slice_too_short` | :161-162 |
| windowed ≠ legacy proof | `test_..._differs_from_legacy` | :160-165 + :87-115 |
| `window_days=None` byte-identical | `test_..._window_none_byte_identical` | :286-297 |

### 7.3 BVA mapping (boundary value analysis — source row 5)

| Boundary point | window_days | Snapshot count | Outcome | Covered by |
|---|---|---|---|---|
| Below-min (param) | 0, 1, 5 | 30 | None (line 145) | Case 1 |
| At-min (data) | 30 | 5 | None (line 161) | Case 2 |
| Nominal (data) | 30 | 60 | float, ≠ legacy | Case 3 |
| Nominal legacy (no window) | None | 30 | live_sharpe == legacy | Case 4 |

---

## 8. Confidence

**High** on:
- File path + import pattern (verbatim from in-repo test_dod4).
- Mock pattern (canonical Python.org doc + matches in-repo style).
- BVA categorisation (rows 4 + 5).

**Medium** on:
- Case 3's exact synthetic series — the divergent first-30/last-30
  pattern is engineered to produce a Sharpe gap, but the precise
  `(windowed > legacy)` direction depends on numpy floating-point
  ordering. The test asserts `windowed != legacy` (the load-bearing
  property) and does not assume direction. If the actual numbers come
  out identical (very unlikely given the construction), fall back to
  a stronger split (e.g., legacy=flat-line, windowed=uptrend).

**Low risk:**
- `compute_paper_sharpe_window` is internal and stable.
  `bq.get_paper_snapshots(limit=...)` is the only external surface
  and is already mocked the same way in test_dod4 (:677, :687, :700).

---

## 9. JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/research_brief_phase_43_0_dod_2_pytest_followup.md",
  "gate_passed": true
}
```

**Hard-blocker checklist:**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (15)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] 3-variant query composition documented
- [x] Existing pyfinagent test pattern enumerated with file:line anchors

`gate_passed: true`.
