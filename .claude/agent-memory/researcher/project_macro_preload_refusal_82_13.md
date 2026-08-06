---
name: macro-preload-refusal-82-13
description: preload_macro has FIVE return paths (a POSITIVE already-warm early return is the trap); the ~40-min hang is unmeasured folklore; cache.py cites its own stale anchor; Python has no must_use so the guard must be a test
metadata:
  type: project
---

Measured 2026-08-05 for masterplan step 82.13 (residual after 82.0). Re-derive line
numbers before citing -- they move.

**`backend/backtest/cache.py::preload_macro` has FIVE return paths, not two.** The naive
`if not preload_macro(): abort` is a trap in BOTH directions:
- `:345` returns a **POSITIVE cached total** on the already-warm early return -- zero rows
  loaded this call. The quant optimizer runs with `skip_cache_clear=True`, so this path fires
  on **every iteration after the first**. Any guard keyed to "rows loaded THIS call" aborts
  every optimizer iteration but the first.
- `:356` returns 0 = empty table (no data). `:418` and `:435` return 0 = **REFUSALS**
  (unparseable dates / per-series SLA staleness). `:461` returns rows loaded = success.
- The ONLY invariant that holds across all five: **`_macro_full` non-empty iff macro is
  available.** Branch on an availability accessor, never on the int.
- The docstring at `:337` ("Returns the total number of rows loaded") is FALSE on 3 of 5
  paths -- arguably the root cause of the discard at the call site.

**The "~40-minute backtest hang" is FOLKLORE.** `CLAUDE.md:30` is the sole origin; re-quoted
in `cache.py:53` and `handoff/harness_log.md:11866`. No benchmark, test, or timing artefact
exists in the repo. The defensible claim is "one uncached 30s-timeout BQ round-trip per
distinct cutoff date" (`cache.py:603-641`, memoised per cutoff, driven by
`historical_data.py:48`). Do not restate the number as measured.

**`cache.py:51` cites `backtest_engine.py:308` for the discard; the real line is `:317`.**
82.0 introduced that stale anchor in the very comment explaining the defect.

**Python has no `#[must_use]` / `[[nodiscard]]` (verified against 2024-2026 literature).**
mypy/pyright do not warn on a discarded non-`None` return; ruff/flake8 have no general
unchecked-return rule. Go's `errcheck` and Rust's `#[must_use]` have no counterpart. So a
discarded-return guard in this repo **must be a test, not a linter** -- and criterion-3-style
call-site censuses must be **AST-derived** (`ast.Call` whose direct parent is `ast.Expr` =
discarded), because grep cannot distinguish a call from a prose mention or a shadowing local
`def`. Measured census: 967 files, 15 `preload_*` call sites, 8 discarded / 7 used.

**Adding a result field is cheap here.** `BacktestResult` (`backtest_engine.py:119-132`) has
all-defaulted fields; `generate_report` returns an opaque dict consumed by 18 call sites;
`result_store.save_result` is `json.dumps(report, default=str)`; `api/backtest.py:1059-1060`
already adds `run_id`/`config` post-hoc (the in-repo precedent); `frontend/src/lib/types.ts`
is structurally typed so an extra runtime key breaks nothing. **Trap:**
`backend/tests/test_phase_75_mcp_truth.py:371-375` asserts `not hasattr(r, X)` for
`dsr`/`return_pct`/`max_drawdown_pct`/`num_trades` -- do not reuse those names.

**Second silent-degradation site on the same path (out of 82.13's scope, file separately):**
`backend/backtest/historical_data.py:269` guards macro features with a bare `if macro:`, so an
empty `cached_macro` silently drops all six macro features from the feature vector.

See also [[project_vacuous_bq_guards_82_12]] (the STRING-column guard that made this refusal
path unreachable until 82.0) and [[project_macro_ingestion_dead_82_0]].
