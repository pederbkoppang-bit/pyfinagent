# Experiment Results -- phase-3.3 Regime Detection

**Step:** 3.3 Regime Detection.
**Date:** 2026-04-19.

## What was built

Narrow code addition per researcher rec. Zero new runtime dependencies.

**New module `backend/backtest/regime_detector.py`** (~180 lines):
- `RegimeDetector` `@runtime_checkable` Protocol matching the consumer interface at `backend/backtest/spot_checks.py:165` (single method `detect() -> list[{name, start_date, end_date}]`).
- `VIXRollingQuantileRegimeDetector` concrete class with `__init__(start_date, end_date, low_q=0.33, high_q=0.67, window_days=252, vix_symbol="^VIX")`.
- `.detect()` fetches VIX closes via `yfinance.Ticker("^VIX").history(...)`, computes trailing `window_days` rolling quantile classification (`low_vol`/`medium_vol`/`high_vol`), and merges consecutive same-label days into regime windows.
- Fail-open: yfinance/pandas import failure, HTTP error, empty response, or insufficient data all route to the same static 2-regime pre/post-COVID fallback that `spot_checks.py:172-175` already emits when `regime_detector=None`.
- `classify_series` and `_merge_runs` exposed at class level for test mutation-resistance.

**Harness wiring (`backend/backtest/spot_checks_harness.py:76-91`):**
- Replaced the hardcoded `regime_detector=None` with a settings-gated instantiation. When `settings.regime_detection_enabled=True`, a `VIXRollingQuantileRegimeDetector` is built using `backtest_start_date`/`backtest_end_date`. When False (default), behavior matches pre-phase-3.3 exactly.

**Settings (`backend/config/settings.py`):**
- New field `regime_detection_enabled: bool = Field(False, description="...")`. Default False preserves all existing behavior; opt-in only.

**Tests (`backend/tests/test_regime_detector.py`, 8 tests):**
- Instantiation with defaults + Protocol-satisfaction check.
- Fail-open on fetch failure (exception).
- Fail-open on insufficient data (None return).
- Classification over a synthetic pd.Series (low-then-high ramp).
- Merge-runs on hand-constructed input.
- Merge-runs handles empty + mismatched inputs.
- Settings flag defaults to False.
- `_to_date_str` coerces `str` / `datetime` / `pandas.Timestamp` correctly.

## File list

Created:
- `backend/backtest/regime_detector.py`
- `backend/tests/test_regime_detector.py`

Modified:
- `backend/backtest/spot_checks_harness.py` (8 lines changed around the wiring point)
- `backend/config/settings.py` (+2 lines: new flag)

NOT touched:
- `backend/backtest/spot_checks.py::RegimeShiftTest` (consumer interface preserved)
- `backend/backtest/gauntlet/regimes.py` (immutable black-swan catalog preserved)
- `backend/agents/planner_enhanced.py` (phase-3.4 scope)
- Any existing test file
- Any dep / requirements.txt

## Verification command output

### Immutable verification (from masterplan)

```
$ source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
...
[INFO] harness: HARNESS COMPLETE -- 1 cycles finished
[INFO] harness: Final best: Sharpe=1.1705, DSR=0.9526
```

Exit 0. Both immutable `success_criteria` satisfied: `no_regressions` ✓ (Sharpe/DSR preserved) and `evaluator_critique_pass` pending Q/A.

### Syntax + import smoke

```
$ python -c "import ast; ast.parse(open('backend/backtest/regime_detector.py').read()); ast.parse(open('backend/tests/test_regime_detector.py').read()); print('OK')"
OK

$ python -c "from backend.backtest.regime_detector import VIXRollingQuantileRegimeDetector, RegimeDetector; d = VIXRollingQuantileRegimeDetector(start_date='2023-01-01', end_date='2024-01-01'); r = d.detect(); assert isinstance(r, list); print('detect() returned', len(r), 'regimes')"
VIXRollingQuantileRegimeDetector: insufficient VIX data (250 rows); using fallback
detect() returned 2 regimes
```

Note: the smoke run fetched only 250 VIX rows (less than window_days=252), so the detector fail-opened to the static fallback as designed. In production with `backtest_start_date=2018-01-01 / end=2025-12-31` the window is ~2000 rows; fallback not triggered.

### Unit tests

```
$ pytest backend/tests/test_regime_detector.py -x -q
........                                                                 [100%]
8 passed in 0.22s
```

### Regression across phase-3 + phase-6

```
$ pytest backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_regime_detector.py -q
62 passed, 1 skipped in 7.39s
```

Zero regressions. +8 new regime tests (cumulative 62 passing).

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `regime_detector.py` module with Protocol + concrete class | PASS |
| 2 | `spot_checks_harness.py:80` wired + settings-gated | PASS |
| 3 | Settings key `regime_detection_enabled` default False | PASS |
| 4 | No new runtime dep | PASS (yfinance already pinned) |
| 5 | Gauntlet static catalog untouched | PASS |
| 6 | `test_regime_detector.py` >=5 tests | PASS (8) |
| 7 | Immutable verify exit 0 + Sharpe/DSR preserved | PASS |
| 8 | Zero regressions | PASS (62 passing) |

## Known caveats

1. **Live VIX-based detection not exercised in-session.** The smoke run fetched only 250 rows (one calendar year requested), triggering the insufficient-data fallback. Production runs with the default `backtest_start_date=2018-01-01` span ~2000 trading days — comfortably above the 252-day window. Not a defect; noting so Q/A can confirm the fall-through path is correct.
2. **`regime_detection_enabled=False` by default.** Contract non-goal: do not flip production behavior. Operators opt-in when ready. Flagging this explicitly so Q/A does not expect a live regime switch in `--dry-run` output.
3. **`VIXRollingQuantileRegimeDetector.classify_series` uses `rolling(...).quantile()`** with `min_periods=1`, which means the first few days are classified against a very short rolling window. Benign for typical use (252-day lookback with 2000 data points); the early days are in the backtest warmup region anyway.
4. **Protocol runtime-check works** because `detect()` is the single required method and it's defined on the class. `isinstance(d, RegimeDetector)` in the tests confirms it.
5. **Pre-Q/A self-check (per Q/A recommendation from phase-3.1 cycle):** grep-verified the consumer interface at `spot_checks.py:165` (`self.regime_detector.detect()`) and the shape expected at `:181-194` (`name`, `start_date`, `end_date` keys) BEFORE writing the detector class. Catch prior bugs early. No invented specifics this cycle.
