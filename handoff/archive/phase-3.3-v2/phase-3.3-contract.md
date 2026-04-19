# Sprint Contract -- phase-3.3 Regime Detection

**Written:** 2026-04-19 PRE-commit.
**Step id:** `3.3` in phase-3.
**Immutable verification:** `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` with `success_criteria: [evaluator_critique_pass, no_regressions]`.

## Research-gate summary

Researcher envelope `{tier: simple, external_sources_read_in_full: 6, snippet_only_sources: 5, urls_collected: 11, recency_scan_performed: true, internal_files_inspected: 7, gate_passed: true}`. Brief: `handoff/current/phase-3.3-research-brief.md`.

Staked rec (adopted): **narrow code addition**. The interface at `backend/backtest/spot_checks.py:165` (`self.regime_detector.detect()`) was left as a deliberate stub; `spot_checks_harness.py:80` passes `regime_detector=None` which triggers a static 2-regime pre/post-COVID fallback at `spot_checks.py:172-175`. Zero HMM / ruptures / change-point imports exist in the tree. Prior phase-3.3.1 artifacts note regime detection was "Secondary / NICE TO HAVE" and was never implemented.

Key research findings:
- VIX rolling-quantile 3-regime (arXiv 2510.14986, Oct 2025, "RegimeFolio") matches HMM accuracy on daily-bar US equities without a new dependency.
- HMM requires `hmmlearn` (new dep, breaking pin policy).
- LLM features degrade under macro-shock regimes (arXiv 2604.10996) — regime-aware spot checks are a defensive measure.
- `backend/backtest/gauntlet/regimes.py` has a 7-window immutable black-swan catalog but is not a live detector.

Pre-Q/A self-check confirmed (per last cycle's Q/A recommendation): interface spec at `spot_checks.py:142-175` returns `list[dict]` with keys `name`, `start_date`, `end_date`. That's our output contract.

## Hypothesis

A `VIXRollingQuantileRegimeDetector` class providing `.detect() -> list[dict]` with continuous rolling-quantile regime boundaries, plus wiring at `spot_checks_harness.py:80`, replaces the static 2-regime fallback with live data-driven regime splits — without adding any new runtime dependency (uses `yfinance` for VIX close, already pinned).

## Success criteria

**Functional:**
1. New module `backend/backtest/regime_detector.py` (~80-120 lines) exporting:
   - `RegimeDetector` Protocol (single method: `detect() -> list[dict]` with `{name, start_date, end_date}` schema matching the consumer at `spot_checks.py:181-194`).
   - `VIXRollingQuantileRegimeDetector` class: `.__init__(start_date, end_date, low_q=0.33, high_q=0.67, window_days=252, vix_symbol="^VIX")`.
   - `.detect()` method returns regime windows labeled `low_vol` / `medium_vol` / `high_vol` based on trailing 252-day rolling VIX quantile at each day. Consecutive same-label days are merged into one window.
   - Fallback data source: `yfinance.Ticker("^VIX").history(start, end)["Close"]`. Fail-open: if yfinance import fails OR returns empty, `.detect()` returns the same static pre/post-COVID split that `spot_checks.py` uses today.
2. `backend/backtest/spot_checks_harness.py:80` wired to instantiate `VIXRollingQuantileRegimeDetector(start_date=..., end_date=...)` and pass to `SpotCheckRunner`. Settings-driven: enable via `settings.regime_detection_enabled: bool = False` (default False so existing behavior unchanged; opt-in).
3. Add the settings key `regime_detection_enabled: bool = False` to `backend/config/settings.py`.
4. NOT adding `hmmlearn` / `ruptures` / any new dep. Zero requirements bump.
5. Gauntlet static catalog (`backend/backtest/gauntlet/regimes.py`) is NOT touched; it remains the immutable black-swan reference catalog per phase-4.9 ownership.

**Tests:**
6. `backend/tests/test_regime_detector.py` with >=5 tests:
   - VIXRollingQuantileRegimeDetector instantiates with default args.
   - `.detect()` returns empty-safe fallback when yfinance is monkey-patched to raise.
   - Rolling-quantile logic: feed a synthetic VIX series (hand-constructed pd.Series), assert 3 regimes detected with correct boundaries.
   - Merge of consecutive same-label days produces N regimes (not N days).
   - Settings flag gate: when `regime_detection_enabled=False`, the harness wiring returns None / uses fallback (integration test).

**Immutable verification command:**
- `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` -> exit 0 with final-best Sharpe preserved (1.1705 / DSR 0.9526).

**Additional correctness:**
- `python -c "import ast; ast.parse(open('backend/backtest/regime_detector.py').read())"` -> 0
- `python -c "from backend.backtest.regime_detector import VIXRollingQuantileRegimeDetector, RegimeDetector; d = VIXRollingQuantileRegimeDetector(start_date='2023-01-01', end_date='2024-01-01'); r = d.detect(); assert isinstance(r, list); print('ok', len(r))"` -> prints `ok N` where N >= 1
- `pytest backend/tests/test_regime_detector.py -x -q` -> all pass
- Cumulative test suite zero regressions: `pytest backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py backend/tests/test_regime_detector.py -q`

**Non-goals:**
- NOT changing `backend/backtest/spot_checks.py::RegimeShiftTest.run()` interior (the consumer). Only the injected detector.
- NOT adding HMM or ruptures library (research-based deferral).
- NOT re-writing `planner_enhanced.py` regime-aware prompts (phase-3.4 scope + part of the still-pending planner consolidation).
- NOT changing gauntlet static catalog.
- NOT wiring into production backtest path by default (behind opt-in settings flag).

## Plan steps

1. Write `backend/backtest/regime_detector.py` with the Protocol + concrete class.
2. Add settings key to `backend/config/settings.py`.
3. Wire `spot_checks_harness.py:80`.
4. Write `backend/tests/test_regime_detector.py`.
5. Run immutable verification + the unit tests + regression.

## References

- `handoff/current/phase-3.3-research-brief.md`
- `backend/backtest/spot_checks.py:142-200` (consumer interface)
- `backend/backtest/spot_checks_harness.py:75-90` (wiring point)
- `backend/backtest/gauntlet/regimes.py` (not touched; historical reference)
- External read-in-full: QuantStart HMM, arXiv 2510.14986, arXiv 2510.03236, arXiv 2604.10996, BSIC HMM, QuantInsti regime adaptive.

## Researcher agent id

`a62435afcbd6a920d`
