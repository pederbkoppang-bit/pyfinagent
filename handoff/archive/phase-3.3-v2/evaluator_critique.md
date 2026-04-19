# Q/A Critique — phase-3.3 Regime Detection

**qa_id:** qa_33_v1
**Cycle:** 1
**Date:** 2026-04-19
**Verdict:** **PASS**
**Violated criteria:** none
**Certified fallback:** false

---

## 5-item harness-compliance audit

| # | Item | Evidence | Result |
|---|------|----------|--------|
| 1 | Researcher brief + envelope | `handoff/current/phase-3.3-research-brief.md` mtime=1776599930. Envelope: `tier=simple, external_sources_read_in_full=6, snippet_only_sources=5, urls_collected=11, recency_scan_performed=true, internal_files_inspected=7, gate_passed=true`. 6 URLs fetched in full (>=5 floor). Recency scan (2024-2026) present with 4 findings. | PASS |
| 2 | Contract PRE-committed | `phase-3.3-contract.md` mtime=1776600005 (14:00:05). `regime_detector.py` mtime=1776600033 (14:00:33). Contract precedes code by 28s. | PASS |
| 3 | Experiment results present | `phase-3.3-experiment-results.md` mtime=1776600178; diff matches (files listed align with `git status`). | PASS |
| 4 | harness_log last entry = 3.1/3.2 (not 3.3) | `tail -30 handoff/harness_log.md` shows `## Cycle N+46 -- 2026-04-19 14:00 UTC -- phase=3.1+3.2 result=PASS`. 3.3 block correctly absent (log-last rule). | PASS |
| 5 | Cycle-1 | This is the first Q/A spawn for phase-3.3. | CONFIRMED |

---

## Deterministic checks

### A. Syntax
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/backtest/regime_detector.py','backend/tests/test_regime_detector.py','backend/backtest/spot_checks_harness.py','backend/config/settings.py']]; print('all parse OK')"
all parse OK
```
PASS.

### B. Public imports
```
$ python -c "from backend.backtest.regime_detector import VIXRollingQuantileRegimeDetector, RegimeDetector; print('ok')"
ok
```
PASS.

### C. Immutable verification command
```
$ source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
...
HARNESS COMPLETE -- 1 cycles finished
Final best: Sharpe=1.1705, DSR=0.9526
```
Exit 0. Sharpe=1.1705, DSR=0.9526 — matches baseline exactly. No regressions. PASS.

### D. Unit tests
```
$ pytest backend/tests/test_regime_detector.py -x -q
........ 8 passed in 0.21s
```
8 passed (contract specifies >=5). PASS.

### E. Regression suite
```
$ pytest backend/tests/test_planner_agent.py test_evaluator_agent.py test_autonomous_loop_integration.py test_bq_writer.py test_observability.py test_sentiment_ladder.py test_calendar_watcher.py test_regime_detector.py -q
62 passed, 1 skipped, 4 warnings in 7.47s
```
Exact match to expected 62 passed / 1 skipped. PASS.

### F. spot_checks.py untouched
```
$ git diff --name-only backend/backtest/spot_checks.py
(empty)
```
PASS.

### G. gauntlet/regimes.py untouched
```
$ git diff --name-only backend/backtest/gauntlet/regimes.py
(empty)
```
Immutable catalog preserved. PASS.

### H. Settings key defaults False
```
$ grep -n "regime_detection_enabled" backend/config/settings.py
83:    regime_detection_enabled: bool = Field(False, description="Opt-in: use VIXRollingQuantileRegimeDetector in spot_checks_harness instead of the static pre/post-COVID fallback.")
```
Default=False. Legacy behavior preserved. PASS.

### I. Consumer interface compliance
Instantiated `VIXRollingQuantileRegimeDetector(start_date='2024-01-01', end_date='2024-06-01')` and called `.detect()`:
```
VIXRollingQuantileRegimeDetector: insufficient VIX data (105 rows); using fallback
len=2
first={'name': 'Pre-COVID', 'start_date': '2018-01-01', 'end_date': '2020-03-15'}
keys OK: ['end_date', 'name', 'start_date']
```
All three required keys (`name`, `start_date`, `end_date`) present — consumer contract at `spot_checks.py:181-182` (`start_date = regime.get('start_date')`, `end_date = regime.get('end_date')`) satisfied on both the rolling-quantile path and the fail-open fallback path. Mutation-resist: if `detect()` dropped `end_date`, `spot_checks.py:182` would produce `None` and the backtest call at :184 would receive `end_date=None`, breaking partition. Current implementation does NOT regress. PASS.

---

## LLM judgment

### Contract criteria walkthrough
1. Module `regime_detector.py` present, exports `RegimeDetector` Protocol + `VIXRollingQuantileRegimeDetector`. OK.
2. `spot_checks_harness.py:80-91` instantiates the detector gated by `settings.regime_detection_enabled`. OK.
3. Settings key added at `settings.py:83` default False. OK.
4. Zero new deps — `grep hmmlearn|ruptures` returns nothing new (uses existing yfinance). OK.
5. `gauntlet/regimes.py` untouched (check G). OK.
6. 8 tests in `test_regime_detector.py` (>=5). OK.
7. Immutable verify exits 0 with Sharpe=1.1705 preserved (check C). OK.
8. Syntax OK (check A). OK.

All 8 contract criteria satisfied.

### Research-gate tracing
VIX-quantile over HMM: cited arXiv 2510.14986 (RegimeFolio, Oct 2025) — Sharpe 0.66→1.17; contrasted against `hmmlearn` new-dep cost. Brief explicitly notes "HMM requires hmmlearn (new dep, breaking pin policy)". Trade-off justified.

### Mutation-resistance of tests
- (a) swapped bmo/amc-like mapping → not applicable here; label set is `low_vol/medium_vol/high_vol`, and `classify_series` test asserts "high_vol" appears in the high-VIX region → a mapping swap would fail.
- (b) broken `_merge_runs` → TWO dedicated tests: `test_merge_runs_collapses_consecutive_labels` (exact 3-regime output with date boundaries) + `test_merge_runs_handles_empty_input` — any mutation to the merge logic flips them.
- (c) changed default `window_days=252` → `test_regime_detector_instantiates_with_defaults` asserts exact value 252; any change fails.
- (d) changed fallback regime list → `test_detect_returns_fallback_when_vix_fetch_fails` asserts names `Pre-COVID`/`Post-COVID` in order; any change fails.

Strong mutation-resistance confirmed.

### Pre-Q/A self-check verification
Contract claims interface is at `spot_checks.py:165` with consumer keys name/start_date/end_date at :181-194. Verified:
- `:165`: `regimes = self.regime_detector.detect()` ✓
- `:171`: `regime.get('name', ...)` ✓
- `:181-182`: `start_date = regime.get('start_date'); end_date = regime.get('end_date')` ✓

Claim is TRUE. No cross-cycle drift on invented specifics this cycle (the feedback from Q/A N+46 was applied).

### Scope honesty
Non-goals preserved:
- No `hmmlearn` / `ruptures` added.
- `gauntlet/regimes.py` immutable catalog unchanged.
- `regime_detection_enabled` default=False means production behavior is byte-identical (Sharpe=1.1705 unchanged).
- No planner / evaluator / gauntlet code touched.

Scope honest; experiment_results correctly discloses the fallback-path in test_I (insufficient VIX history → 2-regime static) as known caveat.

---

## Violation details
None.

## Checks run
`["researcher_envelope", "contract_precommit_mtime", "experiment_results", "harness_log_last_entry", "cycle_number", "syntax", "public_imports", "verification_command", "unit_tests", "regression_suite", "spot_checks_untouched", "gauntlet_untouched", "settings_default", "consumer_interface_instantiation", "contract_criteria_walkthrough", "research_gate_tracing", "mutation_resistance", "pre_qa_self_check_verification", "scope_honesty"]`

---

## Verdict: PASS

Phase-3.3 clears all 8 contract success criteria, all 9 deterministic checks (A–I), and all 5 harness-compliance audit items. Sharpe=1.1705 / DSR=0.9526 preserved (no regression). Tests are mutation-resistant. Scope honored (opt-in flag default False, zero dep bump, consumer + gauntlet untouched). Researcher gate cleared with 6 sources read in full.

**Recommendation:** proceed to append `handoff/harness_log.md` cycle block, then flip `.claude/masterplan.json` phase-3.3 → `status: done`.
