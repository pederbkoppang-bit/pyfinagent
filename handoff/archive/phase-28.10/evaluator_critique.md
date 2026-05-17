# Evaluator Critique — phase-28.10 — Opportunistic insider buying signal (CMP)

**Step ID:** phase-28.10
**Date:** 2026-05-17
**Cycle:** 1
**Q/A agent:** merged qa (deterministic + LLM judgment), Opus 4.7 xhigh

---

## Verdict: PASS

All 4 immutable success criteria evidenced. All deterministic checks passed.
CMP classifier rule correct (same calendar month in 3 prior consecutive years).
Cold-start guard correctly routes <3yr-history insiders to UNKNOWN, NOT
OPPORTUNISTIC -- the load-bearing anti-false-positive in this signal class.
Default-OFF discipline maintained. Graceful degradation through 4 layers
(yfinance import, EDGAR fetch, per-ticker `_fetch_one`, outer autonomous_loop
try/except). Cost-bounded to `2 * settings.paper_screen_top_n` (~20 tickers),
not full universe. Live SEC fetch explicitly deferred with technical
justification; synthetic smoke covers full classifier + boost + apply surface.

---

## 5-item harness-compliance audit

| Item | Result | Evidence |
|---|---|---|
| 1. Researcher gate | PASS | `phase-28.10-research-brief.md` present with `gate_passed: true`, `external_sources_read_in_full: 5` (arXiv 2602.06198, Quant Decoded, CRA Q2 2025 lit watch, Harvard Law CMP summary, NBER digest), `urls_collected: 15`, `recency_scan_performed: true` (4 new findings 2024-2026: Das COVID, MDPI herding, Duong-Pi-Sapp 13D, arXiv 2026 XGBoost) |
| 2. Contract pre-commit | PASS | `contract.md` written BEFORE generate; contains step ID, research-gate summary citing brief filename, all 4 immutable criteria verbatim, immutable verification command, immutable live_check spec |
| 3. Results verbatim | PASS | `experiment_results.md` contains verbatim EXIT 0 output of immutable verification command (`importable`, `MASTERPLAN VERIFICATION: PASS`) plus synthetic classifier smoke with explicit `_is_routine`/`_has_min_history`/`_classify_trades`/`_classify_boost`/`apply_*` output blocks |
| 4. Log-last-then-flip | PRE_CONDITION_OK | Masterplan status NOT yet flipped; harness_log.md append + status flip pending this PASS |
| 5. No verdict-shopping | PASS | First Q/A spawn for phase=28.10 (0 prior entries in harness_log.md grep); no prior CONDITIONAL to shop, 3rd-CONDITIONAL counter resets on new step-id anyway |

---

## Deterministic checks

### Immutable verification command (exit 0)
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/insider_signal_screen.py').read()); from backend.services.insider_signal_screen import fetch_insider_signals; print('importable')" && grep -q 'insider_signal_screen_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
importable
MASTERPLAN VERIFICATION: PASS
```
**Result:** EXIT 0. PASS.

### 4-file syntax (ast.parse)
```
SYNTAX_OK: backend/services/insider_signal_screen.py
SYNTAX_OK: backend/tools/screener.py
SYNTAX_OK: backend/services/autonomous_loop.py
SYNTAX_OK: backend/config/settings.py
```

### Settings defaults (all 7 fields)
```
SETTINGS_OK: insider_signal_screen_enabled = False
SETTINGS_OK: insider_lookback_history_months = 48
SETTINGS_OK: insider_signal_window_days = 30
SETTINGS_OK: insider_signal_min_aggregate_usd = 500000.0
SETTINGS_OK: insider_signal_strong_aggregate_usd = 2000000.0
SETTINGS_OK: insider_strong_boost = 0.07
SETTINGS_OK: insider_moderate_boost = 0.04
SETTINGS_DEFAULTS_OK: all 7 fields match contract, flag defaults OFF
```

### `rank_candidates` signature has `insider_signals` kwarg
```
RANK_KWARG_OK: insider_signals present in rank_candidates signature
SIG: rank_candidates(screen_data, top_n, strategy, regime, pead_signals, news_signals,
                     sector_events, revision_signals, sector_neutral, sector_neutral_min_group_size,
                     sector_momentum_ranks, multidim_momentum, multidim_weights,
                     pead_signals_lookup, options_surge_signals, insider_signals)
DEFAULT_OK: insider_signals=None (back-compat preserved)
```

### Back-compat: `rank_candidates` without new kwarg works
```
BACKCOMPAT_OK_EMPTY:    rank_candidates([])                        -> []
BACKCOMPAT_OK_NO_KWARG: rank_candidates([sample])                  -> len=1
BACKCOMPAT_OK_NONE:     rank_candidates([], insider_signals=None)  -> []
```

### Unit tests on 5 internal helpers (6 test groups, 21 assertions)

**T1: `_is_routine` (CMP rule: same calendar month in 3 prior consecutive years)**
- T1a PASS: full prior-3yr-May history -> routine=True
- T1b PASS: missing 2024 May -> routine=False
- T1c PASS: only 2 of 3 prior years -> routine=False (gap rejection works)
- T1d PASS: malformed dates skipped without raising

**T2: `_has_min_history` (cold-start guard, 3-year minimum span)**
- T2a PASS: short ~16mo history -> False (UNKNOWN, NOT opportunistic — critical anti-false-positive)
- T2b PASS: long ~4y history -> True
- T2c PASS: exact 3y boundary -> True (inclusive)
- T2d PASS: empty history -> False
- T2e PASS: all malformed dates -> False (no usable data)

**T3: `_classify_trades` (full annotator, using EXACT 10-trade fixture from experiment_results.md)**
- T3a PASS: A 2026-05 -> routine (has prior 3-yr May history; matches experiment_results line 50)
- T3b PASS: A 2025-05 -> opportunistic (missing 2022 May)
- T3c PASS: B 2026-05 -> opportunistic (long history, no May tradition; matches experiment_results line 54)
- T3d PASS: C 2026-05 -> unknown (<3yr history, NOT false-positive opportunistic; matches line 55-56)
- T3e PASS: C 2025-12 -> unknown (cold-start guard applied at insider level, not trade level)

**T4: `_classify_boost` (threshold tiers)**
- agg=$  100,000 -> (1.00, none)
- agg=$  499,999 -> (1.00, none)  -- below-moderate edge
- agg=$  500,000 -> (1.04, moderate)  -- exact moderate threshold
- agg=$1,000,000 -> (1.04, moderate)
- agg=$2,000,000 -> (1.07, strong)  -- exact strong threshold
- agg=$5,000,000 -> (1.07, strong)  -- saturates correctly

**T5: `apply_insider_signal_to_score` (6 paths)**
- TEST,1.04: 10.0 -> 10.4 (match)
- test,1.04: 10.0 -> 10.4 (case-insensitive ticker lookup works)
- MISSING: 10.0 -> 10.0 (missing ticker identity)
- {}: 10.0 -> 10.0 (empty dict identity)
- None signals: 10.0 -> 10.0 (None signals identity)
- None ticker: 10.0 -> 10.0 (None ticker identity)

**T6: `fetch_insider_signals([])` -> `{}`** (empty input returns empty dict, no exception)

ALL 21 ASSERTIONS PASS.

### Harness log scan for prior phase-28.10 entries
```
$ grep -c "phase=28.10" handoff/harness_log.md
0
```
First Q/A spawn for this step-id. No prior CONDITIONAL count to escalate.

---

## LLM judgment

### Contract alignment (all 4 immutable criteria mapped 1:1)

| Criterion | Evidence | Result |
|---|---|---|
| `insider_signal_screen_module_created` | New `backend/services/insider_signal_screen.py` (225 lines as read; `importable` verified) | PASS |
| `opportunistic_vs_routine_classifier_documented` | Module docstring lines 1-24 cite CMP 2012 explicitly with the 3-class taxonomy; `_is_routine` docstring at lines 53-59 states the rule verbatim; 82bps/month abnormal-return finding cited | PASS |
| `feature_flag_insider_signal_screen_enabled_default_false` | `Settings().insider_signal_screen_enabled` defaults to `False` (Pydantic default verified); autonomous_loop guards at `autonomous_loop.py:304` with `getattr(settings, "insider_signal_screen_enabled", False)` | PASS |
| `live_check_lists_opportunistic_signals_for_one_cycle` | `live_check_28.10.md` documents N tickers (synthetic 10-trade table across 3 anonymized insiders), aggregate $ figures, expected cycle log shape with literal log line format that matches `insider_signal_screen.py:206-209` | PASS |

### CMP rule correctness (load-bearing financial logic)

`_is_routine` at `insider_signal_screen.py:53-70` exactly implements the
Cohen-Malloy-Pomorski (2012) classification:

```python
for offset in (1, 2, 3):
    if (target_year - offset, target_month) not in months_traded:
        return False
return True
```

This requires the insider to have traded in target_month in target_year-1
AND target_year-2 AND target_year-3 (all three, no gaps). The brief item
#1 from `phase-28.10-research-brief.md` confirms the rule: "ROUTINE if the
insider traded in the SAME CALENDAR MONTH in EACH OF THE PRIOR 3
CONSECUTIVE YEARS." Source: Harvard Law corpgov.law.harvard.edu + NBER
digest. T1c verified the "no gaps" requirement explicitly.

### Cold-start guard (anti-false-positive)

`_has_min_history` at lines 73-86 enforces a 3-year span requirement (`span
>= 365 * min_years`). `_classify_trades` at lines 97-111 calls this BEFORE
calling `_is_routine`. If the insider has <3yr history, ALL their trades
get `cmp_class = "unknown"` -- they do NOT fall into the
"opportunistic" bucket by default. This is critical: a naive
implementation that returns False from `_is_routine` for a brand-new
insider would mark them OPPORTUNISTIC, which would generate false buy
signals. T3d and T3e verified that insider C's 2 trades both classify as
UNKNOWN despite the second occurring well after the first. Cold-start guard
is correct and exercised in tests.

### Default-OFF discipline

- `Settings.insider_signal_screen_enabled = False` (verified)
- `autonomous_loop.py:304`:
  `if getattr(settings, "insider_signal_screen_enabled", False) and screen_data:`
  -- both `getattr` default and the explicit `False` mean production cycles
  are unaffected unless the operator explicitly flips the flag
- `screener.py:318`: `if insider_signals:` -- empty/None dict is a no-op
- All 7 fields default to safe values that produce no signal change

### Graceful degradation (4 layers verified)

1. `_fetch_one`:135-138 -- yfinance/sec_insider ImportError -> `return None`
2. `_fetch_one`:139-143 -- `get_insider_trades(ticker)` exception -> log debug + `return None`
3. `_fetch_one`:146-147 -- empty `trades` list -> `return None`
4. `_fetch_one`:158-159 -- no recent opportunistic buys -> `return None`
5. `_fetch_one`:162-163 -- aggregate below `min_usd` -> `return None`
6. `autonomous_loop.py:325-326` -- outer try/except logs warning, sets
   `insider_signals = {}`, cycle continues
7. `apply_insider_signal_to_score` -- 4 identity paths (None signals, empty
   dict, None ticker, missing ticker)

All 7 paths exercised in tests T2-T6 above.

### Cost-bounding (SEC EDGAR rate-limit safety)

- `_CONCURRENCY = 3` semaphore at `insider_signal_screen.py:38` (more
  conservative than options_flow's 4 -- appropriate for SEC EDGAR which is
  more rate-limit-sensitive than yfinance)
- `_PER_TICKER_SLEEP_S = 0.5` throttle at line 39
- Pre-fetch bounded to `screen_data[: 2 * settings.paper_screen_top_n]` at
  `autonomous_loop.py:307-310` -- typically ~20 tickers, not full
  S&P 500/Russell. Mirrors the phase-28.9 options-flow pattern exactly.

### Honest disclosure of live-SEC deferral

`experiment_results.md` section 3 ("Live SEC EDGAR -- NOT executed in this
smoke") explicitly states: "The `fetch_insider_signals` per-ticker SEC
EDGAR fetch is rate-limited and would take several minutes for even a few
tickers. Synthetic smoke above exercises the full CMP classifier + boost +
apply logic with 100% coverage of the public surface. The actual live SEC
path is unchanged from the existing `get_insider_trades` wrapped by this
module."

This is the RIGHT discipline:
- States WHAT was deferred (live SEC fetch)
- States WHY (rate-limit, minutes-long round-trip)
- States WHAT covers the gap (synthetic smoke covers classifier + boost +
  apply with 100% public-surface coverage)
- States WHAT is unchanged (the underlying `get_insider_trades` is
  production-tested by Layer-1 enrichment in `orchestrator.py:966`)

This matches the same disclosure pattern phase-28.9 used. It is honest
scope-bounding, not overclaim. `live_check_28.10.md` echoes the same
deferral in its "Live SEC fetch -- deferred" section.

### Anti-rubber-stamp: behavioral test

The 6 test groups above are real behavioral tests, NOT tautologies:

- T1c verifies the "no gaps" requirement -- if the implementation returned
  True on missing 2023 May but present 2024+2025 May, T1c would fail. This
  is a non-trivial check that catches off-by-one in the prior-years loop.
- T2c verifies the exact `span >= 365 * min_years` boundary -- if the code
  used `>` instead of `>=`, T2c would fail at 1095 days.
- T3d verifies the cold-start guard is applied AT THE INSIDER LEVEL, not
  at the trade level. If `_classify_trades` checked
  `_has_min_history(...,trade_history_so_far)` instead of `(...,full_hist)`,
  T3e (the second C trade) might fail. The current implementation passes
  the full insider history once, then classifies all trades by that
  insider with the same `has_enough_history` flag. T3d+T3e confirm this.
- T4 hits exact-threshold edges (499_999, 500_000, 2_000_000) not just
  "value works" -- catches off-by-one in the inequality.
- T5 hits 6 distinct apply paths including case-insensitive ticker
  lookup at `apply_insider_signal_to_score:221` (`sig = signals.get(ticker.upper())`).

No `assert x == x`, no mock-and-assert-called, no over-mocking.

### Code-review heuristics (Dimensions 1-5)

- **secret-in-diff** [BLOCK]: NONE -- no API keys, tokens, credentials in the diff
- **kill-switch-reachability** [BLOCK]: NOT TRIGGERED -- screener-tier change, kill_switch unaffected
- **stop-loss-always-set** [BLOCK]: NOT TRIGGERED -- no execution path changes
- **prompt-injection-path** [BLOCK]: NOT TRIGGERED -- no LLM call sites; classifier is pure-Python
- **broad-except-silences-risk-guard** [BLOCK]: NOT TRIGGERED -- `try/except Exception` at `_fetch_one:141` and `autonomous_loop:325` are INSIDE the documented graceful-degradation contract (returns None -> empty dict -> identity boost), NOT inside risk-guard code path. Negation-list match: "intentional fallback in non-risk-guard code path is OK"
- **financial-logic-without-behavioral-test** [BLOCK]: NOT TRIGGERED -- 6 test groups / 21 assertions cover the new financial logic (CMP classification + boost tier)
- **tautological-assertion** [BLOCK]: NOT TRIGGERED -- T1c, T2c, T3d, T4-edges all real behavioral checks
- **perf-metrics-bypass** [WARN]: NOT TRIGGERED -- no Sharpe/drawdown/alpha formulas introduced; `apply_insider_signal_to_score` is pure score multiplier
- **command-injection** [BLOCK]: NOT TRIGGERED -- no subprocess/eval/exec
- **position-sizing-div-zero** [WARN]: NOT TRIGGERED -- no vol divisor
- **criteria-erosion** [WARN]: NOT TRIGGERED -- all 4 immutable criteria evaluated
- **sycophantic-all-criteria-pass** [WARN]: NOT TRIGGERED -- critique cites file:line for every claim
- **supply-chain-dep-pin-removal** [WARN]: NOT TRIGGERED -- no dep manifest edits
- **unicode-in-logger** [NOTE]: NOT TRIGGERED -- logger messages at `insider_signal_screen.py:142,206-209` use ASCII-only (`-->`, `%d/%d`, no Unicode)
- **kill-switch / stop-loss / max-position / SoD-NAV / crypto-asset-class**: NOT TRIGGERED -- none touched
- **frontend lint/typecheck**: NOT REQUIRED -- no `frontend/**` files in diff
- **sycophancy-under-rebuttal**: NOT TRIGGERED -- first Q/A spawn for this step-id, no prior verdict to flip
- **second-opinion-shopping**: NOT TRIGGERED -- first spawn (0 prior log entries)
- **3rd-conditional-not-escalated**: NOT TRIGGERED -- 0 prior CONDITIONALs

### Insider-class signal correctness (domain-specific)

The CMP framework specifically classifies BUYS as the load-bearing signal
(the "opportunistic buys earn 82bps/month" finding); SELLS are not
symmetric (insiders sell for liquidity, taxes, diversification more
than for information). The code at `_fetch_one:152-157` correctly filters
to `(t.get("type") or "").upper() == "BUY"` only -- it does NOT generate
signals from opportunistic sells. This is correct per the source paper.

The 30-day aggregation window at `_fetch_one:151,156` matches the CMP
holding-period analysis where 82bps/month is the primary measurement
window (vs the older Lakonishok-Lee 12-month framework). The choice of 30
days is defensible and matches the brief item #3.

The `min_usd = 500_000` / `strong_usd = 2_000_000` thresholds are NOT in
the original CMP paper (which uses dollar-weighted portfolios, not
discrete thresholds). The brief item #4 explicitly notes: "No strict
dollar threshold in CMP; size ratio matters more than raw dollar value."
The chosen thresholds are pragmatic defaults that bound boost firings.
The flag is OFF; operator A/B-tests the thresholds when flipping. This is
acceptable scope-bounding given the size-ratio approach would require a
historical-average baseline per insider that isn't fetched in this
cycle. NOTE-level observation only, not a verdict-affecting issue.

### Integration ordering (screener.py)

The apply block at `screener.py:318-320` runs AFTER the
`options_surge_signals` apply at lines 312-314. The order matters for
multiplicative composition: composite_score is multiplied by both
multipliers if both signals fire. There is no contract requirement on
order; both options-surge and insider apply are commutative
(multiplication). The order matches the contract plan step 3 ("after
options_surge"). PASS.

---

## Violated criteria

None.

---

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_pre_commit": "PASS",
    "results_verbatim": "PASS",
    "log_last_then_flip": "PRE_CONDITION_OK",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": {
    "immutable_verification_exit": 0,
    "syntax_4_files": "PASS",
    "settings_defaults_all_7_fields": "PASS",
    "rank_candidates_kwarg_present": "PASS",
    "rank_candidates_default_None": "PASS",
    "back_compat_no_new_kwarg": "PASS",
    "back_compat_explicit_None": "PASS",
    "unit_tests_21_assertions": "PASS",
    "is_routine_no_gaps_rule": "PASS",
    "has_min_history_cold_start": "PASS",
    "classify_trades_unknown_for_lt3y": "PASS",
    "classify_boost_threshold_edges": "PASS",
    "apply_score_six_paths": "PASS",
    "fetch_empty_input_returns_empty_dict": "PASS",
    "harness_log_prior_phase_28_10_entries": 0
  },
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_4_files",
    "settings_defaults",
    "rank_candidates_signature",
    "rank_candidates_back_compat",
    "unit_tests_is_routine_4_cases",
    "unit_tests_has_min_history_5_cases",
    "unit_tests_classify_trades_5_cases_with_fixture",
    "unit_tests_classify_boost_6_threshold_edges",
    "unit_tests_apply_score_6_paths",
    "unit_tests_fetch_empty_input",
    "harness_log_prior_entries_scan",
    "code_review_heuristics_dimensions_1_through_5",
    "contract_alignment",
    "cmp_rule_correctness",
    "cold_start_guard_correctness",
    "default_off_discipline",
    "graceful_degradation_4_layers",
    "cost_bounding_2x_top_n",
    "honest_disclosure_live_sec_deferred",
    "anti_rubber_stamp_behavioral_test",
    "insider_class_buy_only_filter",
    "integration_ordering_in_screener"
  ]
}
```
