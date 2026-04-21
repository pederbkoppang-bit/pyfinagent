# Phase-10.6 Evaluator Critique

**qa_id:** qa_106_v2 (v1 died mid-mutation; Main cleaned stale `.pyc` and
restored the source file; this is the fresh independent Q/A)
**Cycle:** v2 (not a second-opinion shop — v1 never emitted a verdict)
**Date:** 2026-04-20
**Verdict:** PASS

## 5-item harness-compliance audit

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher spawned with `gate_passed=true` (>=5 full + recency) | PASS (cited in contract) |
| 2 | `contract.md` mtime (1776718974) precedes `experiment-results.md` mtime (1776719137) | PASS |
| 3 | Immutable verification command quoted verbatim from `.claude/masterplan.json` in contract | PASS |
| 4 | No `handoff/harness_log.md` append before Q/A (log-last discipline) | PASS |
| 5 | This is v2 after a v1 death — no verdict-shopping; v1 never wrote a critique | PASS |

## Deterministic checks_run

- **A. Syntax:** `ast.parse` exit 0 on all three files (module, CLI harness, pytest suite).
- **B. Immutable CLI:** `python scripts/harness/phase10_monthly_sortino_test.py` — exit 0, 7/7 success_criteria reported.
- **C. Pytest module:** `pytest tests/autoresearch/test_monthly_champion_challenger.py -q` — 12/12 passed.
- **D. Neighbors:** `pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q` — 88/88 passed.
- **E. Handoff files:** `phase-10.6-{contract,research-brief,experiment-results}.md` all present.
- **F. Spot-reads confirmed:**
  - `_APPROVAL_WINDOW_HOURS = 48` at line 35 (verified)
  - `"actual_replacement": False` at line 75 — only occurrence in dict literal, no kwarg/setter anywhere (grep-confirmed)
  - `is_last_trading_friday` uses `exchange_calendars` XNYS primary with pure-Python Friday-of-month fallback (lines 244-269)
  - Zero `weekly_ledger.append_row` calls in the monthly module (grep-confirmed)
- **G. Stale `.pyc` hygiene:** cleared at session start AND after every mutation; final file md5 matches `/tmp/mcc_backup.py` (`f48a5f78d43404a5998de4ac73e489cf`).

## Mutation-resistance tests (required by contract)

### M1: `_APPROVAL_WINDOW_HOURS = 48 → 24`
**Expected:** `test_approval_window_48h_expiry` fails.
**Observed:** FAILED as expected with
`assert r_mid["expired"] is False` → `assert True is False` at
`test_monthly_champion_challenger.py:109`. Mutation detected. **Restored; md5 verified.**

### M2: `"actual_replacement": False → True`
**Expected:** `test_no_auto_replacement_hard_coded` fails.
**Observed:** TWO tests failed (belt-and-suspenders):
1. `test_no_auto_replacement_hard_coded` — `assert r["actual_replacement"] is False` failed at line 131.
2. `test_record_approval_transitions` — also asserts `actual_replacement is False` after approval at line 124.

The SR 11-7 invariant is guarded at both the initial-fire and the
post-approval state — stronger than the contract required. **Restored; md5 verified.**

### M3: swap sortino-delta check BEFORE challenger_min_days check
**Expected (per contract):** `test_challenger_min_days_floor` still catches the <20-day input.
**Observed:** All 12 tests STILL PASS. The min_days check is not
strictly load-bearing for test-observable correctness — with
`CHALL_RETURNS[:10]`, the Sortino-delta path either fails the >=0.3
threshold or produces NaN (fail-closed), so the `"days"` substring
still appears because the days gate runs immediately after.

**Analysis:** this is a defensive-redundancy observation, NOT a failure.
The test's `assert "days" in r["reason"]` happens to pass under either
ordering because both gates reject the 10-day challenger. The module's
behavioral invariant ("reject <20-day challenger") holds in both orders.
I flag this as a minor **scope-honesty observation** for the record:
the contract over-stated M3's load-bearing nature. The code still
correctly rejects all invalid inputs — this is not a verdict blocker.
**Restored; md5 verified.**

## LLM judgment

### Holiday-shifted last-Friday (Dec 2026)
`is_last_trading_friday` with `exchange_calendars` XNYS correctly
identifies **Dec 18, 2026 = True** (last trading Friday) and
**Dec 25, 2026 = False** (Christmas, market closed). The pure-Python
fallback would incorrectly return True for Dec 25 standalone, but the
primary xcals path dominates when `exchange_calendars` is importable
(confirmed in venv). Acceptable — documented explicitly in the
docstring (lines 238-241).

### `actual_replacement` hard-coded invariant
- Single occurrence in the source, in the result-dict literal at line 75.
- NOT exposed as a kwarg on `run_monthly_sortino_gate`.
- NOT written by any downstream state transition (`record_approval`
  at lines 201-231 never touches `actual_replacement`).
- Docstring at lines 15-17 and 63 documents the SR 11-7 invariant.
- Two pytest guards (M2 above). **Robust.**

### State-machine transitions
`pending → approved | rejected | expired`, no path to
`actual_replacement=True`. `record_approval` enforces
`status in ("approved","rejected")` at line 213 (raises ValueError
otherwise), and if the expires_at has elapsed it forces `"expired"`
regardless of requested status (lines 221-226). Correct and
fail-closed.

### Deferred Slack posting (`slack_fn` injection)
The stubbed injection path IS exercised by
`test_approval_window_48h_expiry` (observed `slack_calls=2` in the
CLI harness output for the `peder_slack_approval_with_48h_expiry`
criterion). Deferring live Slack wiring to phase-10.6.1 is
scope-honest because the state machine is fully testable via the
callable injection — no contract breach.

### Contract alignment
All 7 immutable success_criteria from `.claude/masterplan.json` are
covered by both the CLI harness (named prints) and the pytest suite.
`reuses_friday_slot_zero_new_slots` is satisfied by NOT calling
`weekly_ledger.append_row` (grep confirms zero references).

## violated_criteria

None.

## violation_details

None.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_106_v2",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax_ast_parse",
    "immutable_verification_command_exit_0",
    "pytest_module_12_of_12",
    "pytest_neighbors_88_of_88",
    "handoff_files_present",
    "spot_read_constants_and_grep_invariants",
    "stale_pyc_cleanup",
    "mutation_M1_approval_window_48h",
    "mutation_M2_actual_replacement_hardcoded",
    "mutation_M3_short_circuit_order_observation",
    "holiday_friday_xcals_dec_2026",
    "state_machine_transition_audit",
    "slack_fn_injection_path_exercised",
    "contract_alignment_all_7_criteria",
    "5_item_harness_compliance_audit"
  ],
  "reason": "All 7 immutable success_criteria met (CLI 7/7, pytest 12/12, neighbors 88/88). Mutation tests M1 and M2 correctly fail; M3 is not load-bearing (flagged as a minor scope-honesty note, not a blocker — the behavioral invariant holds under either gate ordering). actual_replacement hardcoded False and guarded by two tests. is_last_trading_friday correctly handles Dec-2026 Christmas shift via xcals primary. File restored to md5 f48a5f78d43404a5998de4ac73e489cf; stale .pyc cleared. 5-item harness-compliance audit all green. Fresh v2 after v1 death — no verdict-shopping."
}
```

## Non-blocking observations for the record

1. **M3 over-claimed in contract.** The contract hypothesized the
   short-circuit order was load-bearing; in practice the min_days
   gate is defensive redundancy — both orderings reject the 10-day
   input. This is fine (defense-in-depth is good) but worth noting
   so the next Q/A cycle doesn't look for a non-existent failure.

2. **Pure-Python fallback path has a documented limitation.** When
   `exchange_calendars` is NOT installed, Dec 25 2026 (Christmas
   Friday) would incorrectly be treated as the last trading Friday.
   The docstring documents this; xcals is in the venv; production
   risk is low. Consider adding a simple US federal-holiday table
   to the fallback in a future cycle if xcals ever becomes optional.

3. **phase-10.6.1 live Slack wiring is legitimately deferred.** The
   `slack_fn` injection point is well-designed for incremental
   delivery; the state machine is fully exercised without it.
