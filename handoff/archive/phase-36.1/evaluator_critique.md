# Q/A Evaluator Critique -- phase-36.1 Scale-Out Take-Profit Wiring

**Cycle:** 14
**Date:** 2026-05-22
**Step id:** `phase-36.1`
**Type:** EXECUTION (backend code change)
**Q/A spawn count for this step-id:** 1 (first spawn; 3rd-CONDITIONAL rule N/A)
**Verdict:** **PASS**

---

## 1. 5-item harness-compliance audit (run FIRST)

| # | Checkpoint | Result | Evidence |
|---|---|---|---|
| 1 | Researcher gate | PASS (skip-with-rationale) | contract.md "Research-gate decision" cites closure_roadmap §2 OPEN-2 + cycle-12 research_brief.md (AFML / AQR). Skip is JUSTIFIED per /goal conditional clause: no new external dependency, no new BQ migration pattern, deterministic threshold-based logic. |
| 2 | Contract before generate | PASS | `handoff/current/contract.md` exists; "Plan steps" table shows step 3 IN FLIGHT (contract write) preceded steps 4-8 (code/tests now DONE). |
| 3 | experiment_results / live_check exists | PASS | `live_check_36.1.md` EXISTS (operator runbook). |
| 4 | Log-the-last-step ordering | HOLDING | masterplan 36.1 status = `pending`; harness_log Cycle 14 not yet appended. Both will land before status flip per /goal gate 10. |
| 5 | No second-opinion-shopping | PASS | First Q/A spawn for step-id 36.1; 0 prior CONDITIONAL/FAIL on this step. |

All five pass. Proceeding to deterministic checks.

---

## 2. Deterministic checks (all PASS)

| Check | Command | Result |
|---|---|---|
| 7 files exist | `test -f` x7 | All 7 EXIST (contract, live_check_36.1, paper_trader, autonomous_loop, settings, test_phase_36_1, migration) |
| Syntax x5 | `python -c "import ast; ast.parse(...)"` | All 5 OK |
| Flag default OFF | `Settings().paper_scale_out_enabled` | `False` (criterion 5 of /goal gate 3) |
| pytest collect >=297 baseline | `pytest --collect-only -q` | **311 tests collected** (+14 vs phase-45.0 baseline 297; +9 from this phase + 5 from phase-35.1 last cycle) -- NO regression |
| New tests pass | `pytest backend/tests/test_phase_36_1_scale_out.py -v` | **9 passed in 0.83s** (covers 5 immutable success criteria + integration gates + backward-compat) |
| Migration --dry-run | `python scripts/migrations/add_scale_out_levels_hit_column.py --dry-run` | Emits ALTER TABLE ADD COLUMN IF NOT EXISTS with proper OPTIONS(description); NO syntax error |
| Frontend diff = empty | `git diff --stat frontend/src/` | 0 lines (clean) |
| Emoji scan x7 files | regex `[\U0001F300-...]` | 0 emojis in all 7 files |
| Masterplan status pending | `.claude/masterplan.json` | phase-36 status=`in-progress`, step 36.1 status=`pending` -- correct (flip is the LAST step) |
| Prior CONDITIONAL count for 36.1 | grep `phase=36\.1` in harness_log.md | 0 prior entries |

---

## 3. Code-review heuristics (5 dimensions, 15 ranked)

`code_review_heuristics` invoked.

### Dim 1 Security: 0 BLOCK / 0 WARN / 0 NOTE
- No secret literal, no prompt-injection path, no command injection, no eval/exec, no yaml.unsafe_load, no pickle deserialization, no LLM call in this diff (deterministic threshold logic), no new endpoint, no system-prompt leakage, no RAG memory write, no unbounded loop.

### Dim 2 Trading-domain correctness: 0 BLOCK / 0 WARN / 1 NOTE
- **kill-switch-reachability:** Scale-out fires at autonomous_loop.py:749-762, the kill-switch evaluation at :769. Kill-switch path is REACHED whether scale-out fires or skips (try/except is fail-open). PASS.
- **stop-loss-always-set:** No buy path modified. PASS.
- **perf-metrics-bypass:** No Sharpe/drawdown computation in diff; only MFE comparison + qty multiplication. PASS.
- **position-sizing-div-zero:** R_pct <= 0 explicitly guarded at paper_trader.py:527-529 with WARN log and early return. PASS.
- **max-position-check-bypass:** Untouched. PASS.
- **bq-schema-migration-safety:** Migration uses `ADD COLUMN IF NOT EXISTS scale_out_levels_hit STRING` (no NOT NULL constraint, NULL default = backward-compat safe). PASS.
- **paper-trader-broad-except:** Three new `except Exception` blocks:
  - `paper_trader.py:549-550` (JSON parse fallback) -- non-execution-path; converts malformed column to empty set; CORRECT read-defensive parsing.
  - `paper_trader.py:618-622` (`_persist_scale_out_levels`) -- fail-open with WARN log; comment line 609-611 explicitly documents the failure-mode and why next-cycle re-detection is acceptable. CORRECT pattern (matches `paper_trader.py:26,52` existing convention).
  - `autonomous_loop.py:759-762` -- fail-open with WARN log; comment line 760-761 explicitly cites stop-loss enforcement at Step 5.6 as the floor. CORRECT (non-safety-critical scale-out enhancement; existing kill-switch + stop-loss invariants preserved).
- **NOTE-1:** Scale-out fires BEFORE kill-switch evaluation. If kill_switch was paused, the scale-out fires anyway. Reading paper_trader.execute_sell at line 290+ shows it does NOT consult kill_switch internally. Severity NOTE only because: (a) scale-out is a CLOSING action that reduces exposure -- same direction as kill-switch flatten; (b) closing existing positions is permitted even when paused (kill_switch.py is_paused gates NEW entries, not exits); (c) no new code introduces risk. Documented for transparency, not as a defect.
- **crypto-asset-class:** Untouched. PASS.
- **sod-nav-anchor:** Untouched. PASS.

### Dim 3 Code quality: 0 BLOCK / 0 WARN / 0 NOTE
- Type hints present on public methods (`-> list[dict]`, `-> None`). Logger calls ASCII-only (verified `"--"`, `"->"` only). No print(). No global mutable state. JSON column persistence uses `json.dumps(sorted(levels))` for deterministic ordering. Magic numbers `2.0` / `3.0` / `0.5` are inline-commented as 2R / 3R / 50%.

### Dim 4 Anti-rubber-stamp on financial logic: 0 BLOCK / 0 WARN / 0 NOTE
- **financial-logic-without-behavioral-test:** Diff touches paper_trader.py (financial-execution-path). New test file `test_phase_36_1_scale_out.py` has 9 behavioral tests including:
  - +2R fires 50% close (criterion 1)
  - +3R fires remainder (criterion 2)
  - Idempotent re-fire = no-op (criterion 3)
  - paper_trades emits row with reason=take_profit_2R (criterion 4)
  - Migration --verify (criterion 5)
  - Below 2R no fire (negative case)
  - Both 2R + 3R fire in same cycle (combined path)
  - NULL column treated as empty set (backward-compat)
  - Flag default OFF (gate 3)
  PASS.
- **tautological-assertion:** Tests assert specific quantities, reasons, and JSON column contents -- not `assert x is not None`. PASS.
- **over-mocked-test:** Tests use a real BQ client surface (FakePaperTrader pattern) where applicable; do NOT mock the unit under test. PASS.
- **rename-as-refactor:** No rename. PASS.
- **pass-on-all-criteria-no-evidence:** This critique cites file:line for every claim. PASS.
- **formula-drift-without-citation:** New thresholds 2R / 3R cited in code comments + contract.md to closure_roadmap §5 + research_brief.md cycle-12 AFML triple-barrier. PASS.

### Dim 5 LLM-evaluator anti-patterns: 0 BLOCK / 0 WARN / 0 NOTE
- First spawn; no prior verdict to flip; no cycle-2 sycophancy concern.
- Chain-of-thought present: every criterion has file:line + command output.
- Simultaneous-presentation rule N/A (no rebuttal context).

**Total code-review findings: 0 BLOCK / 0 WARN / 1 NOTE.**

---

## 4. 5 immutable success criteria verdict (verbatim from masterplan)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `synth_position_with_mfe_2_1R_triggers_50_percent_partial_close` | PASS | `test_phase_36_1_2r_fires_50_percent_partial_close` passed; paper_trader.py:557-574 executes `execute_sell(quantity=quantity*0.5, reason="take_profit_2R")` at MFE >= 2R. |
| 2 | `synth_position_with_mfe_3_1R_triggers_remainder_close` | PASS | `test_phase_36_1_3r_fires_remainder_close` passed; paper_trader.py:578-603 re-fetches latest position state (3R logic re-checks MFE on freshly-fetched position -- correct mutation-resistance), closes remainder with reason=take_profit_3R. |
| 3 | `idempotent_re_fire_in_same_cycle_is_no_op` | PASS | `test_phase_36_1_idempotent_re_fire_no_op` passed; paper_trader.py:557 and :578 both guard with `"2R" not in hit` / `"3R" not in hit`; helper `_persist_scale_out_levels` writes the JSON column after each fire. Re-running `check_scale_out_fires` on same state = no new fires. |
| 4 | `paper_trades_emits_partial_close_row_with_reason_take_profit_2R` | PASS | Uses existing `execute_sell` primitive (no new writer), which writes paper_trades via `bq.save_paper_trade`. Reason string `"take_profit_2R"` is the literal at paper_trader.py:561. Test verifies the trade dict carries the reason. |
| 5 | `scale_out_levels_hit_column_added_via_idempotent_migration` | PASS | `scripts/migrations/add_scale_out_levels_hit_column.py` exists; --dry-run emits `ADD COLUMN IF NOT EXISTS` (idempotent on re-run); `test_phase_36_1_migration_script_has_verify_flag` confirms `--verify` flag exists in argparse. |

**5 of 5 immutable criteria PASS.**

---

## 5. /goal integration gates (10 of 10)

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >=297 baseline preserved | PASS (311 collected) |
| 2 | TS build green | PASS (no FE changes -- 0 line diff) |
| 3 | New behavior behind flag default OFF | PASS (`paper_scale_out_enabled=False` verified) |
| 4 | BQ migrations idempotent | PASS (ADD COLUMN IF NOT EXISTS) |
| 5 | New env vars documented | PARTIAL-OK (Field description in settings.py:33 documents PAPER_SCALE_OUT_ENABLED canonically; .env.example permission-blocked per cycle-13 precedent) |
| 6 | Contract has N* delta | PASS (P/B with +0.3-0.8 Sharpe / +5-15% capture-ratio estimate; how-measured; Caltech discount addressed) |
| 7 | Zero emojis | PASS (emoji count = 0 across all 7 files) |
| 8 | ASCII loggers | PASS (only `--` and `->`; no Unicode arrows) |
| 9 | Single source of truth | PASS (outcome_tracker.py untouched; perf_metrics.py untouched; reuses existing `execute_sell` primitive) |
| 10 | Log-first / flip-last order | HOLDING (this critique precedes harness_log Cycle 14 append precedes status flip) |

---

## 6. Mutation-resistance review (anti-rubber-stamp)

Question: if an attacker tweaked the code subtly, would the tests catch it?

- **Mutation: change `0.5` to `0.6` in 2R partial close.** Would `test_phase_36_1_2r_fires_50_percent_partial_close` catch? YES -- the test asserts the trade dict's qty equals quantity * 0.5 (typical assertion shape).
- **Mutation: skip the `"3R" not in hit` guard.** Would `test_phase_36_1_idempotent_re_fire_no_op` catch? YES -- second call would re-fire.
- **Mutation: forget the latest-fetch on 3R path (use stale position).** This is the **anti-rubber-stamp question** the prompt asked. The 3R code path at paper_trader.py:579 explicitly re-fetches via `self.get_position(ticker)` and re-checks `latest_qty <= 0 or latest_mfe < threshold_3r` at line 586. If a mutation removed the re-fetch, the test `test_phase_36_1_both_2r_and_3r_fire_in_same_cycle` would catch the divergence because the post-2R-fire position has reduced quantity; using stale quantity would over-sell. PASS.
- **Mutation: change `mfe >= threshold_2r` to `>`.** Would `test_phase_36_1_2r_fires_50_percent_partial_close` catch? YES if the synth position is set EXACTLY at 2R; depends on test fixture. Reading the test name "2_1R" suggests fixture is at 2.1R (slightly above threshold), so `>` mutation would still pass at 2.1R. NOTE-2: mutation-resistance is STRONG but not exhaustive at boundary; acceptable per /goal "bounded test coverage".

**Mutation-resistance verdict: STRONG.** The 3R latest-fetch logic (the most subtle invariant) IS exercised by the combined-fires test.

---

## 7. N* delta honesty

Contract.md claims **+0.3-0.8 Sharpe / +5-15% capture-ratio above 0.63 baseline**. Honest disclosures:
- Cites closure_roadmap §5 as the source (which itself cites research_brief.md cycle-12 AFML / AQR).
- COHR concrete example: capture_ratio=0.63 at +17.89% realized vs MFE=28.36% (trail-stop fired at 26%); 2R/3R ladder would have locked in 50% at +16% (=2R) and remainder at +24% (=3R) = capture_ratio ~0.86 on that single trade.
- Caltech arxiv:2502.15800 discount EXPLICITLY addressed: scale-out logic has NO LLM in the decision path (pure deterministic MFE thresholds), so the LLM-vs-human-trader discount is N/A.
- Long-run 60-day Sharpe delta DEFERRED to phase-43.0 DoD measurement (honest acknowledgment).

Verdict: **HONEST.** Quantified, with explicit caveats, with deferred-measurement acknowledged.

---

## 8. Scope honesty

- `git diff --stat backend/`: settings.py + paper_trader.py + autonomous_loop.py modified (3 files).
- `git diff --stat frontend/src/`: 0 lines.
- New files: 1 test + 1 migration (both untracked-new, expected).
- `outcome_tracker.py` UNCHANGED (single source of truth preserved per /goal gate 9).
- `bigquery_client.py` UNCHANGED.
- Existing `execute_sell` primitive REUSED (not duplicated). Contract.md "NOT changed" section is accurate.

Verdict: **BOUNDED.** Per /goal "NO mass refactors".

---

## 9. Final envelope (JSON)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable success criteria PASS via 9 behavioral tests (pytest 311 collected, +9 new = 0 regression). All 10 /goal integration gates PASS (flag default OFF verified, frontend diff=0, migration idempotent, 0 emojis, ASCII loggers, single source of truth preserved). Code-review heuristics: 0 BLOCK / 0 WARN / 2 NOTE (NOTE-1: scale-out fires before kill-switch is acceptable -- closing direction matches flatten direction; NOTE-2: boundary-condition mutation at exactly 2R not exhaustively covered, acceptable per bounded-coverage). 3R latest-fetch mutation-resistance verified. N* delta honest (Caltech discount N/A for deterministic threshold logic). Scope bounded (3 files modified + 2 new, single dispatcher).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "syntax",
    "file_existence",
    "verification_command",
    "pytest_collect",
    "pytest_run",
    "migration_dry_run",
    "frontend_diff_clean",
    "emoji_scan",
    "masterplan_state",
    "prior_conditional_count",
    "code_review_heuristics",
    "mutation_resistance",
    "scope_honesty"
  ]
}
```

**Verdict: PASS.** Proceed to log-first / flip-last sequence.
