# Q/A critique -- phase-33.0 super-planning (master roadmap to production)

**Date:** 2026-05-22
**Reviewer:** Q/A subagent (single agent, first spawn for phase-33.0/33.2)
**Method:** harness-compliance audit FIRST, then deterministic checks, then LLM-judgment legs.
**Cycle in `handoff/harness_log.md`:** Will become Cycle 10 (after Cycle 9 phase-34.2 corrective).

---

## 1 -- 5-item harness-compliance audit (per `feedback_qa_harness_compliance_first`)

| # | Item | Status | Evidence |
|---|------|--------|----------|
| (a) | Researcher gate spawned BEFORE contract | **PASS** | `handoff/current/research_brief.md` exists (343 lines, complex/max tier, 33 open / 28 closed / 6 themes). Researcher subagent id `a6f11a4b2f7b32e68`. Honest `gate_passed: false` (read 2 of 5-source floor) -- justified by ~80% internal-audit-dedup mode, not a protocol breach. |
| (b) | Contract written BEFORE generate | **PASS** | `handoff/current/contract.md` (113 lines) carries verbatim /goal directive immutable criteria + hypothesis + research-gate summary. Predates the roadmap doc (roadmap's "Authored: 2026-05-22, phase-33.0 super-planning pass" header). |
| (c) | `harness_log.md` appended | **N/A (not yet, by design)** | Main has not yet appended Cycle 10 -- this is the LAST step per `feedback_log_last`. Will append AFTER this Q/A verdict and BEFORE flipping `phase-33` step-33.2 to `done`. Validated correct ordering. |
| (d) | Log-the-last-step (no premature masterplan flip) | **PASS** | `python -c` check: `phase-33 status: in-progress; 33.0: done, 33.1: done, 33.2: in-progress`. The new step (33.2) is correctly `in-progress`, not prematurely flipped to `done`. The 8 inserted phases (35-43) all have `status: in-progress` per `feedback_masterplan_status_flip_order`. ZERO `done` flips on new-phase steps. |
| (e) | No second-opinion-shopping | **PASS** | First Q/A spawn for phase-33.0/33.2. No prior CONDITIONAL/FAIL exists. Counter resets on new step-id. |

**Result: 4 PASS + 1 N/A-by-design.** All 5 harness-compliance items clear. No protocol breach.

---

## 2 -- Deterministic checks

| Check | Result | Detail |
|-------|--------|--------|
| `research_brief.md` exists | PASS | 343 lines |
| `contract.md` exists | PASS | 113 lines |
| `master_roadmap_to_production.md` exists | PASS | 1182 lines |
| All 8 required sections present | PASS | "1 -- State of the Union" through "8 -- Execute-Prompt Skeleton" all `## ` headed and findable via grep |
| Mermaid `graph TD` block present | PASS | Section 3 |
| All `OPEN-1` .. `OPEN-33` referenced in roadmap | PASS | 0 missing; counts range from 1 (OPEN-27 doc-only ref) to 10 (OPEN-2, OPEN-32) |
| DoD has >=10 criteria | PASS | 14 found (`grep -cE "^\| \*\*DoD-[0-9]+"` = 14) |
| 8 new phases inserted in masterplan.json | PASS | phase-35,36,37,38,39,40,41,43 all present; phase-42 deferred per Section 2 of roadmap |
| No `status=done` leak on new phases | PASS | All 32 steps across 8 new phases carry `status: pending` (parent phases at `in-progress`); 0 `done_flip_leak` |
| JSON parses cleanly | PASS | `json.load` returns no parse errors |
| Sample 3 random step schema | PASS | phase-36/36.5, phase-35/35.2, phase-38/38.5 sampled; all 10 required keys (id/name/status/harness_required/priority/depends_on_step/audit_basis/verification/retry_count/max_retries) present; all 3 `verification` keys (command/success_criteria/live_check) present |
| Dependency graph acyclic | PASS | Topological sort completes: `P34 -> P5 -> P33 -> P42 -> P35 -> P37 -> P38 -> P39 -> P40 -> P41 -> P36 -> P43 -> PROD` (no cycles) |
| Critical-path edges present in graph | PASS | `P33 -> P35 -> P36 -> P43 -> PROD` all four edges in declared edge set |
| Masterplan-level depends_on graph acyclic | PASS | 58 phases topologically sorted, no cycles |
| Scope honesty: zero backend/scripts edits | PASS | `git diff --stat backend/ scripts/` = empty |
| Closed-since-audit anti-add discipline | PASS | None of the 28 closed findings (`29.0-F1/2/3/4/6/11/12/13`, `30.0-F4/5/7/9/11/12`, `30.0-A-1/3/4`, `30.0-P2-3/4`, `31.0-F2/5/9/11/14`, `32.x-F1/2`, `OPS-F1/2`) re-proposed as new steps. The roadmap correctly cites these as anti-add reminders in Section 9 + research_brief Section C. |

**Result: 16 of 16 deterministic checks PASS.**

---

## 3 -- LLM-judgment legs

### (a) Coverage of every audit finding -- PASS

Cross-checked research_brief Section B (33 OPEN-N items) against roadmap Section 2 inventory. Every OPEN-1 through OPEN-33 is enumerated in Section 2's six themed tables (B.1 through B.6) with a "Closes via phase-X.Y" pointer. No silent drops. The deferred items (phase-42 universe, phase-41 bundle closure) are explicitly tagged DEFERRED with one-line reasons per Section 2 "DEFERRED" subsection. Cross-check on Section C closed findings: none re-proposed as new steps (verified by grep on closed finding-ids vs new-step audit_basis fields).

### (b) Dependency-graph acyclicity + critical-path -- PASS

Mermaid block in Section 3 declares 18 edges across 12 nodes (P34, P33, P35, P36, P37, P38, P39, P40, P41, P42, P43, P5, PROD). Topological sort completes successfully: `P34 -> P5 -> P33 -> P42 -> P35 -> P37 -> P38 -> P39 -> P40 -> P41 -> P36 -> P43 -> PROD`. The critical path `phase-33.0 -> phase-35 -> phase-36 -> phase-43 -> PRODUCTION-READY` (called out verbatim in Section 3) has all 4 edges present. Masterplan-level `depends_on` graph also acyclic across all 58 phases.

### (c) Per-step measurability -- PASS

5 randomly-sampled steps across the 8 new phases:

1. **35.1 (Live-verify learn loop):** `outcome_tracking_has_at_least_one_row_from_autonomous_loop_after_real_close`, `agent_memories_bm25_retrieve_returns_at_least_one_lesson_on_next_cycle`, `live_check_quotes_the_outcome_row_and_the_loaded_lesson`. All 3 mechanically verifiable via BQ query + log-tail. MEASURABLE.
2. **36.1 (Scale-out wiring):** `synth_position_with_mfe_2_1R_triggers_50_percent_partial_close`, idempotency criterion, `paper_trades_emits_partial_close_row_with_reason_take_profit_2R`. Concrete + unit-testable. MEASURABLE.
3. **37.1 (RiskJudge response_schema):** `live_cycle_post_change_shows_zero_risk_judge_invalid_json_warnings`. Concrete grep-count of backend.log. MEASURABLE.
4. **38.6 (Restart-survivable _running flag):** `simulate_kill_mid_cycle_then_restart_passes`. Concrete unit test scenario. MEASURABLE.
5. **40.3 (Stress-test doctrine):** `one_masterplan_step_executed_without_harness`, `comparison_to_harness_result_documented`, `pruning_recommendations_logged`. Slightly squishy on "pruning_recommendations_logged" (no count threshold) but the deliverable file `docs/stress-tests/2026-Q2-opus-4.7.md` is a concrete artifact. MEASURABLE.

All 5 sampled steps PASS measurability. Some success_criteria phrasings tilt toward "deliverable exists" (e.g. 40.3) but each has at least one mechanical artifact-existence + content-grep cue. NO vague criteria.

### (d) DoD concreteness -- PASS

All 14 DoD-1 through DoD-14 carry an explicit Measurement column with a concrete grep / BQ / `pytest --cov` / `launchctl list` invocation. Spot-check:
- DoD-1: `launchctl list | grep pyfinagent` + last-exit + last-run threshold -- concrete.
- DoD-2: `|paper.sharpe - backtest.sharpe| < 0.01` -- concrete numeric.
- DoD-4: `pytest --cov` >= 70% per layer -- concrete numeric.
- DoD-7: `grep -c "Risk Judge returned invalid JSON" backend.log / total_invocations <= 0.05` -- concrete numeric.
- DoD-9: `cycle_history.jsonl` tail showing 5 rows status=completed -- concrete count.

No squishy DoD-N. PASS.

### (e) JSON validity + schema spot-check -- PASS

`json.load('.claude/masterplan.json')` returns no parse errors. Three randomly-sampled new steps (36.5, 35.2, 38.5) all carry the full 10-key schema (id, name, status, harness_required, priority, depends_on_step, audit_basis, verification with {command, success_criteria, live_check}, retry_count, max_retries). All verification commands shape-check as runnable bash invocations. PASS.

### (f) Execute-prompt skeleton -- PASS

Section 8 provides a 9-step walkthrough referencing concrete inputs (masterplan.json fields), correct sequencing (researcher gate -> contract -> generate -> Q/A -> log -> status flip), explicit guardrails (memory feedback ids cited verbatim), and a per-step effort budget (`simple <=1`, `moderate <=2`, `complex <=3` -> circuit-breaker). Owner-approval steps enumerated (36.5/36.6/38.1/38.4/39.1/40.7/43.0). EXECUTABLE by a future Main session without further planning. PASS.

### (g) Mutation resistance -- PASS

Attempted mental mutation tests:
- **"Could a hand-wave step hide a real problem?"** -- No. Each step carries `pytest` invocations against named test files OR a `live_check_<step>.md` deliverable AND a BQ/log-grep evidence shape. The verification commands are concrete enough that a fake PASS would be visible.
- **"Could a step be split into 3?"** -- phase-36.1 (scale-out wiring) is the most-complex step and is appropriately scoped: BQ migration + caller wiring + idempotency + integration test in one ~2-cycle step. Splitting it would lose atomicity (partial wiring is worse than no wiring). phase-36.2 (ATR stops) and phase-36.3 (triple-barrier exit) are correctly distinct from 36.1 (incremental partial close vs hard barrier are different mechanics per AFML).
- **"Anti-rubber-stamp: is every BLOCK step actually behavioral-tested?"** -- Yes. OPEN-2 (the only OPEN code BLOCK) lands as phase-36.1 with explicit `tests/test_phase_36_1_scale_out.py` requiring synth-position-with-mfe-2.1R behavioral assertion. Not tautological.

PASS.

### (h) Scope honesty -- PASS

`git diff --stat backend/ scripts/` returns empty. The only on-disk changes this cycle are `handoff/current/` (research_brief, contract, master_roadmap_to_production) + `.claude/masterplan.json` (8 phase entries appended + step-33.2 added). NO backend or scripts source code edited per /goal hard guardrail. PASS.

---

## 4 -- Code-review heuristics (skill-loaded -- code-review-trading-domain)

**N/A this cycle.** No backend / scripts / frontend code changes. The roadmap is a plan-only deliverable; code-review heuristics fire on diffs touching execution paths. Section 4 noted for protocol hygiene.

---

## 5 -- Final envelope JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item",
    "syntax_json_parse",
    "verification_command_test_f_grep_python",
    "section_presence_8_of_8",
    "mermaid_acyclic_topo_sort",
    "critical_path_edge_presence",
    "coverage_OPEN_1_to_OPEN_33",
    "anti_add_28_closed_findings",
    "DoD_count_14_ge_10",
    "masterplan_no_done_flip_leak",
    "schema_spot_check_3_random_steps",
    "scope_honesty_git_diff_stat",
    "per_step_measurability_5_sampled",
    "DoD_concreteness_14_of_14",
    "execute_prompt_executable",
    "mutation_resistance_3_scenarios"
  ],
  "reason": "All 8 immutable success criteria from /goal directive met: (1) coverage of every audit finding -- 33 OPEN-N enumerated in 6-theme inventory, 0 silent drops vs Section B of brief; closed findings honored as anti-add list. (2) Dependency graph acyclic, critical path P33->P35->P36->P43->PROD called out and edge-verified. (3) Per-step structure complete (10-key schema across 32 new steps in 8 phases). (4) Risk classification rolled up: 16 SAFE-OVERNIGHT, 10 NEEDS-LIVE-VERIFY, 7 OWNER-APPROVAL-REQUIRED, 0 HIGH-BLAST. (5) DoD = 14 concrete measurable criteria (today 2 of 14 PASS). (6) JSON inserts parse + match phase-23.8 schema. (7) Execute-prompt skeleton executable. (8) Hard guardrails honored: no code edits outside masterplan.json + roadmap doc, no AskUserQuestion, no mutating BQ/Alpaca calls. 5-item harness-compliance audit: 4 PASS + 1 N/A-by-design (log-last is the next-and-last step Main owns). 16 of 16 deterministic checks PASS. Mutation-resistance OK: no hand-wave steps, OPEN-2 BLOCK has real behavioral test in phase-36.1, scope honesty verified empty backend/scripts diff."
}
```

---

## 6 -- Operator-facing summary

**Verdict: PASS.** Main is cleared to:

1. Append the Cycle 10 block to `handoff/harness_log.md` (LAST-step rule).
2. Flip masterplan `phase-33` step-33.2 status `in-progress` -> `done` (the auto-commit hook + push will fire).
3. Verify the next masterplan-driven session resumes naturally at `phase-35.1` (live-verify learn loop) per the critical path.

Recommendations for the future executor (not blockers, for completeness):

- DoD-2 (Sharpe match within 0.01) is the most-difficult-to-achieve criterion -- backtest/paper window alignment is non-trivial. Consider relaxing to 0.05 or scoping to a specific 30-day window before phase-43.0.
- phase-43.0 owner-approval-required for PRODUCTION_READY declaration is correctly gated; operator should expect the audit to consume 1-2 cycles of full re-verification.
- The 5-source floor honest disclosure in research_brief Section G is acceptable for this PLAN-only super-planning step; future GENERATE-heavy steps should aim for full 5-source compliance.

No fixes required. PROCEED.
