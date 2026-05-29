# Evaluator Critique — phase-48.3 (CYCLE-2 re-evaluation)

**Verdict: PASS** | **Cycle:** 2 (fresh Q/A on RESTORED evidence) | **Date:** 2026-05-29

> This is the cycle-2 re-evaluation. The first 48.3 Q/A returned **CONDITIONAL**
> with `violated_criteria: ["masterplan-step-missing", "contract-before-generate"]`.
> Its finding was unambiguous: the ENGINEERING passed every adversarial check; the
> ONLY blockers were two NON-SKIPPABLE protocol artifacts lost to concurrent-writer
> collisions (the scheduled `run_harness.py` clobbered `contract.md`; the masterplan
> 48.3 step had been reverted). The orchestrator re-established BOTH per the cycle-1
> critique's own "Required to clear to PASS" steps 1-3. This pass reads the restored
> evidence — the documented cycle-2 flow (files that did not exist now exist;
> `contract.md` mtime 19:49 is the 48.3 contract, not the prior 48.2 one), NOT
> verdict-shopping on unchanged evidence (Anthropic harness-design file-based
> fresh-respawn; CLAUDE.md cycle-2 block). The verdict change reflects the fix, not a
> different opinion on the same evidence. (Sycophancy-under-rebuttal check, Dim 5:
> the reversal is post-fix — evidence materially changed — so it is the documented
> flow, NOT `Circular_Reasoning`/`sycophancy-under-rebuttal`.)

---

## STEP 1 — The two prior blockers are RESOLVED (focus of this pass)

| Blocker | Cycle-1 finding | Cycle-2 state | Evidence |
|---|---|---|---|
| **B1 masterplan-step-missing** | no phase-48.3 entry in `.claude/masterplan.json`; criteria absent from tracker | **RESOLVED** | `grep -c '"id": "48.3"'` = **1**; `python -c "import json; json.load(...)"` = **VALID**; `status: in-progress` (NOT done — correct, status-flip is after PASS+log); `harness_required: true`; `verification.command` present; **4** `verification.success_criteria` (they live under the `verification` object, not the step root — that is why a naive top-level walk reads 0); `retry_count: 0 / max_retries: 3` (no certified-fallback risk). |
| **B2 contract-before-generate** | `contract.md` was the 48.2 contract; cited the 48.2 brief; `grep research_brief_phase_48_3` = 0 | **RESOLVED** | `head -1 contract.md` = `# Contract — phase-48.3: Live rotation runner + full-kwarg engine_factory`; cites `research_brief_phase_48_3_rotation_runner.md` (**2** refs); the 4 "Immutable success criteria" (`contract.md:19-23`) are a **verbatim** copy of the masterplan `verification.success_criteria`. mtime 19:49 (re-established this cycle). |

---

## STEP 2 — Harness-compliance audit (5 items) + deterministic re-check

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher gate | **PASS** | `research_brief_phase_48_3_rotation_runner.md` (33,607 B, mtime 19:32); envelope `external_sources_read_in_full: 5`, `gate_passed: true`. |
| 2 | Contract pre-commit | **PASS** | B2 above — 48.3 contract, criteria verbatim, cites the 48.3 brief. |
| 3 | experiment_results present | **PASS** | `experiment_results.md` head = phase-48.3; success-criteria mapping + scope-honesty/DEFERRED section; verbatim command output; mtime 19:39. |
| 4 | Log-last (no premature 48.3 block) | **PASS** | `grep -c 'phase=48.3' handoff/harness_log.md` = **0** — correct; the log block is the LAST step (after this PASS, before the status flip). The scheduled-harness "Cycle N" blocks are a separate process, correctly ignored. |
| 5 | No verdict-shopping | **PASS** | Prior 48.3 verdict lived ONLY in `evaluator_critique.md` (0 `phase=48.3` log blocks). Evidence materially changed: masterplan 48.3 step restored + contract re-established 19:49 (both absent in cycle-1). Documented cycle-2 fresh-respawn — `violation_type` NOT applicable. |

**Deterministic re-run (verbatim, exit codes):**

```
$ python -c "import ast; ast.parse(open('backend/autoresearch/rotation_runner.py').read()); print('ast OK')"
ast OK
$ python -m pytest tests/autoresearch/test_phase_48_3_rotation_runner.py -q
........s
8 passed, 1 skipped in 3.22s                                   # EXIT=0  (immutable verification.command)

# full rotation regression (47.6 + 48.1 + 48.2 + 48.3):
$ python -m pytest test_strategy_selector test_phase_48_1_* test_phase_48_2_* test_phase_48_3_* -q
40 passed, 2 skipped in 4.40s                                  # EXIT=0, no regression

$ python -c "import backend.autoresearch.rotation_runner"
import OK: backend.autoresearch.rotation_runner                # no cycle (imports adapter+producer+engine; none import it back)
```

All reproduce the experiment_results.md claims exactly.

---

## STEP 3 — Engineering re-confirmed INDEPENDENTLY (fresh instance; not trusting cycle-1)

- **kwarg-name correctness [the latent-live-TypeError guard].** Via `inspect.signature(BacktestEngine.__init__)` vs an AST walk of the `BacktestEngine(...)` call in `make_rotation_engine`: ctor params (**27**) == passed kwargs (**27**), **set equality, ZERO unknown kwargs**. All 8 named make_engine drops present (`market, train_window_months, test_window_months, embargo_days, starting_capital, target_vol, commission_model, commission_per_share`). A typo'd name — which the test's `**kwargs`-capturing mock would NOT catch — would surface here as an `UNKNOWN`; none found. `rotation_runner.py:114-142` vs `backtest_engine.py:136-173`.
- **target_vol revival.** `target_annual_vol`→`target_vol` map with precedence explicit `target_vol` > `target_annual_vol` > `0.15` at `rotation_runner.py:99-104`. `target_vol` IS the live reader at `backtest_trader.py:89` (`vol_scale = min(self.target_vol / stock_vol, 3.0)`; the `stock_vol <= 0` guard at :87 + `target_vol=0` ⇒ `vol_scale=0` ⇒ sizing disabled; 0.15 ⇒ enabled). Engine threads it ctor→trader at `backtest_engine.py:219`. Test `test_make_rotation_engine_maps_target_annual_vol_to_target_vol` asserts 0.15→0.15, 0→0, explicit-0.2-wins-over-0.1.
- **AUDIT-ONLY / no-deploy [critical].** `git status --porcelain` = exactly **3 untracked files** (`backend/autoresearch/rotation_runner.py`, the brief, the test). **ZERO** edits to autonomous_loop / portfolio_manager / paper_trader / decide_trades. `allocation_pct: 0.0` hard-coded at `_persist_verdict` `:204` ("zero == recorded, NOT deployed"). The only `settings.paper_*` / `promoted_strategies` / `MERGE` tokens in the module are in the **DEFERRED docstring** (`:32`) describing what is NOT built. `load_promoted_params` is imported+called read-only (`:168-169`; def at `autonomous_loop.py:46` is a BQ-promoted-params READ, lazy + fail-open). This runner cannot change live trading.
- **dead-key WARN genuine + honestly disclosed.** `_DEAD_KEYS` (`:55-64`) WARNed (not threaded) at `:106-112`. Independently confirmed inert: the only `trailing_*`/`vol_barrier_multiplier` matches in `backend/backtest/` are the optimizer search-space ranges + the **setter** (`quant_optimizer.py:90-109, :554-562` writing `engine._strategy_params[...]`) — **no engine reader**; blend-weight (`tb/qm/mr/fm_weight`) grep in engine+trader = empty. Trailing half honestly disclosed inert (docstring `:16-24`; results "KEY FINDING"); seeds ~3.5 distinct, flagged follow-up. Test `test_make_rotation_engine_warns_on_dead_keys` asserts the WARN fires.
- **Seam-A genuine wiring (not a hollow stub-all).** `test_seam_A_engine_factory_full_wiring` (`:122`) injects ONLY the `engine_factory` leaf (`_FakeEngine` whose `run_backtest` returns a REAL `BacktestResult` with 3 `WindowResult` + ~45 nav rows). The rest is REAL code: `make_engine_backtest_fn` (48.2 adapter) → `run_strategy_bakeoff` (producer) → `load_seed_strategies` (registry 48.1) + `select_best_strategy` (selector 47.6) at `strategy_candidate_producer.py:147-150`; the producer builds the T×K matrix for `analytics.compute_pbo`. Asserts the verdict shape AND a persisted row at `allocation_pct==0.0`. Seam-B (`:141`) narrowly stubs `adapter_fn` to exercise incumbent + selector wiring.

---

## Code-review heuristic sweep (5 dimensions evaluated — no BLOCK, no WARN)

- **Dim 1 Security:** no secret-in-diff, no command/SQL/path/SSRF injection, no prompt-injection path, no dep-pin removal, no unbounded-llm-loop. The lazy `load_promoted_params` import is a READ. No findings.
- **Dim 2 Trading-domain:** kill-switch / stop-loss / perf-metrics / max-position / crypto invariants all N/A — the module touches NO execution path (git diff confirms). No `llm-output-to-execution` path added. No findings.
- **Dim 3 Code quality:** the 4 `except Exception` (`:151, :170, :215, :220`) are fail-open audit-persistence / incumbent-resolution in a NON-risk-guard/NON-execution module — per the negation list, acceptable (NOT the `paper_trader.py:26,52` swallow anti-pattern). ASCII-only, type-hinted, no `print()`. No findings.
- **Dim 4 Anti-rubber-stamp:** `financial-logic-without-behavioral-test` **N/A** — diff does NOT touch `perf_metrics/risk_engine/backtest_engine/backtest_trader` AND a behavioral test ships regardless. No tautological / `assert_called` / over-mocked (monkeypatches the `BacktestEngine` dependency, never the SUT) / rename-as-refactor. No findings.
- **Dim 5 LLM-evaluator anti-patterns:** verdict cites file:line throughout (not `missing-chain-of-thought`). Reversal is post-fix on changed evidence → documented cycle-2, NOT `sycophancy-under-rebuttal`/`second-opinion-shopping`. This is the FIRST 48.3 CONDITIONAL (0 prior in harness_log) so `3rd-conditional-not-escalated` N/A. No findings.

`checks_run` appends `code_review_heuristics` (all 5 dimensions evaluated, zero findings).

---

## Success-criteria verdict (4/4 MET)

1. `make_rotation_engine` threads the FULL ctor kwarg set with exact names + STRATEGY_REGISTRY raise-on-unknown + `target_annual_vol`→`target_vol` map + dead-key WARN — **MET** (`rotation_runner.py:76-142`; inspect.signature set-equality; 4 dedicated tests).
2. `run_rotation_bakeoff` wires registry→48.2-adapter→producer→`select_best_strategy`, resolves incumbent via `load_promoted_params` (None→first_selection), PERSISTS verdict at allocation_pct=0 (AUDIT ONLY), exposes BOTH seams, fail-open `_persist_verdict` — **MET** (`:225-274`; Seam-A/Seam-B/incumbent/fail-open tests).
3. AUDIT-ONLY/no-deploy: no `promoted_strategies` MERGE, no `settings.paper_*` mutation, `allocation_pct` hard-coded 0.0, only a READ of `load_promoted_params`, ZERO live-module edits, deployment bridge DEFERRED — **MET** (git diff = 3 untracked; `:204`; `:168-169`).
4. $0 deterministic tests (monkeypatched ctor-kwarg capture + injected stub seams + tmp_path) + live ~32-backtest bake-off `@pytest.mark.skip` opt-in + ast clean + pytest green + no import cycle + full rotation regression green — **MET** (8 passed/1 skipped; 40 passed/2 skipped; `test_phase_48_3_rotation_runner.py:187` skip).

---

## Violated criteria

**NONE.** B1 (masterplan-step-missing) and B2 (contract-before-generate) are both RESOLVED; the deterministic checks reproduce; the engineering (full-kwarg factory with exact names, target_vol revival, AUDIT-ONLY no-deploy audit row, genuine Seam-A wiring, honest dead-key WARN) holds; no code-review heuristic fired at BLOCK or WARN.

## checks_run
syntax, file_existence, verification_command, full_rotation_regression, import_cycle, masterplan_step_present, masterplan_json_valid, contract_alignment_verbatim, research_gate_envelope, harness_log_check, kwarg_name_inspect_signature, target_vol_revival_trace, no_deploy_gitdiff, dead_key_inertness, seam_a_wiring, code_review_heuristics, harness_compliance_5item, evaluator_critique_prior, experiment_results
