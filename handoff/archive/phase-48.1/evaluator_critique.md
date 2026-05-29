# Evaluator Critique — phase-48.1: Strategy-rotation foundation (config-driven seed registry + pure per-strategy DSR/PBO producer)

**Q/A agent (merged qa-evaluator + harness-verifier). FIRST Q/A pass on 48.1. Verdict: PASS.**
Deterministic-first, then code-review heuristics, then LLM judgment. Self-evaluation by the orchestrator forbidden; this is an independent verification pass.

---

## STEP 1 — Harness-compliance audit (5 items)

1. **Researcher gate — PASS.** `handoff/current/research_brief_phase_48_1_rotation_foundation.md` exists; JSON envelope (line 47) `"gate_passed": true`, `"external_sources_read_in_full": 8` (>= the 5 floor), `"recency_scan_performed": true`. 4-agent workflow `wf_784c2e77-298`. Contract section "Research-gate summary" (lines 5-12) cites it + the verified selector/gate contract. Sources are Tier-1/2 (Bailey-LdP DSR PDF, PBO/CSCV CRAN, jump-model arXiv 2402.05272, IS/WFA/OOS arXiv 2603.09219, effective-N vertoxquant, buildalpha ensemble).
2. **Contract before generate — PASS.** `handoff/current/contract.md` is the 48.1 contract: step id, research summary, the 4 immutable criteria copied VERBATIM (cross-checked against masterplan `id:"48.1"` success_criteria — exact match), hypothesis, 5 plan steps, out-of-scope/DEFERRED, references.
3. **experiment_results.md present — PASS.** Edits + 2-module/2-test file list + verbatim immutable-command output (`ast OK 2 files` / `23 passed`) + import-spine smoke + success-criteria mapping + explicit "Scope honesty / DEFERRED" block (line 37) naming all 4 deferred items.
4. **Log-last — PASS (correct ordering).** `grep 'phase=48.1' handoff/harness_log.md` -> ABSENT. The log append is the LAST step (after Q/A PASS, before status flip); its absence now is correct, NOT a defect.
5. **No verdict-shopping — PASS.** First Q/A on fresh evidence; no prior 48.1 critique to overturn. (`harness_log.md` has zero `phase=48.1` entries -> no prior CONDITIONAL to count toward the 3rd-CONDITIONAL auto-FAIL rule.)

---

## STEP 2 — Deterministic checks (reproduced, not trusted)

Immutable command (masterplan `id:"48.1"` verification.command), run verbatim:
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/strategy_registry.py','backend/autoresearch/strategy_candidate_producer.py']]; print('ast OK 2 files')" \
  && python -m pytest tests/autoresearch/test_phase_48_1_strategy_registry.py tests/autoresearch/test_phase_48_1_candidate_producer.py tests/autoresearch/test_strategy_selector.py -q
ast OK 2 files
....................... [100%]
23 passed in 0.01s
EXIT_CODE=0
```
23 = 8 (registry) + 7 (producer) + 8 (existing selector) -> **no selector-contract regression** in `test_strategy_selector.py`.

**Selector/gate contract re-read from code (not just the audit):**
- `strategy_selector.py:54-57` — `_strategy_id` resolves id under `strategy_id | strategy | trial_id`. `:95-98` — ranks DSR-desc / PBO-asc. Matches the producer emitting id under key `strategy`.
- `gate.py:28-29` — `PromotionGate.evaluate` drops on `dsr is None or pbo is None` ("missing_dsr_or_pbo"); `:35-38` promotes iff `dsr>=0.95 AND pbo<=0.20`. Matches the producer's mandatory-float dsr+pbo.

**Producer emits the REAL contract + SKIP discipline (independent spot-run, NOT the shipped tests):**
Fed 5 configs to a fixture `backtest_fn`: complete / pbo-OMITTED / raises / non-dict / non-numeric-dsr. Result: only `has_both` emitted; the other 4 ABSENT (each logged a warning, e.g. `"strategy 'no_pbo' metrics missing/invalid dsr|pbo (dsr=0.97 pbo=None); skipping so the gate cannot silently drop it"`). Survivor = `{strategy,dsr,pbo,params,sharpe}` with `isinstance(dsr,float) and isinstance(pbo,float)`. Confirms the producer **SKIPS (no partial dict)** so the gate cannot silently drop a malformed candidate — criterion 2 satisfied at runtime, not just by assertion.

**Registry (LIVE base):** `load_seed_strategies()` off the LIVE `optimizer_best.json` -> `['tb_baseline','mr_short_horizon','qm_trend_tilt','tb_risk_managed']`, types `{mean_reversion, quality_momentum, triple_barrier}` -> >=4 distinct seeds across >=3 strategy TYPES incl. mr+qm+tb. `test_does_not_mutate_module_constant_or_base` + `test_operator_tunable_injected_seeds` + `test_fail_open_empty_base_still_enumerates_ids` cover non-mutation of the module constant, tunability (injected seeds), and fail-open. `load_base_params` reads the inner `params` dict, `except Exception -> {}` (logged) — fail-open confirmed; it overlays `param_overrides` on `{**base, **overrides}` without mutating `SEED_STRATEGIES`.

**No import cycle:** `python -c "import backend.autoresearch.strategy_candidate_producer"` OK. Producer imports only `logging`, `typing`, `strategy_registry`, `strategy_selector` (no engine/BQ/LLM — purity criterion 2). The lone `strategy_candidate_producer` token in registry.py is a docstring reference (line 6), NOT an `import` -> selector and registry do not import the producer back -> acyclic.

**Code-review heuristics (`code_review_heuristics` run, 5 dimensions, no findings that degrade verdict):** no secrets (`secret-in-diff` clean); no bare/silent `except` (`broad-except` clean); no `print`/`eval`/`exec`/`subprocess`/`os.system` (`command-injection`, `insecure-output-handling` clean); logger calls ASCII-clean (`unicode-in-logger` clean); no `requirements`/`pyproject`/`package.json` change (`supply-chain-dep-pin-removal` clean). The two broad `except Exception` (registry:140 fail-open load; producer:94 backtest_fn-raise skip) are the DOCUMENTED fail-open / skip-guard pattern — both LOG a warning and neither sits in a risk-guard/kill-switch/stop-loss/execution path, so the BLOCK-class `broad-except-silences-risk-guard` / `paper-trader-broad-except` do NOT apply. NOTE only, no degradation. No `financial-logic-without-behavioral-test` BLOCK: this cycle does not touch `perf_metrics.py`/`risk_engine.py`/`backtest_engine.py`/`backtest_trader.py`, and the new producer/registry logic ships WITH 15 behavioral tests. No `tautological-assertion` / `over-mocked-test`: the tests use a real in-memory fixture `backtest_fn` and assert ABSENCE of malformed candidates, not mock-called.

---

## STEP 3 — LLM judgment (adversarial)

**HOLLOW-SLICE TRAP — HONEST foundation (the key check, verified against the real `backend/backtest/analytics.py`).** The producer uses an injected fixture `backtest_fn`, so a test PASS does NOT prove live DSR/PBO are computable. I checked the claimed subset against the actual analytics module:
- `generate_report(...)` returns `report["analytics"]` (analytics.py:575-587) containing `"sharpe"` (float = `result.aggregate_sharpe`, :576) and `"deflated_sharpe"` (the DSR in [0,1], float, :577).
- `compute_pbo(pnl_matrix, S=16) -> float` (analytics.py:184) — a SEPARATE function returning PBO as a float.

The producer's `backtest_fn` OUT contract is `{dsr: float, pbo: float, sharpe: float}`. Every one of those three VALUES exists in the real engine and is a float:
  - `sharpe` -> `analytics["sharpe"]` — exact key match.
  - `dsr` (the Deflated Sharpe Ratio) -> `analytics["deflated_sharpe"]` — same quantity, the adapter renames `deflated_sharpe` -> `dsr`.
  - `pbo` -> `compute_pbo(...)` return — a separate float-returning call.
The experiment_results.md:37 wording ("strict SUBSET of `analytics.generate_report()["analytics"]` + `compute_pbo`") is ACCURATE precisely because it conjoins `compute_pbo` — pbo is NOT claimed to come from `generate_report`. The producer docstring (lines 27-32) describes the adapter exactly: `generate_report(...)["analytics"]` for dsr+sharpe, build a per-strategy (T x K-trial) matrix for `compute_pbo`. **Conclusion: the deferral is HONEST and the next-cycle adapter is a genuine drop-in (a thin key-rename + matrix-assembly shim), NOT a rewrite. This is a legitimate foundation, not a hollow slice.**
  - NOTE (no verdict effect): the docstrings phrase it as "generate_report ... for dsr+sharpe" without explicitly flagging that the literal analytics key is `deflated_sharpe` (requiring a one-line rename in the adapter). The value is unambiguously present and the deferral is correctly bounded, so this is a documentation nicety, not a defect.

**DEFERRED documented in BOTH module docstrings — PASS.** registry.py:30-43 + producer.py:24-41 each name: (a) real BacktestEngine adapter (`run_backtest` warm-cache loop -> `nav_history` daily_returns -> `generate_report` DSR + `compute_pbo` T x K matrix), (b) weekly cron, (c) deployment switch + params->settings.paper_* bridge (deploy audit: `best_params` NOT threaded into `decide_trades`/`paper_trader`; flipping a `promoted_strategies` row alone changes only the heartbeat, not live orders), (d) effective-N clustering (plain `num_trials=N` over-deflates — the SAFE direction). Nothing silently dropped (`criteria-erosion` clean).

**Scope / authorization honesty — PASS.** This cycle reversed a SOFT STOP to build Priority 5's completable slice (registry + producer + spine, fixture-tested). What shipped matches the contract; it does NOT overclaim live rotation (experiment_results:37 "No live rotation is implied by this cycle"). `git status --porcelain` diff: only the 2 new modules, 2 new test files, masterplan (the 48.1 block), researcher agent-memory, contract/experiment_results, and append-only audit logs. **NO edits to `autonomous_loop` / `portfolio_manager` / `paper_trader` / `decide_trades`** — confirmed by grep over the porcelain status (`NO live-trading-path file modified`). $0: no LLM/BQ/macro-preload; no requirements/pyproject/package.json change -> no pip cost. `kill-switch-reachability`, `stop-loss-always-set`, `max-position-check-bypass`, `crypto-asset-class`, `llm-output-to-execution-without-validation` all N/A — no execution path touched.

**Mutation-resistance — GENUINE.** The skip-guard tests assert the malformed candidate is ABSENT (`"qm_trend_tilt" not in {c["strategy"] ...}`, `"mr_short_horizon" not in ids`, plus `all(c["pbo"] is not None and c["dsr"] is not None ...)`), not merely "no exception raised". If the producer stopped emitting `pbo`, `test_producer_emits_exact_selector_contract`'s `set(c.keys()) == {"strategy","dsr","pbo","params","sharpe"}` + `isinstance(c["pbo"],float)` fail. If a seed's strategy-type override were dropped, `test_seeds_span_at_least_three_strategy_types` (`{mean_reversion,quality_momentum,triple_barrier} <= types`) + `test_param_overrides_apply_on_top_of_base` fail. If the skip-guard were removed (emitting a partial dict), `test_producer_skips_when_pbo_missing` fails on the absence assertion. Independently re-verified all three skip paths in the spot-run above. Tests bite.

**Anti-churn correctness — PASS (genuinely the below_min_improvement branch).** Independently reproduced `test_bakeoff_anti_churn_retains_incumbent_below_min_improvement`: incumbent `qm_trend_tilt` dsr=0.985 vs top passer `mr_short_horizon` dsr=0.99. `ranked[0]` (`mr_short_horizon`) != incumbent (`qm_trend_tilt`), so this is NOT `incumbent_is_top`. delta=0.005, strictly in (0, 0.01) -> challenger is STRICTLY BETTER but below the 0.01 `min_improvement` -> `reason="below_min_improvement"`, `switched=False`, incumbent retained. The intended anti-churn hysteresis path is exercised, not an accidental `incumbent_is_top`.

---

## Quality-criteria (agent_definitions weights)
- Statistical Validity (40%): DSR>=0.95/PBO<=0.20 gate reused UNCHANGED from `gate.py`; producer emits exactly what the gate consumes; effective-N over-deflation is the SAFE direction and explicitly deferred. **PASS** (no live numbers asserted by design; the contract's claim is composability, which is proven).
- Robustness (30%): fail-open registry + skip-on-malformed producer; spine composes on live + fixture base. **PASS.**
- Simplicity (15%): 2 pure modules, single injected dependency (`backtest_fn`), no new params. **PASS.**
- Reality Gap (15%): touches NO live trading path; the params->settings bridge (the real reality-gap risk) is correctly named as a hard prerequisite for a later cycle, not silently skipped. **PASS.**
No criterion scores below 6.

---

## Verdict

**PASS.** All 4 immutable criteria met and independently reproduced.
1. Registry: >=4 distinct seeds, params = base overlaid with `param_overrides`, >=3 strategy types incl. mr+qm+tb, operator-tunable, fail-open, non-mutating (verified on the LIVE optimizer_best.json + injected base).
2. Producer: PURE (only `backtest_fn` injected, no engine/BQ/LLM import), emits the exact `{strategy,dsr,pbo,params,sharpe}` selector/gate contract (id under `strategy`, dsr+pbo mandatory floats), SKIPS (no partial dict) on raise / non-dict / missing-or-non-numeric dsr|pbo — verified by independent spot-run.
3. `run_strategy_bakeoff` composes registry->producer->`select_best_strategy`: first_selection picks the top-DSR passer (`mr_short_horizon`), a DSR<0.95/PBO>0.20 seed (`tb_risk_managed`) is gate-vetoed, anti-churn retains a DIFFERENT incumbent below `min_improvement`.
4. DEFERRED work documented in BOTH docstrings; `ast OK 2 files`; 23 pytest green incl. the 8 existing selector tests (no regression).
The hollow-slice subset claim is HONEST: every value in the `backtest_fn` OUT contract exists as a float in `generate_report()["analytics"]` (`deflated_sharpe`->`dsr` rename) + `compute_pbo`, so the deferred adapter is a genuine drop-in. No live trading path touched; $0 cost. One NOTE (the `deflated_sharpe`->`dsr` rename is implicit in the docstring) does not degrade the verdict.

violated_criteria: none.
checks_run: syntax, verification_command, evaluator_critique, mutation_test, code_review_heuristics, import_cycle, registry_live_base, producer_skip_guard_spotrun, anti_churn_branch, hollow_slice_subset_verify, live_path_diff_grep, harness_log_absence.
