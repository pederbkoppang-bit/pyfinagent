# Q/A Evaluator Critique — phase-48.2: Rotation real-engine adapter (make_engine_backtest_fn)

**Verdict: PASS** (`ok: true`, zero violated_criteria). FIRST Q/A pass on 48.2.
Single merged Q/A agent (deterministic reproduction + LLM judgment). Replaces the
archived 48.1 verdict wholesale.

---

## STEP 1 — Harness-compliance audit (5/5, evidence cited)

1. **Researcher gate — PASS.** `handoff/current/research_brief_phase_48_2_rotation_adapter.md`
   exists; JSON envelope (line 33) `"gate_passed":true`, `"external_sources_read_in_full":6`
   (>=5 floor), `"recency_scan_performed":true`, 3-agent workflow `wf_2ab3cff3-74f`.
   Contract cites it verbatim in the Research-gate summary (contract.md:5-6) and
   References (contract.md:35).
2. **Contract-before-generate — PASS.** `handoff/current/contract.md` IS the 48.2 contract:
   step id, research summary, the 4 immutable criteria copied verbatim (contract.md:18-22,
   byte-identical to masterplan `success_criteria`), hypothesis, 3 plan steps, refs.
3. **experiment_results present — PASS.** Edits + 1-module+1-test file list, verbatim
   verification output (9 passed/1 skipped + 32-test regression), success-criteria mapping,
   and a DEFERRED/scope-honesty block (experiment_results.md:37-38).
4. **Log-last — PASS (not a defect).** `grep -c 'phase=48.2' handoff/harness_log.md` = 0.
   The +17 harness_log lines are the separate scheduled `run_harness.py` process's
   `## Cycle 1 -- 2026-05-29 05:53 UTC` block + the archived 48.1 closure — correctly
   ignored per instructions. No premature 48.2 block.
5. **No verdict-shopping — PASS.** First Q/A on 48.2, fresh evidence, no prior 48.2 verdict.

---

## STEP 2 — Deterministic checks (reproduced, not trusted)

| Check | Command | Result |
|-------|---------|--------|
| AST | `python -c "import ast; ast.parse(open('backend/autoresearch/strategy_backtest_adapter.py').read()); print('ast OK')"` | `ast OK`, **exit 0** |
| Immutable pytest | `python -m pytest tests/autoresearch/test_phase_48_2_backtest_adapter.py -q` | **9 passed, 1 skipped**, **exit 0** (IMMUTABLE_EXIT=0) |
| Full rotation regression | `pytest test_phase_48_1_* test_phase_48_2_* test_strategy_selector.py -q` | **32 passed, 1 skipped**, **exit 0** (REGRESS_EXIT=0) — no regression |
| Import cycle | `python -c "import backend.autoresearch.strategy_backtest_adapter"` | `import OK`, **exit 0** (only a benign urllib3 RequestsDependencyWarning) |

Immutable command (verbatim from masterplan phase-48.2.verification.command):
`...ast.parse(...) && python -m pytest tests/autoresearch/test_phase_48_2_backtest_adapter.py -q`
— reproduced exactly; ast OK + 9 passed/1 skipped as claimed in experiment_results.md:18-22.

---

## STEP 3 — LLM judgment (adversarial — three cycle-specific checks)

### (a) PBO-METHOD CORRECTNESS (the crux) — CORRECT
Bailey/Borwein/LdP/Zhu Algo 2.3 requires CSCV columns = competing CONFIGURATIONS of
ONE strategy, NOT a single backtest's time windows. Verified the adapter implements
exactly this:
- `_default_param_grid` (strategy_backtest_adapter.py:94-129) holds the `strategy`
  categorical **FIXED** (line 104-108 validates + keeps it; `base = dict(seed_params)`
  copies it into every variant; only `holding_days`/`mr_holding_days`/`tp_pct` knobs are
  jittered at lines 113-127). Columns are therefore K configs of ONE strategy. OK
- `_assemble_pbo_matrix` (lines 132-152) builds (T, N=K) from each variant's
  `nav_history` -> `_daily_returns_from_nav` (line 144) -> `np.column_stack` (line 152),
  then `compute_pbo(matrix, S=pbo_S)` is called on it (line 247). It does NOT use
  per-window scalars as columns — `WindowResult.total_return_pct` (hardcoded 0) is never
  read; only nav-derived daily returns feed the matrix. OK
- Re-confirmed `compute_pbo` (analytics.py:184-236): input `(T, N)`, `T, N = arr.shape`
  (line 204), and the silent `if N < 2 or T < S * 2: return 0.0` at line 205. This is the
  degenerate-0.0 hazard the guard defends against. Orientation matches the adapter's
  column-stack. OK

### (b) LOAD-BEARING GUARD genuineness (most important) — GENUINE, NOT BYPASSABLE
Independently reproduced (NOT via the project's own tests — a separate ad-hoc Q/A probe
using the REAL `BacktestResult`/`WindowResult` dataclasses + REAL `generate_report` +
`compute_pbo`):
- **(a) `_assemble_pbo_matrix` returns None** for N<2 (single usable column) AND for
  T<min_rows: probe with n=10 navs (T=9) -> matrix None; probe with 1 usable col + rest
  empty -> None. OK
- **(b) on None the adapter OMITS `pbo`** (does NOT set 0.0): short-nav probe returned
  `{'dsr':0.0,'sharpe':1.3,'n_variants':4,'n_windows':3}` — **`'pbo' not in out`**;
  single-column probe likewise omitted pbo. OK (matches adapter lines 238-245).
- **(c) producer then SKIPS**: `build_per_strategy_candidates([...], short_fn)` returned
  **`[]`**; the producer logged `metrics missing/invalid dsr|pbo (dsr=0.0 pbo=None);
  skipping so the gate cannot silently drop it` (strategy_candidate_producer.py:107-113,
  the `if dsr is None or pbo is None: ... continue` branch). OK
- **Anti-trivial control:** the HEALTHY path (4 cols x T>=32) DID emit a real
  `pbo=0.9893` (valid float in [0,1]) — so the guard is not a blanket always-omit; it
  fires ONLY on undersized/degenerate matrices. **The guard cannot be bypassed by
  short-nav or single-column inputs; pbo=0.0 cannot leak through.** OK

This is the correctness lynchpin of the cycle and it holds: an overfit/undersized
strategy can never false-pass the `pbo<=0.20` gate via a silent 0.0 — it is skipped.

### (c) Mock validity / hollow-slice — VALID, deferral HONEST
- `_extract_dsr_sharpe` (lines 155-164) reads `generate_report(seed_result,...)
  ["analytics"]["deflated_sharpe"]` — NOT a hardcoded fake. Verified `generate_report`
  (analytics.py:536-568) genuinely computes `dsr = compute_deflated_sharpe(observed_sr=
  result.aggregate_sharpe, num_trials, variance_of_srs=sr_variance, ...)`. My live probe
  showed dsr VARYING across inputs (0.0 on a degenerate/short series, 1.0 on a healthy
  one), proving it is a real computation. Test `test_extract_dsr_sharpe_matches_generate_report`
  (test:95-101) asserts `dsr == rep["analytics"]["deflated_sharpe"]` directly. OK
- `compute_pbo` IS exercised on a real (T>=32, N>=2) matrix: `test_compute_pbo_happy_path_hand_matrix`
  (test:104-107) uses a hand-built (40,4) matrix; `_assemble_pbo_matrix` healthy branch
  (test:115-117) asserts `shape[0] >= 32 and shape[1] == 4`. The tests mock ONLY
  `engine.run_backtest` (returning a hand-built REAL `BacktestResult` with real
  `WindowResult`s + ~45 nav rows) and run the REAL pure-numpy generate_report+compute_pbo
  — NOT over-mocked (does not patch the module under test). OK
- **Deferral honesty:** the LIVE multi-run bake-off (4 seeds x K~8 = ~32 real backtests,
  tens of minutes) is honestly DEFERRED behind `@pytest.mark.skip` (test:172-184) with a
  documented future live_check; it is slow/compute-bound, not hiding a defect. The
  make_engine-kwarg-subset risk (vanilla `run_harness.make_engine` threads no
  target_vol/trailing/blend -> a live run would silently ignore `tb_risk_managed` overrides)
  is HONESTLY DISCLOSED in the module docstring (lines 49-53), the contract (line 32), and
  experiment_results.md:38 — flagged as the live caller's responsibility, not buried. OK

### (d) Scope / authorization — IN SCOPE, $0, NO live path
- Operator "continue you have my approval" -> Priority-5 follow-on #1 (replace the 48.1
  injected `backtest_fn` with the real engine adapter). In scope. OK
- **Diff scope** (`git status --porcelain`): only NEW files
  `backend/autoresearch/strategy_backtest_adapter.py` + `tests/autoresearch/test_phase_48_2_backtest_adapter.py`
  are code; `.claude/masterplan.json` + handoff files are bookkeeping. **Grep for
  `autonomous_loop|portfolio_manager|paper_trader|decide_trades|kill_switch|risk_engine|
  backtest_engine.py|analytics.py` -> NONE.** No live trading path touched; no engine/
  analytics source edited (the adapter only IMPORTS them). OK
- **$0**: no real backtest/BQ/LLM/macro — engine.run_backtest is mocked; only pure-numpy
  generate_report+compute_pbo run on fakes. OK
- Warm-cache clear-once: `test_adapter_emits_full_metrics_and_clears_cache_once` (test:121-132)
  asserts `calls["clear"] == 1`; adapter calls `_clear()` ONCE in the `finally`
  (lines 217-228). Strategy-name reject: `test_adapter_unknown_strategy_raises_and_producer_skips`
  (test:147-153) asserts `pytest.raises(ValueError)` + producer returns `[]`. Both genuinely
  tested. OK

---

## Code-review heuristics (5 dimensions evaluated) — no BLOCK/WARN findings
- Security: no secret-in-diff, no command-injection (no subprocess/eval/exec), no
  prompt-injection path, no dep-pin removal. Adapter imports NO settings/BQ. NOTE only.
- Trading-domain: adapter is OFF the execution path (no kill_switch/stop-loss/paper_trader
  edits); the load-bearing PBO guard strengthens (does not weaken) the overfit gate. OK
- Code quality: type hints present on the public factory; per-variant try/except is scoped
  (drops a column, not a silent risk-guard swallow — it logs a warning); ASCII-only
  logger calls verified (no Unicode). The broad `except Exception` at adapter:212,227 is
  acceptable — it is a degrade-one-variant / clear-cache-cleanup path with a WARN log, NOT
  a risk-guard/kill-switch swallow (negation-list exempt).
- Anti-rubber-stamp: financial logic (PBO/DSR wiring) HAS behavioral tests exercising the
  real path; no tautological assertions (tests assert real equalities + shapes + skip
  behavior); not over-mocked.
- LLM-evaluator anti-patterns: first verdict, fresh evidence, no rebuttal context, every
  finding cited to file:line + reproduced command output.

---

## checks_run
syntax, verification_command, regression_suite, import_cycle, load_bearing_guard_probe
(independent), pbo_method_correctness, mock_validity, diff_scope, code_review_heuristics,
research_brief, contract_alignment, experiment_results, harness_log_absence

## violated_criteria
(none) — all 4 immutable criteria MET:
1. adapter factory + 4 pure helpers + producer-boundary shape; dsr from generate_report,
   pbo from a separate per-strategy (T x K) compute_pbo on K-variant nav-derived returns. OK
2. LOAD-BEARING undersize guard -> no pbo -> producer skips (independently reproduced;
   pbo=0.0 cannot leak; healthy path still emits a real pbo). OK
3. strategy-name validated vs STRATEGY_REGISTRY (raise->skip, no silent triple_barrier);
   warm-cache clear-once; no settings/BQ import / no cycle. OK
4. $0 mock test (real generate_report+compute_pbo on fakes), undersize/reject/end-to-end;
   live test @pytest.mark.skip; ast clean; pytest green (9 passed/1 skipped). OK

## NOTE (non-blocking, does not degrade verdict)
The producer docstring (strategy_candidate_producer.py:24-39) still describes the adapter
as DEFERRED ("The REAL backtest_fn: ... is a drop-in next cycle"). Now that 48.2 ships it,
a future doc-touch could update that DEFERRED bullet to point at strategy_backtest_adapter.py.
Cosmetic only.
