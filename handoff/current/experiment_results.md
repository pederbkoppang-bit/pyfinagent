# Experiment Results — phase-48.3: Live rotation runner + full-kwarg engine_factory

**Cycle:** 14 (Priority 5 follow-on #2; operator-approved). **LLM spend:** $0 (monkeypatched BacktestEngine + injected stub factory/adapter + tmp_path persistence; no real backtest/BQ/LLM). **Result:** ready for Q/A.

## What was built (1 new module + 1 new test file)
`backend/autoresearch/rotation_runner.py` — the live-wiring glue over registry(48.1)→adapter(48.2)→producer(48.1)→selector(47.6):
- `make_rotation_engine(params, settings, bq, *, start_date, end_date, progress_callback)` — full-ctor-kwarg `BacktestEngine` factory. Threads the **8 kwargs `run_harness.make_engine` drops** (market, train/test_window_months, embargo_days, starting_capital, **target_vol**, commission_model, commission_per_share). Validates `strategy` vs `STRATEGY_REGISTRY` FIRST (raise on unknown — no silent triple_barrier fallback). **Maps the seed's `target_annual_vol` → the LIVE `target_vol` ctor arg** (read at trader:89; 0 disables) — this REVIVES tb_risk_managed's vol-targeting (0.15 ON vs tb_baseline 0 OFF), which the brief had thought fully dead. WARNs (does NOT cargo-cult) the genuinely-inert keys (`trailing_*`, `vol_barrier_multiplier`, blend weights — readers reverted in 9fbd9cd6).
- `run_rotation_bakeoff(settings, bq, *, seeds, incumbent, num_param_variants=8, num_trials, start_date, end_date, persist=True, engine_factory=None, adapter_fn=None, log_path, bq_fn, clear_cache_fn)` — builds the full-kwarg factory → 48.2 adapter → `run_strategy_bakeoff`; resolves the incumbent via `load_promoted_params`; PERSISTS the verdict at `allocation_pct=0` (AUDIT ONLY — NO deploy). Two test seams (`engine_factory` full wiring / `adapter_fn` narrow).
- `_resolve_incumbent` (live strategy NAME + optimizer_best DSR; None→first_selection) + `_persist_verdict` (fail-open JSONL row + optional fail-open `bq_fn`).

`tests/autoresearch/test_phase_48_3_rotation_runner.py` — 8 tests + 1 `@pytest.mark.skip` opt-in live smoke.

## KEY FINDING (research-surfaced, reshapes the seed set — flagged for follow-up)
The vol-targeting + trailing-stop engine readers were REVERTED in commit `9fbd9cd6`, so `tb_risk_managed`'s `target_annual_vol`/`trailing_*` overrides write to `engine._strategy_params` with NO reader. The factory's `target_annual_vol`→`target_vol` MAPPING revives the vol-targeting half (target_vol IS a live ctor arg), so tb_risk_managed now differs from tb_baseline by `tp_pct`(6) + `target_vol`(0.15 vs 0). The **trailing-stop half stays inert** (engine logic reverted). **Follow-up:** re-enable the reverted trailing/vol-barrier readers (own cycle) OR swap tb_risk_managed for a seed differentiated purely on live knobs — until then the 4 seeds are ~3.5 distinct (trailing inert). The runner WARNs on the dead keys so this is never masked.

## Verbatim verification output (immutable command + regression)
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/rotation_runner.py').read()); print('ast OK')"
ast OK
$ python -m pytest tests/autoresearch/test_phase_48_3_rotation_runner.py -q
........s                                                                [100%]
8 passed, 1 skipped in 3.25s

# full rotation regression (47.6 + 48.1 + 48.2 + 48.3):
40 passed, 2 skipped in 4.34s
# import OK; no cycle (rotation_runner imports adapter+producer+engine; none import it back)
```

## Success-criteria mapping (masterplan phase-48.3)
1. make_rotation_engine threads the full ctor kwarg set + validates strategy + maps target_annual_vol→target_vol + WARNs dead keys — **MET** (test_make_rotation_engine_threads_full_kwargs, _maps_target_annual_vol_to_target_vol, _raises_unknown_strategy, _warns_on_dead_keys).
2. run_rotation_bakeoff wires factory→adapter→producer→selector, resolves incumbent via load_promoted_params, persists verdict at allocation_pct=0 (no deploy), dual test seams — **MET** (test_seam_A_engine_factory_full_wiring, test_seam_B_adapter_fn_and_incumbent, test_resolve_incumbent_from_loader).
3. $0 deterministic tests (no real backtest/BQ/LLM); live smoke @pytest.mark.skip; fail-open persistence; ast+pytest green; no import cycle; no live-trading-path edit — **MET** (8 passed/1 skipped; test_persist_verdict_failopen_and_persist_false; full-suite 40 passed/2 skipped).
4. records verdict for AUDIT only — explicitly NO promoted_strategies MERGE / NO settings.paper_* mutation; deployment bridge documented DEFERRED — **MET** (allocation_pct=0; DEFERRED block in the docstring).

## Scope honesty / DEFERRED (documented in the module docstring + masterplan + contract)
The $0 tests prove the WIRING, not live DSR/PBO values (engine.run_backtest mocked; the actual ~32-backtest bake-off is the `@pytest.mark.skip` opt-in). DEFERRED: the LIVE bake-off run; the weekly cron; **the deployment bridge (params→settings.paper_* + promoted_strategies MERGE — the keystone that makes rotation change live orders; this runner only RECORDS the verdict)**; re-enabling the reverted trailing/vol-target readers; effective-N clustering; CPCV multi-path. **No live trading path touched** this cycle (no edits to autonomous_loop/portfolio_manager/paper_trader/decide_trades).

## Files
backend/autoresearch/rotation_runner.py, tests/autoresearch/test_phase_48_3_rotation_runner.py, .claude/masterplan.json (phase-48.3), handoff/current/{contract.md, research_brief_phase_48_3_rotation_runner.md}.
