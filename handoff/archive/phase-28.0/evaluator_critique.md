# Evaluator Critique — phase-28.0

Q/A subagent: `qa`, 2026-05-17, single pass on cycle-1 evidence.

## Verdict: PASS

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    {
      "cmd": "source .venv/bin/activate && python -c \"import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')\"",
      "exit": 0,
      "output_snippet": "PASS: min_market_cap is either used or removed"
    },
    {
      "cmd": "python -c \"import ast; ast.parse(open('backend/tools/screener.py').read()); print('OK')\"",
      "exit": 0,
      "output_snippet": "OK"
    },
    {
      "cmd": "python -c \"from backend.tools.screener import screen_universe; import inspect; print(list(inspect.signature(screen_universe).parameters))\"",
      "exit": 0,
      "output_snippet": "['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']"
    },
    {
      "cmd": "grep -rnE 'min_market_cap' backend/ scripts/ tests/",
      "exit": 0,
      "output_snippet": "backend/tools/screener.py:83:    phase-28.0 (2026-05-17): removed unused `min_market_cap` parameter. The"
    },
    {
      "cmd": "grep -c 'screen_universe' backend/services/autonomous_loop.py backend/api/backtest.py tests/services/test_screener_sector_propagation.py tests/verify_phase_23_1_13.py",
      "exit": 0,
      "output_snippet": "backend/services/autonomous_loop.py:2\nbackend/api/backtest.py:2\ntests/services/test_screener_sector_propagation.py:10\ntests/verify_phase_23_1_13.py:6"
    }
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 5
}
```

## Audit (5-item harness-compliance — done FIRST)

1. **researcher_gate**: PASS — `handoff/current/phase-28.0-research-brief.md` exists; JSON envelope shows `gate_passed: true`, `external_sources_read_in_full: 5`, `recency_scan_performed: true`, `urls_collected: 15`. Three-variant query discipline visible (current-year frontier, last-2-year, year-less). Contract.md "Research gate summary" cites the brief by path.
2. **contract_before_generate**: PASS — contract.md hypothesis (`remove dead min_market_cap; one-line non-breaking change`) is consistent with what was THEN done in experiment_results.md (signature edit + docstring note). Logical ordering intact.
3. **results_verbatim**: PASS — experiment_results.md shows verbatim shell capture of the immutable verification command (with the `RequestsDependencyWarning` and exact `PASS: min_market_cap is either used or removed` output), plus verbatim smoke output for `screen_universe(tickers=['AAPL','MSFT','NVDA'])`. Not paraphrased.
4. **log_last**: PASS — grep of `handoff/harness_log.md` returns no `phase=28.0 result=PASS` line yet. Main correctly held the log-append for after this Q/A pass.
5. **no_verdict_shopping**: PASS — prior `evaluator_critique.md` contained a phase-27.6 verdict (CONDITIONAL); no phase-28.0 prior verdict existed. This is the first Q/A spawn for phase-28.0.

## Deterministic checks (immutable verification cmd + supporting)

All 5 commands ran to EXIT 0 with expected output. Detail in the JSON above. Notably:
- The signature `['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']` confirms `min_market_cap` is gone.
- The only remaining `min_market_cap` occurrence in `backend/ scripts/ tests/` is line 83 of `screener.py` — inside the docstring, explaining the removal. That is the documented note from the contract, not dead code.
- All 4 caller sites still reference `screen_universe` (counts: autonomous_loop=2, backtest=2, test_screener_sector_propagation=10, verify_phase_23_1_13=6).

## LLM judgment

- **Contract alignment**: PASS. Contract called for "remove `min_market_cap: float = 1e9,` from `screen_universe` signature (line 65 in pre-edit numbering). Update docstring with phase-28.0 note explaining the removal and the $22.7B S&P 500 floor." git diff on `backend/tools/screener.py` shows exactly that: one-line param removal + 6-line docstring note citing "S&P DJI 2024 methodology update".
- **Mutation resistance**: ACKNOWLEDGED-WEAK BY DESIGN. The verification command (`'min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src`) only checks REMOVED-or-USED. The check would not catch a re-added dead param later if the new param contained the substring `market_cap` in the body (e.g., a comment). However, that's the immutable spec for THIS step — the prompt explicitly notes this is fine for a one-time drift fix. Not a defect.
- **Scope honesty**: PASS for the contract-scoped file. `git diff backend/tools/screener.py` is exactly the 8-line change described in experiment_results.md (1 line removed from signature + 7 lines added to docstring). Side note: `git status` shows pre-existing uncommitted changes to `backend/services/autonomous_loop.py` (315-line diff). Inspection confirms those are NOT phase-28.0 work — the inline comments tag them as `phase-27.5.1 + 27.6.5: parallelize per-ticker analysis with PER-PROVIDER bounded concurrency`. These are untracked changes from earlier phase-27 cycles (consistent with `git status` showing many `handoff/archive/phase-27.x/` untracked dirs). Main did NOT smuggle these into phase-28.0; they pre-date it and are simply being held in the working tree.
- **Research-gate compliance**: PASS. Brief has 5 sources read in full (PEP 702, PyAnsys deprecation guide, Seth Larson blog, CFI S&P 500, Python deprecations docs), three-variant query discipline (current-year/last-2-year/year-less), recency scan present, 15 URLs collected, internal code inventory covers all 6 caller files with file:line anchors. Hits the spec floor exactly.
- **Reference-case impact**: NONE. Sandisk (SNDK) / oil majors / defense reference cases are downstream of the screener returning candidates. Removing a dead parameter that filtered nothing changes nothing about which candidates appear. Pure cleanup.

## Code-review heuristics

- No security findings (no secret, no command injection, no broad-except added).
- No trading-domain correctness regressions (kill_switch, stop_loss, perf_metrics, risk_engine paths untouched).
- No anti-rubber-stamp violations (the diff has no financial-logic change to require a behavioral test — it's a dead-code drop).
- No LLM-evaluator anti-patterns (first spawn, real evidence cited).

## Conclusion

XS dead-code removal executed cleanly. Verification command rc=0, signature confirmed clean, docstring note correctly cites the S&P 500 $22.7B inclusion floor (per research brief). All 5 harness-compliance items PASS. Main may now append the harness_log cycle entry, then flip `phase-28.steps[0].status` to `done` in `.claude/masterplan.json`.
