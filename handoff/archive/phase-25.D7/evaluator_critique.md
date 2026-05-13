---
step: phase-25.D7
cycle: 99
cycle_date: 2026-05-13
verdict: PASS
ok: true
---

# Q/A Critique -- phase-25.D7 (cycle 99)

## Harness-compliance audit (5 items)

1. **Researcher spawned?** Yes -- `handoff/current/research_brief.md`,
   tier=simple. Main-authored from inspection of cache.py:184-228 --
   acceptable for a thin constant + guard.
2. **Contract before generate?** Yes -- `handoff/current/contract.md`
   step=25.D7, immutable criteria copied verbatim.
3. **experiment_results present?** Yes.
4. **Masterplan status still pending?** Yes (not yet flipped).
5. **No verdict-shopping?** First spawn; no prior phase=25.D7 entries in
   harness_log.md.

## Deterministic checks

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D7.py
[PASS] 1. macro_max_age_days_constant_35
        -> Found MACRO_MAX_AGE_DAYS=35
[PASS] 2. preload_macro_checks_max_age_days_35_before_caching
        -> compare=True refuse_msg=True
[PASS] 3. warning_log_emitted_on_stale_data_refuse_to_preload
        -> return=0 warning_records=1 cache_populated=False
[PASS] 4. fresh_data_caches_normally
        -> return=3 series_count=2
ALL 4 CLAIMS PASS

$ python3 -c "import ast; ast.parse(open('backend/backtest/cache.py').read())"
AST OK

$ grep -n "MACRO_MAX_AGE_DAYS\|stale data" backend/backtest/cache.py
26:MACRO_MAX_AGE_DAYS = 35
218:    # date across all series with today - MACRO_MAX_AGE_DAYS.
231:        if age_days > MACRO_MAX_AGE_DAYS:
233:                "preload_macro: stale data, refusing to cache "
237:                MACRO_MAX_AGE_DAYS,
```

| Check | Result |
|-------|--------|
| Constant `MACRO_MAX_AGE_DAYS = 35` at module scope | PASS |
| Comparison logic `age_days > MACRO_MAX_AGE_DAYS` | PASS |
| Warning text "preload_macro: stale data, refusing to cache" | PASS |
| Behavioral stale-path: return=0, warning emitted, cache NOT populated | PASS |
| Behavioral fresh-path: return=3, cache populated with 2 series | PASS |
| AST syntax cache.py | OK |

## LLM judgment

- **Contract alignment**: Files (`backend/backtest/cache.py` +
  `tests/verify_phase_25_D7.py`) match the contract Files table.
- **Mutation-resistance**: Claims 3 & 4 are LIVE invocations of
  `preload_macro()` with a mocked BQ client + captured log records,
  exercising BOTH branches (stale-refuse and fresh-success). Removing
  the guard or the warning would fail claim 3; removing the cache
  populate would fail claim 4.
- **Scope honesty**: Per-series staleness (e.g. DGS10 daily vs CPILFESL
  monthly) explicitly deferred. The 35-day default is conservative for
  FRED-monthly (latest observation typically lags by ~30 days).
- **Caller safety**: Stale path returns 0 without populating
  `_macro_full`. Callers already handle 0-row case at cache.py:207, so
  no upstream change required.
- **Research-gate compliance**: research_brief.md present and cited.

## Violations

None.

## Verdict

PASS.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Both immutable criteria met (35-day guard + warning-on-stale refuse-to-preload). 4/4 verification claims PASS including LIVE stale-path + fresh-path invocations. AST OK. Harness-compliance audit clean; no prior CONDITIONALs.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "grep_guard_constant", "mutation_test_stale_path", "mutation_test_fresh_path", "contract_alignment", "scope_honesty", "research_gate"]
}
```
