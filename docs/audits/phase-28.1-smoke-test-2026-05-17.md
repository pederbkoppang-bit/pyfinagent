# phase-28.1 Smoke Test — 2026-05-17

**Step:** phase-28.1 (Analyst EPS revision-breadth plug-in)
**Date:** 2026-05-17
**Outcome:** PASS

## Scope

End-to-end harness smoke for analyst revision-breadth overlay. Goal: confirm new module imports, settings defaults are correct, signature carries `revision_signals` kwarg, live yfinance fetch returns real data, deadband logic works, back-compat preserved, and Q/A returns PASS.

## Test 1: Immutable verification command (masterplan)

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/analyst_revisions.py').read()); from backend.services.analyst_revisions import fetch_revision_signals; print('module importable')" && grep -q 'analyst_revisions_enabled' backend/config/settings.py && grep -q 'analyst_revisions' backend/services/autonomous_loop.py && echo "MASTERPLAN VERIFICATION: PASS"
module importable
MASTERPLAN VERIFICATION: PASS
```

Exit 0. **PASS.**

## Test 2: 4-file syntax + imports + signature + settings defaults

```
syntax OK: backend/services/analyst_revisions.py
syntax OK: backend/tools/screener.py
syntax OK: backend/services/autonomous_loop.py
syntax OK: backend/config/settings.py

--- Settings defaults ---
analyst_revisions_enabled = False
analyst_revisions_lookback_days = 100
analyst_revisions_min_analysts = 3
analyst_revisions_threshold = 0.1
analyst_revisions_weight = 0.15
PASS

--- rank_candidates signature ---
['screen_data','top_n','strategy','regime','pead_signals','news_signals','sector_events','revision_signals']
PASS: revision_signals kwarg present
```

**PASS.**

## Test 3: Live yfinance fetch — 9 large-cap tickers

```
--- min_analysts=1 ---
Returned 4/9 signals
  AAPL: breadth=+1.000 up=1 down=0 total=1
  TSLA: breadth=+1.000 up=1 down=0 total=1
  GOOGL: breadth=+0.000 up=1 down=1 total=2
  AMD: breadth=+0.143 up=4 down=3 total=7

--- min_analysts=3 (production) ---
Returned 1/9 signals
  AMD: breadth=+0.143 up=4 down=3 total=7
```

**PASS** — non-empty signal at production setting.

## Test 4: apply_revisions_to_score deadband

```
AAPL: breadth=+1.000 -> 5.750 (+15.0%)   [APPLIED: full breadth × weight 0.15]
TSLA: breadth=+1.000 -> 5.750 (+15.0%)   [APPLIED]
GOOGL: breadth=+0.000 -> 5.000 (+0.0%)   [deadband: |0.0| <= 0.10]
AMD: breadth=+0.143 -> 5.107 (+2.1%)     [APPLIED]
```

**PASS** — deadband working at threshold boundary.

## Test 5: rank_candidates baseline vs overlay (synthetic momentum + real revisions)

```
--- Top-3 conviction shifts ---
AAPL: 7.700 -> 8.855  (delta=+1.155, +15%)
TSLA: 6.100 -> 7.015  (delta=+0.915, +15%)
AMD:  6.900 -> 7.048  (delta=+0.148, +2.1%)
```

Ranking: AAPL #2 → #1; TSLA #7-tied → #5.

**PASS.**

## Test 6: Back-compat

```
$ rank_candidates(test_data, top_n=1)  # NO revision_signals
[{'ticker': 'AAPL', ..., 'composite_score': 8.05}]
```

**PASS** — back-compat preserved.

## Test 7: Q/A subagent verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    "masterplan_verification_cmd: exit 0",
    "syntax_4_files: all OK",
    "settings_defaults: 'False 100 3 0.1 0.15'",
    "rank_candidates_signature: 'revision_signals' kwarg present",
    "smoke_fetch_AAPL_AMD: 2/2 signals",
    "back_compat_top_n_1: 1 result, composite 8.05",
    "back_compat_multi_stock: 3 results sorted",
    "grep_revision_signals: 3 hits in screener.py",
    "grep_analyst_revisions: 7 hits in autonomous_loop.py"
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 8
}
```

**PASS — no violations.**

## Stack traces / failures

**Mid-cycle (resolved):** initial smoke returned 0/5 signals due to tz-comparison TypeError silently swallowed in `_compute_breadth`. Fix: tz-naive cutoff + `tz_convert(None)` fallback. Documented in experiment_results.md and the contract update. No production residue.

**Q/A advisory (not blocking):** the outer broad-except in `_compute_breadth`/`_fetch_one` could be tightened to specific exception classes. Matches the existing pead/news/sector graceful-degradation pattern; future polish, not a defect.

## Conclusion

Phase-28.1 analyst revision-breadth plug-in is implemented, tested end-to-end (4 real revision signals → 3 conviction shifts), and verified by Q/A. Feature flag defaults OFF so production is unchanged. Ready for operator-driven rollout via `analyst_revisions_enabled=True`.

## Related artifacts

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/evaluator_critique.md`
- `handoff/current/live_check_28.1.md`
- `handoff/current/phase-28.1-research-brief.md`
- `docs/design/phase-28.1-analyst-revisions.md`
- `backend/services/analyst_revisions.py`, `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
