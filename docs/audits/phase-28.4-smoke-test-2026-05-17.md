# phase-28.4 Smoke Test — 2026-05-17

**Step:** phase-28.4 (Sector-neutral momentum scoring)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import rank_candidates; print('importable')" && grep -qE 'sector.{0,40}rank|percentile' backend/tools/screener.py && echo "MASTERPLAN VERIFICATION: PASS"
importable
MASTERPLAN VERIFICATION: PASS
```

Exit 0. **PASS.**

## Test 2: Settings defaults + signature

```
sector_neutral_momentum_enabled = False
sector_neutral_min_group_size = 3
rank_candidates params: ['screen_data', 'top_n', 'strategy', 'regime', 'pead_signals', 'news_signals', 'sector_events', 'revision_signals', 'sector_neutral', 'sector_neutral_min_group_size']
```

**PASS.**

## Test 3: Smoke — 15 candidates / 4 sectors

| Mode | Tech | Energy | Financials | Health Care |
|---|---|---|---|---|
| Absolute (top-10) | 5 | 3 | 1 | 1 |
| Sector-neutral (top-10) | 3 | 3 | 2 | 2 |

| Top-10 churn | Absolute only | Sector-neutral only |
|---|---|---|
| Dropped | CRM, MSFT (mid-tier Tech) | — |
| Added | — | JPM, JNJ (mid-tier Financials + Healthcare) |

**PASS** — sector concentration reduced; mid-tier names in over-represented sectors swap out for mid-tier names in under-represented sectors.

## Test 4: Edge cases (Q/A independently verified)

- 12-candidate 3-sector dataset with one news_only candidate (no sector) — news_only routed to `_UNKNOWN_` global pool; received pct=1.0 (sole member of pool).
- Sector with N<3 members → merged into global pool.
- Empty `scored` → no-op (no exception).

**PASS.**

## Test 5: Back-compat

```
$ rank_candidates(test_data, top_n=10)  # no new kwargs
[... 10 results with no composite_score_raw field ...]
```

**PASS** — default-OFF mode is byte-identical to pre-phase-28.4 behavior.

## Test 6: Mutation test

Enabling `sector_neutral=True` produces different top-10 ordering than `sector_neutral=False` on the same dataset (CRM, MSFT swap for JPM, JNJ). Confirms the feature actually does something.

**PASS.**

## Test 7: Q/A subagent verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {...},
  "deterministic_checks": {
    "immutable_verification_command_exit": 0,
    "three_file_syntax": "OK",
    "settings_defaults": "False 3",
    "signature_has_new_kwargs": true,
    "back_compat_old_signature_works": true,
    "smoke_12_candidate_3_sector": "PASS",
    "edge_case_small_groups_missing_sector": "PASS",
    "news_only_interaction": "PASS",
    "mutation_test_sn_changes_ordering": "PASS"
  },
  "violated_criteria": [],
  "violation_details": "WARN-not-BLOCK: dim-4 financial logic test lives in handoff artifacts rather than tests/ dir -- acceptable for this cycle, recommend formalizing in a future cycle as backend/tests/test_screener_sector_neutral.py",
  "certified_fallback": false,
  "checks_run": 10
}
```

**PASS** — one WARN-not-BLOCK (formalize the smoke as a `tests/` unit test in a future cycle).

## Stack traces

None.

## Conclusion

Sector-neutral momentum scoring is implemented, tested with real diversification benefit visible in synthetic 4-sector data, and Q/A-verified. Feature flag defaults OFF — current Sharpe 1.1705 preserved until operator A/B-validates.

**This completes the pre-go-live tier (7/7: 28.0, 28.5, 28.1, 28.2, 28.3, 28.6, 28.4).**

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.4.md`, `phase-28.4-research-brief.md`
- `docs/design/phase-28.4-sector-neutral.md`
- `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
