# Experiment Results — phase-10.4 (Friday promotion gate)

**Step:** 10.4 **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate): 6 sources in full, 14 URLs, recency scan, gate_passed=true. Brief at `handoff/current/phase-10.4-research-brief.md`. Grounded DSR/PBO ranking, clarified "5% starting allocation" as ledger-notes marker (not direct `position_size` call), and carried forward the Thursday-append edge flagged by qa_103_v1.
2. Contract authored at `handoff/current/phase-10.4-contract.md`.
3. Created `backend/autoresearch/friday_promotion.py` (143 lines):
   - Public `run_friday_promotion(week_iso, *, candidates, top_n=1, max_n=3, starting_allocation_pct=0.05, gate=None, ledger_path=None) -> dict`
   - Returns `{promoted_ids, rejected_ids, allocations, already_fired, error}`
   - **Fail-closed** on missing Thursday row or empty `thu_batch_id` → `error="no_thursday_batch_on_ledger"`
   - **Idempotent:** second call with same `week_iso` when `fri_promoted_ids` is populated returns `already_fired=True`
   - **Ranking:** DSR desc, PBO asc; take `min(top_n, max_n)` from passed
   - **Notes preservation:** reads prior `notes` (e.g. Thursday's `"kicked_off"`), appends `"; starting_alloc=0.05"`, avoids losing provenance
   - **Ledger write preserves all prior columns** (`thu_batch_id`, `thu_candidates_kicked`, `cost_usd`, `sortino_monthly`) rather than defaulting them
   - ASCII-only logger messages
4. Created `scripts/harness/phase10_friday_promotion_test.py` (~150 lines) — 4 cases matching the masterplan success_criteria verbatim.
5. Created `tests/autoresearch/test_friday_promotion.py` (~160 lines) — 9 pytest cases:
   - 4 masterplan-mirror cases
   - `test_fail_closed_when_thursday_row_missing`
   - `test_fail_closed_when_thu_batch_id_empty`
   - `test_empty_candidates_does_not_raise`
   - `test_preserves_thursday_notes_kicked_off` — regression guard for the notes concatenation
   - `test_ranks_by_dsr_desc_then_pbo_asc` — explicit tie-breaker test

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/friday_promotion.py','scripts/harness/phase10_friday_promotion_test.py','tests/autoresearch/test_friday_promotion.py']]; print('AST OK')"
AST OK

$ python scripts/harness/phase10_friday_promotion_test.py
[PASS] routine_consumes_exactly_1_slot  (r1.af=False, r2.af=True, rows=1)
[PASS] reuses_phase_8_5_5_dsr_pbo_gate  (promoted=['g1', 'g2'], rejected=['b1', 'b2'])
[PASS] promotion_at_5pct_starting_allocation  (notes='kicked_off; starting_alloc=0.05', allocations=[0.05])
[PASS] top_n_default_1_max_3  (default=1, three=3, capped=3)

ALL PASS  (4/4)
(exit 0)

$ pytest tests/autoresearch/test_friday_promotion.py -q
.........                                                                [100%]
9 passed in 0.42s

$ pytest tests/autoresearch/ tests/slack_bot/ -q
.................................................................        [100%]
65 passed in 1.73s
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `routine_consumes_exactly_1_slot` | PASS — r2 returns already_fired=True; 1 row; matching promoted_ids |
| 2 | `reuses_phase_8_5_5_dsr_pbo_gate` | PASS — dsr=0.90 rejected; dsr>=0.95 promoted (gate is imported + used at line 70) |
| 3 | `promotion_at_5pct_starting_allocation` | PASS — `notes='kicked_off; starting_alloc=0.05'`; `allocations=[0.05]` |
| 4 | `top_n_default_1_max_3` | PASS — default=1, top_n=3 -> 3, top_n=5 capped at max_n=3 |

## Responds to phase-10.3 Q/A carry-forward

The qa_103_v1 critique flagged that `thursday_batch.py` can return `already_fired=False` even when `append_row` returned `ok=False`. Phase-10.4 addresses this at the consumer side:
- `run_friday_promotion` reads the ledger row directly (not the batch-return value)
- If the row is missing OR `thu_batch_id` is empty string, returns `error="no_thursday_batch_on_ledger"` — fail-closed, never silently promotes nothing
- Covered by `test_fail_closed_when_thursday_row_missing` + `test_fail_closed_when_thu_batch_id_empty`

## Carry-forwards (out of scope)

- Risk-parity / IR-weighted allocations across multiple promoted strategies — v1 is uniform 0.05 each; future phase can compute per-strategy allocation via vol-normalization
- Downstream `Promoter.position_size()` integration — the actual live-capital sizing layer; Friday only records intent
- Auto-rollback on DD breach — phase-10.7 kill-switch wiring
- Monthly Champion/Challenger — phase-10.6
