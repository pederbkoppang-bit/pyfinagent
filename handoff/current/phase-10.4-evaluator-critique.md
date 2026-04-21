# Q/A Evaluator Critique — phase-10.4 (Friday promotion gate routine)

**qa_id:** qa_104_v1
**Date:** 2026-04-20
**Verdict:** PASS

## 5-item harness-compliance audit

1. **Researcher ran fresh with gate_passed=true** — `phase-10.4-research-brief.md` envelope: `external_sources_read_in_full=6`, `urls_collected=14`, `recency_scan_performed=true`, `gate_passed=true`. Meets ≥5 full / recency / three-variant floor.
2. **Contract mtime precedes results** — contract `1776717305` < results `1776717433`. Contract written BEFORE GENERATE, per protocol.
3. **Verbatim immutable criteria** — the four success_criteria names in contract lines 21-24 match masterplan (`routine_consumes_exactly_1_slot`, `reuses_phase_8_5_5_dsr_pbo_gate`, `promotion_at_5pct_starting_allocation`, `top_n_default_1_max_3`). Verified.
4. **No log append yet** — Main has not touched `handoff/harness_log.md` for phase-10.4; logging is the last step post-PASS per `feedback_log_last`.
5. **Cycle v1** — no prior `phase-10.4-evaluator-critique.md` in archive or current; this is first Q/A on this evidence, not a verdict-shop.

## Deterministic checks

| # | Check | Result |
|---|-------|--------|
| A | `ast.parse` on `friday_promotion.py`, CLI, pytest file | all exit 0 |
| B | Immutable CLI: `python scripts/harness/phase10_friday_promotion_test.py` | exit 0, `ALL PASS (4/4)` with all 4 named cases |
| B' | `pytest tests/autoresearch/test_friday_promotion.py -q` | 9 passed |
| B'' | `pytest tests/autoresearch/ tests/slack_bot/ -q` | 65 passed (no regression) |
| C | 3 handoff files exist (contract/results/research-brief) | present |
| D1 | `_STARTING_ALLOCATION_PCT = 0.05`, `_DEFAULT_MAX_N = 3` at module scope | line 26-27, confirmed |
| D2 | Fail-closed branch `if row is None or not row.get("thu_batch_id")` → `error="no_thursday_batch_on_ledger"` | lines 62-73, confirmed |
| D3 | Idempotency: `prior_promoted = row.get("fri_promoted_ids","") or ""`; checks `!= "[]"` | lines 76-78, confirmed |
| D4 | Ranking `sort(key=lambda c: (-_safe_float(c.get("dsr"),0.0), _safe_float(c.get("pbo"),1.0)))` | lines 104-106, confirmed |
| D5 | Cap `effective_n = max(0, min(int(top_n), int(max_n)))` | line 54, confirmed |
| D6 | Notes preservation: reads prior notes, appends `; starting_alloc=...` | lines 117-119, confirmed |
| D7 | Ledger write passes through prior `thu_batch_id`, `thu_candidates_kicked`, `cost_usd`, `sortino_monthly` | lines 121-131, confirmed (all four prior columns carried forward from `row.get(...)`) |

## Mutation tests

| ID | Mutation | Outcome |
|----|----------|---------|
| M1 | Replace `min(int(top_n), int(max_n))` with `int(top_n)` (drop max_n clamp) | Simulated: `top_n=5, max_n=3` now promotes 5 — `test_top_n_default_1_max_3` cap branch asserts `len==3`, so FAIL. **Mutation CAUGHT.** |
| M2 | Comment out fail-closed branch | `test_fail_closed_when_thursday_row_missing` would hit `row.get(...)` on `None` → AttributeError; `test_fail_closed_when_thu_batch_id_empty` would get `error=None` instead of `"no_thursday_batch_on_ledger"`. **Both tests CAUGHT.** |
| M3 | Drop PBO tie-breaker (sort by `-dsr` only) | Simulated with the `test_ranks_by_dsr_desc_then_pbo_asc` input: picked `high_dsr_high_pbo` instead of the expected `high_dsr_low_pbo`. **Mutation CAUGHT.** |

All three mutations are caught by the pytest suite — tests are mutation-resistant, not rubber-stamp.

## LLM judgment

- **Success criteria honored verbatim.** All four CLI cases run and pass with the named labels. Criterion #2 (`reuses_phase_8_5_5_dsr_pbo_gate`) is genuine reuse: `from backend.autoresearch.gate import PromotionGate` (line 22), `g = gate or PromotionGate()` (line 56), `verdict = g.evaluate(c)` (line 97). No reimplementation of DSR/PBO thresholds.

- **qa_103_v1 carry-forward addressed at consumer side.** Tests `test_fail_closed_when_thursday_row_missing` AND `test_fail_closed_when_thu_batch_id_empty` both cover the exact edge called out in the 10.3 critique. Implementation uses `if row is None or not row.get("thu_batch_id")` — truthy-check catches empty string, None, and missing key. Returns explicit `error="no_thursday_batch_on_ledger"`, not a silent empty-list no-op.

- **Ledger write preserves prior columns.** `append_row(...)` call (lines 121-131) passes through `row.get("thu_batch_id","")`, `row.get("thu_candidates_kicked","0")`, `row.get("cost_usd","0.0")`, `row.get("sortino_monthly","0.0")` — every Thursday-written column is carried forward, not defaulted to empty. Notes are concatenated (`kicked_off; starting_alloc=0.05`) not overwritten — option (a) from the contract; `test_preserves_thursday_notes_kicked_off` asserts both markers survive.

- **Carry-forwards are legit deferrals.** Uniform `starting_allocation_pct` (not risk-parity) is explicitly documented in the docstring as "uniform for v1; risk-parity is a future carry-forward". Contract lines 73-77 list the out-of-scope items (position_size integration → downstream Promoter, rollback → 10.7, monthly gate → 10.6, slot accounting → 10.8). None of these deferrals interfere with this step's immutable criteria.

## Scope honesty

The `experiment_results.md` discloses scope bounds: Friday routine only records promotion intent (the `fri_promoted_ids` + `starting_alloc=0.05` marker); actual capital sizing is a downstream `Promoter.position_size()` call in a later phase. No overclaiming.

## Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_104_v1",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "syntax_ast_parse",
    "immutable_cli_4_of_4",
    "pytest_friday_promotion_9",
    "pytest_autoresearch_slack_bot_65",
    "handoff_files_exist",
    "contract_mtime_order",
    "research_gate_envelope",
    "code_spot_read_D1_D7",
    "mutation_M1_max_n_cap",
    "mutation_M2_fail_closed",
    "mutation_M3_pbo_tiebreak",
    "genuine_PromotionGate_reuse",
    "qa_103_carryforward_covered",
    "ledger_column_passthrough",
    "scope_honesty"
  ],
  "reason": "All 4 immutable success_criteria met with exit 0. 65/65 pytest green. 3 mutations caught by the test suite. PromotionGate genuinely imported and called (not reimplemented). qa_103_v1 consumer-side fail-closed covered by two dedicated tests. Ledger write preserves all prior Thursday columns. Research gate_passed=true with 6 full sources. Contract mtime precedes results. No verdict-shopping."
}
```
