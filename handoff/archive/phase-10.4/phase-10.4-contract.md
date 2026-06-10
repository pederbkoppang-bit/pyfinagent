# Sprint Contract — phase-10.4 (Friday promotion gate routine)

**Step id:** 10.4 **Date:** 2026-04-20 **Tier:** moderate **Harness-required:** true

## Why

phase-10.3 set up the Thursday batch. phase-10.4 is the paired Friday step: read the ledger's `thu_batch_id`, evaluate candidates through the phase-8.5.5 gate, promote the top-N (default 1, max 3) at a 5% starting allocation, and persist `fri_promoted_ids` / `fri_rejected_ids` back to the ledger.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.4-research-brief.md` — 6 sources in full, 14 URLs, three-variant queries, recency scan, gate_passed=true.

Key grounding:
- DSR-desc / PBO-asc ranking is canonical (Bailey & Lopez de Prado 2014); no 2024-2026 work supersedes
- "5% starting allocation" = capital slice recorded in ledger `notes`; actual sizing is downstream in `Promoter.position_size(capital=0.05*AUM)`
- Max N=3 is an operational count cap, not a statistical constraint
- **Q/A-flagged edge from 10.3:** must fail-closed if `thu_batch_id` missing/empty in the ledger row

## Immutable success criteria (masterplan-verbatim)

1. `routine_consumes_exactly_1_slot` — second call for same `week_iso` returns `already_fired=True`, no duplicate ledger row
2. `reuses_phase_8_5_5_dsr_pbo_gate` — routine calls `PromotionGate.evaluate()`; candidates below DSR/PBO thresholds are rejected
3. `promotion_at_5pct_starting_allocation` — ledger `notes` column contains `starting_alloc=0.05` after a successful fire
4. `top_n_default_1_max_3` — default `top_n=1` promotes exactly 1; `top_n=3` promotes 3; `top_n=5` caps at 3

## Plan

1. Create `backend/autoresearch/friday_promotion.py`:
   - Public `run_friday_promotion(week_iso, *, candidates, top_n=1, max_n=3, starting_allocation_pct=0.05, ledger_path=None) -> dict`
   - Validate `1 <= top_n` and `top_n <= max_n` (clamp with `min(top_n, max_n)` rather than raise, per 10.3 idiom)
   - **Fail-closed on missing Thursday batch:** look up `week_iso` row in ledger; if row missing or `thu_batch_id` empty, return `{promoted_ids: [], rejected_ids: [], allocations: [], already_fired: False, error: "no_thursday_batch_on_ledger"}`
   - **Idempotency:** if row has `fri_promoted_ids` populated and != `"[]"`, return `already_fired=True` (with parsed list)
   - Evaluate each candidate via `PromotionGate().evaluate(c)`; split passed / rejected
   - Rank passed by `(-dsr, +pbo)`, take `min(top_n, max_n)`
   - Write `append_row(week_iso, fri_promoted_ids=[...], fri_rejected_ids=[...], notes=f"starting_alloc={starting_allocation_pct}")` preserving the existing `thu_batch_id` + candidates count (append_row already does upsert-by-week_iso; however `notes` overwrites — research brief notes this; must preserve prior notes too)
   - ASCII-only logger messages per security.md
2. Create `scripts/harness/phase10_friday_promotion_test.py`:
   - 4 cases matching success_criteria names verbatim
   - Each case wrapped in `tempfile.TemporaryDirectory()`
   - Seed the Thursday row before Friday tests (use `thursday_batch.trigger_thursday_batch(week_iso, ledger_path=lpath)` then overwrite candidates via the test's own fixtures)
   - Reuse gate fixtures from `autoresearch_gate_test.py` (good: `dsr=0.99/pbo=0.10`; bad: `dsr=0.90/pbo=0.10`)
   - Exit 0 iff all four PASS
3. Create `tests/autoresearch/test_friday_promotion.py` — ≥6 pytest cases mirroring CLI + edge cases:
   - All 4 CLI cases
   - Fail-closed when Thursday row missing (returns `error="no_thursday_batch_on_ledger"`)
   - Fail-closed when Thursday row exists but `thu_batch_id` empty
   - `candidates=[]` returns `promoted_ids=[]` without raising
4. Run verification (ast.parse + immutable CLI + pytest new file + pytest autoresearch+slack_bot neighbor suites)
5. Spawn fresh Q/A. **If CONDITIONAL/FAIL with violated_criteria:** Main reads critique, fixes, updates handoff files, spawns fresh Q/A on updated evidence (cycle-2 flow per user directive).
6. Log, flip masterplan, close task.

## Ledger-write notes-preservation detail

The existing `weekly_ledger.append_row(week_iso, ..., notes="")` overwrites the row's `notes` field. Thursday wrote `notes="kicked_off"`. If Friday writes `notes="starting_alloc=0.05"`, the Thursday marker is lost. Options:
- **(a)** Concatenate: read prior `notes`, append `; starting_alloc=0.05`
- **(b)** Overwrite: `starting_alloc=0.05` replaces `kicked_off`; rely on `fri_promoted_ids` populated as the "Friday ran" signal

Decision: **(a)** — preserves full provenance. Implementation: `new_notes = f"{prior_notes}; starting_alloc={pct}".strip("; ")` before calling `append_row`.

## References

- `handoff/current/phase-10.4-research-brief.md` (6 in full, 14 URLs, gate_passed=true)
- `backend/autoresearch/gate.py` (PromotionGate — 8.5.5)
- `backend/autoresearch/weekly_ledger.py` (append_row upsert — 10.2)
- `backend/autoresearch/thursday_batch.py` (trigger_thursday_batch — 10.3; known edge)
- `backend/autoresearch/candidate_space.yaml` (7-dim candidate tuples — 8.5.1)
- `backend/autoresearch/sprint_calendar.yaml` (fri_promotion slot_id — 10.1)
- `scripts/harness/autoresearch_gate_test.py` (gate fixtures — 8.5.5)
- `scripts/harness/phase10_thursday_batch_test.py` (test-scaffold template — 10.3)

## Carry-forwards (out of scope)

- Actual live capital deployment — downstream `Promoter.position_size()` machinery; Friday only records the intent
- Auto-rollback if a promoted strategy breaches DD — phase-10.7 kill-switch wiring
- Monthly Champion/Challenger gate — phase-10.6
- Slot accounting to `harness_learning_log` — phase-10.8
- The `thursday_batch.py:82-91` edge (returns already_fired=False when append_row fails) — address in 10.4 at the consumer side (fail-closed on missing ledger row); original producer-side fix is a separate ticket
