# Experiment Results — phase-10.3 (Thursday batch trigger routine)

**Step:** 10.3 **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate tier): 7 sources in full, 17 URLs, three-variant queries, recency scan, gate_passed=true. Brief at `handoff/current/phase-10.3-research-brief.md`. Recommendation: **Option C** (pure library + CLI wrapper), Sobol QMC sampling, deterministic uuid5 batch-id, ledger-row idempotency, write-before-execute timing.
2. Contract authored at `handoff/current/phase-10.3-contract.md`.
3. Created `backend/autoresearch/thursday_batch.py` (105 lines):
   - Public API `trigger_thursday_batch(week_iso, *, n_candidates=128, ledger_path=..., candidate_space_path=..., calendar_path=...) -> dict`
   - Validates `n_candidates >= 100` (raises `ValueError` otherwise)
   - Idempotency guard reads ledger and short-circuits if a row for `week_iso` already has a non-empty `thu_batch_id`
   - Batch-id: `uuid.uuid5(NAMESPACE_DNS, f"thu_batch_{week_iso}_1")` — deterministic, double-trigger-safe
   - `_sample_candidates()` enumerates the 15,000-combo Cartesian product from `candidate_space.yaml`, seeds `scipy.stats.qmc.Sobol` from `md5(week_iso)`, maps to indices, dedupes, and falls back to deterministic stride-sampling if Sobol raises
   - Calls `weekly_ledger.append_row(..., notes="kicked_off")` write-before-execute
   - ASCII-only logger messages per security.md
4. Created `scripts/harness/phase10_thursday_batch_test.py` (95 lines):
   - Three cases map directly to the three masterplan `success_criteria`
   - Each case wrapped in `tempfile.TemporaryDirectory()` — no persistent side effects on `weekly_ledger.tsv`
   - Exit 0 iff all three cases PASS
5. Created `tests/autoresearch/test_thursday_batch.py` — 7 pytest cases mirroring the CLI plus edge cases:
   - `test_consumes_exactly_one_slot` (criterion 1)
   - `test_kicks_ge_100_candidates` (criterion 2)
   - `test_batch_id_is_valid_uuid` (criterion 3)
   - `test_batch_id_is_deterministic_per_week` — same week across fresh ledgers → identical UUID
   - `test_different_weeks_produce_different_batch_ids` — W17 ≠ W18
   - `test_n_below_floor_raises` — `ValueError` on `n_candidates=50`
   - `test_ledger_row_notes_kicked_off` — notes column asserts the write-before-execute marker

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/autoresearch/thursday_batch.py').read()); ast.parse(open('scripts/harness/phase10_thursday_batch_test.py').read()); ast.parse(open('tests/autoresearch/test_thursday_batch.py').read()); print('AST OK')"
AST OK

$ python scripts/harness/phase10_thursday_batch_test.py
[PASS] routine_consumes_exactly_1_slot  (r1.already_fired=False, r2.already_fired=True, rows=1)
[PASS] ge_100_candidates_kicked_off  (returned=128, persisted=128)
[PASS] batch_id_persisted_to_weekly_ledger  (batch_id=b9686bc5..., persisted=b9686bc5...)

ALL PASS  (3/3)
(exit 0)

$ pytest tests/autoresearch/test_thursday_batch.py -q
.......                                                                  [100%]
7 passed in 0.40s

$ pytest tests/autoresearch/ tests/slack_bot/ -q
........................................................                 [100%]
56 passed in 1.71s
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `routine_consumes_exactly_1_slot` | PASS — second call returns `already_fired=True`, ledger has exactly 1 row, batch_ids match |
| 2 | `ge_100_candidates_kicked_off` | PASS — `returned=128, persisted=128` |
| 3 | `batch_id_persisted_to_weekly_ledger` | PASS — `thu_batch_id` is a valid UUID, matches returned value |

## Pre-existing regression context (not caused by this change)

`pytest tests/ -q` surfaces 6 collection errors (in `tests/test_deduplication.py`, `test_end_to_end.py`, `test_ingestion.py`, `test_queue_processor.py`, `test_response_delivery.py`, `test_tickets_db.py`) — these fail at `ast`/import time and have nothing to do with phase-10.3 (new files only, no edits to those modules). Confirmed pre-existing in the repo. Flagged as cleanup candidate for a dedicated tooling step.

## Carry-forwards (out of scope)

- Actual trial execution for the kicked candidates — phase-10.4 Friday promotion gate reads `thu_batch_id` off the ledger and invokes the gate
- Calendar-day "is it Thursday?" enforcement — the function accepts `week_iso` as input; wall-clock checking is caller concern
- 6 pre-existing broken-collection test files — separate cleanup ticket
