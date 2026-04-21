# Sprint Contract — phase-10.3 (Thursday batch trigger routine)

**Step id:** 10.3 **Date:** 2026-04-20 **Tier:** moderate
**Harness-required:** true

## Why

phase-10.0 retired the nightly cron. phase-10.1 established the 2-slot-per-week sprint calendar. phase-10.3 implements the Thursday slot: a routine that consumes **exactly 1 slot**, samples **≥100 candidates** from the 15,000-combo space, and **persists a `batch_id` to the weekly ledger**. This unblocks phase-10.4 (Friday promotion gate) which reads the Thursday batch_id off the ledger.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.3-research-brief.md` — 7 sources in full, 17 URLs, three-variant queries, recency scan, gate_passed=true. Design anchors:
- **Option C** confirmed (pure library + CLI wrapper); matches autoresearch-module idiom; no APScheduler dependency
- **"1 slot" semantics:** idempotent `weekly_ledger.append_row(week_iso, ...)` IS the slot counter (write-once-per-week; guard via `read_rows` pre-check)
- **Batch-id:** `uuid.uuid5(NAMESPACE_DNS, f"thu_batch_{week_iso}_1")` — deterministic, double-trigger-safe
- **Sampling:** `scipy.stats.qmc.Sobol` seeded from `md5(week_iso)`, n=128 (power-of-2 floor ≥100), indices drawn against the enumerated 15,000-combo Cartesian product
- **Timing:** write ledger row with `notes="kicked_off"` BEFORE candidates dispatched (AWS Powertools idempotency pattern — INPROGRESS before execution)

## Immutable success criteria (copied verbatim from masterplan.json phase-10.3)

1. `python scripts/harness/phase10_thursday_batch_test.py` exit 0
2. Test must assert **`routine_consumes_exactly_1_slot`** — second invocation for same `week_iso` returns `already_fired=True` and does NOT write a new ledger row
3. Test must assert **`ge_100_candidates_kicked_off`** — `thu_candidates_kicked >= 100` in the persisted row
4. Test must assert **`batch_id_persisted_to_weekly_ledger`** — `thu_batch_id` in the row is a valid UUID string and matches the value returned by `trigger_thursday_batch`

## Plan

1. Create `backend/autoresearch/thursday_batch.py`:
   - Public: `trigger_thursday_batch(week_iso: str, *, n_candidates: int = 128, ledger_path: Path | None = None, candidate_space_path: Path | None = None, calendar_path: Path | None = None) -> dict`
   - Returns `{batch_id, week_iso, candidates_kicked, slot_num, already_fired}`
   - Validates `n_candidates >= 100` (defends criterion #3 at the source)
   - Guards re-fire via `read_rows` → existing row with non-empty `thu_batch_id` for `week_iso` → return `already_fired=True` without re-writing
   - Loads `candidate_space.yaml`, enumerates the 7-dim Cartesian product (5×4×3×2×5×5×5 = 15,000), samples `n_candidates` indices via Sobol seeded from `md5(week_iso)`
   - Computes `batch_id = uuid.uuid5(NAMESPACE_DNS, f"thu_batch_{week_iso}_1")`
   - Calls `weekly_ledger.append_row(week_iso=..., thu_batch_id=str(batch_id), thu_candidates_kicked=n, notes="kicked_off")`
   - ASCII-only logger messages per security.md
2. Create `scripts/harness/phase10_thursday_batch_test.py`:
   - CLI entrypoint following the `run_harness.py` pattern (sys.path.insert + argparse + `raise SystemExit(main())`)
   - Three test cases, each using `tempfile.TemporaryDirectory()` for an isolated ledger:
     - `test_consumes_exactly_one_slot` — fire twice for same week_iso → second call returns `already_fired=True`, ledger has exactly 1 row
     - `test_kicks_ge_100_candidates` — fire once → `thu_candidates_kicked >= 100`
     - `test_batch_id_persisted` — fire once → ledger row `thu_batch_id` is a valid UUID and equals returned `batch_id`
   - Each case prints PASS/FAIL; exit 0 iff all three pass
3. Write a matching pytest at `tests/autoresearch/test_thursday_batch.py` (mirroring the 3 CLI cases + an edge case: `n_candidates=50` should raise `ValueError`)
4. Run verification:
   - `python -c "import ast; ast.parse(open('backend/autoresearch/thursday_batch.py').read())"`
   - `python scripts/harness/phase10_thursday_batch_test.py` (immutable command)
   - `pytest tests/autoresearch/test_thursday_batch.py -q`
   - `pytest tests/ -q` (no regression)
5. Spawn fresh Q/A. **If Q/A returns CONDITIONAL or FAIL with violated_criteria: Main reads the critique, fixes the blockers, updates the handoff files, then spawns a NEW Q/A on the updated evidence.** Not second-opinion-shopping — canonical cycle-2 flow.
6. Log; mark task #63 complete.

## References

- `handoff/current/phase-10.3-research-brief.md` (7 in full, 17 URLs, gate_passed=true)
- `backend/autoresearch/weekly_ledger.py` (consumer; COLUMNS tuple is the ledger row contract)
- `backend/autoresearch/sprint_calendar.yaml` (Thursday slot definition, `thu_batch` slot_id)
- `backend/autoresearch/candidate_space.yaml` (15,000-combo Cartesian enumerated)
- `scipy.stats.qmc.Sobol` (sampling primitive)
- `uuid.uuid5(NAMESPACE_DNS, ...)` (batch-id primitive)

## Carry-forwards (out of scope)

- Actual trial execution for the kicked candidates — this step only persists the batch-id + candidate count; actual runs are phase-10.4's input via the ledger
- Cost tracking per-candidate-run — reuses `cost_budget_watcher` on the BQ side; no new primitive needed
- Calendar-day Thursday enforcement — the function accepts `week_iso` as input; whether "today is Thursday" is checked is the caller's concern (unit tests don't care about wall-clock)
