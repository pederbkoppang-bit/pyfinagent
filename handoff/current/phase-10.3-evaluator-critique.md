# Q/A Critique — phase-10.3 (Thursday batch trigger) — qa_103_v1

**Verdict: PASS**  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_103_v1",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["ast_parse_3_files", "immutable_cli_exit0_3of3_pass", "pytest_thursday_batch_7of7", "pytest_autoresearch_plus_slack_56of56", "handoff_files_exist", "contract_mtime_le_results_mtime", "verbatim_success_criteria", "spot_read_production_code", "mutation_1_min_candidates_floor", "mutation_2_idempotency_guard", "mutation_3_uuid5_determinism", "llm_judgment_contract_alignment", "research_gate_compliance"],
  "reason": "All 3 immutable success_criteria pass. All 3 mutations (floor=0, guard removed, uuid4) caught by test suite — tests are real, not cosmetic. Mutation 2 also caught by the immutable CLI itself (exit 1). Researcher gate_passed=true (7 in full, 17 URLs, recency scan). Contract mtime 19:51 <= results mtime 19:57. Idempotency design correct (early return before _sample_candidates). Sobol sampling genuinely used (n=128 distinct indices with dedup + stride fill)."
}
```

## Protocol audit (5/5 PASS)

1. Researcher moderate tier, 7 in full, 17 URLs, three-variant, recency, gate_passed=true.
2. Contract mtime 19:51 <= results mtime 19:57.
3. Success criteria copied verbatim from masterplan; CLI case names match character-for-character.
4. No harness_log.md append yet.
5. qa_id=qa_103_v1; cycle v1.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse (3 files) | exit 0 |
| immutable CLI `phase10_thursday_batch_test.py` | 3/3 PASS, exit 0 |
| pytest test_thursday_batch.py | 7/7, exit 0 |
| pytest autoresearch + slack_bot neighbor suites | 56/56, exit 0 |
| Handoff triplet | all 3 present |

## Mutation tests (Q/A ran all 3)

| Mutation | Target | Result |
|---|---|---|
| M1: `_MIN_CANDIDATES = 0` | floor constant load-bearing? | `test_n_below_floor_raises` FAILED (DID NOT RAISE) — **caught** |
| M2: comment out idempotency guard | guard load-bearing? | `test_consumes_exactly_one_slot` FAILED + **immutable CLI exit 1** (FAILED 2/3) — caught at two layers |
| M3: `uuid.uuid5(...)` → `uuid.uuid4()` | determinism genuine? | `test_batch_id_is_deterministic_per_week` FAILED (UUID mismatch) — caught |

File restored byte-identical post-mutation; main re-verified `_MIN_CANDIDATES = 100`, `uuid.uuid5(uuid.NAMESPACE_DNS, ...)`, and idempotency guard all present; CLI 3/3 PASS exit 0 on restored file.

## Spot-reads confirmed

- `_MIN_CANDIDATES = 100` at line 29 (module scope)
- Idempotency guard at lines 54-69 checks `row.get("thu_batch_id")` truthiness (not just presence) — empty string would not short-circuit, correct
- Batch-id `uuid.uuid5(uuid.NAMESPACE_DNS, f"thu_batch_{week_iso}_1")` at line 52 — deterministic
- Seed `int(hashlib.md5(week_iso.encode("utf-8")).hexdigest(), 16) % (2**32)` at line 117 — reproducible per week
- Sobol exception path (lines 141-144) falls open to deterministic stride-sample

## Non-blocking observations (not violated_criteria)

1. `scipy` UserWarning on Sobol "balance properties" when n is not a power of 2. Benign (n=128 IS a power of 2; warning is about the algorithm in general).
2. `ok = weekly_ledger.append_row(...)` return is logged on failure but `trigger_thursday_batch` still returns `already_fired=False` even if the write failed. Not a masterplan violation here, but **phase-10.4 design should handle the case where a returned batch_id isn't actually persisted.** Worth a note for the 10.4 contract — flag, not blocker.

## Carry-forwards (legit deferrals)

- Trial execution → phase-10.4 (reads batch_id off the ledger)
- Calendar "is-it-Thursday?" wall-clock check → caller's concern
- 6 pre-existing collection errors in unrelated test modules → separate cleanup ticket

Cleared for log append + masterplan status flip.
