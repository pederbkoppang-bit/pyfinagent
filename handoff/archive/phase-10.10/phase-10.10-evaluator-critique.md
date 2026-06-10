# phase-10.10 Evaluator Critique — qa_1010_v2

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_1010_v2",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "ast_parse_both_files",
    "grep_new_test_and_monkeypatch_pattern",
    "grep_mf_write_precedes_shutil_move_in_source",
    "clear_pyc_and_pycache",
    "pytest_full_suite_baseline",
    "cli_harness_phase10_housekeeping_test",
    "mutation_M3_write_after_move_test_fails",
    "restore_and_reverify_9_of_9_green"
  ],
  "reason": "All five deterministic checks pass (A-E). M3 mutation (writing manifest inside try block AFTER shutil.move) caused test_manifest_written_before_move_crash_resilience to fail with the predicted 2 != 3 assertion. Restoring canonical order (write BEFORE move) returned 9/9 green. The v1 gap — invariant documented but not tested — is now closed by a real, mutation-resistant test.",
  "certified_fallback": false
}
```

## Deterministic evidence

### A. AST parse — PASS
Both `scripts/housekeeping/quarantine_phantom_archives.py` and
`tests/housekeeping/test_quarantine.py` parse cleanly under Python 3.14.

### B. CLI harness + pytest — PASS
- `python scripts/harness/phase10_housekeeping_test.py` → 4/4 PASS
  (phantom_dirs_moved_not_deleted, canonical_dirs_untouched,
  manifest_written_with_sha256_per_dir, quarantine_is_reversible).
- `pytest tests/housekeeping/test_quarantine.py -q` → 9 passed in 0.03s.

### C. New test present with the claimed pattern — PASS
`test_manifest_written_before_move_crash_resilience` at line 137 uses
`monkeypatch.setattr(qpa.shutil, "move", flaky_move)` exactly as the
cycle-2 patch described. The `flaky_move` closure increments a
call-count and raises `RuntimeError("simulated mid-run crash")` on the
2nd of 3 moves (v2 ok, v3 crash, v4 ok). Three phantoms are built,
exactly two are expected to land in quarantine, and all three must
appear in the manifest.

### D. Source write-order invariant — PASS
In `quarantine_phantom_archives.py` lines 141-149:
- line 145: `mf.write(json.dumps(entry, sort_keys=True) + "\n")`
- line 146: `mf.flush()`
- line 148: `try:`
- line 149: `shutil.move(str(phantom), str(dest))`

The write+flush unambiguously precede `shutil.move`, with an inline
safety comment referencing the test by name.

### E. Mutation M3 — PASS (invariant is actually tested)
I performed the documented M3 mutation end-to-end:
1. Backed up the canonical source to `/tmp/qpa_orig.py`.
2. Moved `mf.write` + `mf.flush` INSIDE the `try` block AFTER
   `shutil.move(str(phantom), str(dest))`.
3. Cleared all `housekeeping`-scoped `__pycache__`.
4. Ran only the new test:

```
FAILED tests/housekeeping/test_quarantine.py::test_manifest_written_before_move_crash_resilience
E  AssertionError: manifest-before-move invariant violated: 2 entries
   after 2 successful moves + 1 crash; crashed phantom would be
   unrecoverable
E  assert 2 == 3
```

The failure signature matches the critique's prediction exactly
(`2 != 3`). The crashed phantom (v3) has no manifest entry under the
mutation — proving the test binds the invariant.

5. Restored from `/tmp/qpa_orig.py`, re-cleared `__pycache__`,
   re-ran the full file: `9 passed in 0.03s`.

## Contract alignment

The contract (`handoff/current/phase-10.10-contract.md`) lists the
immutable success criteria for phase-10.10 housekeeping quarantine:
reversible-not-destructive, SHA-256 manifest, canonical paths
untouched, idempotent. All four are asserted by the original 8 tests
(still green). The cycle-2 patch adds a 9th test binding the
crash-recovery invariant that the code's docstring claims but v1
did not enforce. This is a strict superset of v1 evidence, not a
substitution, so no criterion is weakened.

## Anti-rubber-stamp (mutation resistance)

qa_1010_v1 flagged the missing M3 mutation-resistance test as a
protocol gap: an invariant in prose is not an invariant under CI.
v2 resolves the gap by adding a monkeypatched-`shutil.move` test and
I verified independently (not just trusting Main's claim) that the
mutation fails the test and the restore re-greens it. The test is
load-bearing.

## Research-gate compliance

`handoff/current/phase-10.10-research-brief.md` is present and the
contract references it. Not re-evaluating research-gate mechanics in
a cycle-2 remediation — the research scope did not change between
v1 and v2; only the test surface did.

## Second-opinion-shopping check

This is a legitimate cycle-2 spawn on **updated evidence**:
- New file content: `tests/housekeeping/test_quarantine.py` grew
  from 8 to 9 tests (added crash-resilience test).
- New code state: `quarantine_phantom_archives.py` has an inline
  safety comment tying the write-order to the new test.
- New verification artifacts: `experiment-results.md` has a
  cycle-2 patch section documenting the mutation walk.
- Stale `.pyc` cleared.

Per `CLAUDE.md` "canonical cycle-2 flow", fresh-Q/A-on-updated-
evidence is the documented pattern, not verdict-shopping.

## Scope honesty

The fix is narrow (one test + one comment, no behaviour change in
production code paths). No overclaim in experiment-results.

## Verdict

**PASS.** qa_1010_v2 approves phase-10.10 REMEDIATION v2. Main may
append to `handoff/harness_log.md` and flip
`.claude/masterplan.json` step phase-10.10 to `status: done`.
