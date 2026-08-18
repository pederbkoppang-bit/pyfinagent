STATUS: INCOMPLETE -- not a verdict
STEP: 75.11.4
WRITTEN: 2026-08-17T19:34:44Z

# Q/A write-first record -- step 75.11.4 (cycle 3 evidence)

Observed at start: two prior WIP records exist in verdicts/
- verdict_wip_75.11.4__20260817T185113Z.md (15,710 bytes, 21:02 local)
- verdict_wip_75.11.4__20260817T191121Z.md (15,359 bytes, 21:27 local)
Neither is a verdict. They are evidence for this spawn.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git scope, lint, syntax
C. Independent mutation re-derivation (do NOT accept the claimed 10/10)
D. Criterion-by-criterion MET/NOT MET

## Findings (appended as established)

### Prior-attempt evidence
- qa_wip.py --spawned-at 2026-08-17T19:34:44Z: attempt_number=3, prior_attempts=2,
  source_present=true, attempt_number_status=ok, records_retained=3 (gauge).
- verdict_history_86_21.py --step 75.11.4 --evidence-only: status=ok,
  "2 verdict(s) from the ledger", verdicts: FAIL -> FAIL.
- CROSS-CHECK: prior_attempts(2) == ledger rows(2) -> ledger is NOT stale.

### Deterministic
- IMMUTABLE CMD `.venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`
  => 27 passed in 0.87s, EXIT=0 (captured bare, not through a pipe).
- Derived .py scope (git diff HEAD + git ls-files --others, xargs, non-empty asserted):
  backend/api/sovereign_api.py, backend/tests/test_phase_75_11_4_backfill_status_aware.py,
  scripts/housekeeping/{backfill_handoff_archive,handoff_naming,quarantine_misattributed_archives,verify_handoff_layout}.py
  => uvx ruff check --select F821,F401,F811 : "All checks passed!", exit=0.
- Step boundary honoured: git status --short -- scripts/housekeeping/ backend/tests/ shows
  ONLY M backfill_handoff_archive.py, M verify_handoff_layout.py, ?? handoff_naming.py,
  ?? quarantine_misattributed_archives.py, ?? test_phase_75_11_4_backfill_status_aware.py.
  NO file under .claude/hooks/** modified (confirmed separately below).

### C10/C12 independent re-derivation (my own script, not the author's)
- 845 phase-* archive dirs. classify() histogram: agree=440, mismatch=156,
  unclassified=222, no_contract=27.
- MISATTRIBUTION_NOTICE.md count = 156. mismatch-without-marker = 0.
  marker-on-non-mismatch = 0. EXACT 1:1. Author's 156 reproduces.
- NOTE for judgement: 249 of 845 dirs (29.5%) are unclassified/no_contract -- the
  census cannot speak to them. Check whether the artifacts disclose this bound.
- No circularity: classify() reads only contract.md / contract_*.md, never the
  notice, so writing the marker cannot flip a verdict. Verified by reading
  scripts/qa/derive_archive_misattribution_86_29.py::classify.
