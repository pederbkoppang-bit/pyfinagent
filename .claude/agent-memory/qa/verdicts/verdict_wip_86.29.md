STATUS: INCOMPLETE -- not a verdict
STEP: 86.29
WRITTEN: 2026-08-11T06:38:03Z

# Q/A write-first record -- step 86.29, CYCLE 1

Launch: Workflow structured-output rail. Author = pyfinagent-51 (Main).
Contract authored by peer session pyfinagent-06 (c806cad6) BEFORE GENERATE (974297ce).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable cmd `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'`
C. Re-run the two re-runnable scripts myself
D. Attack the 4 self-nominated weak points + criterion-by-criterion judgment

## Findings log (appended as established)

### A. Harness compliance (5 items) -- ALL CLEAN
- research-gate-before-contract: `handoff/current/research_brief_86.29_rerun.md`
  envelope `brief_status COMPLETE`, `gate_passed: true`, `external_sources_read_in_full: 7`,
  `recency_scan_performed: true`, internal_files 7. The FIRST brief
  (`research_brief_86.29.md`) honestly records gate FAILED (no envelope, stopped
  mid-loop) and was re-run -- that is the rail working, not a skip.
- contract-before-generate: PROVABLE FROM GIT (split authorship).
  c806cad6 (2026-08-10 22:53) touches ONLY `handoff/current/contract_86.29.md`.
  974297ce (2026-08-11 08:37) touches the hook + 2 scripts + results + live_check.
  Contract commit strictly precedes and contains NO code.
- experiment_results present: `handoff/current/experiment_results_86.29.md` (13151 B).
- log-last: `handoff/harness_log.md` -- to be checked. masterplan 86.29 status=pending,
  retry_count=0. NOT yet flipped.
- no-verdict-shopping: CYCLE 1, no prior verdict for 86.29 exists. Nothing to shop.

### B. Deterministic
- IMMUTABLE CMD `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n ...'` -> **EXIT=0**
- hook sha256[:16] = `6dc68f781edb4fd0` -- MATCHES the digest stated in live_check line 5.
- `git diff --name-only HEAD` = only agent-memory/audit-jsonl/heartbeat churn.
  NO unintended production change. GENERATE commit `974297ce` touched exactly the
  5 files experiment_results section 1 claims. Verified with
  `git diff --name-only 974297ce^ 974297ce`.
- `python scripts/qa/prove_archive_provenance_86_29.py` -> RESULT: PASS (0 problems),
  EXIT=0. Reproduced BY ME. 4/4 mutants KILLED, 3/3 control checks GREEN,
  BEFORE half declares '82.54' from the git-recovered pre-fix hook.
  Isolation: real hook digest unchanged, archive dir list unchanged (819).
- `python scripts/qa/derive_archive_misattribution_86_29.py` -> EXIT=0.
  Reproduced BY ME at tree 33255004: **153 mismatch / 387 agree / 255 unclassified
  / 24 no_contract over 819 dirs**. Recall 2/2, controls 4/4, precision 1.0000,
  0 suspects. EXACTLY the "after" row of experiment_results section 6.

### B2. Live-witness claim (attack point 2) -- MAIN'S READING IS CORRECT
Read `.claude/hooks/archive-handoff.sh:241-244`:
```
for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
    [ -f "$CURRENT_DIR/$f" ] || continue
    [ -f "$target/$f" ] && continue          # derived branch already won
    if rolling_declares_step "$CURRENT_DIR/$f" "$short_sid"; then
```
For 86.31 the derived branch supplied contract/experiment_results/evaluator_critique/
research_brief, so those four hit the `[ -f "$target/$f" ] && continue` early-out;
`handoff/current/research.md` is ABSENT (verified) so it hits the first `continue`.
=> `rolling_declares_step` was NEVER CALLED. `rolling_skipped=0` in
`handoff/archive/phase-86.31/PROVENANCE.md` is consistent with that and with nothing
else. Main's statement "the live run proves the DERIVATION branch ONLY, not the
guard" is EXACT -- neither over- nor under-claimed.
Independently verified the live dir: 5 artifacts + PROVENANCE.md, every head -1
declares 86.31 (`# Contract -- step 86.31` etc.). First archive dir since
2026-08-06 holding its own step's files -- confirmed by me, not taken from the report.
