# Experiment Results -- phase-91.18
Step: Cron/Logs page resolves ablation's log paths without `/logs/`

## What was built/changed
1. `backend/api/cron_dashboard_api.py:163-168` -- `_log_paths()`'s `ablation` and `ablation_launchd`
   dict values now include the `logs/` path component, matching `run_ablation.sh:14` and the
   `com.pyfinagent.ablation` plist's `StandardOutPath`/`StandardErrorPath`.
2. New file `backend/tests/test_phase_91_18_ablation_log_path.py` -- 2 tests asserting the allowlist
   resolves under `handoff/logs/` and matches the real writer (`run_ablation.sh`'s `LOG=` line) verbatim.

## File list
- `backend/api/cron_dashboard_api.py` (modified, 2 lines changed + 4-line comment)
- `backend/tests/test_phase_91_18_ablation_log_path.py` (new)

## Verbatim verification command output

### Unit test, mutation-tested red-first
```
$ python -m pytest backend/tests/test_phase_91_18_ablation_log_path.py -v
backend/tests/test_phase_91_18_ablation_log_path.py::test_ablation_log_paths_resolve_under_handoff_logs PASSED
backend/tests/test_phase_91_18_ablation_log_path.py::test_ablation_log_paths_match_the_real_writers PASSED
2 passed in 0.10s
```

Mutation control (revert to the pre-fix values via Edit, since `git stash` is deny-listed on this
project -- see `feedback_no_git_stash_with_active_hooks`):
```
$ python -m pytest backend/tests/test_phase_91_18_ablation_log_path.py -v   # mutant: pre-fix paths
FAILED test_ablation_log_paths_resolve_under_handoff_logs
FAILED test_ablation_log_paths_match_the_real_writers
2 failed in 0.11s
```
Both tests correctly discriminate the bug. Restore verified byte-identical via SHA-256:
`3c4ffe7ccf538079528214951fe670d2c5b205426ec7a0687250d8decbeb4231` (fixed file, both before and
after the mutation-then-restore round trip).

### Immutable command -- DEFERRED, code not yet live
```
curl -s 'http://localhost:8000/api/logs/tail?log=ablation&lines=10'
```
**NOT YET CAPTURED.** Per CLAUDE.md's restart-batching rule ("Backend restarts are batched to
SESSION END... never claim a config is live because the file says so; read the value from the
RUNNING process"), the running backend process (started before this edit) does not have this
Python change loaded -- `_log_paths()` in memory still resolves the OLD (buggy) values. Running the
curl now would observe stale behavior and would be a false-positive live_check.

**Plan:** this step's GENERATE is complete and mutation-verified at the code level. Q/A and the
status flip are held until the single batched restart planned for the end of this round of fixes
(alongside 91.9/91.13/91.22, none of which need a backend restart). After that restart, the live
curl + a Playwright capture of the Cron/Logs page will be captured into
`handoff/current/live_check_91.18.md` and a fresh Q/A spawned against the completed evidence.

## Artifact shape
- Code diff: 2 dict values corrected, 4-line explanatory comment added.
- Test: 2 new pytest functions, both red-first verified against the actual pre-fix code (not a
  synthetic mutant), both green against the fix, byte-identical restore confirmed.
- Live evidence: pending the batched restart (see above).
