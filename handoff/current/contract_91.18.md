# Sprint Contract -- phase-91.18
Step: Cron/Logs page resolves ablation's log paths without `/logs/`, always showing "does not exist yet" for real, populated logs

## Research Gate
researcher (tier=simple) gate_passed=true.
Brief: `handoff/current/research_brief_91.18.md`.
- 6 external sources read in full (floor is 5).
- Recency scan (last 2 years) performed: one new finding (2026-02-25 practitioner post on post-rotation ownership), no supersession of the core launchd semantics.
- Key findings:
  - Relative `StandardOutPath` resolves against `RootDirectory`/`/`, NOT `WorkingDirectory` -- a plist and its wrapper script can silently diverge on where the "same" relative path lands. Not live here (every plist inspected uses absolute paths), but a discipline to preserve.
  - launchd creates the file on first write and never reopens it after a rename-style rotation -- a missing file means "never wrote here," not necessarily "wrong path."
  - Apple prescribes NO log-location convention; **the plist itself is the only source of truth for where a job logs**, never a repo-internal convention comment.
  - The safest tail-endpoint shape is a server-side KEY -> Path allowlist (never client-supplied paths) -- pyfinagent's `cron_dashboard_api.py:130-134` already implements this correctly; the defect is a wrong VALUE in the allowlist, not a traversal risk.
  - Measured on disk 2026-08-20: 4 of 10 `_log_paths()` allowlist entries point at files that do not exist. Two (`harness`, `mas_harness_launchd`) are dangling for an unrelated, already-tracked reason (job not installed, phase-85.1) and are OUT OF SCOPE here. The two IN SCOPE are `ablation` (`:163`, points at `handoff/ablation.log`, should be `handoff/logs/ablation.log` -- real file is 12,227 B, mtime 2026-08-20 03:00) and `ablation_launchd` (`:164`, points at `handoff/ablation.launchd.log`, should be `handoff/logs/ablation.launchd.log`).
  - The research surfaced one additional, non-required but well-justified recommendation: because `get_log_tail` returns HTTP 200 with `exists:false` for a wrong path (not an error), a wrong allowlist entry is silent by construction and this exact drift already happened once (phase-23.3.5 fixed 3 keys, introduced 2 new wrong ones). A small regression test walking `_log_paths()` would make the next drift loud. This is an enhancement beyond the immutable criteria below, not a substitute for them -- included in the plan as a value-add, not required for PASS.

## Hypothesis
`backend/api/cron_dashboard_api.py:163-164` hardcodes `ablation` -> `handoff/ablation.log` and `ablation_launchd` -> `handoff/ablation.launchd.log`, one directory level short of where `scripts/ops/run_ablation.sh:14` and the `com.pyfinagent.ablation` launchd plist actually write (`handoff/logs/`). Correcting those two dict values to include the `logs/` component will make the Cron/Logs page's ablation log tabs show real content instead of "log file does not exist yet," with no other behavior change.

## Success Criteria (immutable)
```
curl -s 'http://localhost:8000/api/logs/tail?log=ablation&lines=10'
```
Plus sub-criteria (copied verbatim from `.claude/masterplan.json` phase-91 step 91.18):
- `backend/api/cron_dashboard_api.py`'s `_log_paths()` (or equivalent) resolves `ablation.log` and `ablation.launchd.log` under `handoff/logs/`, matching where `run_ablation.sh` and the plist actually write them
- the command above returns real log content, not "does not exist yet," when `handoff/logs/ablation.log` has content
- a Playwright screenshot of the Cron/Logs page with `ablation.log` selected shows real log lines

## Plan (PRE-commit; will NOT diverge in Generate)
1. Edit `backend/api/cron_dashboard_api.py:163-164`: change the `ablation` and `ablation_launchd` dict values from `handoff/ablation(.launchd).log` to `handoff/logs/ablation(.launchd).log`.
2. Add a small regression test (new file `backend/tests/test_phase_91_18_ablation_log_path.py`) asserting `_log_paths()["ablation"]` and `["ablation_launchd"]` resolve under `handoff/logs/` -- a red-first guard against this exact class of drift recurring (per the research's finding that it already happened once silently).
3. Verify: run the immutable curl command against the live backend and confirm real content (not `exists:false`); run the new pytest file; capture a live Playwright screenshot of the Cron/Logs page with `ablation.log` selected.

## Scope honesty / out-of-scope
- The two dangling `harness`/`mas_harness_launchd` entries are NOT touched -- they are dangling because the mas-harness launchd job is not installed (already tracked by phase-85.1 and related steps), a different root cause than this step's ablation path bug.
- No change to the KEY-not-path API shape (`get_log_tail`) -- research confirmed it already implements the correct security pattern; this step is a data-correctness fix (wrong allowlist value), not an API redesign.
- Not fixing `get_log_tail`'s HTTP-200-on-missing-file behavior generally -- that's a broader API design question outside this step's immutable criteria; the added regression test targets the specific `_log_paths()` drift instead.

## References
- Research brief: `handoff/current/research_brief_91.18.md`
- Filed from: `.claude/masterplan.json` phase-91 step 91.18 (originally 86.144, renumbered during the same-day phase-91 split)
- Related, out-of-scope: phase-85.1 (mas-harness launchd job not installed)
