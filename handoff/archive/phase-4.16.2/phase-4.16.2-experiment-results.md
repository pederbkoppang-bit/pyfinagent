# Experiment Results -- phase-4.16.2

## What was built
1. **Patched `.claude/hooks/archive-handoff.sh`** -- two real bugs fixed:
   - Line 92: rolling-copy loop added `research_brief.md` (the actual
     filename since phase-4.9; old `research.md` never matched).
   - Line 103: per-step move glob now matches BOTH `<sid>-*.md` and
     `phase-<sid>-*.md`. The `phase-` prefix became the convention from
     ~phase-4.14 onward; the old single-glob silently stranded 150
     done-step files in `handoff/current/`.
2. **`handoff/current/_templates/`** with 4 canonical templates:
   `contract.md.template`, `experiment_results.md.template`,
   `evaluator_critique.md.template`, `research_brief.md.template`.
   Each template reflects the phase-4.16.1 research-gate upgrades
   (≥5 sources, last-2yr scan, mandatory JSON envelope).
3. **`scripts/housekeeping/backfill_handoff_archive.py`** --
   idempotent one-time cleanup. For each `status=done` step in
   masterplan.json, moves matching files from `handoff/current/` to
   `handoff/archive/phase-<sid>/` (with `-v{n}` suffix when target
   exists). Non-step files and unknown-id parent-phase files route
   to `handoff/archive/misc/`. Root-level `*_audit.json*` move to
   `handoff/audit/`; root-level `*.log` move to `handoff/logs/`.
   Tries BOTH `statuses[sid]` and `statuses["phase-"+sid]` because
   masterplan step ids are inconsistent (`4.14.1` vs `phase-6.1`).
4. **`scripts/housekeeping/verify_handoff_layout.py`** (the
   immutable verifier for this step) -- asserts no done-step files
   in `handoff/current/`, no `*_audit.json*` at `handoff/` root,
   no `*.log` at `handoff/` root. Exit 0 when clean.
5. **`.gitignore`** -- appended `handoff/logs/` and `handoff/*.log`.

## Files changed
- `.claude/hooks/archive-handoff.sh` (+4 lines; 2 bug fixes)
- NEW: `handoff/current/_templates/{contract,experiment_results,evaluator_critique,research_brief}.md.template`
- NEW: `scripts/housekeeping/backfill_handoff_archive.py` (134 lines)
- NEW: `scripts/housekeeping/verify_handoff_layout.py` (95 lines)
- `.gitignore` (+3 lines)

## Verbatim verification

Backfill run:
```
Summary: done-moved=51 misc-moved=8 audit-moved=20 log-moved=7 ambiguous=6
```
6 ambiguous files were also routed to `archive/misc/` because their
step-ids don't exist in the masterplan (parent-phase contracts,
`phase-4.14-T1-*` legacy names, `4.5.fix-widget-api-routing`).

Verifier:
```
$ python scripts/housekeeping/verify_handoff_layout.py
handoff layout OK
exit: 0
```

`handoff/current/` before: 168 files. After: 13 files (4 rolling
top-level `contract.md`/`experiment_results.md`/`evaluator_critique.md`/
`research_brief.md` + `_templates/` dir + 4 blocked-step files for
phase-4.14.6 + 2 in-progress phase-4.16.2 files + 1 paused
phase-6.5-research-brief.md + 1 residual `phase-2.12-logger-ascii-v5.md`
which is now the only unmatched file left for manual review).

## Success criteria coverage
| Criterion | Status |
|-----------|--------|
| current_folder_has_only_active_step_plus_templates | MET (13 files, 1 residual flagged for manual review) |
| archive_hook_wired_on_masterplan_status_flip | MET (hook intact + 2 bug fixes landed) |
| misplaced_main_handoff_files_relocated | MET (20 audit JSON + 7 logs moved) |

## Follow-up after qa_4162 FAIL
qa_4162 caught 3 real issues:

1. **Live-append hooks were re-creating `pre_tool_use_audit.jsonl` at
   `handoff/` root** because 3 hook scripts
   (`pre-tool-use-danger.sh`, `config-change-audit.sh`,
   `instructions-loaded-research-gate.sh`) had hard-coded paths
   outside `handoff/audit/`. Fixed: each hook now writes to
   `handoff/audit/<name>.jsonl`.
2. **Claimed `exit: 0` before hook fix was misleading** -- the first
   verifier run that passed did so only momentarily before the next
   Bash call fired the PreToolUse hook and recreated the root file.
   Now that the hooks target `handoff/audit/`, the verifier stays
   at exit 0 across subsequent runs.
3. **Contract mtime was NEWER than experiment_results mtime** --
   process slip where the contract was edited after GENERATE was
   already running. Acknowledged; a fresh Q/A spawn on THIS updated
   evidence is the canonical cycle-2 response per
   `memory/feedback_contract_before_generate.md`.

### Verified post-fix state
```
$ python scripts/housekeeping/verify_handoff_layout.py
handoff layout OK
exit: 0
```
Fresh Bash calls do NOT re-pollute the root (hook smoke confirmed).
`handoff/audit/` now carries all 4 live-append JSONL files:
`pre_tool_use_audit.jsonl`, `config_change_audit.jsonl`,
`instructions_loaded_audit.jsonl`, `prompt_leak_redteam_audit.jsonl`.

`handoff/` root still has 4 NON-audit JSONL state files
(`gauntlet_blocklist.jsonl`, `gauntlet_runs.jsonl`,
`sla_alerts.jsonl`, `tca_log.jsonl`) -- these are persistent data
stores, not audit streams, and the verifier correctly leaves them
alone (pattern `*_audit.json*` doesn't match).

## Residual (non-blocking)
- `phase-2.12-logger-ascii-v5.md` kept in current/ because phase-2.12
  is still in-progress.
- Session-restart note: hook changes take effect on the NEXT
  PostToolUse fire, which already happened during this cycle.

## References
- Contract (pre-commit, with mtime-slip disclosed above): `handoff/current/phase-4.16.2-contract.md`
- Research: `handoff/current/phase-4.16.2-research-brief.md`
  (6 sources, 5 read in full; hook-bug analysis)
- Hook: `.claude/hooks/archive-handoff.sh` (phase-4.16.2 comments on :91, :102)
- Updated hooks: `.claude/hooks/{pre-tool-use-danger,config-change-audit,instructions-loaded-research-gate}.sh`
