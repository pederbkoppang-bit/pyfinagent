# Sprint Contract -- phase-4.16.2
Step: Handoff folder cleanup + archive-on-done hook

## Research Gate
researcher_4162 gate_passed=true. 5 sources read in full (git hooks,
Python shutil, 12-factor, keepachangelog, semver). Brief:
`handoff/current/phase-4.16.2-research-brief.md`.

Key findings:
- **Real bug in `archive-handoff.sh:103`**: glob `${sid}-*.md` misses
  `phase-<sid>-*.md` naming used since ~phase-4.14. Explains the 150
  stranded done-step files in `handoff/current/`.
- **Second bug at `archive-handoff.sh:92`**: rolling-file loop copies
  `research.md` not `research_brief.md` (actual file name since
  phase-4.9).
- **168 `handoff/current/` files break down**: 4 rolling (keep), 150
  done-step files (archive), 4 blocked-step files (keep as in-flight
  for phase-4.14.6), 10 non-conforming names (move to `archive/misc/`).
- **42 JSON + 8 JSONL + 7 log files in `handoff/` root** -- audit
  outputs belong in `handoff/audit/`, logs in `handoff/logs/` +
  `.gitignore`. Some JSONL (`pre_tool_use_audit.jsonl`,
  `instructions_loaded_audit.jsonl`) are actively appended by hooks --
  move with care but no lock required because appends to moved path
  re-create the file harmlessly (hooks re-`mkdir -p` on each call).
- **6963 archive dirs already** -- `-v{n}` suffix idempotency works.
  Backfill must replicate.

## Hypothesis
Patching both hook bugs, writing a one-time backfill script, a
layout-verifier, and a `_templates/` subfolder closes phase-4.16.2.

## Success Criteria (immutable)
```
python scripts/housekeeping/verify_handoff_layout.py
```
Plus 3 sub-criteria:
- current_folder_has_only_active_step_plus_templates
- archive_hook_wired_on_masterplan_status_flip (hook still intact + fixed)
- misplaced_main_handoff_files_relocated

## Plan (PRE-commit; will not diverge)
1. Patch `.claude/hooks/archive-handoff.sh`:
   - Add `research_brief.md` to the rolling-file copy list.
   - Change glob `${sid}-*.md` to match BOTH `${sid}-*.md` AND
     `phase-${sid}-*.md`.
2. Create `handoff/current/_templates/` with 4 canonical templates:
   `contract.md.template`, `experiment_results.md.template`,
   `evaluator_critique.md.template`, `research_brief.md.template`.
3. Write `scripts/housekeeping/backfill_handoff_archive.py`:
   - Reads masterplan.json; for each `status=done` step, gather matching
     `handoff/current/phase-<sid>-*.md` and `<sid>-*.md` files, move
     them to `handoff/archive/phase-<sid>/` (with `-v{n}` suffix if
     target exists).
   - Move 10 non-conforming files listed in the brief to
     `handoff/archive/misc/`.
   - Move `handoff/*_audit.json*` and `handoff/*_audit.jsonl*`
     (and a safelist of other `*.json` + `*.jsonl`) into
     `handoff/audit/`.
   - Move `handoff/*.log` into `handoff/logs/`.
   - Does NOT touch in-progress step files (current phase-4.16.x
     + phase-6.5-research-brief.md flagged AMBIGUOUS in the brief
     -- keep the 6.5 brief since that step is paused, not done).
4. Write `scripts/housekeeping/verify_handoff_layout.py`:
   - Asserts no `phase-<sid>-*.md` in `handoff/current/` where sid
     has status=done in masterplan (ignoring `_templates/`).
   - Asserts no `*_audit.json` at `handoff/` root.
   - Asserts no `*.log` at `handoff/` root.
   - Returns exit 0 when clean, 1 with a diff list when not.
5. Update `.gitignore` to exclude `handoff/logs/` and `handoff/*.log`.
6. RUN the backfill from repo root.
7. RUN the verifier -- must exit 0.

## Scope honesty
- `pre_tool_use_audit.jsonl` and siblings are live-append files. If the
  backfill fails mid-move because of an active append, the re-run is
  idempotent (already-moved files won't re-move).
- Templates are minimal scaffolds, not documentation -- a separate
  docs pass can expand them.
- Ambiguous files (phase-5.5-contract.md, phase-6.5-research-brief.md)
  are surfaced in the backfill report and not silently dropped.

## References
- Research brief: `handoff/current/phase-4.16.2-research-brief.md`
- `.claude/hooks/archive-handoff.sh` (bug sites at :92, :103)
- `.claude/masterplan.json` (authoritative step-status registry)
