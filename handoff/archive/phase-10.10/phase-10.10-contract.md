# Sprint Contract — phase-10.10 (Housekeeping: quarantine phantom archive dirs)

**Step id:** 10.10 **Date:** 2026-04-21 **Tier:** moderate **Harness-required:** true

## Why

`handoff/archive/` holds 12,997 dirs: 213 canonical (pattern `phase-X.Y`) + **12,784 phantom** (pattern `phase-X.Y-vN`) created by the archive-handoff hook race. Phantoms are byte-identical duplicates of canonicals (4 rolling files each). 245 MB of dead weight, 73% of 37,325 files untracked by git.

User directive: **"make sure you don't destroy anything"** — non-destructive quarantine only.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.10-research-brief.md` — 8 sources in full, 18 URLs, gate_passed=true.

Key grounding:
- Phantoms are byte-identical (confirmed via internal audit); zero unique content
- `shutil.move` same-FS = atomic `os.rename()` — 12,784 dirs linear in count
- Manifest-before-move (JSONL, flush-per-entry) enables crash recovery + reversibility
- SHA-256 per-dir granularity via hashlib walk
- Regex gates enforce canonical-untouched invariant
- Git-tracked files survive via rename; untracked files survive because `git gc` never touches them

## Immutable success criteria (masterplan-verbatim)

Test: `python scripts/harness/phase10_housekeeping_test.py`

1. `phantom_dirs_moved_not_deleted` — after run, phantom dirs exist under `_quarantine_*/` subdir, not at original paths; count matches
2. `canonical_dirs_untouched` — all canonical dirs (pattern `^phase-\d+(?:\.\d+)+$`) still at original paths with unchanged `dir_sha256`
3. `manifest_written_with_sha256_per_dir` — `MANIFEST.jsonl` exists with one entry per moved phantom, each entry has `dir_sha256`, `original_path`, `quarantine_path`, `size_bytes`, `file_count`, `moved_at_iso`
4. `quarantine_is_reversible` — calling the restore companion script using the manifest restores every phantom to its original path, with `dir_sha256` matching the manifest value

## Plan

1. Create `scripts/housekeeping/quarantine_phantom_archives.py`:
   - Public `quarantine_phantom_dirs(*, archive_root, quarantine_subdir="_quarantine_2026-04-21", dry_run=True, manifest_path=None) -> dict`
   - Returns `{moved, skipped_canonical, manifest, reversible}`
   - Regex phantom: `re.compile(r'^phase-\d+(?:\.\d+)+-v\d+$')`
   - Regex canonical: `re.compile(r'^phase-\d+(?:\.\d+)+$')`
   - `_dir_sha256(path)`: walk files sorted by relative path; SHA-256 each file's bytes; combine
   - Manifest-before-move pattern: write JSONL entry + flush → `shutil.move()` → next
   - Default `dry_run=True`; only moves on `dry_run=False`
   - CLI entry: argparse with `--archive-root`, `--quarantine-subdir`, `--no-dry-run`, `--manifest`
   - ASCII-only logs
2. Create `scripts/housekeeping/restore_from_quarantine.py`:
   - Reads manifest JSONL
   - For each entry: `shutil.move(quarantine_path, original_path)`; recompute `dir_sha256`; assert matches manifest
   - Returns `{restored, mismatches, skipped}`
3. Create `scripts/harness/phase10_housekeeping_test.py`:
   - 4 cases matching success_criteria verbatim, all in `tempfile.TemporaryDirectory`
   - Synthetic fixture: 3 canonical + 5 phantom dirs, each with 4 rolling files
   - Case 4 calls the restore script to verify reversibility
4. Create `tests/housekeeping/test_quarantine.py` — pytest mirror (≥6 cases)
5. Verify: ast + immutable CLI + pytest + neighbor regression
6. **Do NOT run the script against the real `handoff/archive/` in this step** — that's a separate operator decision. This step ships the tool; the user runs it when ready.
7. Spawn fresh Q/A. Cycle-2 flow if gaps surfaced.
8. Log, flip, close task #94.

## References

- `handoff/current/phase-10.10-research-brief.md` (8 full, 18 URLs, gate_passed=true)
- `scripts/housekeeping/backfill_handoff_archive.py` (existing idiom)
- `scripts/housekeeping/verify_handoff_layout.py` (existing idiom)
- `.claude/hooks/archive-handoff.sh:32-36` (root cause documentation)
- `.claude/archive-handoff.disabled` (hook guard)

## Safety rails (non-negotiable)

- `dry_run=True` default
- Regex-gated: canonical pattern match NEVER moved
- Manifest written BEFORE move (crash-recoverable)
- `skipped_canonical` counter tracked and returned
- Test assertions on canonical integrity (Case 2)
- Restore script validates post-restore SHA-256 matches manifest

## Carry-forwards (out of scope)

- Running the script against the real `handoff/archive/` — operator decision after this step PASSes
- Actually deleting the quarantine dir after a cooling-off period — separate ticket
- Re-archiving the `-v2+` phase dirs if any were legitimate — not applicable (phantoms are confirmed byte-identical)
