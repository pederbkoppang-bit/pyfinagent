# Research Brief: phase-10.10 — Non-destructive quarantine of phantom archive dirs

**Tier:** moderate  **Accessed:** 2026-04-21

## Read in full (8 sources; gate floor ≥5)

| URL | Kind | Key finding |
|-----|------|-------------|
| https://github.com/NLNZDigitalPreservation/Safe_mover | Code/tool | 18-field CSV manifest per file; restore via logged source paths |
| https://orbiscascadeulc.github.io/digprezsteps/fixity-deep.html | Digital-preservation doc | BagIt `manifest-sha256.txt`; Level-1 = verify fixity on ingest |
| https://www.dpconline.org/handbook/technical-solutions-and-tools/fixity-and-checksums | DPC handbook | SHA-256 for tamper-resistance; manifest files valid checksum storage |
| https://thelinuxcode.com/python-shutilmove-a-practical-guide-to-safe-file-and-directory-moves-2026/ | 2026 blog | `shutil.move` same-FS = atomic `os.rename`; 12k dirs fast |
| https://thelinuxcode.com/git-move-files-in-2026-reliable-renames-clean-history-and-real-world-workflows/ | 2026 blog | `git mv` fails on untracked; bulk OS-level moves + `git add -A` |
| https://ioos.github.io/ncei-archiving-cookbook/practices.html | NOAA NCEI | SHA-256 concurrent with data; SHA-1/MD5 deprecated |
| https://pypi.org/project/checksumdir/ | Package doc | `dirhash(path, 'sha256')` = single hash for entire tree |
| https://git-scm.com/docs/git-gc | Git official | `git gc` never touches untracked files — quarantine is git-safe |

## Recency scan (2024-2026)

Two 2026-dated sources confirm current behavior of `shutil.move` same-FS fast-path and `git mv` limitations. No peer-reviewed 2024-2026 academic literature on this specific pattern. No finding supersedes canonical approach.

## Key findings

1. **Phantoms are byte-identical duplicates** of canonical counterparts. Every phantom has only 4 rolling files (`contract.md`, `evaluator_critique.md`, `experiment_results.md`, `research_brief.md`) identical to canonical. Zero unique content.
2. **Hook race already stopped** via `.claude/archive-handoff.disabled` flag (confirmed present).
3. **Root cause:** `archive-handoff.sh:32-36` idempotency suffix loop fires repeatedly when HEAD masterplan wasn't committed.
4. **`shutil.move` on APFS same volume = atomic `os.rename()`** — metadata-only, no data copy. 12,784 dirs linear in count.
5. **OS-level move preferable to `git mv`** for 37,325 untracked files.
6. **JSONL manifest format:** append-safe, streamable, per-directory metadata.
7. **SHA-256 per-dir granularity:** `checksumdir.dirhash` or equivalent hashlib walk.
8. **Manifest-first pattern:** write entry BEFORE move; flush per entry; enables mid-run crash recovery.
9. **No `.gitignore` change needed** — `handoff/archive/` already tracked; moves produce git renames/new-untracked, no data loss.
10. **Script home:** `scripts/housekeeping/` — matches existing `backfill_handoff_archive.py` + `verify_handoff_layout.py` idiom.

## Final recommendation

**Script:** `scripts/housekeeping/quarantine_phantom_archives.py` (not a backend module).

**Public function:**
```python
def quarantine_phantom_dirs(
    *,
    archive_root: Path,
    quarantine_subdir: str = "_quarantine_2026-04-21",
    dry_run: bool = True,
    manifest_path: Path | None = None,
) -> dict:
    """Non-destructively move phantom `-vN` dirs into a quarantine subdir.
    Returns {moved, skipped_canonical, manifest, reversible}."""
```

**Regex gates:**
- Phantom (move): `^phase-\d+(?:\.\d+)+-v\d+$`
- Canonical (never touch): `^phase-\d+(?:\.\d+)+$`

**Manifest format (JSONL, one line per phantom dir, written BEFORE move, flushed):**
```json
{"original_path": "...", "quarantine_path": "...", "dir_sha256": "...", "size_bytes": N, "file_count": N, "moved_at_iso": "..."}
```

**Companion restore:** `scripts/housekeeping/restore_from_quarantine.py` reads manifest, calls `shutil.move(quarantine_path, original_path)`, verifies post-restore `dir_sha256`.

**Test scaffold:**
- Build synthetic tempdir with 3 canonicals + 5 phantoms
- `quarantine_phantom_dirs(archive_root=td, dry_run=False)`
- Assert: (1) 5 phantoms moved to quarantine; (2) 3 canonicals untouched at original paths; (3) manifest has 5 entries each with `dir_sha256`; (4) restore script reverses all moves

## Risk matrix

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Accidental canonical move | Low | Regex gate + `skipped_canonical` counter + `--dry-run` default |
| Mid-run crash | Medium | Manifest-before-move + flush-per-entry |
| Git-tracked files show deleted | Certain | Operator runs `git add` post-script if commit desired |
| Quarantine dir already exists | Low | Timestamp suffix in default name; never clobber |

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-10.10-research-brief.md",
  "gate_passed": true
}
```
