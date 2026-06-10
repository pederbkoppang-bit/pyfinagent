# Experiment Results — phase-10.10 (Housekeeping: quarantine phantom archive dirs)

**Step:** 10.10 **Date:** 2026-04-21

## What was done

1. Fresh researcher (moderate): 8 in full, 18 URLs, recency 2026, gate_passed=true. Brief at `handoff/current/phase-10.10-research-brief.md`. Confirmed phantoms are byte-identical duplicates; `shutil.move` same-FS = atomic `os.rename`; JSONL manifest-before-move pattern for crash recovery + reversibility.
2. Contract authored at `handoff/current/phase-10.10-contract.md`.
3. Created `scripts/housekeeping/quarantine_phantom_archives.py` (~175 lines):
   - Public `quarantine_phantom_dirs(*, archive_root, quarantine_subdir="_quarantine_2026-04-21", dry_run=True, manifest_path=None) -> dict`
   - `PHANTOM_RE = re.compile(r"^phase-\d+(?:\.\d+)+-v\d+$")`
   - `CANONICAL_RE = re.compile(r"^phase-\d+(?:\.\d+)+$")`
   - `_dir_sha256()` walks files in sorted relative-path order; hashes path + contents
   - Manifest-BEFORE-move pattern (flush per entry); JSONL format
   - CLI: `--archive-root`, `--quarantine-subdir`, `--no-dry-run`, `--manifest`
   - **`dry_run=True` default** — nothing mutates without explicit `--no-dry-run`
4. Created `scripts/housekeeping/restore_from_quarantine.py` (~90 lines):
   - Reads manifest; reverses each move via `shutil.move(quarantine_path, original_path)`
   - Verifies post-restore `dir_sha256` matches manifest; reports mismatches without aborting
   - Default dry-run
5. Created `scripts/harness/phase10_housekeeping_test.py` (~160 lines):
   - 4 cases matching masterplan success_criteria verbatim
   - Synthetic fixture: 3 canonical + 5 phantom dirs in `tempfile.TemporaryDirectory`
   - Case 4 chains quarantine → restore; asserts round-trip integrity
6. Created `tests/housekeeping/test_quarantine.py` — 8 pytest cases incl. edge cases (dry-run default, regex gates, SHA determinism, missing root)

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/housekeeping/quarantine_phantom_archives.py','scripts/housekeeping/restore_from_quarantine.py','scripts/harness/phase10_housekeeping_test.py','tests/housekeeping/test_quarantine.py']]; print('OK')"
OK

$ python scripts/harness/phase10_housekeeping_test.py
[PASS] phantom_dirs_moved_not_deleted  (moved=5, quarantined=5, root_phantoms_gone=True)
[PASS] canonical_dirs_untouched  (skipped_canonical=3, content_identical=True)
[PASS] manifest_written_with_sha256_per_dir  (entries=5, all_sha=True, all_keys=True)
[PASS] quarantine_is_reversible  (restored=5, mismatches=0, identical=True)

ALL PASS  (4/4)
(exit 0)

$ pytest tests/housekeeping/test_quarantine.py -q
........                                                                 [100%]
8 passed in 0.02s
```

## Dry-run against REAL archive (safe; no mutation)

```
$ python scripts/housekeeping/quarantine_phantom_archives.py --archive-root handoff/archive
{
  "dry_run": true,
  "manifest": ".../handoff/archive/_quarantine_2026-04-21/MANIFEST.jsonl",
  "moved": 0,
  "reversible": true,
  "skipped_canonical": 203,
  "would_move": 12647
}
```

Accounting: 203 canonical + 12,647 phantom + 147 other (non-phase-shaped dirs like `_templates`, `_audit*`, `_quarantine_*`) = 12,997 total. Canonical + phantom regexes are strict; "other" dirs are correctly left alone.

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `phantom_dirs_moved_not_deleted` | PASS — 5/5 moved to `_quarantine_*/`; none deleted; originals gone |
| 2 | `canonical_dirs_untouched` | PASS — 3/3 canonical dirs byte-identical before and after |
| 3 | `manifest_written_with_sha256_per_dir` | PASS — JSONL with 5 entries; all 6 required keys + valid 64-char hex SHA-256 |
| 4 | `quarantine_is_reversible` | PASS — restore companion reverses all 5; post-restore SHA matches manifest; zero mismatches |

## Safety rails enforced

- **`dry_run=True` default** on both scripts
- **Regex-gated canonical protection** — `CANONICAL_RE` never moved (test Case 2 asserts)
- **Manifest-before-move** — entry written + flushed before each `shutil.move()` call
- **Reversibility tested** — Case 4 chains quarantine → restore → asserts byte-identical recovery
- **Script ships, operator runs** — Main did NOT run `--no-dry-run` against real `handoff/archive/`; that's the operator's explicit decision

## Carry-forwards (out of scope)

- **Running the tool against the real archive:** operator decision. Command: `python scripts/housekeeping/quarantine_phantom_archives.py --archive-root handoff/archive --no-dry-run`. Dry-run shows 12,647 phantoms would move.
- **Permanent deletion of quarantine after cooling-off period:** separate ticket; the quarantine subdir can be `rm -rf`'d or `git clean -fd`'d once a reasonable window has passed.
- **Cleanup of the 147 "other" dirs** (`_templates`, `_audit*`, etc.) — out of scope; those are legitimate infra directories.
