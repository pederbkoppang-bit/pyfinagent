# phase-38.5.1 + 38.5.2 -- experiment results (Cycle 42)

**Date:** 2026-05-23
**Cycle:** 42
**Steps batched:** phase-38.5.1 (sweep) + phase-38.5.2 (CI hard-gate flip)

---

## What was built/changed

### v1 sweep (`scripts/qa/sweep_ascii_logger.py`, NEW 155 lines)
- Bulk character-replacement using REPLACEMENTS map (35 entries).
- Operates only on lines containing `logger.`.
- AST-verifies each file post-edit.
- Result: swept 22 files, 116 lines.

### v2 sweep (`scripts/qa/sweep_ascii_logger_v2.py`, NEW 120 lines)
- JSON-driven targeted sweep using `ascii_logger_check.py --json` output.
- Handles multi-line `logger.*()` calls where emoji is on a continuation line.
- Result: swept 4 additional files, 10 lines.

### v3 remediation (`scripts/qa/sweep_ascii_logger_v3.py`, NEW 85 lines)
- Catch-all `?` cleanup -- removes the leading `"? ` from logger string literals.
- Regex: `(logger\.\w+\(\s*(?:[fFrR]...)?(["\']))\?\s+`
- AST-verifies each file.
- Result: 13 files, 24 substitutions; ALL `?`-prefixes eliminated.

### CI workflow flip (`.github/workflows/ascii-logger-lint.yml`)
- `continue-on-error: true` -> `false`
- Comments updated to cite phase-38.5.2 closure.

### Test rename (`backend/tests/test_phase_38_5_ascii_logger_check.py`)
- `test_phase_38_5_known_existing_violations_surface_in_real_codebase` -> `test_phase_38_5_real_codebase_clean_post_sweep`.
- Assertion flipped: was "expect 50-500 violations", now "expect 0 violations".

---

## Verbatim verification command output

```
$ python3 scripts/qa/ascii_logger_check.py --roots backend scripts
OK: 521 files, 1752 logger calls, 0 violations
exit 0
```

```
$ pytest backend/tests/test_phase_38_5_ascii_logger_check.py -v
9 passed in 0.68s
```

```
$ pytest backend/ --collect-only -q | tail -2
473 tests collected
```

```
$ grep "continue-on-error:" .github/workflows/ascii-logger-lint.yml | grep -v "^#"
    continue-on-error: false
```

---

## Cycle-2 remediation note (Q/A round-1 -> round-2)

Round-1 Q/A returned CONDITIONAL on 3 issues:
1. Researcher SKIPPED -- spawned retroactively (`handoff/current/research_brief_phase_38_5_1.md`)
2. experiment_results.md was stale phase-34 content -- THIS file is the refreshed version
3. Semantic-loss: 24 catch-all `?` substitutions -- v3 script removed all 24

Round-2 Q/A spawned on UPDATED evidence per CLAUDE.md cycle-2 flow.

---

## File touch summary

26 source files modified (150 line edits: 126 v1+v2 sweep + 24 v3 remediation).

| Category | Files | Lines |
|---|---|---|
| backend/services/* | 8 | 60 |
| backend/slack_bot/* | 7 | 47 |
| backend/* (other) | 5 | 27 |
| scripts/* | 4 | 9 |
| Other (config, tests) | 2 | 7 |
| **TOTAL** | **26** | **150** |

---

## Artifact: ASCII-clean codebase

521 Python files; 1752 logger calls; 0 non-ASCII string literals in any logger.*() call argument. CI lane now hard-fails on any new violation.
