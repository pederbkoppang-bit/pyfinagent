# Experiment results — Step 75.19 (recalibrate the masterplan preflight gate, then triage its true residue)

Date: 2026-07-24. Execution model: opus-tagged step → Main (Fable 5) GENERATE
directly; Researcher gate + Q/A via Workflow structured-output at opus/max.
Research gate PASSED (wf_209bbec6-ace, envelope in contract.md).

## What was built

1. **`scripts/meta/preflight_verify_masterplan.py` — full recalibration rewrite** (452-line diff).
   - Status-aware: only `status=="done"` steps can be reported broken; every other
     status lands in a counted excluded bucket.
   - Annotation-aware: `superseded_record` holders are dispositioned (75.2.1/75.17
     semantics), excluded and counted.
   - Container-explicit walk `iter_steps()`: scans `phases[].steps[]` AND
     `phases[].subphases[]` (live), excludes-but-counts `archived_legacy_steps[]` /
     `archived_dropped_steps[]`, never descends into `superseded_record` values.
     Every reported id comes from a real node — zero '?' ids by construction.
   - Adjudication REUSED from `scripts.qa.sweep_absent_verification_paths`
     (`verif_commands`, `_extract_candidates`, `_clean`, `fp_reason`, `git_classify`)
     per the 75.17 charter; no adjudicator re-implemented. `classify()` itself is not
     called because its `flat_steps` misses `subphases[]` (measured recall gap) and it
     has no import leg — documented in the module docstring.
   - Import leg kept (sweep is path-only), gated behind the same status/annotation filters.
   - Shlex-independent scanning: regex extraction runs regardless; shlex failures go to
     a `shlex-untokenizable(regex-scanned)` NOTE bucket, never "broken". Commands untouched.
   - Internally-consistent summary + `check_consistency()` self-check (rows re-derived vs
     summary; any mismatch → INTERNAL-INCONSISTENCY, exit 2). Per-step de-dup kills the
     old duplicate-line emission (8.4 was emitted twice).
   - CLI compat: positional `path` + `--quiet` survive (16.38's immutable consumer);
     `--json` added for future status-aware CI wiring.
   - Testable core: `build_report(masterplan, repo_root, *, git_classify_fn, repo_basenames)`.
2. **`backend/tests/test_phase_75_19_preflight_calibration.py` — 33 tests** (new file; this
   is the step's immutable verification target). Fixture matrix: per-status pins (6),
   transient/non-source classes by construction (5: handoff/ output, gitignored log,
   `/openapi.json`, `lib/icons.ts`, `/Library/LaunchAgents/com.py`), POSITIVE
   genuine-defect fixture with absence-guard, annotation exclusion, `subphases[]`
   inclusion + archive exclusion (by container), list/dict shape BY SHAPE, shlex-fallback
   (bucketed AND still scanned; tokenization alone never broken), status-gated import leg,
   per-step de-dup, summary/rows consistency (fixture + live), live zero-unresolved-ids,
   live-clean reality pin, 14-holder/13-done annotation census, CLI exit codes 1/0/2 + --quiet.
3. **Triage outcome (work item d): NO new `superseded_record` owed.** Genuine residue on
   the live masterplan is 0 by two independent instruments (recalibrated preflight
   `genuine=0`; 75.17 sweep `CLEAN`). No annotation was added; `git diff
   .claude/masterplan.json` contains only the 75.19 status flip → byte-identity of every
   verification block holds trivially. go_live_drills (75.17-owned) untouched.

## Files changed

- `scripts/meta/preflight_verify_masterplan.py` (rewritten)
- `backend/tests/test_phase_75_19_preflight_calibration.py` (new)
- `.claude/masterplan.json` (75.19 status pending→in_progress only)
- `handoff/current/{contract.md, research_brief_75.19.md, live_check_75.19.md, experiment_results.md}` (harness artifacts)

## Verbatim verification output

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_19_preflight_calibration.py -q
.................................                                        [100%]
33 passed in 1.93s
```

Lint (scope derived from `git status --short | grep '\.py$'` AFTER the last edit):

```
$ uvx ruff check scripts/meta/preflight_verify_masterplan.py backend/tests/test_phase_75_19_preflight_calibration.py
All checks passed!
```

Syntax: `python -c "import ast; ast.parse(...)"` → `ast OK` (both files).

## Before/after artifact shape (full verbatim in live_check_75.19.md)

- BEFORE (exit 1): `scanned 863 steps, 151 broken, 8 unparseable` + 222 BROKEN lines
  (151 distinct ids: 82 done [12 already annotated] / 56 non-done / 13 in subphases).
- AFTER (exit 0): `live_steps=872 scanned(done+unannotated)=710 genuine=0 lines across
  0 steps; excluded: archived=4 annotated(superseded_record)=13
  non-done{blocked=1 deferred=15 dropped=5 in_progress=1 merged=2 pending=122 superseded=3};
  shlex-untokenizable(regex-scanned)=8`.

## Mutation matrix (qa.md §4c)

7 mutations (5 production + 1 FIXTURE + 1 STUB/harness), **7 killed, 0 survivors**,
post-restore suite green. Full table + verbatim FAILED lines in live_check_75.19.md §6.

## Measured corrections to prior claims (measure-don't-assert)

- The step text's 819/141/212/28 figures were 2026-07-20 stale; live baseline re-measured
  as 863/151/222/8+13-subphase ids. Documented, not trusted.
- My own first test asserted `annotated_excluded == 14` from the 75.17 census without
  measuring status: holder 68.5 is `pending`, so done-annotated is 13. Caught by the
  suite's own run; test now asserts both counts (14 holders, 13 done) with the reason.
