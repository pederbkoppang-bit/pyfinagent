# live_check_75.19 — recalibrated masterplan preflight gate

All output below is verbatim from live runs on 2026-07-24 (Main session, repo root
`/Users/ford/.openclaw/workspace/pyfinagent`, `.venv` Python 3.14).

## 1. BEFORE — the mis-calibrated baseline (pre-75.19 code, captured before the rewrite)

```
$ .venv/bin/python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json
exit=1
preflight_verify_masterplan: scanned 863 steps, 151 broken, 8 unparseable
```

stderr: **222 `[BROKEN]` lines + 8 `[WARN]` lines** (full capture retained in the session
scratchpad `preflight_baseline_stderr.txt`; first lines):

```
[BROKEN] step=4.7.2: missing path 'handoff/lighthouse_home.json'
[BROKEN] step=4.8.6: missing path 'docs/runbooks/$f.md'
[BROKEN] step=5.5.6: missing path 'handoff/current/phase-5.5-contract.md'
[BROKEN] step=phase-6.5.3: missing path 'backend/tests/extractors/test_institutional.py'
[BROKEN] step=8.4: missing path 'handoff/current/phase-8-decision.md'
[BROKEN] step=8.4: missing path 'handoff/current/phase-8-decision.md'   <- duplicate emission
...
[WARN] step=23.2.7: unparseable command (No closing quotation)
```

Measured decomposition of the 222 lines (reproduction commands in
`research_brief_75.19.md` + `experiment_results.md`):

- 151 distinct step ids (summary counts steps, lines count tokens — relation never stated or tested)
- by status: 82 done / 40 pending / 10 deferred / 5 dropped / 1 superseded / 13 under `subphases[]`
- 12 of the 82 done ids already carried `superseded_record` (annotation-blind)
- 5 import lines, all on PENDING phase-5 steps (`backend.markets.*` not yet built by design)

## 2. AFTER — recalibrated run on the live masterplan

```
$ .venv/bin/python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json
exit=0
preflight_verify_masterplan (recalibrated phase-75.19): live_steps=872 scanned(done+unannotated)=710 genuine=0 lines across 0 steps; excluded: archived=4 annotated(superseded_record)=13 non-done{blocked=1 deferred=15 dropped=5 in_progress=1 merged=2 pending=122 superseded=3}; shlex-untokenizable(regex-scanned)=8
```

stderr (all 8, NOTE-bucketed, still scanned via regex extractors, never "broken"):

```
[NOTE] step=23.2.7: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=23.3.2: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=70.2: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=70.3: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=70.5: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=71.2: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=71.4: shlex-untokenizable (No closing quotation); scanned via regex extractors
[NOTE] step=71.5: shlex-untokenizable (No closing quotation); scanned via regex extractors
```

16.38's immutable invocation shape survives:

```
$ .venv/bin/python scripts/meta/preflight_verify_masterplan.py .claude/masterplan.json --quiet
quiet-exit=0        (stdout empty; NOTE lines on stderr only)
```

## 3. Verification command (immutable) — exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_19_preflight_calibration.py -q
.................................                                        [100%]
33 passed in 1.93s
```

## 4. git diff --stat (working tree at capture time)

```
 .claude/masterplan.json                       |   4 +-
 backend/tests/test_phase_75_19_preflight_calibration.py (new, ~320 lines)
 handoff/audit/pre_tool_use_audit.jsonl        | 178 ++++++++++   (hook stream)
 handoff/current/contract.md                   |  71 ++--        (75.19 contract)
 scripts/meta/preflight_verify_masterplan.py   | 452 ++++++++++++++++----------
```

## 5. Criterion-5 evidence: NO new superseded_record owed (byte-identity trivially holds)

Triage of the post-recalibration residue found **zero genuinely-unrunnable done steps
without an annotation**, reproducible two independent ways:

```
$ .venv/bin/python scripts/qa/sweep_absent_verification_paths.py
phase-75.17 sweep: 731 done/unannotated steps scanned; shape census dict=720 str=126 list=13 none=24
phase-75.17 sweep: CLEAN -- no genuine absent-path defects
```

plus the recalibrated preflight's own `genuine=0` (section 2), which additionally covers
`subphases[]` (13 steps the sweep's `flat_steps` cannot see) and the import leg.

Holder census: exactly **14** `superseded_record` holders repo-wide (13 done + 68.5
pending), unchanged by this step. `git diff .claude/masterplan.json` contains ONLY the
75.19 status-flip line — no verification block, no success_criteria, no annotation was
touched, so byte-identity of every verification block vs pre-step state holds by
`git diff` emptiness on those regions. The go_live_drills cluster (75.17-owned) was not
re-touched. Asserted in-suite by `test_live_annotated_count_matches_the_75_17_census`
(14 holders / 13 done) and `test_live_masterplan_is_currently_clean`.

## 6. Mutation matrix (qa.md §4c) — 7 mutations, 7 killed, 0 survivors

Runner: scratchpad `run_mutations.py` (applies mutation → full suite → restores → post-restore
sanity). Full verbatim log: scratchpad `mutation_matrix_75_19.txt`. Summary line, verbatim:

```
SUMMARY: 7 mutations, 7 killed, survivors: NONE
=== post-restore sanity: pytest exit 0 ===
```

| # | Mutation (applied to real code, executed) | Killed by (verbatim FAILED lines) |
|---|---|---|
| M1 | production: `if status != "done":` → `if status == "__m1_never__":` (status filter dropped) | `test_non_done_step_with_absent_path_is_not_reported[pending/deferred/dropped/superseded/in_progress/blocked]`, `test_unimportable_module_on_pending_step_is_not_reported`, + 4 live-masterplan tests — **11 failed** |
| M2 | production: `"superseded_record" in step` → `"__m2_no_such_key__" in step` (disposition dropped) | `test_superseded_record_step_is_dispositioned_not_reported`, `test_live_masterplan_is_currently_clean`, `test_live_annotated_count_matches_the_75_17_census` — **3 failed** |
| M3 | production: archive containers `child = "archived"` → `child = "live"` | `test_archive_containers_excluded_but_counted[archived_legacy_steps/archived_dropped_steps]` — **2 failed** |
| M4 | production: `if fp_reason(...) is None:` → `if True:` (adjudication bypassed) | all 5 `test_transient_and_nonsource_refs_excluded_by_construction[...]` params, `test_done_step_with_existing_path_is_clean`, `test_shlex_untokenizable_alone_is_not_broken`, +5 more — **12 failed** |
| M5 | production: summary `"genuine_lines": len(lines)` → `len(genuine)` (rows/summary decoupled) | `test_fixture_summary_agrees_with_rows` — **1 failed** |
| M6 | **FIXTURE** (criterion 6): `ABSENT` flipped to existing `backend/main.py` | `test_fixture_target_is_absent_and_positive_control_exists` + 8 downstream positives — **9 failed** |
| M7 | **STUB/harness**: consistency-detector corruption `+= 1` → `+= 0` (detector test made vacuous) | `test_consistency_checker_actually_detects_a_broken_summary` — **1 failed** |

Each guard has a named concrete mutation that flips it; both a FIXTURE mutation (M6) and a
harness-stub mutation (M7) are included per the 75.2.1 lesson and qa.md §4c.
