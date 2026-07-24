# experiment_results (draft) -- Step 75.17: absent-path verification family (triage + repair)

**Executor**: Sonnet GENERATE pass (8th delegated). Repo-file edits only;
no commit/push/env edits/network. Main to review + re-measure before
spawning Q/A.

Baseline commit for every byte-identity / git-show comparison:
`7739922d8ab9ed8398dbb97da1a9a8d6ed6894ae` (pinned SHA, not "HEAD" --
mirrors `test_phase_75_2_1_push_approval.py`'s `BASELINE_COMMIT` pattern
so the claim stays meaningful after later commits land).

## What was built

1. **`scripts/qa/sweep_absent_verification_paths.py`** (new, ~340 lines) --
   the committed, importable classifier. Core API: `classify(masterplan:
   dict, repo_root: Path) -> dict` returns `{genuine, shape_census,
   steps_scanned}`. Refactored from the researcher's reference
   (`handoff/current/census_75_17_reference.py`) into a pure-ish
   core + thin CLI, mirroring the `scripts/qa/sweep_ascii_logger.py`
   idiom (module docstring, `REPO_ROOT = Path(__file__).resolve().parents[2]`,
   `main()` + `sys.exit`). Handles all 4 verification shapes
   (`verif_commands()`), 4 independent extractors feeding one adjudicator
   (`fp_reason()`), git-classification (`git_classify()`: never-existed
   vs retired, naming the retiring commit). A step already carrying a
   `superseded_record` is treated as dispositioned and excluded --
   this is what makes the classifier return an empty genuine set once
   annotated, and the pre-annotation genuine set against an older
   masterplan snapshot, without any hardcoded step-id allowlist.
   Docstring documents importability for step 75.19's preflight-gate
   recalibration; no nightly CI wiring added (per the research brief's
   explicit recommendation -- that is 75.19's job).
2. **10 `superseded_record` sibling annotations** in `.claude/masterplan.json`,
   inserted via a python round-trip (`json.load` -> insert -> `json.dumps(indent=2,
   ensure_ascii=False)` + trailing newline), mirroring the 4.17.9 house
   shape at masterplan:4870 (NOT the 68.5 originals-preserved variant,
   which was never touched).
3. **`backend/tests/test_phase_75_17_verification_paths.py`** (new, 45
   tests) -- the guard suite satisfying the node's immutable verification
   command.
4. **One in-scope collateral fix**: `backend/tests/test_phase_75_ci_gates.py`'s
   pre-existing collection-count canary (`test_backend_not_requires_live_collection_count_is_stable`)
   hard-codes an exact `N/M tests collected` string; adding 45 new tests
   bumped it from 1518/1534 to 1563/1579 (deselected unchanged at 16).
   Updated the string + added a phase-75.17 comment line, exactly the
   situation the existing phase-75.16 comment anticipated ("baseline
   moved... N new tests, none carrying requires_live, deselected count
   unchanged"). This is the only file touched outside the three above.

## Annotation table (10 steps x on_disk_equivalent, independently re-verified)

| Step | Plan-side name (never existed) | Class | on_disk_equivalent | Exists? | git --diff-filter=A |
|------|--------------------------------|-------|---------------------|---------|----------------------|
| 4.17.2 | researcher_smoke_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_2.py | YES | empty |
| 4.17.3 | qa_smoke_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_3.py | YES | empty |
| 4.17.4 | handoff_e2e_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_4.py | YES | empty |
| 4.17.5 | coala_memory_layers_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_5.py | YES | empty |
| 4.17.6 | signal_evidence_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_6.py | YES | empty |
| 4.17.7 | paper_trade_e2e_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_7.py | YES | empty |
| 4.17.8 | slack_bot_smoke_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_8.py | YES | empty |
| 4.17.11 | openclaw_runtime_test.py | (i) | scripts/go_live_drills/smoke_test_4_17_11.py | YES | empty |
| 4.17.12 | f1_recovery_drill.py | (i) | scripts/go_live_drills/smoke_test_4_17_12.py | YES | empty |
| 4.14.26 | (retired, not name-mismatch) | (ii) | n/a -- retired_by_commit f7e24d0a (phase-26.4) | n/a | n/a |

All 9 class-(i) checks and the class-(ii) retirement were independently
re-verified by this executor (not trusted from the research brief) via
direct `git log --all --diff-filter=A/D` calls -- see section 9 of
`live_check_75.17.md`.

Excluded, untouched (verified byte-identical `superseded_record` vs
baseline): 4.14.4, 4.14.24, 4.17.9, 68.5 (4 pre-existing holders + 10 new
= 14 total repo-wide, confirmed by `grep -c` and by the guard test's
duplicate-key-aware parse).

## Byte-identity proof method

For each of the 10 touched steps: `json.dumps(verification, sort_keys=True)`
compared between `git show 7739922d:.claude/masterplan.json` and the live
file. All 10: `command_match=True criteria_match=True`. Reproduced two
ways -- a standalone script (captured in `live_check_75.17.md` section 3)
and the parametrized pytest guard
(`test_verification_byte_identical_to_baseline`, x10) plus a second
parametrized guard for the 4 pre-existing holders confirming THEY were
not re-touched.

## Classifier results: empty (live) vs exactly-10 (baseline)

```
Live (post-annotation) masterplan  -> genuine = {}                (CLEAN, rc=0)
git-show baseline masterplan       -> genuine = {4.17.2, 4.17.3, 4.17.4,
                                       4.17.5, 4.17.6, 4.17.7, 4.17.8,
                                       4.17.11, 4.17.12, 4.14.26}   (exactly 10)
```
Shape census (live): `dict=720 str=126 list=13 none=24` -- matches the
research brief's corrected figures exactly (the node's "674 dict" is the
stale 75.2.1-era count over the 837-step, pre-phase-75 masterplan).

## Verbatim tails + exits

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_17_verification_paths.py -q
.............................................                            [100%]
45 passed in 2.36s                                                       exit=0

$ ruff check --select F821,F401,F811 scripts/qa/sweep_absent_verification_paths.py \
    backend/tests/test_phase_75_17_verification_paths.py backend/tests/test_phase_75_ci_gates.py
All checks passed!                                                       exit=0

$ PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false PAPER_SWAP_CHURN_FIX_ENABLED=false \
  .venv/bin/python -m pytest backend/tests/ -q -m "not requires_live"
1555 passed, 2 skipped, 16 deselected, 5 xfailed, 1 xpassed, 1 warning in 100.01s   exit=0

$ .venv/bin/python -m pytest backend/tests/ -q   (raw, no filters)
8 failed, 1553 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 123.50s
```

## Regression comparison vs the 8/9-red environment-dependent baseline

The raw-suite 8 failures are a **strict subset** of the documented 9-red
baseline (`tree_fails_75_15.txt`):
```
FAILED test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh
FAILED test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
FAILED test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence
FAILED test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
FAILED test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
FAILED test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
FAILED test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
FAILED test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
```
Missing vs the 9-red baseline: `test_phase_23_2_15_known_pass_scripts_still_pass`
-- the already-documented PATH-shell-dependent test (baseline's own
disclosure: "your shell may show 8"). Zero new failures. CI-equivalent
selection (`-m "not requires_live"` + 3 env overrides false) is fully
green: 0 failed.

## Masterplan diff purity

```
$ git diff HEAD --stat -- .claude/masterplan.json
 .claude/masterplan.json | 148 ++++++++++++++++++++++++++++++++++++++++++++----
 1 file changed, 138 insertions(+), 10 deletions(-)
```
All 10 "-" lines are trailing-comma artifacts (JSON requires a comma on
the previously-last property when a new sibling key is appended); each is
paired 1:1 with an identical "+" line plus the comma -- e.g.
`-  "max_retries": 3` / `+  "max_retries": 3,` immediately followed by
`+  "superseded_record": {`. Zero real content deletions, proven both by
manual grep and by the guard test
`test_masterplan_diff_touches_only_the_ten_sibling_insertions`.

## Mutation matrix M1-M10 (full table)

See `live_check_75.17.md` section 8 for the complete verbatim JSON and
discussion. Summary: **12/12 rows killed-or-expected-no-op, survivors: NONE**.
Every real-file mutation (M3, M6, M10, M7, and the M4/M5 resolver family)
was applied to an exact-count-1 pattern with a guaranteed restore,
verified byte-identical via shasum before and after the full run. M1
(masterplan parse drift), M2 (planted defect), M8 (byte-identity break),
and M9 (duplicate-key) were proven via standalone logic that never
touches the real 1.4MB masterplan file, even transiently -- M8 mutates a
`tempfile.NamedTemporaryFile` copy; M9 uses a small synthetic JSON string
reproducing the exact duplicate-key shape against the identical
`object_pairs_hook` detection logic the real invariant test uses.

**Notable finding, disclosed not smoothed over**: the M4 (URL-fragment)
resolver test required FOUR independent layers removed together to kill,
not the two originally planned -- the realistic
`curl http://localhost:8000/openapi.json`-shaped command is defended by
the well-formedness gate, the leading-slash branch, an incidental
boundary/mis-split gate, AND the dedicated localhost-URL regex. Two
intermediate mutation rows are RECORDED as measured non-kills (not
silently dropped), and the fourth-layer row is the one that actually
kills the test, proving it is non-vacuous rather than immune.

**Construction incident, disclosed**: an earlier draft of the M4+M5
mutation logic had a restore-order bug (writing the same target file
under two different variable names, so the second "restore" clobbered the
first with an intermediate half-mutated state) that left the committed
sweep module corrupted mid-session in a partially-mutated state. Caught
by re-running the guard suite immediately after, fixed by restoring the
exact original content and rewriting the mutation script to keep exactly
one original-backup per file across any compound mutation. Final
pre/post-run shasums on both real touched files are identical, confirmed
in `live_check_75.17.md` section 8.

## Deviations from the plan (all disclosed)

1. **One collateral file touched beyond the three named in the plan**:
   `backend/tests/test_phase_75_ci_gates.py`'s collection-count canary
   needed its hard-coded string bumped (1518/1534 -> 1563/1579) because
   this step legitimately added 45 tests. This is in-scope test-suite
   hygiene, not a scope creep -- the canary's own comment anticipated
   exactly this update pattern from the prior step (75.16).
2. **PRIOR_HOLDERS in the guard test includes `68.5`**, not just
   `4.14.4/4.14.24/4.17.9` as the contract's prose enumerates in one
   place -- the masterplan has 4 pre-existing `superseded_record` holders
   at HEAD (confirmed by `grep -c` = 4 before this step's edits: lines
   3908/4245/4870/15530), and the contract's own point 5 lists all four
   ("current holders: 4.14.4/4.14.24/4.17.9/68.5"). The test enforces the
   verified reality (14 total after +10), not the narrower 3-item list
   that appears elsewhere in the contract prose.
3. **Resolver unit-test assertions for the URL-fragment and
   truncated-plist cases assert `is not None` (excluded) rather than a
   specific FP-class label** (`"url-route"` / `"abs-host-path"`). Measured:
   a leading-slash token fails the well-formedness gate FIRST (returns
   `"malformed-token"`) before ever reaching the semantically-named
   branches -- this matches the researcher's own reference implementation
   byte-for-byte (same gate ordering), so it is not a deviation from the
   adopted design, just a correction to my own test's initially-assumed
   label. The behavioral contract ("excluded from genuine") holds either
   way and is what the immutable success criteria actually require.

## Coordination note for 75.19

The classifier is importable (`from scripts.qa.sweep_absent_verification_paths
import classify`); its docstring names 75.19 as the intended consumer for
the preflight-gate recalibration. No nightly workflow was added here. The
go_live_drills annotations belong entirely to 75.17 (done); 75.19 should
not re-annotate any of the 10.

## Nothing incomplete

All 5 verification-command-relevant success criteria independently
re-measured MET: (1) sweep handles all 4 shapes without crashing, FP
classes excluded by construction with fixtures; (2) every genuine hit
classified with evidence recorded per row; (3) all 10 gained a
`superseded_record` mirroring the 75.2.1 shape, byte-identity proven; (4)
4.14.4/4.14.24/4.17.9 excluded and untouched, exactly-one-per-step proven
repo-wide; (5) counts backed by sweep output, not asserted; (6) every
guard mutation-tested including the mandatory fixture mutation (M7).
