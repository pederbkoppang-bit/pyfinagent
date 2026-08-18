STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 75.11.4
WRITTEN: 2026-08-17T19:11:21Z
COMPLETED: 2026-08-17T19:41:05Z

# Q/A write-first record -- step 75.11.4 (handoff archive status-aware backfill)

Spawn started 2026-08-17T19:11:21Z. Workflow rail.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, syntax
C. Mutation re-derivation (independent, not accepting the author's matrix)
D. Criterion-by-criterion MET/NOT MET
E. Claim auditing over the prose (numbers must reproduce)

## Findings log (appended as established)

### Prior-attempt / sequence evidence
- `qa_wip.py 75.11.4 --spawned-at 2026-08-17T19:11:21Z`: source_present=true,
  attempt_number=2 (status ok, not lower bound), prior_attempts=1,
  records_retained=2 (GAUGE), prior record
  `verdict_wip_75.11.4__20260817T185113Z.md`.
- `verdict_history_86_21.py --step 75.11.4 --evidence-only`: status=ok,
  1 verdict from the ledger, sequence = `FAIL`.
- Cross-check: prior_attempts (1) == ledger rows (1) -> ledger NOT stale.
  sequence = FAIL (one row).

### B. Deterministic
- IMMUTABLE COMMAND `.venv/bin/python -m pytest
  backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`
  -> **exit 0, 22 passed in 0.70s**. (bare re-run confirmed
  `IMMUTABLE_CMD_EXIT=0`.)
- Ruff gate, scope DERIVED (`git diff --name-only HEAD -- '*.py'` UNION
  `git ls-files --others --exclude-standard -- '*.py'`, xargs, non-empty
  guard): 6 files, `All checks passed!`, exit 0.
  NOTE: `git diff --name-only HEAD` ALONE misses the 3 NEW untracked files
  (handoff_naming.py, quarantine_misattributed_archives.py, the test) --
  union with ls-files --others was required (my own memory
  `derived_scope_misses_untracked_files`).
- Adjacent regression suites 36.7 + 36.8: **77 passed**, exit 0.
- `verify_handoff_layout.py` on the live tree: **exit 1, 455 violations**.
  Composition re-derived: 452 `current/<file> belongs to <closed> step`
  (443 done / 6 dropped / 3 superseded) + 3 root-level
  (autoresearch.log, autoresearch.launchd.log, prompt_leak_redteam_audit.jsonl).
  Matches GENERATE's "452 closed-step files" claim exactly.
- 156 untracked `MISATTRIBUTION_NOTICE.md` files confirmed by
  `git status --short -uall | grep -c MISATTRIBUTION_NOTICE.md` = 156.

### C. Independent mutation matrix (my own harness, NOT the author's)
Harness: hermetic mirror of scripts/housekeeping/{backfill,handoff_naming,verify}
+ optional mirrored hook; test module's HOUSEKEEPING/HOOK repointed via
`pytest.main(argv, plugins=[P()])` in a fresh subprocess. Repo never written.
**CONTROL (null mutant, mirrored dir AND mirrored hook): 22 passed, rc=0** --
so the harness measures the subject, not relocation.

| cell | mutation | result |
|---|---|---|
| M1 | `if is_archivable(status):` -> `if True:` | 5 failed / 17 passed **KILLED** |
| M2 | `if name in protected:` -> `if False:` | 2 failed **KILLED** |
| M3 | no-step-id branch sweeps to MISC | 2 failed **KILLED** |
| M4 | unknown-step branch sweeps to MISC (warn kept) | 2 failed **KILLED** |
| M-INV | `dry_run=not args.execute` -> `dry_run=args.execute` | 3 failed **KILLED** |
| N5b | protection reads only `live_check` (drops `command`) | 2 failed **KILLED** |
| **N5a** | **protection reads only `command` (drops `live_check`)** | **22 passed -- SURVIVED** |

All five cycle-1 findings independently re-derived as FIXED.

**N5a IS A REAL, NON-EQUIVALENT SURVIVOR (behavioural differential run):**
fixture = `census_99.4.json` (belongs to DONE step 99.4) referenced ONLY by
step 99.3's `verification.live_check`.
- CONTROL `--execute`: current/=['census_99.4.json'], archive=[],
  `[protected]` printed = True
- MUTANT N5a `--execute`: current/=[], archive=
  ['handoff/archive/phase-99.4/census_99.4.json'], `[protected]` printed = False
...and the whole 22-test suite stays GREEN.

Root cause: `_referenced_tree` plants the protected file in
`verification.command`; the `live_check` it also writes points at
`handoff/current/live_check_99.3.md`, which the fixture never creates. So the
live_check half of `_masterplan_referenced_names` is exercised by nothing.

LIVE EXPOSURE (re-derived from .claude/masterplan.json with the function's own
rule): 381 protected basenames; 215 from `command`, 174 from `live_check`,
**166 protected ONLY by the live_check half**. 18 of those exist in
handoff/current/ right now and **15 belong to CLOSED steps**, i.e. 15 real
files whose only thing keeping them in place is a guard half no test can fail
(live_check_75.20.1.md, 75.5.12, 76.9.2, 76.9.3, 78.0, 78.16, 78.2, 79.55,
80.1, 80.2, 80.27, 80.3, 80.31, 80.4, 80.5).

=> Criterion 7 names EXACTLY this mutation ("point a step's
verification.live_check at a file the classifier would otherwise sweep -> the
protection test goes red when the guard is removed"). It does NOT go red.
Criterion 7 NOT MET. Criterion 5's `or verification.live_check` half NOT MET.

### Extended matrix (all cells CONTROL-anchored)
| cell | mutation | result |
|---|---|---|
| N6 | ARCHIVABLE_STATUSES += "pending" | 5 failed **KILLED** |
| N7 | `is_archivable` -> `status is not None` | 5 failed **KILLED** |
| N7b | drop "superseded" | 1 failed **KILLED** |
| N7c | drop "dropped" | 1 failed **KILLED** |
| N8 | SUFFIX_RE broken | 12 failed **KILLED** |
| N9 | VARIANT_RE broken | 2 failed **KILLED** |
| N10 | PREFIX_RE (legacy) broken | 1 failed **KILLED** |
| N11 | `_move` dry_run early-return removed | 2 failed **KILLED** |
| N13 | HANDOFF_ROOT_KEEP emptied (36.7 regression) | 1 failed **KILLED** |
| N16 | `_move` copies instead of moving (kills convergence) | 10 failed **KILLED** |
| H1 | hook: `rolling_declares_step` -> `true` | 1 failed **KILLED** (test_c13) |
| H2 | hook: derived suffix source disabled | 2 failed **KILLED** |
| H3 | hook: derived branch writes wrong target name | 1 failed **KILLED** |
| H4 | hook: `live_check` dropped from derived base list | 2 failed **KILLED** |
| N12 | `_is_rolling_keep` exact-name branch -> False | 22 passed SURVIVED (likely EQUIVALENT: resolve_step_id returns None for those names anyway) |
| **N14** | **verifier: done-step arm `elif is_archivable(status)` -> `elif False`** | **22 passed SURVIVED** |
| **N15** | **verifier reverts to the dead PREFIX-only regex, KEEPING the `from handoff_naming import` line** | **22 passed SURVIVED** |
| **H5** | **hook: derived file written EMPTY (`: > target`) instead of copied** | **22 passed SURVIVED** |

**N14/N15 non-equivalence proven on the LIVE tree:**
shipped verifier -> `handoff layout FAIL -- 455 invariant violation(s)`;
N14 -> `FAIL -- 3`; N15 -> `FAIL -- 3`. Both restore EXACTLY the pre-fix state
the module docstring names as the defect ("the done-step arm became
unreachable ... the invariant could not fire"). **No test in the suite ever
executes `verify_handoff_layout.main()`** -- `_load_script` is only ever called
with `"backfill_handoff_archive"`. The verifier's only guard is the
byte-presence pin `assert "from handoff_naming import" in src`, which N15
satisfies while reverting the behaviour (vacuity shape #3, literal kept /
behaviour stripped).

**H5 non-equivalence:** the hook writes a zero-byte `contract.md` /
`live_check.md` into the archive dir and test_c8_c11 stays green, because it
asserts existence plus the NEGATIVE `"ANOTHER STEP" not in ...`, which an empty
file satisfies. A genuine behavioural guard coexists (H2/H3/H4 kill), so this is
WARN-level, not sole-coverage.

### Criterion 12 / 13 probes
- c12 fixture mutation: synthetic mismatched dir WITH marker -> 1 passed;
  WITHOUT marker -> 1 failed. **The marker assertion IS load-bearing.**
- c13 detection half: `classify()` on a planted foreign contract ->
  `('mismatch','82.54')`; on its own -> `('agree','77.8')`.
  `run_controls()`=True, `run_precision_controls()`=True (4 synthetic controls
  forcing BOTH answers). Detection is real; the suite's own c13 tests
  PREVENTION, and H1 kills it.

### Claim reproduction (cycle-2 corrections)
- docstring triple re-derived with the function's OWN rule:
  total=**577**, distinct_paths=**386**, into_current=**415**,
  distinct_basenames=**381** -> matches the corrected docstring EXACTLY
  (cycle-1's 557/373/395 is gone).
- live census: **156** mismatches, precision **0.9936**, contestable **43**,
  845 phase-dirs, **156** MISATTRIBUTION_NOTICE files, 24 PROVENANCE dirs.
  The "165" figure cycle-1 flagged is gone.
- ruff F401 on `importlib.util`: gone (exit 0).
- "On the live tree 19 files are held back" -> measured **20**
  (the 20th is this step's own `live_check_75.11.4.md`, created during the
  step). Self-explaining drift; NOTE only.

### Dry-run side effect (measured on the LIVE tree)
A bare (dry-run) invocation calls `_move` -> `dest_dir.mkdir(parents=True)`
BEFORE the `if dry_run: return` guard, so a "dry run" CREATES directories.
Hermetic probe: a no-arg run created handoff/archive, archive/misc,
archive/phase-99.2, archive/phase-99.5, audit, logs while moving 0 files.
LIVE EVIDENCE: `handoff/archive/phase-80.5`, `phase-81.1`, `phase-82.23` are
EMPTY with mtime **2026-08-17T18:42:23Z** -- i.e. created by this step's own
dry run. Disclosure 5 says "the backfill was never run with --execute against
the live tree" (true) but the dry run did write to it.
CLASSIFIED AGAINST HEAD: the mkdir-before-dry_run-check is PRE-EXISTING
(`git show HEAD:...` has the identical `_move`). Not introduced here, but it
is newly load-bearing because this step makes dry-run the DEFAULT and stakes
criterion 6 on it. NOTE/WARN, not a criterion miss. One-line fix: move the
mkdir below the dry_run early return.

### A. Harness compliance (5/5)
1. research-gate-before-contract: brief 18:35:04Z < contract 18:38:30Z
   (original; contract mtime now 19:05:54Z because cycle 2 corrected the
   numbers in it). Envelope: brief_status=COMPLETE, gate_passed=true,
   sources_read_in_full=18 (floor 5), urls=72, recency scan present,
   coverage.dry=true.
2. contract-before-generate: original ordering held (verified by cycle 1 and
   consistent with the cycle-2 edit times).
3. experiment_results present (19:09:41Z), live_check present (19:10:38Z).
4. log-last: `grep -nF "phase=75.11.4 result=" handoff/harness_log.md` -> 0;
   masterplan status still `pending`, retry_count=0, max_retries=3.
5. no-verdict-shopping: evidence CHANGED materially (19->22 tests, `_fake_repo`
   subprocess harness, 2 new fixture classes, F401 removed, 3 numbers
   corrected). Independently confirmed: all three cycle-1 survivors are now
   KILLED in MY harness, not just the author's.
Gates 1b (frontend) and 1c (UI) N/A -- no frontend/** or UI claim in this
step's diff or criteria. 1d: the only backend/** file is the test module,
which pytest imports and executes (22 passed).

### Scope
`.claude/hooks/**` UNTOUCHED (git status clean). `.claude/masterplan.json`
UNTOUCHED. Step's own changes = 2 modified housekeeping scripts + 3 new files
+ 156 additive markers. Peer-session uncommitted work
(backend/api/sovereign_api.py, 5 frontend/src files, and -- not named in
disclosure 6 -- backend/services/experiments/perf_results.tsv) is NOT this
step's and must not ride its commit.

### D. Criterion-by-criterion
1 MET (M1/M3/M4/N6/N7/N7b/N7c/N8/N9/N10 all KILLED; warn line + counters asserted)
2 MET (test_c2 both directions in one run; M1 kills)
3 MET (M1 KILLED independently; test_c3 fixture flip)
4 MET (test_c4; N16 copy-instead-of-move KILLED -> assertion load-bearing)
5 **PARTIAL** -- `command` half proven (M2, N5b KILLED); `live_check` half
  UNPROVEN (N5a survives; fixture's live_check names a file it never creates)
6 MET (M-INV + N11 KILLED; real `__main__` driven as a subprocess) + dry-run
  mkdir NOTE above
7 **NOT MET** -- the criterion NAMES this mutation and the suite has no such
  cell; performing it leaves 22/22 green with a proven behavioural differential
8 MET (H2/H3/H4 KILLED; real hook driven; deviation disclosed with reason)
9 **NOT MET as a behavioural guard** -- no test ever executes
  verify_handoff_layout; N14 and N15 both survive and both restore the exact
  pre-fix state (455 -> 3 violations); the only tie is a byte-presence pin
10 MET in its provable half (whole-tree checker with recall gate + 4 controls,
   not a spot check); the literal property is FALSE (156 dirs) and is
   explicitly disclosed and routed to criterion 12's marker alternative
11 MET, WARN -- H5 (hook writes an EMPTY archived file) survives because the
   assertion is existence + a NEGATIVE substring; genuine guards coexist
12 MET (156 markers; count derived from classify() at run time; fixture
   mutation proves the marker assertion is load-bearing: with marker GREEN,
   without RED)
13 MET (H1 KILLED; guard discriminates; detection separately validated --
   classify() returns mismatch/agree correctly and run_controls()/
   run_precision_controls() both True)

### Verdict reasoning
10 of 13 criteria genuinely MET with guards I independently killed. Every
cycle-1 finding is genuinely fixed and every corrected number reproduces
exactly. But criterion 7 is a MUTATION criterion whose named mutation is
absent from the suite and survives when performed -- a criterion MISS, the
same class (and same severity) as the cycle-1 criterion-6 finding. Criterion 5
shares its root cause and criterion 9 has an independently-proven vacuous
guard. Product code is CORRECT in all three cases.
=> FAIL.

### Fix list for the next cycle (all small; no product behaviour change needed)
1. Criterion 7/5: add a fixture whose protected file is referenced ONLY by a
   step's `verification.live_check` (and which belongs to a done step so it
   would otherwise move), assert it stays + `[protected]` prints, and add the
   cell `for k in ("command","live_check")` -> `("command",)` proving it goes
   red. 15 real files in handoff/current/ depend on this half today.
2. Criterion 9: drive the verifier behaviourally --
   `_load_script("verify_handoff_layout")`, point it at a temp tree holding a
   done-step suffix-named file, assert `main() == 1` and that the file is
   named in the failure list. Confirm N14 (`elif False`) and N15 (revert to
   the PREFIX-only regex while keeping the import line) both go red.
3. Criterion 11 (WARN): assert the archived `contract.md`/`live_check.md`
   CONTENT equals the source, not just that the file exists (H5).
4. NOTE: move `dest_dir.mkdir(...)` below the `if dry_run:` return in `_move`
   so a dry run is genuinely read-only; and clean up the 3 empty dirs the dry
   run created (phase-80.5 / 81.1 / 82.23).
5. NOTE: "On the live tree 19 files are held back" -> 20; "SURVIVORS: none"
   is a global claim from a 3-cell matrix -- scope it to the cells run.

### Closing state
- FINAL immutable command re-run at the END of the evaluation: 22 passed,
  exit 0.
- Mutation residue: `grep -c QA_PROBE` and `grep -c MUTANT` = 0 in
  backfill_handoff_archive.py, handoff_naming.py, verify_handoff_layout.py and
  .claude/hooks/archive-handoff.sh. Every cell ran on hermetic copies under
  tempfile.mkdtemp(); test-module constants were repointed via
  `pytest.main(argv, plugins=[...])`, so no plugin/conftest file was written.
- HEAD = 8f281795 at the start AND at the end of this evaluation.
- `.claude/hooks/`, `.claude/masterplan.json` and `scripts/qa/` all clean.
- My only writes were to this file.

