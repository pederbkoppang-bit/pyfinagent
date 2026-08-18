STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.97
WRITTEN: 2026-08-16T20:33:48Z

# Q/A cycle-3 write-first record for step 86.97

Spawn context: cycle 3 (Main discloses attempt_number=3, prior verdicts
[CONDITIONAL, CONDITIONAL], run ids wf_3be25861-bde / wf_2dd1efc9-d0c).
Main's disclosure is ADVISORY ONLY -- to be re-derived from qa_wip.py +
verdict_history_86_21.py --evidence-only.

## Log

- [t0] Read .claude/agents/qa.md in full. Created this record.

## Prior-attempt / prior-verdict EVIDENCE (gathered, not applied)

qa_wip.py 86.97 --spawned-at 2026-08-16T20:33:48Z:
  source_present=true, attempt_number=3, attempt_number_status="ok",
  attempt_number_is_lower_bound=true, prior_attempts=2, records_retained=3 (GAUGE),
  prior_records = verdict_wip_86.97__20260816T201133Z.md,
                  verdict_wip_86.97__20260816T195546Z.md
verdict_history_86_21.py --step 86.97 --evidence-only:
  status = no_rows_for_step ; verdicts = (none)
CROSS-CHECK: attempt_number (3) > ledger verdict count (0) => THE LEDGER IS STALE
  for this step. Sequence per the authoritative source: UNKNOWN-from-ledger.
  Main's advisory disclosure [CONDITIONAL, CONDITIONAL] is consistent with
  prior_attempts=2 but is NOT independently corroborated by the ledger.

## Deterministic checks

1. IMMUTABLE COMMAND: bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'
   -> stdout "parses", exit 0.  [MET]
2. Real decision log md5 BEFORE any of my runs: 1f9a558f492123fcc13e962813560b02 (29 lines)
3. scripts/qa/verify_decision_log_86_97.py -> ALL GREEN: 35 passed, 0 failed, exit 0.
   Re-derived at execution time in MY run: window pinned to 2026-08-16T08:23:33Z,
   commits=57 decision lines=29 gap=28 recursion-guard commits=29.
4. Real decision log md5 AFTER: 1f9a558f492123fcc13e962813560b02 -- ISOLATION HOLDS
   under my own independent measurement, not just the script's self-assertion.

## Independent mutation re-derivation (in-memory, zero tree writes)

Technique: exec each checker's source with `HOOK_SRC = HOOK.read_text(...)`
replaced by an injected string. No file in the repo was modified.

- CRITERION 1 (second half), REPRODUCED BY ME:
    86.91 checker, CONTROL (unmutated hook)  -> ALL GREEN 42 passed, 0 failed, exit 0
    86.91 checker, delete-the-call MUTANT    -> ALL GREEN 42 passed, 0 failed, exit 0
  The mutant SURVIVES the pre-existing guard, with the control observed GREEN FIRST.
- CRITERION 4, REPRODUCED BY ME:
    86.97 checker vs the SAME delete-the-call mutant -> exit 1, 27 passed / 5 failed.
    Lead failure: "[3] a decision line is WRITTEN TO THE FILE ... no decision-log
    file was produced". The guard is bound to the real hook, not to a copy.

## Lint / scope

- Derived .py scope of the STEP (git diff 52358053..HEAD -- '*.py') =
  scripts/qa/verify_decision_log_86_97.py only. ruff F821,F401,F811 -> All checks
  passed, exit 0. Full default ruff on that file -> All checks passed, exit 0.
- `git diff --name-only HEAD -- '*.py'` returns backend/api/sovereign_api.py, which
  is a PEER SESSION's uncommitted sovereign-UI work (1y window enum). It is NOT in
  any 86.97 commit -- verified by `git diff --name-only 52358053 HEAD`, which lists
  ZERO backend/ or frontend/ files. No `git add -A` sweep occurred.
- Files changed by the three 86.97 commits (derived, 52358053..HEAD):
  .claude/agent-memory/researcher/project_uncalled_function_86_97.md,
  .claude/hooks/post-commit-changelog.sh, .claude/masterplan.json, CHANGELOG.md,
  handoff/current/{contract_86.91,contract_86.97,experiment_results_86.91,
  experiment_results_86.97,live_check_86.97,research_brief_86.97}.md,
  scripts/qa/verify_decision_log_86_97.py.
- Hook diff 52358053..HEAD is DOCSTRING-ONLY inside `_log_decision`. No code line
  changed. Confirmed by reading the full diff.
- Masterplan id->status map: 1397 -> 1398 steps. ADDED ['86.103']. REMOVED [].
  STATUS CHANGED {}. 86.97 itself still `pending`. NO verdict artifact in the diff.

## Independent enumeration (criterion 2) -- SYMMETRIC DIFFERENCE, not counts

My own tokenising enumeration (split on ; & | then do; first-word == exit),
computed independently of the shipped RULE regex:
  mine   = [28, 33, 37, 394, 396, 397]
  theirs = [28, 33, 37, 394, 396, 397]
  SYMMETRIC DIFFERENCE = []   <- exact member agreement, not merely equal counts
Correctly EXCLUDED: :228 (a comment inside the heredoc that contains the words
"exit 0"), :362 and :368 (`sys.exit(0)` inside the Python, both AFTER
`_log_decision(bump_type)` at :278, so they cannot cause a missing decision line).
KNOWN-MEMBER RECALL: all three known pre-detector members found.

## Independent vacuity probes on the guards that STILL have no shipped cell

M1 second-heredoc          -> "[1] exactly ONE heredoc" RED           (load-bearing)
M2 duplicate-terminator    -> "[1] terminator found exactly once" RED (load-bearing)
M3 no-exits-outside        -> "[2] rule finds non-zero" + PRE + POST RED
M5 drop-`reason=` from the written line -> "[3] the decision line carries a reason"
   RED, and it is the ONLY failure in that run -- independently load-bearing.
=> ZERO vacuous guards found. The criterion-6 residual is COVERAGE+DISCLOSURE,
   not a guard that cannot fail.

## Sibling gates re-run BY ME (not read from the artifact)

verify_changelog_flip_86_91.py     ALL GREEN 42 passed, 0 failed, exit 0
verify_workflow_args_boundary.mjs  ALL GREEN 96 passed, 0 failed, exit 0
verify_research_gate_workflow.mjs  ALL GREEN 124 passed, 0 failed, exit 0
All three reproduce the artifact's numbers exactly.

## *** FINDING A -- A SURVIVING MUTANT, AND THE DISCLOSURE'S JUSTIFICATION IS FALSE ***

CLAIM UNDER TEST, experiment_results_86.97.md:184-187 (the "Scope honesty" section):
  "`bump_type = _flip_magnitude()` (hook `:214`) is the second call site the
   research surfaced. It is covered incidentally by the end-to-end driver
   (IF IT WERE DELETED THE HOOK WOULD FAIL), but it has no dedicated mutation
   cell. Stated rather than implied."

MEASURED BY ME (in-memory mutant, anchor unique, bytes changed 16965 -> 16940):
  mutant = replace
      'if bump_type != "major":\n    bump_type = _flip_magnitude()'
    with
      'if bump_type != "major":\n    pass'

  DRIVEN DIRECTLY, base   : rc=0  log='... bump=none   reason=no_flip     created_done=- transitioned_done=-'
  DRIVEN DIRECTLY, mutant : rc=0  log='... bump=minor  reason=unrecorded  created_done=- transitioned_done=-'
  SHIPPED CHECKER vs mutant: ALL GREEN 35 passed, 0 FAILED, exit 0  <-- SURVIVES

So: (i) the hook does NOT fail -- rc=0 and a decision line IS written, so the
stated justification does not reproduce (qa.md 4b Contradiction); (ii) the mutant
is NOT equivalent -- behaviour differs in the exact direction phase-86.68 exists to
prevent, a spurious `bump=minor` on a `feat:` commit that flipped nothing, plus
`reason=unrecorded`, which is the unexplained-decision class 86.91 criterion 4
exists to close. The guard asserts THAT a line exists and that it contains
"reason=", never WHAT the decision was, so `reason=unrecorded` passes.
This is the step's OWN class recurring one line away: it closed one call site's
observable effect and left its sibling, while asserting the sibling was covered.

## *** FINDING B -- CRITERION 5 HAS AN UNCORRECTED MEMBER ***

handoff/current/live_check_86.91.md:104 :
    "## 3. Criterion 4 -- every decision now explains itself"
- The section body states NO bound. I grepped the WHOLE file for
  reach / pre-detector / bash exit / recursion guard / bound / 86.97: zero hits
  that bound criterion 4. A reader of that section meets only the unbounded claim.
- The file was NOT touched by ANY 86.97 commit (git diff --name-only 52358053 HEAD
  -- handoff/current/live_check_86.91.md is EMPTY).
- It is NOT covered by the stated exemption. experiment_results_86.97.md:161 and
  live_check_86.97.md exempt only "the verbatim checker output in
  live_check_86.91.md:167". :104 is authored PROSE, not captured output, and :167's
  labels I agree are accurate about the detector's internal branches.
- IT IS A KNOWN MEMBER. experiment_results_86.91.md:441 -- authored BEFORE this
  step, by the same author, and cited by this step's own §5 rewrite -- says
  verbatim: criterion 4's "every decision explains itself" holds only for
  invocations that REACH the detector. The author had the exact phrase in hand.
- WHY THE SWEEP MISSED IT: live_check_86.97.md §F sweeps for "every invocation".
  The surviving site says "every DECISION". A completeness probe built from the
  author's own wordings rather than the claim's semantics -- the same class as
  criterion 2's own rule ("a scan that cannot find its own known members is a
  FAILED gate"), which this step applies rigorously to bash exits and not to prose.

CONSIDERED AND NOT CHARGED:
- verify_changelog_flip_86_91.py:184-185 "NO UNEXPLAINED 'none'" -- 'none' is a
  value only the DETECTOR produces; a pre-detector exit never yields one. Correctly
  scoped by its own vocabulary. (Same conclusion the cycle-1 record reached.)
- live_check_86.91.md:167 assertion labels -- verbatim checker output, accurate.
- contract_86.91.md:150 "An unexplained `none` becomes impossible" -- reads
  "**within the detector**" in place. Corrected.

## CREDIT -- both prior WARNs are genuinely fixed (verified, not read)

W1: re-derived with the SHIPPED extractor at BOTH commits --
    52358053 : base 7597 B / f7458a6ab1f5fe96 ; call-deleted IDENTICAL
    HEAD     : base 8617 B / 072056e58af2befa ; call-deleted IDENTICAL
    All six sites now name the commit (checker docstring :18-19, contract :50,
    experiment_results :28-29 + :181-183, live_check :43 + :357-368,
    research_brief :122-125, researcher agent-memory :17-20). Fixed.
W2: section [5] ships and is load-bearing. `analyse(src)` is consumed by BOTH the
    shipped [1]/[2] assertions and the [5] cells -- NOT a re-implementation
    (verified by reading the call sites). Each cell changes EXACTLY the key it
    credits: funcs [] -> [40]; reorder [] -> [40]; unexplained [] -> [(40, ...)];
    unclassified [] -> [(41, ...)]. No mis-attributed mechanism (shape #11 clear).
    Fixed.
NOTE also verified fixed: the mis-named diagnostics (bash-n vs compile leg), the
accompany-form pointer at experiment_results_86.91.md:186 ("no longer expressible
BY THE DETECTOR" in place), and the isolation probe is now a pure predicate
`isolation_holds()` with no scoreboard patching.

## Harness-compliance audit (5/5 CLEAN)

1 research-gate-before-contract: research_brief_86.97.md envelope brief_status
  COMPLETE, gate_passed true, 8 read-in-full (>=5), 23 URLs (>=10),
  recency_scan_performed true, 9 internal files. Contract cites the enforced run
  wf_71bc038d-45a with the enforcer's own numbers.
2 contract-before-generate: all four artifacts entered in ONE commit (3894ac71),
  so git ordering cannot decide it; mtime chain is CONSISTENT and unviolated
  (brief 22:29:14 = contract 22:29:14 < checker 22:31:35 < live_check 22:32:46 <
  experiment_results 22:33:01). All 7 immutable criteria appear VERBATIM in the
  contract (checked programmatically, 7/7).
3 experiment_results present: yes, 10,582 B.
4 log-last: `grep -F "phase=86.97" handoff/harness_log.md` -> ZERO rows;
  masterplan 86.97 status `pending`. Correct.
5 no-verdict-shopping: evidence CHANGED -- commit 64ca8160 (22:33:16) rewrites the
  checker (+304/-92 region) and 6 artifacts. Not a re-spawn on unchanged evidence.

## Other measurements

- Gap guard sensitivity (it is NOT vacuous): live |gap-recursion|=1; if 3 lines
  were lost it still passes (=2), if 5 were lost it FAILS (=4), if 10 lost FAILS.
- Criterion-6 accounting: 34 check() sites; 11 are oracle/control/mutation lines.
  Assertions with NO shipped cell: heredoc-count, terminator-uniqueness,
  region-ordering, rule-nonzero, pre>0, post>0, reason-carried, gap-explained.
  I probed 4 of them (M1/M2/M3/M5) -- ALL load-bearing, ZERO vacuous. The residual
  is coverage+disclosure, and is NOT charged as a criterion miss.
- 1c live UI gate: N/A, no UI claim and no frontend file in any 86.97 commit.
- 1d backend smoke: N/A for backend/**, but the hook itself was DRIVEN end-to-end
  (base + recursion-guard subject + 3 mutants), which is the equivalent.

## CRITERIA

1 MET     - reproduced BY ME: 86.91 control 42/0 GREEN first, delete-the-call
            mutant 42/0 SURVIVED; gap re-derived live at 57/29/28/29 (a fresh
            number, not the filed 10-vs-5 nor any figure in the artifact).
2 MET     - written-down rule; SYMMETRIC DIFFERENCE [] vs my independent
            tokeniser; known-member recall 3/3 pre + 3/3 post; correct exclusions.
3 MET     - recursion guard DRIVEN (rc=0, no line), classified LEGITIMATELY-SILENT
            as a BOUND with reason; corroborated by the gap arithmetic.
4 MET     - deleting `_log_decision(bump_type)` turns the guard RED (exit 1, 27/5)
            -- verified BY ME. Extraction not patched.
5 NOT MET - Finding B: live_check_86.91.md:104 uncorrected, no bound anywhere in
            that file, not covered by the stated exemption, and a KNOWN MEMBER
            named verbatim in the author's own experiment_results_86.91.md:441.
6 MET+NOTE- control-GREEN-first both sections, UNSCORABLE arm wired, oracle
            two-sided, mechanisms correctly attributed, ZERO vacuity found in 4
            independent probes. NOTE: undisclosed residual subsetting.
7 MET     - masterplan add-only (1397->1398, ADDED ['86.103'], STATUS CHANGED {}),
            86.97 still pending, no verdict artifact touched, hook diff is
            docstring-only, zero backend/frontend files in any 86.97 commit.

PLUS Finding A (Contradiction): a measured surviving mutant at hook :213-214 whose
disclosure carries a justification that does not reproduce. The contract itself
named this call site at PLAN time as "a second exposed call site ... likewise
absent from the extraction ... precisely how the production call went unnoticed"
(contract_86.97.md:72-75).

## VERDICT SHAPE

Worst-of-lenses: correctness PASS; does-it-reproduce FAIL (the :214 justification
does not reproduce); scope-honesty FAIL (a false justification inside the
scope-honesty section itself, plus an uncorrected completeness member).
=> FAIL. The engineering is strong and both prior WARNs are genuinely fixed, but
one immutable criterion's universal quantifier is violated by a demonstrated
member, and a mutant of the step's own class survives behind a claim that is
measurably false.

COMPLETED: 2026-08-16T20:46:10Z

