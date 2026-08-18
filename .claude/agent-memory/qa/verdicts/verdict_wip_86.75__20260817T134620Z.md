STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.75
WRITTEN: 2026-08-17T13:46:20Z
COMPLETED: 2026-08-17T14:07:00Z

# Q/A cycle-2 evaluation of step 86.75 (harness best-practice audit)

Spawned via Workflow rail. qa.md read in full at 13:46:20Z.

## Attempt / sequence evidence (gathered, NOT applied)
- `qa_wip.py 86.75 --spawned-at 2026-08-17T13:46:20Z` -> attempt_number=2,
  prior_attempts=1, attempt_number_status="ok", attempt_number_is_lower_bound=false,
  source_present=TRUE (checked FIRST), records_retained=2 (GAUGE, not counter),
  records_pruned_known=null. Prior record:
  verdict_wip_86.75__20260814T025732Z.md.
- `verdict_history_86_21.py --step 86.75 --evidence-only` -> status=no_rows_for_step,
  verdicts=(none). detail: "nothing writes this ledger automatically yet, so absence
  here is weak evidence."
- CROSS-CHECK: attempt_number (2) > ledger verdict count (0) => LEDGER IS STALE for
  this step. Sequence treated as UNRELIABLE. Secondary cross-check harness_log:
  1 row for phase=86.75, token result=IMPLEMENTED-PENDING-REVIEW (not a verdict).
- The on-disk cycle-1 critique (handoff/current/evaluator_critique_86.75.md) records
  attempt 1 = CONDITIONAL. That is a FILE observation, not the ledger.

## Deterministic checks

### Immutable verification command (from .claude/masterplan.json, NOT the prompt's paraphrase)
```
bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_research_gate_workflow.mjs | tail -1'
-> ALL GREEN: 124 passed, 0 failed
IMMUTABLE_EXIT=0
```
(The spawn prompt quoted a DIFFERENT command; I ran the masterplan one. The prompt's
variant also returns green/exit 0.)

### Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.75.md 04:49:40 <
   research_gate_86.75_verdict.md 04:52 < contract_86.75.md 04:53:50. ORDER OK.
2. contract-before-generate: contract 2026-08-14 04:53 < experiment_results
   2026-08-17 15:11. OK for the RE-DERIVATION (the original GENERATE predates the
   contract -- disclosed breach, contained not repaired).
3. experiment_results present: yes, 11,110 bytes, cycle-2 section at tail.
4. log-last: masterplan 86.75 status=pending (NOT flipped). BUT harness_log.md:34389
   carries a PRE-EVALUATE `## Cycle 86.75 -- 2026-08-13 -- phase=86.75
   result=IMPLEMENTED-PENDING-REVIEW` row from commit 9a59a4fa. Token is not a
   verdict. NOW DISCLOSED in experiment_results cycle-2 item 4 (the cycle-1 critique
   required this). Order breach real, contained, non-fabricating.
5. no-verdict-shopping: evidence CHANGED. 7bbd6574 (2026-08-17 15:11:13 +0200) added
   +81 lines across experiment_results_86.75.md and live_check_86.75.md. Both mtimes
   2026-08-17 15:11 vs cycle-1 critique 2026-08-14 05:08. NOT a re-run on unchanged
   evidence.

### No unintended production change
`git show --stat 7bbd6574` -> 2 files, both handoff/current/*.md, +81/-0.
ZERO .py/.js/.ts/.tsx in the cycle-2 commit.
Working-tree .py delta = backend/api/sovereign_api.py only (concurrent work, not
86.75); `uvx ruff check --select F821,F401,F811` -> "All checks passed!", exit 0.

## Criterion-by-criterion re-derivation (mine, not the artifact's)

### C1 -- MET (was the cycle-1 Missing_Assumption; now cited AND independently verified)
- 86.64 attempt 3 notes, verbatim from evaluator_critique_86.64.md:
  "ATTEMPT NUMBER: 3 (of F1b's 5-attempt cumulative budget). qa_wip.py 86.64 ->
   source_present=TRUE (CHECKED FIRST), records_retained=3,
   prior_records=[verdict_wip_86.64__20260814T020057Z.md,
   verdict_wip_86.64__20260814T014304Z.md] -- two genuine prior spawns plus my own."
  => the >=2-prior-records half, DRIVEN, correct derived number.
- Disk corroborates: 3 x verdict_wip_86.64__*.md; qa_wip.py 86.64 records_retained=3.
- 86.68: 2 WIP files; critique carries ATTEMPT NUMBER 1 (records_retained=1,
  prior_records=[]) and ATTEMPT NUMBER 2 (records_retained=2, 1 prior). qa_wip 86.68
  records_retained=2.
- 0-prior half: 86.75's own cycle-1 spawn, "ATTEMPT NUMBER: 1 ... records_retained=1,
  prior_records=[], source_present=TRUE" -> unaffected.
- The "Not yet run" text is QUOTED-AND-REPLACED at the site (experiment_results:167),
  which is the correction-must-replace discipline.

### C2 -- MET
Positive control on the grep is present AND the artifact reports disagreement rather
than adopting audit_basis: headers 1,227 (audit_basis) vs 1,229/1,230 re-derived;
result=CONDITIONAL 35 vs 26 anchored / 36 unanchored, with the reasoning that 35 sits
between them so was taken unanchored. Negative control 99.99 -> 0. The self-match
finding (probe matching its own documentation, live_check:18-27) is an ADDITIONAL
independent reason the log-grep counter was unsound.

### C3 -- MET
`grep -n '| Contract completeness | gate |' .claude/agents/qa.md`
-> `606:| Contract completeness | gate | EVERY immutable criterion mapped to covering
    evidence in experiment_results.md (uncovered = Missing_Assumption, caps verdict) |`
LIVE table row. (Artifact cites :570; the line MOVED to :606 as qa.md grew. Stale
line number, live row.)

### C4 -- MET, and I have STRONGER evidence than the artifact
- `grep -n "const FLOOR_SOURCES\|const FLOOR_URLS" .claude/workflows/research-gate.js`
  -> 433: const FLOOR_SOURCES = 5 ; 434: const FLOOR_URLS = 10.
- `git log --oneline -S "const FLOOR_SOURCES = 5" -- .claude/workflows/research-gate.js`
  -> ONLY 22582714 (phase-36.27, the creating commit). Same for FLOOR_URLS.
  NEVER MODIFIED since creation. (Note: the artifact/cycle-1 cite `git log -L 213,214`;
  those lines have MOVED to 433/434, so that exact command now returns a DIFFERENT
  commit -- 98c5b6ab. The literal -S search is the non-stale derivation.)
- Lower-floor scan over a 31-file live-doctrine population (shell ARRAY, all 31
  asserted to exist) -> 3 hits, all non-floor:
  ARCHITECTURE.md:502 (records the RAISE from >=3), cycle_prompt.md:28 (correction
  note quoting removed text), researcher.md:264 (">=2 sources from ADJACENT domains"
  = cross-domain triangulation, an ADDITIVE requirement, not a gate floor -- read at
  :255-270 to confirm).
  My regex found a hit the artifact's did not; substantive claim ("no live rule states
  a lower floor") HOLDS. The artifact still does not state its C4 regex -- the same
  NOTE the cycle-1 evaluator raised, not fixed.

### C5 -- MET
`node scripts/qa/verify_research_gate_workflow.mjs` -> ALL GREEN: 124 passed, 0 failed.
124 >= 121 floor. Green-by-deletion impossible: the verifier's last touching commits
are 8b520f6c / 6b4df8f9 / 23270f29 / 133060b0 / d3bb1dfb (phases 86.81, 86.37) -- NOT
86.75. Count has GROWN 121 -> 124.
RESIDUAL: the artifact reports 121 and the cycle-2 section does NOT refresh it. Stale
but not false for its date, and the criterion is a FLOOR.

### C6 -- MET, reproduces byte-for-byte
`git grep -l "context/research-gate" -- . | wc -l` -> 21. `| sort` -> list IDENTICAL
to live_check:216-236 (all 21 entries, same order). `.claude/context/research-gate.md`
-> ABSENT confirmed. Population rule STATED beside the count, and the count-moves
caveat disclosed (11 at cycle-1 re-check under an archive-excluding rule, 21 today
under the all-tracked rule). Live pointers: 0 -- 20 are deletion notes / audit records
/ this step's own artifacts (self-match class disclosed); the single .py
(scripts/autoresearch/run_memo.py) says in its own docstring nothing reads the path.

### C7 -- see WIP continuation
### C8 -- see WIP continuation

## Findings so far
F1 (evidence-quality, NOT a criterion miss): live_check_86.75.md:242-243, inside a
block headed "## Cycle-2 captures (2026-08-17; exits unpiped)", shows
`$ grep -n "records_retained" .claude/agents/qa.md | head -2` followed by the PROSE
"(current wording -- the gauge correction; quoted in the GENERATE)" instead of the
command's output. The real output is qa.md:669 / :672. An edited/narrated line in a
block labelled as captures = the qa.md 4b Invalid_Precondition shape. The underlying
claim is TRUE (verified independently), so this is presentation, not fabrication.

F2 (NOT a finding -- a claim I tried to falsify and could not): "six commits touching
qa.md net +268/-19". Sum-of-per-commit numstats = +327/-78 (my first derivation), but
`git diff --numstat 9a59a4fa^..7bbd6574 -- .claude/agents/qa.md` = 268 / 19 EXACTLY.
"net" is the correct word for the cumulative diff. Claim REPRODUCES. Commit count at
7bbd6574 = exactly SIX (77f15b4d, 85127353, 9b4d5281, 2e40e8c7, 89e254fc, 9a59a4fa).

### C7 -- MET, with the attestation limit stated
ROSTER HALF -- verified BY ME, and more strongly than the artifact did. I am an
agentType:'qa' spawn, so qa.md is supplied as my SYSTEM PROMPT from the session's
roster snapshot (deletions are inert until restart; only additions are live via the
runtime read). Both 86.75 DELETIONS are ABSENT from my snapshot:
  - the weighted quant rubric (Statistical Validity 40 / Robustness 30 / Simplicity 15
    / Reality Gap 15 + "Score below 6 on ANY criterion = FAIL") is GONE; the table
    carries ONLY the Contract-completeness gate row, followed by "Why the weighted
    table was removed (phase-86.75, 2026-08-13)".
  - the anchoring clause "If an evaluator verdict is FAIL or CONDITIONAL, that is
    ground truth. Do NOT override it." appears in my snapshot ONLY as a QUOTE inside
    the "Why this changed (phase-86.75)" note, never as an instruction; the live text
    is "A prior evaluator verdict is EVIDENCE, not ground truth."
  - my snapshot also carries the 1777cc8d addition (research_router.py, committed
    2026-08-17 13:24:47Z, 22 min before my 13:46:20Z spawn) -> snapshot is current.
  - verify_qa_roster_live.sh step-3 probe answered directly: my snapshot DOES contain
    "### 1b. Frontend lint + typecheck" with the expected phase-23.2.24 opening lines.
  - I re-ran `bash scripts/qa/verify_qa_roster_live.sh` myself: "On-disk + git checks
    PASSED. Behavioral check is operator-driven." (script has no rm/mv/redirect).
OPERATOR-REVIEW HALF -- attested by Main, timing-corroborated by me, content NOT
independently verifiable by ANY evaluator: handoff/audit/pre_tool_use_audit.jsonl
records AskUserQuestion at 2026-08-17T13:04:11Z and 13:09:37Z -- 2 and 7 minutes
before the cycle-2 commit 7bbd6574 at 13:11:13Z. The stream records only
{ts, tool, verdict, reason}; it NEVER records the question text or the operator's
answer. So occurrence is corroborated, content is attested-only. STATED AS A LIMIT,
not smuggled in as proof. Cross-check: goal_next_2026-08-16.md:22 lists 86.84 as
"owed: separation-of-duties review" and does NOT list 86.75 -- consistent.

### C8 -- MET, re-derived on the CURRENT qa-verdict.js (not the cycle-1 line numbers)
  - sole verdict-producing assignment: :800 `const verdict = await
    agentRetryingDrops(PROMPT, {...})`.
  - drop path :813 `if (verdict == null || typeof verdict !== 'object') return verdict`
    -- returned unmistakably empty-of-verdict. NO VERDICT, never PASS.
  - enum :445 ['PASS','CONDITIONAL','FAIL'] intact; :519 blind run verdict:null.
  - `grep -nE "verdict\s*[:=]\s*['\"]PASS['\"]"` -> NONE. No path synthesises PASS.
  - escalation sits BESIDE the verdict with a RUNTIME leak guard that THROWS
    (:837 leaked / :848 leakedR), and `verdict_unmodified` is COMPUTED
    (`Object.keys(verdict).every(k => merged[k] === verdict[k])`), not attested.
  - `would_auto_fail = n >= 2 && verdict?.verdict === 'CONDITIONAL'` -- can only push
    CONDITIONAL -> FAIL. Monotone AWAY from PASS.
  - direction of 86.75's own change: the counter moved from a source that reads LOW
    (log 0 for 86.33) to one that reads TRUE (ledger 3); a higher count fires the
    auto-FAIL EARLIER. No mechanism reaches PASS.
  - the one honest caveat, which the artifact itself states at live_check:170-175:
    deleting the anchoring clause DOES permit overturning a prior FAIL -- but only on
    CHANGED evidence, and the separate "Never second-opinion-shop" constraint survives
    intact in the current file AND in my snapshot. Not an overclaim.

## Guard-vacuity pass (qa.md 4c) -- the mutation that would make each guard fail
  C1: delete one verdict_wip_86.64__*.md -> qa_wip returns 2, contradicting the note's
      "records_retained=3, prior_records=[2 named files]". The notes name SPECIFIC
      filenames that must exist on disk; I verified all 3 exist. NON-VACUOUS.
  C2: break the `^## Cycle ` anchor -> positive control 36.17 returns 0 instead of 3.
      I RAN the control: 3. NON-VACUOUS.
  C3: delete the table row -> grep exits 1. Criterion-MANDATED literal scan (skill
      negation list); strongest available guard for a doc-line existence claim.
  C4: weaken a floor in the source -> verify_research_gate_workflow.mjs:691-720 runs an
      EXECUTING 9-cell mutation matrix incl. 'FLOOR_SOURCES 5 -> 1' and
      'FLOOR_URLS 10 -> 1', each with an anchor-present AND actually-applied check.
      NON-VACUOUS -- this is real mutation machinery, not a source scan.
  C5: is the printed count hardcoded? NO -- :39-42 `function check(name, cond)`
      does `pass++` per passing call and :1037 prints `${pass}`. COMPUTED. Corroborated
      empirically: the file's own comments record the count at 61 / 73 / 92 / 117 in
      past cycles and it is 124 today, so it tracks content. NON-VACUOUS.
  C6: add an `open(".claude/context/research-gate.md")` anywhere -> my probe fires.
      I ran the probe WITH A WORKING POSITIVE CONTROL (same regex vs masterplan.json
      -> 3 hits, so the probe is live): ZERO code opens the deleted path. The only
      code hits for "research-gate" are verify_workflow_args_boundary.mjs:252,475
      reading SCRIPTS['research-gate.js'] -- the .js WORKFLOW, not the deleted .md.
  C7: roster half has a real mutation (an older snapshot would still carry the deleted
      rubric -- mine does not). Operator half has NO mutation available to any
      evaluator; that is the inherent limit, stated.
  C8: spread `...escalation` into the verdict -> the runtime leak guard THROWS. The
      file's own comment records that this exact flattening SURVIVED the checker before
      phase-86.78 cell QA-F, which is why it is now a runtime check. NON-VACUOUS.

## Deterministic gate results (all run by me)
  immutable command (masterplan)  : exit 0, "ALL GREEN: 124 passed, 0 failed"
  ruff F821,F401,F811 (step .py)  : "All checks passed!", exit 0
    (backend/config/model_tiers.py, scripts/autoresearch/run_memo.py)
  ruff (working-tree .py delta)   : exit 0 (backend/api/sovereign_api.py, concurrent)
  ast.parse                       : OK both
  runtime smoke (1d)              : `import backend.config.model_tiers` OK, 9 keys
  frontend tsc --noEmit           : exit 0
  frontend eslint .               : exit 1; ERRORS by top dir =
                                    {".next-audit-36-12":13,".next-functional":13},
                                    **0 in src/**. Pre-existing build-artifact dirs;
                                    86.75 touches ZERO frontend files.
  1c live UI capture              : N/A -- no UI claim in contract, criteria or diff.

## Corpus figures TODAY vs the artifact (reported, not adopted)
  ## Cycle headers          : audit_basis 1,227 | artifact 1,229/1,230 | ME 1,266
  anchored result=CONDITIONAL: audit_basis 35    | artifact 26          | ME 40
  unanchored                 :                   | artifact 36          | ME 50
  The artifact's inference ("35 sits BETWEEN 26 and 36, so it was taken unanchored")
  SURVIVES today's growth: harness_log.md is append-only, so the anchored count is
  monotone; it was 26 on 08-14, therefore <=26 on 08-13 when the audit_basis was
  written, therefore 35 could not have been an anchored count then. My 40 today is
  3 days of appends, not a contradiction.

## RESIDUALS (evidence-quality; none changes any criterion's MET/NOT MET)
  R1 live_check_86.75.md:242-243 -- a NARRATED line where output belongs, inside a
     block headed "## Cycle-2 captures (2026-08-17; exits unpiped)". The entry reads
     `$ grep -n "records_retained" .claude/agents/qa.md | head -2` followed by
     "(current wording -- the gauge correction; quoted in the GENERATE)". Real output
     is qa.md:669 + :672. qa.md 4b: an edited capture in a block labelled verbatim is
     an Invalid_Precondition finding REGARDLESS of whether the underlying command
     passed. The underlying claim IS true -- I verified qa.md:669 carries the gauge
     correction and my own snapshot carries it too. FIX: regenerate the two lines.
  R2 stale figures: experiment_results:5 and :65 and live_check:124 report
     "121 passed"; today it is 124. Criterion 5 sets 121 as a FLOOR, so 124 clears it,
     and the cycle-2 section does not refresh the number. Same for the C2 corpus table.
  R3 stale line numbers: artifact cites qa.md:570 (row is now :606) and qa.md:622
     (gauge text now :669); cycle-1 cites `git log -L 213,214` for the floors, but the
     floors moved to :433/:434 so that exact command now returns 98c5b6ab instead of
     the creating commit. The non-stale derivation is
     `git log -S "const FLOOR_SOURCES = 5"` -> ONLY 22582714. Facts hold.
  R4 the cycle-1 NOTE that the artifact never states its C4 lower-floor regex is still
     unfixed. My own broader regex found a THIRD hit the artifact does not list --
     researcher.md:264 ">=2 sources from ADJACENT domains" -- which I read at :255-270
     and confirmed is cross-domain triangulation, an ADDITIVE requirement, not a gate
     floor. Substantive claim ("no live rule states a lower floor") HOLDS.
  R5 the operator approval is a point-in-time snapshot already superseded: 1777cc8d
     (2026-08-17 13:24:47Z, phase-86.72) added +4/-2 to qa.md 13 min AFTER the approved
     set. That is 86.72's separation-of-duties obligation, not 86.75's -- noted so the
     snapshot nature of the approval is visible.

## Harness compliance (5 items) -- NOT CLEAN, and the reasons are historical
  1 research-gate-before-contract : CLEAN for the re-derivation (gate wf_c1b10b08-07c
    PASSED; brief 04:49 < gate verdict 04:52 < contract 04:53).
  2 contract-before-generate      : BREACHED for the ORIGINAL GENERATE (work predated
    both gate and contract), CLEAN for the re-derivation. DISCLOSED by Main
    ("The breach is NOT repaired -- only contained"). UNREPAIRABLE retroactively.
  3 experiment_results present    : CLEAN.
  4 log-last                      : BREACHED -- harness_log.md:34389 carries a
    pre-EVALUATE `## Cycle 86.75 -- 2026-08-13 -- phase=86.75
    result=IMPLEMENTED-PENDING-REVIEW` row from 9a59a4fa. Token is NOT a verdict; no
    verdict was self-authored; masterplan still status=pending. The cycle-1 critique
    required this be DISCLOSED and it now is (experiment_results cycle-2 item 4).
    UNREPAIRABLE retroactively.
  5 no-verdict-shopping           : CLEAN. Evidence CHANGED: 7bbd6574 added +81 lines
    across both graded artifacts (mtime 2026-08-17 15:11 vs cycle-1 critique 08-14
    05:08). Not a re-run on unchanged evidence.

## VERDICT REASONING (recorded before I looked at the sequence's implications)
  All EIGHT immutable criteria: MET on my own re-derivation.
  No unintended production change: cycle-2 commit = 2 handoff .md files, +81/-0.
  harness_compliance_ok = FALSE (items 2 and 4 above are true facts; I will not erase
  them by reporting true).
  The launcher conditions PASS on "harness compliance is clean". It is not clean. I
  cannot honestly certify it, and qa.md says prefer the conservative verdict when
  uncertain. Plus R1 is a genuine artifact-integrity defect of the exact class qa.md
  4b names, in a block that advertises "exits unpiped".
  => CONDITIONAL. NO CRITERION WORK REMAINS. The fixable gaps are R1 (regenerate two
  lines) and R2 (refresh two stale figures). The compliance breach is unrepairable and
  already disclosed exactly as the cycle-1 critique demanded.
