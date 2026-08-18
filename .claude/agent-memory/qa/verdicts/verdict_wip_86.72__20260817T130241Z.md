STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.72
WRITTEN: 2026-08-17T13:02:41Z
COMPLETED: 2026-08-17T13:15:34Z
(the first value written on this line was a guessed clock and was corrected to
the value actually returned by `date -u` -- never narrate a clock you did not read)

# Q/A write-first record -- step 86.72 (research-on-demand / re-research leg)

Spawn context: first evaluation claimed by Main. Workflow rail, agentType qa.
Read `.claude/agents/qa.md` in full at runtime (it now carries the
"Research-on-demand (phase-86.72 ...)" section -- that section IS part of the
work under evaluation, so I loaded the artifact I am grading; noted).

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command + git scope + lint + tests
C. eight immutable criteria, each MET/NOT MET with cited evidence
D. independent mutation work (fixture + harness shapes, not just the author's cells)

## Findings (appended as established)

### Attempt / sequence evidence
- `qa_wip.py 86.72 --spawned-at 2026-08-17T13:02:41Z`: source_present=true,
  attempt_number=1 (status ok, not a lower bound), prior_attempts=0,
  prior_records=[]. records_retained=1 (gauge, includes this record).
- `verdict_history_86_21.py --step 86.72 --evidence-only`: status
  `no_rows_for_step`, verdicts `(none)`. prior_attempts 0 == ledger 0 -> no
  staleness contradiction detectable for this step.

### Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.72.md exists (37,763 B),
   mtime 11:50:14 < contract 12:22:17 < experiment_results 15:01:36. ORDER OK.
2. contract-before-generate: contract committed 65e38c3b 12:35:43; GENERATE
   commit 77f15b4d 15:02:10. OK.
3. experiment_results present + live_check present. OK.
4. log-last: `grep -cF "phase=86.72" handoff/harness_log.md` = 0; masterplan
   status = "pending". NOT yet logged/flipped. OK.
5. no-verdict-shopping: first spawn, no prior verdict. N/A.

### DISCREPANCY -- the spawn prompt's verification command is NOT the immutable one
- Spawn prompt gave: `node --check .claude/workflows/qa-verdict.js`
- masterplan.json 86.72 verification.command is:
  `bash -c 'source .venv/bin/activate && node --check .claude/workflows/research-gate.js && echo parses'`
- I ran BOTH. Both print `parses`, exit 0. No material impact, but the caller
  substituted a different command than the immutable field; recorded.

### Deterministic
- Immutable (masterplan, research-gate.js): `parses`, exit 0.
- Prompt-supplied (qa-verdict.js): `parses`, exit 0.
- `node scripts/qa/verify_prompt_render_86_90.mjs` -> exit 0,
  "ALL GREEN: 126 passed, 0 failed" (reproduced by me).
- md5 after my whole mutation run, repo untouched:
  qa-verdict.js 94164d41f77b5d53d2cb5378fbd110a6
  research-gate.js e26dc258bc862beead7f4a336c978480
  verify_prompt_render_86_90.mjs db1b2702c79e835f5ea5316d1c8dfbfa
  `git status --short -- .claude/workflows scripts/qa` = EMPTY.

### Independent mutation results (my own, on temp copies only)
CONTROL green first: all 9 section-[8] assertions green on unmutated source.

A. The author's four cells RE-SCORED -- all four are GENUINE behavioural kills
   (each mutant BUILT and the control oracle went red on a named assertion):
   - 8-recorder-neutered      -> oracle RED [pos->true, evidence recorded]
   - 8-recorder-hardcoded-false -> oracle RED [pos->true]
   - 8-routing-signal-dropped -> oracle RED [on: spec echoed, on: guidance at-most-2]
   - 8-tmax-bound-removed     -> oracle RED [on: guidance at-most-2]
   None were credited by the build-break catch. (But see harness defect below.)

B. HARNESS DEFECT (latent, did not fire here): section [8]'s cell loop scores a
   mutant that fails to BUILD as `DETECTED (mutant broke the build: ...)` ->
   counted as KILLED (verify_prompt_render_86_90.mjs:638). Section [7] of the
   SAME file scores the identical situation `UNSCORABLE` (:534). A future cell
   whose mutant is a syntax error would be falsely credited.

C. FIVE INDEPENDENT SURVIVORS on the guidance string (all 9 assertions stayed
   green; control was green first):
   - IM-1: guidance drops "Spawn the research gate BEFORE the next GENERATE"
     AND the research-gate.js scriptPath, keeping only "at most 2"/"stagnation"
     -> SURVIVED. The guard pins the BOUND, never the SPAWN INSTRUCTION -- i.e.
     never the thing criterion 3 is about.
   - IM-5: guidance rewritten to point at `.claude/workflows/qa-verdict.js`
     instead of research-gate.js (route back to the JUDGE instead of the
     RESEARCHER -- the exact inversion of the step's purpose) -> SURVIVED.
   - IM-4: the "stagnation" clause deleted -> SURVIVED. The control assertion
     checks only `includes('at most 2')`; the `stagnation` half is exercised
     only inside the 8-tmax cell's own red-predicate, never as a control.
     (ANDed-guard half left untested.)
   - IM-6: the FLOORS sentence (">=5 sources ... >=10 URLs ... recency scan")
     removed from the guidance -> SURVIVED.
   - IM-7: spec FABRICATED instead of echoed -> SURVIVED, because the fixture
     uses non-distinctive values o/f/t/b that a fabricator reproduces exactly.
   KILLED controls (so the oracle is not inert): IM-2 (spec dropped) killed by
   [on: spec echoed]; IM-3 (absent coerced to false) killed by [off: null/null].

D. FULL-SCRIPT DRIVE (runDriver replica, whole script, agent() stubbed):
   - CONTROL, no field: returns keys ... escalation, research_routing,
     verdict_unmodified; routing = {null,null,null}; no throw.
   - CONTROL, research_needed=true: research_needed/research_brief_spec appear
     INSIDE the verdict, routing nested beside it, verdict_unmodified=true.
   - LG-1 spread `...research_routing` into merged -> THROWS the phase-86.72
     invariant. The runtime leak-guard is REAL (I proved it; the author's
     matrix did not). Sections [0]/[2] assert `!r.threw`, so this is killed by
     the existing checker.
   - LG-2 delete the leakedR guard (no violation present) -> silent (expected).
   - LG-3 delete the guard AND spread -> SILENT: `next_action_on_research_needed`
     surfaces as a TOP-LEVEL sibling of ok/verdict/reason and nothing catches
     it; verdict_unmodified stays true. No output-SHAPE assertion exists.
   - LG-4 make the routing line mutate `verdict` in place -> SILENT, and
     `verdict_unmodified` still reports TRUE. That check compares `merged[k]`
     against `verdict[k]` where merged was built FROM verdict, so a pre-merge
     in-place mutation is invisible to it. experiment_results credits
     "verdict_unmodified still computes" as part of the criterion-7
     demonstration; it cannot detect that class.

### *** CRITERION 2 DOES NOT REPRODUCE -- the claim is CONTRADICTED, not confirmed ***
live_check §2 prints, as verbatim output:
  86.85: qa=12 researcher=0 / 86.84: 8/0 / 86.74: 7/0 / 86.94: 6/0 / 86.97: 5/0 / 86.71: 5/0
and experiment_results concludes "all show ZERO researcher re-engagement.
The audit-basis claim is CONFIRMED on today's corpus."

I re-derived it TWO independent ways.
 A) The artifact's OWN stated population rule (every agent-*.jsonl under
    */subagents/workflows/wf_*/, first user message, marker roles): 1352 agent
    files. Result: 86.85 res=1, 86.84 res=0, 86.74 res=1, 86.94 res=2,
    86.97 res=2, 86.71 res=1.
 B) INDEPENDENT operationalisation, no regex: the 604 workflow run records
    (`workflows/*.json`), role from `scriptPath` containing research-gate.js /
    qa-verdict.js, step from `args.step_id` (the dispatched field itself).
    Result: 86.85 res=1, 86.84 res=0, 86.74 res=1, 86.94 res=2, 86.97 res=2,
    86.71 res=1. A and B AGREE.
=> `researcher=0` is wrong on 5 of the 6 rows. Only 86.84 is 0.

Worse, the CONCLUSION is inverted on the two newest high-run steps. Full
chronology from the run records (all status=completed):
  86.97: researcher 08-16T19:47 (tier=moderate) -> qa x3 ->
         *** researcher 08-17T08:35 (tier=simple) *** -> qa x2
  86.94: researcher 08-16T21:08 (moderate) -> qa x3 ->
         *** researcher 08-17T07:07 (moderate) *** -> qa x3
That is a SECOND, COMPLETED research-gate launch AFTER Q/A cycling -- exactly
the "re-engagement" the audit_basis says never happens. Criterion 2 required
the claim be "confirmed OR CORRECTED"; it was confirmed where the corpus
corrects it.

Also, the "six highest-spawn steps" set does not reproduce: under the stated
rule the top qa counts include 36.8 (9), 78.2 (8), 86.28 (8), 75.5 (7) -- each
>= several of the six listed -- so the published six is a subset selected by an
unstated filter. A today-only filter does not rescue the zeros either
(today-only: 86.71 res=1, 86.94 res=1, 86.97 res=1).

### Consumer / wiring audit
- `grep -rn "research_routing|next_action_on_research_needed|enforceResearchRouting"`
  over *.js/*.mjs/*.py/*.md/*.json (excl node_modules/.venv): the ONLY
  occurrences are qa-verdict.js itself, the checker, and the 86.72 handoff
  artifacts. NO consumer.
- `grep -rni "tmax|re-research|rounds per step|stagnation"` over *.js/*.mjs/*.py:
  no code anywhere counts rounds or enforces the bound. The Tmax=2 bound is
  PROSE inside a returned string. qa.md's new section nonetheless tells the
  judge "the caller enforces at most 2 re-research rounds per step" -- there is
  no such caller code.
- Neither CLAUDE.md nor docs/runbooks/per-step-protocol.md was updated to tell
  MAIN (the only actor that can act on the signal) to read `research_routing`.
  The commit touched 7 files; no runbook among them. (grep over both files for
  research_needed|research_routing|re-research|research-on-demand returns ONLY
  CLAUDE.md:462, the pre-existing run_harness PLANNER leg -- not the Q/A leg.)
- `handoff/harness_log.md` carries NO separation-of-duties flag for the qa.md
  edit; last commit 1d9a360e 14:56 (phase-86.96) and `git diff HEAD` on it is
  EMPTY. experiment_results states in the PAST TENSE "Flagged in harness_log for
  operator review". Not true as written/committed. (LOG runs after EVALUATE, so
  it is schedulable -- but it is a past-tense claim in a GENERATE artifact.)

### Other gates
- Python lint: the GENERATE commit touched ZERO .py. Derived union of commit
  .py + `git diff --name-only HEAD -- '*.py'` + untracked = ONE file,
  backend/api/sovereign_api.py (86.69, in flight). `uvx ruff check
  --select F821,F401,F811` -> "All checks passed!", exit 0. Non-empty set
  asserted before reading the exit code.
- Frontend (binding because the diff touches .claude/agents/qa.md):
  `npx tsc --noEmit` exit 0. `npx eslint .` exit 1 -- 26 errors, ALL in the
  build-output dirs .next-audit-36-12/ and .next-functional/
  (@next/next/no-assign-module-variable); ZERO errors under src/. Pre-existing,
  already-queued class; this step touched no frontend/**.
- UI capture gate (1c): N/A -- no UI claim, no frontend in the diff.
- Backend runtime smoke (1d): N/A -- no backend/** in the diff.
- Research gate: brief COMPLETE, 8 sources read in full (>=5), 28 URLs (>=10),
  dedicated "Recency scan (2024-2026) -- PERFORMED" section, gate_passed true.
- ASK-1 cost basis reproduces: 86 research-gate runs carry totalTokens,
  p50 = 192,778; today's six = [165534, 187355, 189842, 199411, 206640, 262895].
  "~190-210K/run moderate" is consistent; the deep 1.5-2.5x is labelled
  "plausibly" and is not passed off as measured.

## Per-criterion conclusion
1. MET -- controls reproduce exactly on my re-run (5 / 0 / 7 / 0 / 0); the
   brief's D4 reports its disagreement with the audit_basis rather than
   adopting it silently.
2. NOT MET -- Contradiction. See the block above.
3. NOT MET -- the mechanism exists and I drove the routing function, but the
   demanded end-to-end demonstration ("causes a researcher spawn before the
   next GENERATE") was not performed in either arm, and NO consumer exists in
   code or in Main's own instruction files.
4. MET -- caller-supplied tier retained, justified in writing against cited
   sources, and demonstrably caller-supplied in shipped code
   (research-gate.js VALID_TIERS at :411, tier from args).
5. MET -- 'deep' absent from VALID_TIERS; ASK-1 numbered with a cost basis I
   reproduced.
6. MET -- FLOOR_SOURCES=5 / FLOOR_URLS=10 / recency checks unchanged; every
   added line of research-gate.js in this commit is a comment (verified by
   filtering non-comment '+' lines -> empty); .claude/rules/research-gate.md
   untouched since 2026-08-13.
7. MET -- I drove it: FAIL+research_needed -> FAIL, CONDITIONAL -> CONDITIONAL,
   verdict keys pass through, routing nested. WARN: experiment_results credits
   `verdict_unmodified` as part of the demonstration; that check catches only
   LG-5 (mutation of `merged` before the computation) and misses LG-4 and LG-6.
8. NOT MET -- the new runtime leak-guard is absent from the matrix (I mutated
   it myself; it fires), the guidance guard pins only "at most 2" and survives
   five independent mutants including a full inversion of the routed target,
   and section [8] scores a non-building mutant as KILLED.

Product code assessed CORRECT. The misses are in the evidence, the guard
coverage, and one substantive factual inversion (criterion 2).
