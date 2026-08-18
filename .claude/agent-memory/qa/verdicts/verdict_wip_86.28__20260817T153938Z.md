STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.28
WRITTEN: 2026-08-17T15:39:38Z

# Q/A write-first record -- step 86.28 (cycle 7, closing evaluation)

## Plan
1. Read qa.md (DONE -- read in full at start).
2. Harness-compliance audit (5 items).
3. Prior-attempt / prior-verdict evidence (qa_wip.py --spawned-at, verdict_history_86_21.py --evidence-only).
4. Immutable criteria read from .claude/masterplan.json (governing text).
5. Deterministic: immutable verification command exit code; git status/diff scope.
6. Re-derive the checker check-count claims; symmetric difference vs baseline.
7. Mutation matrix [7b] re-run + independent adversarial mutants.
8. Grade the nine criteria.

## Findings (appended as established)

### D1. Immutable verification command -- EXIT 0
`bash -c 'source .venv/bin/activate && node scripts/qa/verify_research_gate_workflow.mjs'`
EXIT=0, tail: `ALL GREEN: 124 passed, 0 failed`. Reproduced by me at HEAD 36e42227.

### D2. Prior-attempt / verdict evidence (gathered, not applied)
- `qa_wip.py 86.28 --spawned-at 2026-08-17T15:39:38Z`: source_present=true,
  attempt_number=1, attempt_number_status=ok, prior_attempts=0,
  records_retained=1 (gauge), records_pruned_known=null, prior_records=[].
- `verdict_history_86_21.py --step 86.28 --evidence-only`: status=ok,
  7 verdicts: CONDITIONAL -> CONDITIONAL -> NO_VERDICT -> FAIL -> CONDITIONAL
  -> CONDITIONAL -> NO_VERDICT.
- CROSS-CHECK: prior_attempts (0) is NOT > ledger rows (7) => the ledger is NOT
  stale by the qa.md rule. The DIVERGENCE is the other direction: the WIP sink
  holds 0 prior records for a step with 7 ledger rows, and
  `records_pruned_known: null` means pruning loss is unaccounted. So the WIP
  attempt number for THIS step is an undercount; the ledger is the richer source
  and governs. Reported as-is, no aggregate computed.

### D3. Baseline INDEPENDENTLY reproduced (criterion 1)
Mirrored tree in scratchpad: checker + research-gate.js both at base commit
`089726f9` -> `ALL GREEN: 40 passed, 0 failed`, EXIT=0. The claimed baseline is
real, not asserted.

### D4. Symmetric difference of check NAMES (criterion 1)
- baseline(089726f9)=40 names vs HEAD=124 names: **0 REMOVED**, 84 added.
- 86.28's own last checker commit `d2e987f1` = **97 passed, 0 failed** (matches
  the live_check S9 ladder tail).
- d2e987f1(97) vs HEAD(124): **0 REMOVED**, 27 added => the 97->124 growth is
  other steps, independently derived by me.
- FINDING (NOTE): live_check "Closing re-capture" says "The growth 92 -> 124 is
  OTHER steps' additions". True span for other steps is **97 -> 124 (27)**; the
  92 -> 97 (5 checks) was 86.28's OWN cycle-7 fix, documented in S9 directly
  above that sentence. Attribution imprecision, not a criterion miss: 124 > 40,
  0 removed, total stated.
### D5. "No pre-existing check WEAKENED" -- the semantic half (criterion 1)
- Of the 29 baseline checks with LITERAL names, **29/29 exist at HEAD and 29/29
  have byte-identical assertion bodies** (the single "body changed" hit was my
  extractor spilling into a following comment; the assertion
  `!/Monitor\(/.test(src)` is identical in both revisions).
- The baseline `[7]` mutant TABLE (6 cells) is fully retained at HEAD, with 3
  cells ADDED (tier_unsupported / recency corroboration / urls corroboration).
- NOTE: running the BASELINE 40-check suite against TODAY's research-gate.js
  goes red (32/8) -- but `[1] control passes` is itself among the failures, so
  that run indicts the PROBE, not the product: 86.28 deliberately made the
  fixture brief compliant (recency section + snippet table), which the old
  fixture lacked. Not a finding.

### D6. MY OWN mutation matrix -- production source (control GREEN 124/0 first)
All 7 KILLED, each with the killing guard named:
Q1 recency corroboration vacuous -> KILLED (122/2)
Q2 urls over-claim corroboration vacuous -> KILLED (122/2)
Q3 UNSUPPORTED tier no longer refuses -> KILLED (115/7)
Q4 ABSENT misclassified as UNSUPPORTED -> KILLED (120/4)
Q5 main return hardcodes tier_supported:true -> KILLED (119/4)
Q6 main return hardcodes tier_applied moderate -> KILLED (121/2)
Q7 'deep' ADDED to VALID_TIERS -> KILLED (116/8)

### D7. MY OWN fixture/harness mutants (qa.md 4c evaluator duty)
H1 fixture always emits the recency section -> suite RED (123/1)
H2 snippet URLs 17->100 -> suite RED (123/1)
H3 spawn RECORDER blinded -> suite RED (119/5, known-positive fires first)
H4 verifyBrief hardcodes recency_section_present:true -> suite RED (123/1)
=> the fixtures and the recorder are LOAD-BEARING, not decorative. verifyBrief
derives recency_section_present and distinct_urls_in_brief FROM THE FILE.

### D8. B1 vacuity probe (a mutant that cannot build would score as a kill)
Reconstructed the checker's B1 mutant myself: it BUILDS and spawns **2** agents
(not the -2 build-failure sentinel). The `b1Spawns !== 0` assertion is a genuine
behavioural kill here.

### D9. Criterion 4 fail-closed, 7 shapes driven directly through enforceGate
verification = undefined / null / string / array / number / true -> gate_passed
false with "brief verification did not run"; {} -> fails closed on
brief-not-found. Fail-closed preserved unchanged.

### D10. Criterion 9 -- the LIVE run, verified against the run record itself
`wf_23d9ed4b-22c.json` (status completed, 2026-08-10T06:49:16Z) result matches
the live_check quote. I then drove TODAY's research-gate.js with tier:'deep':
**0 spawns**, and the return is FIELD-IDENTICAL to the 2026-08-10 live result on
all 10 compared keys. `wf_60de95f7-5dc` is the step's own gate (gate_passed
true) but PRE-change -- its checks[] lack the new corroboration labels, matching
the artifact's own disclosure that the full stage-1+2 path was not re-run
post-change.
- FINDING (NOTE): live_check S5's sentence "HEAD is byte-identical to it" no
  longer reproduces -- research-gate.js went 40,582 -> 65,098 bytes across 11
  commits by OTHER steps since 86.28's freeze (294a9a09). The SUBSTANCE holds
  (measured above), but the sentence is a carried-forward residual gone stale.

### D11. Scope -- no unintended production change
86.28's committed file set across its 6 commits: research-gate.js,
verify_research_gate_workflow.mjs, researcher.md, CLAUDE.md, its own handoff
artifacts, verdict_ledger.jsonl. Nothing else. Working-tree modifications
(sovereign_api.py, 5 frontend files, audit jsonl) belong to other in-flight
work; ruff F821/F401/F811 on the one changed .py = exit 0, "All checks passed!".
OBSERVATION for Main: the auto-commit hook does `git add -A`, so flipping 86.28
to done would sweep those unrelated files into a commit titled 86.28.

- FINDING (NOTE): `experiment_results_86.28.md` still states "Current: **92
  passed, 0 failed**. Ladder 40 -> 61 -> 64 -> 73 -> 78 -> 92" and "Six evaluate
  cycles" -- stale by the cycle-7 delivery (97) and by today's tree (124). The
  live_check carries the current numbers; the GENERATE artifact was not
  regenerated for cycle 7.

### D12. FAIL-CLOSED-DIRECTION mutants (positive controls exercised)
C1 recency corroboration fires even when the section IS present -> KILLED (107/17)
C2 urls corroboration fires for any count -> KILLED (106/18)
C3 refusal violation drops the tier names -> KILLED (123/1)
C4 enforceGate ABSENT branch removed -> KILLED (121/3)
C5 refusal ALSO pushes empty_or_errored_return -> KILLED (122/2)
C7 fail-closed guard neutered -> KILLED (run ABORTS, exit 1; my first tally read
   "0 red" -- that was a CRASH, not a survivor. Probe indicted itself.)
C9 enforceGate tier_unsupported violation removed -> KILLED (120/4)
F2 ABSENT label branch moved above unsupported -> KILLED (118/6)
F3 empty_or_errored_return violation removed -> KILLED (121/3)
F1a/F1b/F1c FIXTURE tier drift (3 variants) -> ALL CAUGHT (121/3, 122/2, 119/5)
   incl. the "BRANCH-STEERING fields" fidelity check I had flagged as unreached.

### D13. TOTAL independent matrix: 22 mutants applied, ZERO survivors
Control re-verified GREEN (124/0) before every batch. Coverage census over the
57 checks 86.28 added at its own close: 35 reached by my open-direction matrix
alone; the fail-closed-direction batch + F-series reached most of the rest; the
residual not-reached set is dominated by the mutation APPARATUS cells
(`anchor is UNIQUE`, `mutant X is KILLED`), which are self-demonstrating.

### D14. Criteria 3 / 6 / 7 / 8
C3: VALID_TIERS = ['simple','moderate','complex']; every 'deep' occurrence is a
comment (asserted green); only TWO `await agent(` sites (stage 1 researcher,
stage 2 Explore verifier) -- no fan-out. Divergence disclosed live_check S7 with
options (a)/(b) for the operator.
C6: `const floors = (opts && opts.floors) || {...}` and the coverage.dry logic
are byte-identical to baseline. Reasons recorded in both artifacts.
C7: researcher.md:93 and CLAUDE.md:280 both say agentType:'researcher';
CLAUDE.md's self-contradiction retracted in place at :291-292; code pins it at
research-gate.js:962; checker asserts it (green). Remaining 'general-purpose'
strings are in qa.md and concern the Q/A role, not this rail.
C8: [8] all green (model opus, no Monitor, 0 static imports, exactly 1 export,
no minimum/minItems, gate_passed not const:true, agentType researcher). NOTE:
the checker never asserted "no internal research-to-re-grade loop" -- not in the
baseline either, so nothing was weakened; grep confirms no such loop exists.

### D15. Harness compliance (5/5 clean)
1. brief 08:38:58 < contract 09:09:12 < first code commit 09:26:39 (all
   2026-08-10). Envelope gate_passed true; the LIVE gate run wf_60de95f7-5dc
   RECOMPUTED it (sources_floor_ok 7>=5, urls_floor_ok 34>=10, recency_scan_ok,
   all 7 claimed sources present in the brief). Recency section at brief:92.
2. contract precedes the generated artifact. 3. experiment_results present.
4. log-last: `grep -c 'phase=86.28' handoff/harness_log.md` = 0; masterplan
   status still `pending`. 5. no verdict-shopping: commit 3315546c (today
   15:39:07Z) added 22 lines to live_check + 7 ledger rows, and the prior spawn
   was a NO_VERDICT drop.

### D16. Hygiene
Secret scan over all 6 of 86.28's commits: 0 hits. `node --check` green on both
files. Immutable command re-run a second time: green, exit 0. HEAD unchanged
throughout the evaluation (36e42227).

### GRADE (all nine MET; five evidence-quality residuals for queueing)
1 MET  2 MET  3 MET  4 MET  5 MET(residual)  6 MET  7 MET  8 MET  9 MET(residual)
R1 experiment_results stale ("Current: 92", "Six evaluate cycles").
R2 live_check closing sentence attributes 92->124 wholly to other steps; true
   other-step span is 97->124.
R3 live_check S5 "HEAD is byte-identical to it" no longer reproduces (40,582 ->
   65,098 bytes over 11 other-step commits); substance re-measured and holds.
R4 ledger backfill omits the cycle-7 drop wf_e03ec2d0-c07 that live_check S9
   documents (its source table omits it too, per CLAUDE.md).
R5 no STANDING mutant cell for the positive-control checks (I demonstrated them
   killable ad hoc via C1-C5/F-series).
Verdict returned: PASS with residuals named for queueing.

COMPLETED: 2026-08-17T15:55:57Z
