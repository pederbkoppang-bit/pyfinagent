STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.37
WRITTEN: 2026-08-17T14:22:11Z

# Q/A write-first record -- step 86.37, cycle 5 re-evaluation

## Plan
1. Harness-compliance audit (5 items)
2. Read masterplan immutable criteria (governing text)
3. Immutable verification command: `node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs`
4. git status / diff scope check
5. Prior-attempt evidence: qa_wip.py + verdict_history_86_21.py --evidence-only
6. Read handoff artifacts (experiment_results_86.37, live_check_86.37, evaluator_critique_86.37, contract_86.37, operator_asks)
7. Independent mutation matrix (guard-vacuity check on the drop/marker/recovery guards)
8. Per-criterion MET/NOT MET

## Findings (appended as established)

### D1. Immutable verification command -- GREEN
`bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'`
EXIT=0, `ALL GREEN: 124 passed, 0 failed`. Run by me, unpiped.

### D2. Prior-attempt / verdict evidence
- `qa_wip.py 86.37 --spawned-at 2026-08-17T14:22:11Z`: source_present=true,
  attempt_number=3 (INCLUSIVE, status ok, is_lower_bound=true), prior_attempts=2,
  records_retained=3 (gauge).
- `verdict_history_86_21.py --step 86.37 --evidence-only`: status=ok,
  `FAIL -> CONDITIONAL -> CONDITIONAL` (3 rows).
- Main's own artifacts call this "cycle 5" (4 prior cycles). Ledger holds 3.
  attempt_number(3) is NOT > ledger count(3), so the staleness rule does not
  fire, but both under-count Main's own claim; attempt_number is a stated
  lower bound. Report as observed; no aggregate derived.

### F1 (BLOCKING-CLASS, verified) -- cycle-5 claim #3 does NOT reproduce:
the "false +3 attribution" was NOT corrected at the site.
- experiment_results_86.37.md, Cycle-5 GENERATE item 3, states verbatim:
  "**The false +3 attribution is corrected at the site** (live_check
  section 6)".
- MEASURED: `git show 936dc97e -- handoff/current/live_check_86.37.md`
  touches ONLY the residual-re-triage paragraph (@@ -116,12 +116,15 @@).
  `git log -- handoff/current/live_check_86.37.md` -> last touch 936dc97e;
  the cycle-5 commit 651e1f78 does NOT touch it; `git status` shows it clean.
- live_check_86.37.md lines 105-107 at HEAD STILL read: "the +3 are
  phase-86.28's cycle-5 additions to the same file (its own artifact derives
  73->78 on its different baseline; both derivations are per-tree and
  reproduce)."
- My OWN independent derivation (symmetric difference of check() titles,
  23270f29 -> HEAD): 92 -> 95, +3/-0, and the three are
  "a SINGLE stochastic drop is RETRIED...", "...and the retried run reports
  NO rail_dropped...", "...and the recovered run PASSES the gate...".
  `git log -- scripts/qa/verify_research_gate_workflow.mjs` attributes them
  to 6b4df8f9 / 8b520f6c (2026-08-14, phase-86.81). The 86.28 commits
  (d2e987f1 and earlier, 2026-08-10) PREDATE the cycle-3 baseline.
  => the correct attribution is phase-86.81; the FALSE one is what is still
  on disk in the MASTERPLAN-NAMED artifact.
- Shape: correction ACCOMPANIES rather than REPLACES, and lives in a
  DIFFERENT file from the one it claims to have corrected -- i.e. the exact
  "remediation by file substitution" class this same step diagnosed and named
  at cycle 3 ("the two absent items ... cycle 2 supplied them in
  experiment_results instead").

### D3. INDEPENDENT hermetic mutation matrix (built by me, mini-repo in os.tmpdir,
### tracked tree never written; control run FIRST)
| cell | parses | exit | outcome | named reds |
|---|---|---|---|---|
| C0 CONTROL (null mutant, relocated copy) | true | 0 | SURVIVED (expected) | 124 passed, 0 failed -- relocation is inert |
| M1 UNWRAP stage-1 try/catch [crit 6a] | true | 1 | KILLED | 16, incl. "a stage-1 DROP does not kill the workflow -- the driver RESOLVES (kills QA-RETHROW)" |
| M2 RESURRECT (post-loop compliant literal) | true | 1 | KILLED | 6 -- but by rail_dropped assertions, NOT gate_passed => MIS-ATTRIBUTED kill, rebuilt as M2b |
| M2b FAITHFUL: `gate_passed: railDropped ? true : ...` at the RETURN [crit 6b] | true | 1 | KILLED | 5, first = "a DROPPED run returns gate_passed === false even with a PERFECT stage-2 verification (kills QA-RESURRECT) -- gate_passed=true" |
| M2c SILENT DROP (rail_dropped: null at return) | true | 1 | KILLED | 7 incl. "rail_dropped is returned as its OWN field" |
| M3 MARKER FAIL-OPEN (INCOMPLETE accepted) | true | 1 | KILLED | 2, both marker-named |
| M4 rail_dropped field renamed away | true | 1 | KILLED | 7 |
| M5 ABSENT-marker fail-open | true | 1 | KILLED | 4 incl. "ABSENT is reported DISTINCTLY from INCOMPLETE" |
| H1 FIXTURE mutant: makeBrief stops emitting brief_status COMPLETE | true | 1 | KILLED | 16 -- the fixture IS load-bearing |
| M9 DELETE born-inert teaching from STAGE-1 prompt | true | 0 | **SURVIVED** | 124/0 green |
| M10 DELETE brief_status instruction from STAGE-2 prompt | true | 0 | **SURVIVED** | 124/0 green |

M9/M10 reproduce the DISCLOSED residual (c), widened by Main at live_check S6.
They are a guard-COVERAGE gap on the prompt teaching, not a product defect: the
teaching is present at HEAD and the marker SEMANTICS are demonstrated (M3/M5/H1).

### D4. Scope / immutability / lint
- masterplan 86.37 block byte-identical HEAD vs working tree; status still
  `pending`; 6 criteria; all 6 verbatim in contract_86.37.md.
- 86.37-attributable commits (936dc97e, 651e1f78) touch 0 .py, 0 backend/,
  0 frontend/. 651e1f78 additionally touches qa-verdict.js + verify_escalation_86_78.mjs
  (step 86.78, disclosed in the spawn prompt); research-gate.js untouched by it.
- Lint gate derived scope: `git diff --name-only HEAD -- '*.py'` =
  backend/api/sovereign_api.py (a PEER session's uncommitted Red Line Monitor
  work, NOT this step). `uvx ruff check --select F821,F401,F811` -> "All checks
  passed!" exit=0.
- Uncommitted `.claude/masterplan.json` diff = a peer session appending NEW step
  86.109 only. No 86.37 criterion amended.
- harness_log: one row `Cycle 1204 ... phase=86.37 result=PARKED`; no PASS/FAIL
  row -> log-last holds.

### D5. Research gate (REUSED, operator-ratified)
- No researcher spawned for 86.37; contract S5 cites research_brief_86.31.md.
- I re-verified the brief: envelope external_sources_read_in_full=12,
  snippet_only=52, urls_collected=64, recency_scan_performed=true,
  gate_passed=true; 66 distinct http(s) URLs on disk >= 64 claimed (no over-claim).
- ASK #1 RULED: operator_asks_2026-08-11.md ASK #1 header carries
  "ANSWERED 2026-08-17 (attended session, AskUserQuestion): 'Ratify the reuse
  (Recommended)'". Recorded by Main; I cannot independently observe the
  AskUserQuestion, but the ruling is an operator decision and is on disk.

### D6. Envelope-placement reconciliation (prompt criterion 5) -- MET
- .claude/rules/research-gate.md:254-288 now teaches WRITE IT EARLY / born
  inert and QUOTES the retired wording: "This section previously read 'Every
  brief ENDS with this envelope' and its example carried no `brief_status`."
- .claude/agents/researcher.md:321 mirrors it: "This section used to say only
  'emit this envelope at the tail of every brief'."
- KNOWN-MEMBER RECALL over *.md/*.js/*.mjs/*.py (excl. node_modules, archive):
  every remaining hit is a QUOTE-TO-CORRECT (researcher.md:321,
  research-gate.md:256, contract_86.37.md:53, evaluator_critique_86.37.md:51).
  The only other hit is research_brief_86.96.md:35 -- a DIFFERENT step's brief
  summary pointing at its own FINAL envelope, not doctrine.

### D7. Retry-seam mutants (phase-86.81 code, newest seam)
| cell | exit | outcome |
|---|---|---|
| R1 `isDrop := true` (retry EVERY stage-1 error 3x) | 0 | **SURVIVED** 124/0 |
| R2 `isDrop := false` (retry never fires) | 1 | KILLED, 3 named reds |
| R3 `STAGE1_MAX_ATTEMPTS 3 -> 1` | 1 | KILLED, 3 named reds |

R1 is a NEW finding neither prior evaluator reported. research-gate.js states the
property in its own comment -- "Retry ONLY the stochastic drop. Any other error --
a real bug, a refusal, an abort -- must surface on the first occurrence, not be
re-run 3x" -- and NO assertion drives a non-drop error while counting spawns. The
positive half (drops ARE retried) is guarded; the NEGATIVE half is not.
NOT a 86.37 criterion miss: it is phase-86.81 code, and it fails SAFE (extra
tokens only; enforceGate still recomputes, so it cannot manufacture a pass).
Queueable.

### D8. Residual-(b) claim INDEPENDENTLY CORROBORATED
Main's re-triage says (b) is CLOSED because checker :534 drives the real driver
asserting `recovered.gate_passed === true`. Verified: :528-537 builds
`dropsOnceThenSucceeds`, drives the REAL driver, and :534 asserts exactly that.
My M1 turned that very assertion red, so it is load-bearing. Main's claim 4
reproduces.

### F2 (queueable) -- live_check S6 heading still says "Cycle-4 re-capture" while
carrying cycle-5 re-triage prose. Cosmetic.

### CRITERION MAPPING (masterplan .claude/masterplan.json, 6 criteria + live_check)
- C1 REPRODUCE FIRST -- **MET**. Method stated by author (drive the whole
  workflow with a throwing stage-1 stub, d3bb1dfb~1 vs shipped). Independently
  reproduced by me via M1: unwrapping the try/catch at HEAD makes the driver
  THROW `agent({schema}): subagent completed without calling StructuredOutput`
  with no return, while the shipped tree RESOLVES a structured object.
- C2 DROPPED run => gate_passed:false ALWAYS -- **MET**. M2b (faithful, at the
  RETURN) KILLED by the named assertion; 4 further error-spelling cells; the
  fixture brief satisfies every floor.
- C3 recovery report, DISTINCT field -- **MET**. M2c + M4 KILLED, incl.
  "rail_dropped is returned as its OWN field, not folded into gate_passed".
- C4 born-inert marker + caller shown checking -- **MET**. M3/M5 KILLED with
  marker-named assertions; H1 fixture mutant KILLED (16 reds) so the fixture is
  load-bearing. Teaching present at HEAD in researcher.md and the stage-1 prompt.
  Guard-coverage gap M9/M10 disclosed + queued by the author; reproduced by me.
- C5 floors + anti-trust UNCHANGED -- **MET**. Immutable command exit 0,
  124 passed / 0 failed, run by me unpiped; blocks [2]/[4]/[6] enforce the
  floors, the over-claim rejection and the recompute-not-trust discipline.
- C6 MUTATION-TESTED (both mandated cells) -- **MET**. M1 (revert try/catch) and
  M2b (drop path -> gate_passed:true), both KILLED with NAMED assertions, in my
  own hermetic matrix with a green control run first.
- live_check field's 5 named items -- **ALL PRESENT** in live_check_86.37.md
  (S1 before/after, S2 dropped-run object verbatim, S3 marker demo, S4 green
  run, S5 mutation output, S6 dated re-capture).

### THE ONE BLOCKING-CLASS FINDING
F1 above: cycle-5 GENERATE claim #3 does not reproduce. This was one of exactly
TWO one-line corrections the cycle-4 evaluator named as needed to clear to PASS
("re-attribute the +3 in live_check section 6 to 6b4df8f9/8b520f6c"); the other
(drop residual (b)) DID land. `grep "86.81\|6b4df8f9\|8b520f6c\|retry\|RETRY"
handoff/current/live_check_86.37.md` -> exit 1, ZERO matches. Commit 936dc97e's
SUBJECT also asserts the correction. Correct content lives in a DIFFERENT
artifact; the named site still carries the false claim, unretracted.

### VERDICT: CONDITIONAL
Not FAIL: no immutable criterion is missed; the product is correct and
genuinely mutation-resistant (9 of my 11 cells killed with named assertions).
Not PASS: a claim made in THIS cycle about THIS cycle's remediation does not
reproduce, in the masterplan-named artifact, in the exact "remediation by file
substitution" shape this step already paid for at cycle 2.
Everything else I found (R1 retry-scope negative half, M9/M10 teaching
survivors, the stale S6 heading) is EVIDENCE-QUALITY ONLY and appropriate to
queue rather than iterate, per the operator's 2026-08-17 directive.

### FINAL RE-CHECK AT RETURN
HEAD 9aa2f64e (unchanged since spawn). live_check_86.37.md md5
9adb565694ecb8df0d0ee246ad94f6c0, mtime Aug 17 16:19:28, clean vs HEAD.
research-gate.js md5 e26dc258bc862beead7f4a336c978480 -- BYTE-IDENTICAL to the
md5 the cycle-4 evaluator recorded, so the PRODUCT did not change between
cycles 4 and 5; only the evidence layer and the operator ruling did.
Immutable command re-run at return: EXIT=0, "ALL GREEN: 124 passed, 0 failed".

COMPLETED: 2026-08-17T14:31:41Z
