STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.37
COMPLETED: 2026-08-10T15:59:14Z
WRITTEN: 2026-08-10T15:48:49Z
CYCLE: 2 (predecessor cycle-1 verdict was FAIL; this file OVERWRITES cycle-1's WIP at 17:42 local)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command; git status; lint; scoped tests
C. Mutation matrix: QA-RETHROW, QA-RESURRECT (faithful form), third evasion
D. Criteria 1-6 MET/NOT MET

## Log (append-only, as established)

### Deterministic
- IMMUTABLE CMD `bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'`
  -> **exit=0, ALL GREEN: 117 passed, 0 failed**. Re-derived assertion count by
  `grep -cE "^  (ok|FAIL) "` = **117** (exact match, so all 7 new assertions RUN).
  All 7 new drop assertions appear in stdout as `ok`.
- git status: no uncommitted production change; only audit/heartbeat jsonl + untracked archive dir + my WIP.
- harness_log grep -F "86.37" = 0 hits -> LOG-LAST satisfied; masterplan status=pending.
- 3rd-CONDITIONAL counter: 0 prior CONDITIONALs for 86.37 (cycle 1 was FAIL).

### Mutation matrix (hermetic mkdtemp mini-repos; repo tree NEVER written)
| cell | parses | result |
|---|---|---|
| CONTROL | yes | ALL GREEN 117/0 (expected) |
| QA-RETHROW (`throw e` last stmt of catch) | yes | **KILLED -- 7 failed**, incl. "a stage-1 DROP does not kill the workflow -- the driver RESOLVES" |
| QA-RESURRECT faithful (compliant literal injected AFTER stage 2, beyond 600 chars; old regex sees it = false) | yes | **KILLED -- 2 failed** |
| T0 QA-RESURRECT one-line-after-catch (Main's described placement; old regex sees it = false) | yes | **KILLED -- 2 failed** |
| T3 DROP-SKIPS-STAGE2 | yes | KILLED -- 2 failed (recovery-report assertions are load-bearing) |
| T4 RESURRECT-VIA-CONST (assign inside catch from a far-away literal; regex-invisible) | yes | KILLED -- 2 failed |
| **T1 UNCONDITIONAL-NULL** (`envelope = null` after the try/catch) | yes | **SURVIVED -- ALL GREEN 117/0** while the gate can NEVER pass |
| **T5 PROMPT-STEP-0B-REMOVED** (stage-1 prompt no longer teaches the born-inert marker) | yes | **SURVIVED -- ALL GREEN 117/0** |

FINDING A (claim audit): Main's artifact states QA-RESURRECT killed with **3 failed**.
I measure **2 failed** under BOTH constructions (mine and Main's described placement).
The KILL reproduces; the COUNT does not.

FINDING B: T1/T5 are both "the guard covers the CONSUMER, not the PRODUCER" --
no driver-level happy-path assertion exists, so any mutation that kills the rail
outright is invisible. Both fail CLOSED. Classify pre-existing vs new next.

### Per-assertion attribution (each of the 7 new assertions has its OWN kill)
- M-DROPPED-FALSE (`dropped: true` -> `false`): 116/1, kills ONLY "rail_dropped.dropped === true"
- M-BLANK-ERROR (`error: ''`): 116/1, kills ONLY "rail_dropped carries the ERROR TEXT"
- T3 kills #5 "STILL carries brief_verification" + #7 "reports the on-disk brief"
- QA-RESURRECT/T0/T4 kill #2 "gate_passed === false" + #6 "names at least one violation"
- QA-RETHROW kills #1 "the driver RESOLVES" (and, as a consequence, all 7)
=> NO vacuous assertion among the 7. Criterion 6 satisfied on both mandated cells.

### THIRD EVASION -- FOUND: E3 SELECTIVE-CATCH  (SURVIVES, ALL GREEN 117/0)
Mutation: `catch (e) { if (!/StructuredOutput/.test(String(e.message))) throw e; ... }`
- parses (node --check exit 0); suite ALL GREEN 117 passed 0 failed.
- Behavioural differential vs baseline, measured by driving both:
    SHIPPED   drop="...StructuredOutput"  -> RESOLVED gate_passed=false
    SHIPPED   drop="max_tokens reached"   -> RESOLVED gate_passed=false
    E3-MUTANT drop="...StructuredOutput"  -> RESOLVED gate_passed=false
    E3-MUTANT drop="max_tokens reached"   -> THREW (workflow destroyed, NO return)
=> the SHIPPED code is CORRECT (unconditional catch, survives every error shape).
   The GUARD is single-shaped: the behavioural test drives exactly ONE error
   string, so a future narrowing of the catch is undetectable. qa.md 4c shape:
   "a guard from the instance is not a guard against the class". WARN, not a
   criterion miss (criterion 6 mandates two cells; both die).

### Classification of the survivors
- T1 RAIL-DEAD (`enforceGate(null, ...)` always): PRE-step checker vs PRE-step
  workflow = ALL GREEN 97/0; POST = ALL GREEN 117/0. **PRE-EXISTING blind spot**,
  not a regression. Root cause: `driveRecording`'s agentStub returns null in BOTH
  checkers, so no driver-level HAPPY-PATH (gate_passed===true end to end) exists.
- T5 PROMPT-STEP-0B-REMOVED: no guard asserts the stage-1 PROMPT still teaches the
  born-inert marker. Fails CLOSED (every gate would fail on ABSENT).

### Criterion 1 -- reproduced INDEPENDENTLY by driving both versions
PRE-FIX  d3bb1dfb~1 -> THREW, NO RETURN VALUE
POST-FIX working tree -> RESOLVED
  {"gate_passed": false,
   "rail_dropped": {"dropped": true, "error": "agent({schema}): subagent completed without calling StructuredOutput"},
   "violations": ["empty_or_errored_return"], "brief_verification_present": true}
Matches Main's artifact exactly. Baseline re-derives: pre-step checker on pre-step
workflow = **97 passed** (so 97 -> 110 -> 117 is honest).

### Criterion-by-criterion
1 MET (drive PRE/POST, above). 2 MET (drop + PERFECT stage-2 verification still
false; enforceGate(null) hard-returns gate_passed:false with no reachable
exception; T0/T4/RESURRECT all die). 3 MET (`rail_dropped` is its own return
field; brief_verification still computed on a drop; T3 proves both assertions
load-bearing). 4 MET (marker is a HARD gate checked before counts, fail-closed on
omitted/unrecognised values; stage-2 reads it from the file; 7 marker assertions
green). 5 MET (git diff d3bb1dfb~1..HEAD shows NO change to FLOOR_SOURCES/
FLOOR_URLS/recency/over-claim/agentType/model -- only indentation; over-claim
assertion green). 6 MET (both mandated mutations now die: RETHROW 7 failed,
RESURRECT 2 failed).

### Harness compliance
1. Research gate REUSED not re-run (research_brief_86.31.md, gate_passed:true,
   12 sources, 64 URLs, recency true -- verified in the file). Disclosed in the
   contract s1 and experiment_results s2. DEVIATION from the standing operator
   rule "ALWAYS spawn per step".
2. Contract-before-generate: contract 17:25:58 < researcher.md 17:29:10 <
   research-gate.js (cycle-1 edit) -- OK. Cycle-2 edits (17:45:15/17:45:27/
   17:46:01) POSTDATE the cycle-1 critique 17:44:55 = correct cycle-2 flow.
   All 6 criteria copied VERBATIM into the contract (programmatic diff, 6/6).
3. experiment_results present, with an explicit CYCLE 2 section.
4. LOG-LAST: `grep -F 86.37 handoff/harness_log.md` = 0 hits; masterplan pending.
5. NO verdict-shopping: evidence CHANGED (commit 133060b0, 6 files).
6. 3rd-CONDITIONAL counter = 0 (cycle 1 was FAIL), so CONDITIONAL is permitted.

### FINDING C (the blocker) -- live_check_86.37.md is STALE and INCOMPLETE
mtime 17:34:06 = CYCLE 1; never regenerated. Against its own immutable
`verification.live_check` spec (5 named items):
  1 "before/after behaviour of a stage-1 failure"            ABSENT
  2 "the dropped-run return object VERBATIM"                 ABSENT
  3 "the born-inert marker demonstration"                    present
  4 "the green verify_research_gate_workflow.mjs run"        STALE: says
      "ALL GREEN: 110 passed" / "exit=0 (110 passed, was 97)" -- tree yields 117
  5 "the mutation output"                                    STALE: cycle-1's 5
      cells only; the two cells this cycle is ABOUT are missing
The file's own header claims "Verbatim machine output, regenerated by running the
command shown. 2026-08-10" -- a present-tense claim that no longer reproduces.
Items 1 and 2 are EXACTLY what the cycle-1 Q/A cited against this artifact; cycle
2 supplied them in experiment_results_86.37.md instead = remediation by FILE
SUBSTITUTION. `.claude/hooks/lib/live_check_gate.py:72` is existence-only
(`return "passed" if artifact.exists() else "skip"`), so nothing automated catches
this.

### FINDING D (claim audit) -- one measured count does not reproduce
experiment_results + commit message: "QA-RESURRECT (faithful) KILLED -- 3 failed".
I measure **2 failed** under THREE independent constructions (injected after
stage 2; injected one line after the catch = Main's described placement; assigned
inside the catch from a far-away const). The KILL reproduces and the named
assertion is the right one; the COUNT does not.

### Lens verdicts (worst-of-N)
correctness = PASS | does-it-reproduce = PASS | scope-honesty = CONDITIONAL
=> verdict = CONDITIONAL.
