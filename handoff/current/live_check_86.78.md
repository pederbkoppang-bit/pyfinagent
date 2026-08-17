# live_check — step 86.78

Verbatim tool output. Re-runnable:

```
node scripts/qa/verify_escalation_86_78.mjs    # 51 checks, exit 0 (cycle-4 refresh; was "37 checks" -- the checker grew across cycles and 86.72's sibling key briefly turned it red via a whole-line literal, repaired to a property assertion)
node scripts/qa/mutation_matrix_86_78.mjs      # 13 cells, exit 0 (cycle-4 refresh; was "10 cells")
```

`.claude/agents/qa.md` is **not modified by this step** — §5.

---

## §0. Control — the immutable command, before and after

```
$ bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'
parses
exit=0        (identical before the change and after)
```

## §1. Criterion 1 — the exposure RE-DERIVED, with the enumeration command stated

### 1a. The current `qa.md` text, quoted from disk (grep-anchored, not from memory)

```
$ grep -n 'MUST state the derived attempt number' .claude/agents/qa.md
702:  **You MUST state the derived attempt number and the prior-verdict

$ grep -n 'return \*\*FAIL\*\* instead of a third' .claude/agents/qa.md
665:  PASS or FAIL**, return **FAIL** instead of a third. Stacking a third

$ grep -n 'recommend operator escalation' .claude/agents/qa.md
685:  budget**: at 5+, say so in `notes` and recommend operator escalation rather
```

### 1b. Spawn-prompt census — THREE CLASSES, counted separately

The command is in the output. **The first pass of this census was WRONG and the
correction is the point of separating the classes** — see §1c.

```
POPULATION RULE: first record (type=user, no isMeta) of every file matching
  ~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/subagents/workflows/wf_*/agent-*.jsonl
  classed Q/A iff that prompt matches /pyfinagent Layer-3 Q\/A evaluator/i
  transcripts scanned: 1352    classed Q/A: 370

CLASS A -- rule merely NAMED (a pointer; NOT a consequence): 365 of 370 prompts (98.6%)
    rule named in a pointer            365

CLASS B -- consequence explicitly STATED: 25 of 370 prompts (6.8%)
    you MUST return FAIL               17
    return FAIL instead of a third     9
    at 5+ ...                          8
    ESCALATION BOUNDARY header         1
    recommend operator escalation      1

CLASS C -- self-derived attempt number DEMANDED: 16 of 370 prompts (4.3%)
    State the derived attempt number   16

CONTROLS: negative 0/370 (must be 0); positive 370/370 (must equal population)
```

### 1c. The false number this census replaces, and why it was false

A first pass reported **"365 of 370 (98.6%) supplied a consequence"**. That is
**false**. All 365 hits came from a single probe matching
`.claude/workflows/qa-verdict.js:92`, which reads:

```
...follow it as your operating instructions -- it is the single source of truth
 for the Q/A role (verification order, ..., the no-auto-PASS clause, the
 3rd-CONDITIONAL auto-FAIL rule, and the no-second-opinion-shopping rule).
```

That is a **pointer that NAMES the rule** in a table-of-contents sentence. It does
not tell the judge what its verdict triggers. Counting it as consequence-framing
conflates a contents listing with the stakes sentence arXiv 2604.15224 actually
manipulates. Corrected figures: **Class A 365 (98.6%) named**, **Class B 25 (6.8%)
stated**, **Class C 16 (4.3%) demanded**.

### 1d. The census STRUCTURALLY UNDERCOUNTS, and that is not a caveat to bury

A prompt census can only see the caller's prompt. **The dominant channel is
`qa.md`, which every spawn reads at runtime** — so the true exposure is
**370 of 370**, not 25. First-party proof obtained today on step 86.79:
its cycle-1 spawn prompt supplied **no** attempt number, **no** prior sequence and
**no** consequence, and the Q/A still opened its notes with *"ATTEMPT NUMBER: 1"*.
**Suppressing the caller's prompt alone is not sufficient.**

---

## §2. Criteria 1/3/4/5 — checker output, verbatim

```

==========================================================================
C1 -- the consequence is GONE from the rail prompt; qa.md still carries it
==========================================================================
  [PASS] rail prompt no longer states the 3rd-CONDITIONAL trigger -- "return FAIL instead of a" occurs 0x
  [PASS] rail prompt no longer states the F1b escalation consequence -- "recommend operator" occurs 0x
  [PASS] rail prompt no longer states the budget threshold -- "at 5+" occurs 0x
  [PASS] rail prompt no longer states the self-count demand -- "State the derived attempt number" occurs 0x
  [PASS] positive control: the EVIDENCE pointer is still there (not a blanket deletion)
  [PASS] the rail says explicitly that the consequence is withheld ON PURPOSE
  [PASS] qa.md STILL carries the consequence (disclosed, operator-gated, NOT fixed here) -- 3 of 3 probes hit -- this check asserts the residual EXISTS, so it goes red if someone quietly edits qa.md without the operator

==========================================================================
C3 -- the threshold is computed caller-side, from data the judge never sees
==========================================================================
  [PASS] empty history -> 0 consecutive, not armed
  [PASS] 1 prior CONDITIONAL -> not armed -- consecutive=1
  [PASS] 2 prior CONDITIONALs + a third -> ARMED (the loop terminates) -- consecutive=2
  [PASS] a PASS RESETS the run -- consecutive=0
  [PASS] a FAIL RESETS the run
  [PASS] NO_VERDICT is a dropped ATTEMPT: it neither extends nor resets the run -- consecutive=2 -- a dropped spawn is not a verdict
  [PASS] the judge is never given any of this: the function runs AFTER agent() returns
  [PASS] ...and the flag recording that fact is present and false
  [PASS] step 36.17 (C,F,F,C,C) -> a 6th attempt is NOT force-failed -- the attempt-count trigger would have denied the PASS 36.17 earned at attempt 6

==========================================================================
C4 -- verdict semantics UNCHANGED; no path turns a FAIL into a PASS
==========================================================================
  [PASS] a PASS input verdict is never mutated by the threshold, under any sequence
  [PASS] a CONDITIONAL input verdict is never mutated by the threshold, under any sequence
  [PASS] a FAIL input verdict is never mutated by the threshold, under any sequence
  [PASS] a NO_VERDICT input verdict is never mutated by the threshold, under any sequence
  [PASS] would_auto_fail can only ARM on a CONDITIONAL -- never on a FAIL
  [PASS] ...and never on a PASS either (arming a PASS would be a downgrade path)
  [PASS] the returned object carries NO writable verdict field of its own
  [PASS] a dropped rail return is passed through unchanged (NO VERDICT, never PASS)

==========================================================================
C4b -- an uncomputable sequence yields null, NEVER 0
==========================================================================
  [PASS] sequence not supplied -> null, not 0 -- status=not_supplied
  [PASS] unparseable sequence -> null, not 0 -- status=unparseable
  [PASS] a non-array -> unusable, null, not 0
  [PASS] a spurious 0 would falsely report "no consecutive run" -- assert it is absent

==========================================================================
C5 -- the two law-of-the-case safeguards
==========================================================================
  [PASS] safeguard 1: the BURDEN is named, and sits on the departing party -- the party departing from the computed escalation
  [PASS] safeguard 2: an override SLOT exists on the caller side
  [PASS] ...and it defaults to null -- an override must be recorded, never implied
  [PASS] the JUDGE cannot record one: VERDICT_SCHEMA is additionalProperties:false
  [PASS] the input is echoed back, so what the caller supplied is auditable

==========================================================================
C5b -- the attempt budget is also caller-side, and also fails closed
==========================================================================
  [PASS] no attempt number supplied -> budget_exhausted is null, not false
  [PASS] attempt 4 of 5 -> not exhausted
  [PASS] attempt 5 of 5 -> exhausted
  [PASS] exhaustion does NOT touch the verdict (it escalates, it never passes)

==========================================================================
RESULT
==========================================================================
  checks run : 37   (cardinality floor 30)
  failed     : 0

  ALL CHECKS PASS
```

## §3. Criterion 6 — mutation matrix, verbatim

```
subject : .claude/workflows/qa-verdict.js  sha256[:16]=26124f817e6d9bb7
checker : scripts/qa/verify_escalation_86_78.mjs

[CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (37 checks)

==========================================================================
MUTATION MATRIX
==========================================================================
  KILLED            M1-THRESHOLD-OFF-BY-ONE            by: 2 prior CONDITIONALs + a third -> ARMED (the loop terminates)
  KILLED            M2-NO-RESET-ON-PASS                by: a PASS RESETS the run
  KILLED            M3-FAIL-OPEN-WITH-ZERO             by: sequence not supplied -> null, not 0
  KILLED            M4-ARM-ON-ANY-VERDICT              by: ...and never on a PASS either (arming a PASS would be a downgrade path)
  KILLED            M5-NO_VERDICT-RESETS-THE-RUN       by: NO_VERDICT is a dropped ATTEMPT: it neither extends nor resets the run
  KILLED            M6-BUDGET-FAILS-OPEN               by: no attempt number supplied -> budget_exhausted is null, not false
  KILLED            M7-BURDEN-SAFEGUARD-REMOVED        by: safeguard 1: the BURDEN is named, and sits on the departing party
  KILLED            M8-OVERRIDE-DEFAULTS-TO-APPLIED    by: ...and it defaults to null -- an override must be recorded, never implied
  KILLED            M9-CONSEQUENCE-RESTORED-TO-THE-PROMPT by: rail prompt no longer states the 3rd-CONDITIONAL trigger
  KILLED            M10-INPUT-NOT-ECHOED               by: the input is echoed back, so what the caller supplied is auditable

  subject sha256[:16] before=26124f817e6d9bb7 after=26124f817e6d9bb7 -> tracked file UNCHANGED
  cells: 10   killed: 10   survived/unearned: 0

  ALL CELLS KILLED
```

---

## §4. Two defects in MY OWN checker, found by the matrix and fixed

Recorded because both would have produced a false green, and both are repeat
instances of lessons already on the board.

**(1) `M9-CONSEQUENCE-RESTORED-TO-THE-PROMPT` scored `SURVIVED` on the first run.**
The mutation *had* been applied; the checker simply was not reading it. `SRC` was
`fs.readFileSync(WORKFLOW)` — the **tracked** file — while only the imported function
came from the override. Every source-text assertion was therefore blind to a mutant.
This is the same defect fixed hours earlier in `verify_counter_86_79.py` and
reintroduced here. Fixed: `SRC` now reads the subject under test.

**(2) `M8-OVERRIDE-DEFAULTS-TO-APPLIED` scored `RED-WRONG-REASON`** — it had been
killed correctly, but the matrix's `[FAIL]`-line regex stripped a ` -- ` suffix, and
that assertion's **label itself contains ` -- `**, so the label was truncated and never
matched. Fixed: capture the whole line, match by **prefix**.

Both were caught only because the matrix refuses to score a kill it did not earn.

## §5. Scope — what this step touched

```
$ git status --porcelain -- .claude/workflows/ scripts/qa/ handoff/current/
 M .claude/workflows/qa-verdict.js
?? handoff/current/contract_86.78.md
?? scripts/qa/mutation_matrix_86_78.mjs
?? scripts/qa/verify_escalation_86_78.mjs

$ git diff --stat -- .claude/agents/qa.md
(no output — zero-line diff)
```

**Cross-step regression.** `qa-verdict.js` was also edited earlier today by step
86.79. That step's gate is still green against the shared file:

```
$ python scripts/qa/verify_counter_86_79.py   -> exit 0
```

## §6. What I could NOT verify — stated plainly

1. **The `qa.md` half is NOT fixed, and it is the larger half.** `:663-668` (the
   trigger), `:684-688` (the escalation consequence) and `:702-704` (*"You MUST state
   the derived attempt number"*) are untouched and operator-gated. The checker asserts
   this residual **still exists**, so it goes red if anyone edits `qa.md` quietly.
2. **No claim is made that verdicts changed.** PMC5557596 (N=30,674, 12 years) found
   removing the cue from a human marker moved the ethnicity gap only **5.67 → 5.30 pp**
   and gender **3.92 → 3.27 pp**, and gaps also narrowed on **oral** exams which cannot
   be anonymised. The criteria here are deliberately **architectural**.
3. **The stakes result is single-lab and under review.** arXiv 2604.15224 has **no
   independent replication** of the stakes finding; 2509.26072v2 corroborates only the
   *unacknowledged* half, with different cues. Strong directional evidence, not settled.
4. **The caller supplies the sequence, and the caller is the constrained party.** The
   Workflow runtime has no filesystem access, so `enforceEscalation` cannot read the
   ledger itself. Mitigated only by echoing the input back (`sequence_supplied`) so it
   is auditable. This is the 86.21 objection — 35 of 35 ledger rows `recorded_by: main`
   — and it is **disclosed, not solved**.
5. **`enforceEscalation` has no production caller yet other than this workflow's own
   return path.** Nothing downstream consumes `escalation`; wiring a consumer is not in
   this step's scope.
6. **The "override must be RECORDED" safeguard is NOT sourced from the legal
   material.** Both fetched legal sources are silent, and the two that would settle it
   returned 403/301. It is attested only in the clinical analogue (the DMC documents
   its recommendation and rationale to the sponsor). Stated rather than upgraded.

---

## §7. MEASURED MID-EVALUATION: the named-workflow rail runs a SESSION-SNAPSHOTTED script

**This was found by checking rather than assuming, and it changes what "live" means
for every `Workflow({name: ...})` launch.**

The first Q/A spawn for this step was launched as `Workflow({name: 'qa-verdict'})`.
Before reading its verdict, the launched copy was inspected — and it did **not**
contain this step's change:

```
$ S=.../workflows/scripts/qa-verdict-wf_0471dd22-909.js
$ grep -c 'enforceEscalation' "$S"                                        -> 0
$ grep -c 'THE CONSEQUENCE OF YOUR VERDICT IS DELIBERATELY NOT STATED HERE' "$S"  -> 0
$ grep -c 'return FAIL instead of a' "$S"                                 -> 1
$ grep -c 'recommend operator' "$S"                                       -> 1
$ grep -c 'State the derived attempt number' "$S"                         -> 1
```

That run was **stopped** rather than allowed to produce a verdict on a rail that does
not exist on disk.

**The scope is every named launch this session, not just this one.** Enumerated over
all three `qa-verdict` scripts the session persisted, oldest first:

| launched script | carries 86.79's fix? | carries 86.78's fix? |
|---|---|---|
| `qa-verdict-wf_61338c26-b90.js` (86.79 cycle 1) | **no** | no |
| `qa-verdict-wf_44776e5d-ca3.js` (86.79 cycle 2) | **no** | no |
| `qa-verdict-wf_0471dd22-909.js` (86.78, stopped) | **no** | **no** |
| `.claude/workflows/qa-verdict.js` **on disk** | **yes (1)** | **yes (2)** |

**Conclusion: no edit made to `qa-verdict.js` today has ever been live on the
`name:`-dispatched rail.** The registration snapshots at session start, exactly as the
Agent-tool roster does. `scriptPath` does **not** snapshot — relaunching with
`scriptPath: .claude/workflows/qa-verdict.js` reports that exact path as the script
file, and that is how the graded run was obtained.

### What this does and does not invalidate

- **It does NOT invalidate 86.79's two verdicts.** Both Q/As verified that step's
  `qa-verdict.js` change by **reading the file from disk** (grep), which is why the
  cycle-2 Q/A could correctly report *"its only two surviving `records_retained`
  mentions (:156, :158) now say it is NOT the attempt number."* Their findings concern
  file content, not the prompt they were handed.
- **It DOES bound this step's live-test claim.** Criterion 3's demonstration is valid
  only for a run launched by `scriptPath`. Until a session restart, a
  `Workflow({name: 'qa-verdict'})` launch still hands the judge the **old prompt with
  the consequence in it**.
- **`CLAUDE.md` is incomplete on this point.** It says the Workflow launch *"has the
  Q/A read `qa.md` from disk at runtime, so a `qa.md` edit is live immediately on this
  path; only the Agent-tool roster snapshots at session start."* True for `qa.md` — the
  agent reads it with a tool call — but **the workflow SCRIPT itself is snapshotted**,
  and the sentence reads as though nothing on this path is. Queued as a defect rather
  than edited here, since `CLAUDE.md` is outside this step's scope.

**This is the "committed is NOT in force" class**: the commit is real, the file on disk
is correct, and the running system was still using the old copy.

---

# CYCLE 2 — the two mutants the cycle-1 Q/A found surviving

Its own battery of 7 cells KILLED 4 and left 3. Two were WARN findings
(QA-F, QA-C) and one a note (QA-D). **All three are now closed and pinned.**

| its cell | what survived | fix | now |
|---|---|---|---|
| **QA-F** | `{...verdict, escalation}` -> `{...verdict, ...escalation}` passed all 37 checks. "Alongside, never merged" was asserted in prose and **nowhere guarded** | a **runtime throw** in the shipped code if any escalation key surfaces top-level, PLUS a checker assertion that detects the spread itself | cell **M11** KILLED |
| **QA-C** | a **REWORDED** consequence tripped none of the four literal probes | the withheld-on-purpose block is **content-pinned** (normalised length) and the region between the criteria sentence and that block must be **empty** | cell **M12** KILLED |
| **QA-D** | `verdict_unmodified: true` was a hardcoded attestation that would read true even if the verdict HAD changed | now **COMPUTED** from the keys | cell **M13** KILLED |

**My first fix for QA-F did not work, and the matrix caught it.** The check asserted
only that the runtime `leaked` throw EXISTS in the source — which the mutation leaves
untouched — so M11 **SURVIVED** again. *A check that a guard exists is not a check that
the property holds.* Replaced with a check that detects the spread.

**Two of my new probes were miscalibrated and went red against a CORRECT subject:**
a guessed `EXPECTED_LEN = 1180` (measured: 886) and a gap anchored at the START of
the criteria line instead of its end. Both indicted the probe, not the code — verified
before changing anything.

## §8. Cycle-2 checker output (verbatim)

```
  checks run : 43   (cardinality floor 41)
  failed     : 0
  ALL CHECKS PASS

subject : .claude/workflows/qa-verdict.js  sha256[:16]=7d2942310c695762
checker : scripts/qa/verify_escalation_86_78.mjs

[CONTROL] unmutated checker -> exit 0
  ok -- GREEN control established (43 checks)

==========================================================================
MUTATION MATRIX
==========================================================================
  KILLED            M1-THRESHOLD-OFF-BY-ONE            by: 2 prior CONDITIONALs + a third -> ARMED (the loop terminates)
  KILLED            M2-NO-RESET-ON-PASS                by: a PASS RESETS the run
  KILLED            M3-FAIL-OPEN-WITH-ZERO             by: sequence not supplied -> null, not 0
  KILLED            M4-ARM-ON-ANY-VERDICT              by: ...and never on a PASS either (arming a PASS would be a downgrade path)
  KILLED            M5-NO_VERDICT-RESETS-THE-RUN       by: NO_VERDICT is a dropped ATTEMPT: it neither extends nor resets the run
  KILLED            M6-BUDGET-FAILS-OPEN               by: no attempt number supplied -> budget_exhausted is null, not false
  KILLED            M7-BURDEN-SAFEGUARD-REMOVED        by: safeguard 1: the BURDEN is named, and sits on the departing party
  KILLED            M8-OVERRIDE-DEFAULTS-TO-APPLIED    by: ...and it defaults to null -- an override must be recorded, never implied
  KILLED            M9-CONSEQUENCE-RESTORED-TO-THE-PROMPT by: rail prompt no longer states the 3rd-CONDITIONAL trigger
  KILLED            M11-MERGE-ESCALATION-INTO-THE-VERDICT by: escalation is NESTED in the return, not spread into it
  KILLED            M12-REWORDED-CONSEQUENCE           by: nothing sits between the criteria sentence and the withheld-on-purpose block
  KILLED            M13-VERDICT-UNMODIFIED-HARDCODED   by: verdict_unmodified is COMPUTED, not a hardcoded attestation
  KILLED            M10-INPUT-NOT-ECHOED               by: the input is echoed back, so what the caller supplied is auditable

  subject sha256[:16] before=7d2942310c695762 after=7d2942310c695762 -> tracked file UNCHANGED
  cells: 13   killed: 13   survived/unearned: 0

  ALL CELLS KILLED
```

---

# §9. THE SCRUB IS INERT THIS SESSION — measured, and it invalidates a documented claim

**Found by the cycle-3 Q/A run that DROPPED**, preserved only because write-first made it
write its findings before it died, and then **confirmed independently** by a fresh probe.

## The measurement

A minimal `agentType: 'qa'` subagent was spawned and asked, without letting it read any
file, which strings were in **its own system prompt**:

| marker | in qa.md at session start (pre-fix) | on disk NOW (post-fix) | in the fresh Q/A's SYSTEM PROMPT |
|---|---|---|---|
| `return **FAIL** instead of a third` | yes | **0** | **yes** |
| `recommend operator escalation` | yes | **0** | **yes** |
| `You MUST state the derived attempt number` | yes | **0** | **yes** |
| `3rd-CONDITIONAL auto-FAIL` | yes | **0** | **yes** |
| `and it is authoritative` | yes | **0** | **yes** |
| `--evidence-only` | no | 4 | **no** |
| `attempt_number_is_lower_bound` | no | 1 | **no** |
| `--spawned-at` | no | 2 | **no** |

**5 of 5 pre-fix markers present; 0 of 3 post-fix markers present.** The judge's system
prompt IS `qa.md` as it stood at session start.

## Why this defeats the fix, specifically

`.claude/workflows/qa-verdict.js` launches with **`agentType: 'qa'`** — it *is* the
Agent-tool `qa` type, and that type's definition is snapshotted at session start. The
prompt's STEP 0 makes the judge *read* `qa.md` at runtime, which is why the rail was
believed live. But:

> **A runtime read can only ADD text. It cannot RETRACT text already in the system
> prompt.** So an ADDITIVE qa.md change is live immediately on this rail, and a
> **DELETION-based** change — which is exactly what phase-86.78 is — is **INERT until
> the session restarts.**

The judge therefore still received, as binding system-level instruction: the
**consequence** ("return FAIL instead of a third"), the **threshold** (3 consecutive; 5
attempts), the **unit**, and an **order to compute its own position**.

## The documented claim this falsifies

`.claude/workflows/qa-verdict.js:13-16` and `CLAUDE.md` both state that the Workflow
launch *"has the Q/A read `qa.md` from disk at runtime, so a `qa.md` edit is live
immediately on this path; only the Agent-tool roster snapshots at session start."*

**That is false as written**, and falsified by its own `agentType: 'qa'`. It is true only
for ADDITIONS. This is the second snapshot defect found today — §7 is the same class one
level up (the workflow SCRIPT snapshots on a `name:` launch).

## What it means for this step's status

**Criterion 3 is architecturally complete and NOT YET IN FORCE.** Every artifact on disk
is correct; the running session is still serving the old instructions. This is the
`committed is NOT in force` class in its sharpest form yet — the commit is real, the
file is right, the probe agrees, and the judge is still being told the consequence.

**It cannot be closed from inside this session.** The remedy is a session restart, after
which `scripts/qa/verify_qa_roster_live.sh` (which exists for exactly this, and quotes
Anthropic's *"Subagents are loaded at session start… restart your session to load it"*)
should be run before the next Q/A is trusted on this step.


---

## 10. Cycle-3 census (2026-08-17): the exposure re-derived at execution time, two-level

**Level 1 -- the standing RULE text (not the exposure).** Scanning every
Workflow agent transcript's FIRST user message across all session dirs of this
project (enumeration command in the fenced block below): 413 of 413 qa-verdict
spawn prompts contain "3rd-CONDITIONAL"/"auto-FAIL" -- because the prompt
embeds qa.md itself. A judge KNOWING the rule is the design; this is not the
bias channel and the first census version that counted it (413/413) is
disclosed as the overcount it was.

**Level 2 -- caller-supplied POSITIONAL claims (the exposure).** Slicing each
prompt to its caller-controlled regions (EVIDENCE / ADDITIONAL CONTEXT) and
matching positional language (attempt N of M, FIFTH counted, "rail binds
this", "next launch will be denied", ...):

```
qa-verdict spawn prompts: 413
prompts whose CALLER-controlled sections carry POSITIONAL consequence claims: 8
  2026-08-11  wf_1a848968-750  ['Attempt 3 of 5']
  2026-08-11  wf_20a27baa-df8  ['ATTEMPT 5 OF 5']
  2026-08-11  wf_cd7339e2-5d9  ['Attempt 3 of 5']
  2026-08-11  wf_db40da8a-9db  ['Attempt 2 of 5']
  2026-08-17  wf_4e01adc8-9d8  ['FIFTH counted', 'if this cycle does not close', 'next launch will be denied']
  2026-08-17  wf_6821f477-9d7  ['FIFTH counted', "This launch is the step's", 'if this cycle does not close']
  2026-08-17  wf_86449fa1-bad  ['attempt 1 of 3']
  2026-08-17  wf_c5326358-53c  ['Attempt 2 of 3']
```

Population rule: every `agent-*.jsonl` under
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/subagents/workflows/wf_*/`;
a prompt is a qa-verdict spawn iff its first user message contains
"IMMUTABLE SUCCESS CRITERIA". The census script is reproduced in
`scratchpad/census_8678_v2.txt`'s generator (12-line Python; SLICE regex
`EVIDENCE.*?(?=IMMUTABLE|$)|ADDITIONAL CONTEXT.*`, POS regex as listed).

**The uncomfortable rows are MINE and are the point:** wf_4e01adc8 (86.71
cycle 5, returned PASS) and wf_6821f477 (86.85 cycle 12, returned
CONDITIONAL) are Main's own 2026-08-17 spawns -- the caller that relocated
the counter still leaked position in its extra.note. Verdicts of all 8:
2 PASS (wf_20a27baa 2026-08-11; wf_4e01adc8 = the 86.71 close), 1 FAIL,
4 CONDITIONAL, 1 no-result. Criterion-2 consequence: both PASSes are queued
for a re-grade DECISION in step 86.93 (the reconstructed-evidence re-grade
queue) rather than quietly kept; the 86.71 PASS's mitigation is that its
substance was independently EXECUTED by the evaluator (12-process drive,
4,368-cell sweep, its own mutants), so the framing disclosed stakes it did
not need to borrow.

**The recorder gap this measured:** `qa-verdict.js:580` hardcodes
`judge_was_told_consequence: false` with a comment delegating to caller
self-declaration -- and no caller has ever declared it (all 8 leaky spawns
recorded nothing). The cycle-3 fix computes it AT RENDER TIME from the same
caller-controlled strings this census sliced, converting self-declaration
into measurement at the seam.


---

## 11. Cycle-4 captures (2026-08-17; exits unpiped)

```
$ node scripts/qa/verify_escalation_86_78.mjs > /tmp/ve78.txt 2>&1; echo VERIFY_EXIT=$?
VERIFY_EXIT=0
$ tail -2 /tmp/ve78.txt

  ALL CHECKS PASS
$ grep -c "GREEN control established (51 checks)" /tmp/mm78b.txt; node scripts/qa/mutation_matrix_86_78.mjs > /tmp/mm78b.txt 2>&1; echo MATRIX_EXIT=$?
MATRIX_EXIT=0
$ grep -E "M11|cells:" /tmp/mm78b.txt
  KILLED            M11-MERGE-ESCALATION-INTO-THE-VERDICT by: escalation is NESTED in the return, not spread into it
  cells: 13   killed: 13   survived/unearned: 0
```
