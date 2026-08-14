# live_check — step 86.78

Verbatim tool output. Re-runnable:

```
node scripts/qa/verify_escalation_86_78.mjs    # 37 checks, exit 0
node scripts/qa/mutation_matrix_86_78.mjs      # 10 cells, exit 0
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
