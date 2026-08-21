# Experiment Results -- step 90.1

**Step:** 90.1 -- an attempt row cannot tell a graded attempt from a rail drop, and the
token half of the budget has never been able to fire.
**Date:** 2026-08-20. **Contract:** `handoff/current/contract_90.1.md`.
**Research gate:** PASSED (enforced), `wf_db313c3d-b75`,
`handoff/current/research_brief_90.1.md`.

---

## 1. What was built

| File | Change |
|---|---|
| `scripts/harness/attempt_outcomes.py` | **NEW.** Resolves what an attempt PRODUCED and what it COST from the Workflow run record. Backfill CLI, masterplan membership set, lazy per-step resolution. |
| `scripts/harness/attempt_gate.py` | `extract_step_id_claim` / `extract_step_id` split; masterplan membership check; unknown-step-id DENY + its escalation body; reason-named `write_escalation` with the forged-exhaustion fallback REMOVED; `build_state` now passes tokens and prefers the row's own outcome; `--status` reports `verdicts_seen`/`dropped`/`outcome_mix`/`max_tokens`; self-test extended and CONTAINED. |
| `scripts/qa/mutation_matrix_90_1.py` | **NEW.** 11 cells (1 null + 10 mutants), 21 checks, control-green-first. |
| `handoff/audit/attempt_budget_audit.jsonl` | Backfilled: 4 keys added to all 89 attempt rows. Purely additive, proven. |

## 2. Criterion-by-criterion evidence

### Criterion 1 -- outcome + total_tokens, re-runnable backfill, UNKNOWN stated

Verbatim, `python3 scripts/harness/attempt_outcomes.py --backfill`:

```
{
  "attempt_rows": 89,
  "dry_run": false,
  "ledger": "/Users/ford/.openclaw/workspace/pyfinagent/handoff/audit/attempt_budget_audit.jsonl",
  "outcome_counts": {
    "CONDITIONAL": 45,
    "FAIL": 10,
    "NO_VERDICT": 18,
    "PASS": 11,
    "UNKNOWN": 5
  },
  "reason_counts": {
    "completed_without_result": 2,
    "graded": 66,
    "no_run_record": 5,
    "not_an_evaluation": 16
  },
  "rows_total": 93,
  "tolerance_s": 30
}

UNKNOWN = 5 (used ONLY where no run record matched; an ambiguous match also resolves UNKNOWN and is never guessed)
```

**UNKNOWN = 5, and all five are the synthetic `999.2` pipetest rows** for which no run
record exists. That is the only reason UNKNOWN is used.

**The criterion says "all 92 existing rows"; the ledger holds 93** (4 of them
`operator_extension`, 89 `attempt`). The count moved between filing and execution because
the gate is live and kept recording -- one of the new rows is **this step's own research
gate**. Extension rows are passed through verbatim and are not attempt rows, so 89 is the
resolvable population. Stated rather than quietly reconciled to 92.

Additive-only, verified against the pre-write `.bak` **independently of the writer's own
assertion**:

```
rows before 93 after 93 | count preserved: True
rows whose ORIGINAL fields changed or lost a key: 0 -- purely additive
keys ADDED: ['outcome', 'outcome_reason', 'run_id', 'total_tokens']
order preserved (ts sequence identical): True
```

A resolved row:

```json
{"ts": "2026-08-18T18:57:26Z", "type": "attempt", "step_id": "74.0",
 "workflow": "research-gate.js", "tool_use_id": "toolu_01Q7rtjSh5PRuN3jzxQHHAFh",
 "session_id": "ad20ebbd-32bf-445b-8501-6734674c33b1", "attempt_number_inclusive": 1,
 "note": "recorded at launch (PreToolUse); outcome unknown at this seam",
 "outcome": "NO_VERDICT", "outcome_reason": "not_an_evaluation",
 "total_tokens": 297590, "run_id": "wf_98c646a4-8b1"}
```

**A defect this measurement found in my own first implementation.** The first classifier
put all 18 no-verdict rows in one bucket, `no_verdict_other`. Breaking that bucket down
showed it was **16 research-gate launches that COMPLETED successfully** (a different rail,
which never had a verdict to give) plus **2 qa-verdict runs that completed returning
nothing**. Calling the 16 "drops" would have overstated the drop rate by 8x on this
ledger. `outcome_reason` now names six distinct classes and the criterion's five-value
`outcome` vocabulary is untouched.

**Zero `structured_output_drop` rows exist in the live ledger.** The 46 drops the brief
measured are in the wider 617-record corpus and predate this gate's 2026-08-17 wiring.
Stated so nobody reads "0 drops" as "drops stopped happening".

### Criterion 2 -- reason-named escalations, pre-existing record byte-identical

Run against the **real files at the real path**, not a fixture. A non-exhaustion denial
(claim `86.118.1`) with all four real exhaustion records in the blast zone:

```
BEFORE                                                             AFTER (identical)
9180bf317cec3aac24c825c5288f2c38514e358c2d29f4bd6f431518508a3e05   escalation_attempt_budget_75.11.4.md
670edd040b83e22603012ea12471469c1d7ac3b6b4e91a58a1e82811f53858bb   escalation_attempt_budget_86.47.md
1d8a53e58131e2a20eff9dcf04f1b816b7c1e053155d0dc967217d052c1779ab   escalation_attempt_budget_86.85.md
6fbbec66810478aaae8bb6d980c72894fb9cc3bf47fcd2cb8bd047aa06879610   escalation_attempt_budget_999.2.md
ALL FOUR BYTE-IDENTICAL
ledger untouched: a denied launch consumed no budget
```

`86.85` is the load-bearing one: it is **hand-authored by the operator** and sits at
exactly the path the old fixed-path + forged-body code would have overwritten.

The denial wrote its own artifact instead:
`handoff/current/escalation_unknown_step_id_86.118.1.md`, kept as evidence.

Exhaustion keeps `reason="attempt_budget"`, so the four existing files keep their exact
names and nothing is orphaned. The `# BUDGET EXHAUSTED` fallback body is **deleted**;
`write_escalation` now raises rather than forging an exhaustion record for a step that is
not exhausted.

### Criterion 3 -- the token ceiling FIRES (decided by running it, not by reading it)

**Decision: ENFORCE.** Shown by execution, both directions:

```
ok  ONE attempt costing 1,200,001 tokens is DENIED on the TOKEN ceiling with 4 of 5 attempts still unused
ok  and one token UNDER the ceiling is still allowed -- so the check discriminates rather than always denying
```

The under-the-line cell is there because a ceiling that denies everything is not a
ceiling. The defect it fixes: `attempt_gate` called `state.record(outcome)` with no
`tokens=`, so `tokens_used` was a constant 0 -- which is why every escalation file on disk
prints `tokens used : 0 / 1,200,000` verbatim.

**Operational consequence, measured on the live ledger BEFORE shipping.** Old logic
re-implemented as a control and compared step by step across all 27 live ids:

```
DECISION CHANGES: NONE -- every live step decides identically
```

Five steps now sum at or above 1.2M (75.11.4 1,500,493; 86.108 1,241,203; 86.59
1,208,831; 86.78 1,207,469; 86.116 1,203,394) and **every one is already denied by the
attempt ceiling or already CLOSED_PASS**. Switching the token ceiling on denies nothing
that is not already denied. It is a bound going forward, not a retroactive one.

**Population, stated because it moves the answer.** The step's own audit_basis says "441
qa-verdict runs" and then quotes "18 steps over 1.2M, max 2,677,199". Both figures are
real but belong to different populations: 18 / 2,677,199 reproduces only on the 540-run
all-workflows superset; restricted to the 441 qa-verdict runs it is **13 / 2,506,619**.
The ledger holds both rails, so the enforced denominator is the all-workflows one.

### Criterion 4 -- a claimed step id must resolve

The four named cells, run against the real module:

```
  86.118       -> ADMITTED  (extract_step_id='86.118')
  86.118.1     -> DENIED    (extract_step_id=None)
  86.1180      -> DENIED    (extract_step_id=None)
  999.99       -> DENIED    (extract_step_id=None)
```

The rule is a split, so the escape hatch survives: **no `step_id` at all** stays allowed
and uncounted (81 of 617 historical launches use it); **a `step_id` that does not resolve**
is a loud DENY.

**Blast radius, measured over all 617 historical launches before choosing this:** 531
resolve, 81 carry none, and only **5 do not resolve** -- `82.3+82.4`,
`PLAYWRIGHT-SUBAGENT-PROBE`, `86.90-PROBE`, and two `86.28-LIVETEST-*`. All five are
one-off probes and all five were already refused by the shape regex. **Zero production
Q/A evaluations are affected.**

**The self-test ids -- and a correction to this step's own filing.** The masterplan notes
say `_self_test` "builds synthetic rows with step_id '9.1' that is not a masterplan id".
**That is false on today's plan: 9.1, 9.2, 9.3, 9.4 and 9.5 are all real masterplan
steps.** So membership validation would have let every one pass **silently**, which is
what the criterion's last clause forbids -- and a leaked self-test row has already raised
a real step's allowance once (`read_ledger`'s docstring records the 9.4 incident). Fix:
an `ATTEMPT_GATE_MASTERPLAN` override points the self-test at a synthetic plan of record
holding its own ids. They are exempt **by construction** and can never touch a real
allowance again.

The researcher proposed an OTel-style *visible overflow bucket* instead of a hard refusal.
Criterion 4 is immutable and demands a loud DENY, so DENY is what shipped; the blast-radius
measurement is why that costs nothing. Recorded, not silently dropped.

### Criterion 5 -- mutation matrix, control GREEN first

```
KILLED 10 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
real tree untouched (md5 before == after): True
```

Control observed GREEN across all 21 checks before any cell ran. The two cells the
criterion names by name:

- **M1** (a NO_VERDICT attempt recorded as a graded outcome) -- **KILLED**
- **M2** (the unresolvable-step-id DENY turned into exit 0) -- **KILLED**

A mutant that does not run scores **ERROR**, never a kill. `N0` is a comment-only null
mutant that must SURVIVE; if it were killed, every other kill in the run would be void.

**Two cells survived the first run, and both were real holes in my checks, not weak
mutants.** M4 (restoring the forged `# BUDGET EXHAUSTED` fallback) survived because every
call site passes an explicit body, so the guard was never executed -- an unexercised guard
is indistinguishable from an absent one. M10 (collapsing the join tolerance to 0) survived
because every drive used an EMPTY run-record dir, so everything resolved UNKNOWN regardless
of tolerance. Both are now driven directly: `drive_forge` reaches the guard, and
`drive_join` plants a run record 900ms off the row's `ts` (inside 30s, outside 0s) with a
deliberately distant `timestamp` so a join on the wrong field cannot find it. Both cells
now KILL.

### Criterion 6 -- verdict semantics unchanged

sha256 of `handoff/verdict_ledger.jsonl` taken around the **whole** cell run:

```
BEFORE: fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
AFTER : fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2
CRITERION 6: verdict ledger BYTE-IDENTICAL across the whole cell run
```

Structurally: every write-capable call site in the three changed files was enumerated. The
only one targeting `VERDICT_LEDGER` is at `attempt_gate.py:577`, proven by AST to lie
inside `_self_test()` (lines 518-741), which rebinds `VERDICT_LEDGER` to a temp path before
writing. In production the verdict ledger is **read only**, via `emit_sequence` at
`attempt_gate.py:208`. No new code path can emit a verdict value.

## 3. Immutable verification command -- verbatim

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
EXIT CODE: 0
```

Self-test: 32 checks, `SELF-TEST PASSED`. Matrix: control green, 10 killed, 0 survivors,
0 errors, null mutant survived, real tree untouched.

## 4. A defect this work introduced and then fixed

The first revision of the extended self-test called `write_escalation` with only `LEDGER`
rebound, and it wrote a real `escalation_unknown_step_id_9.9.md` into production
`handoff/current/`. That is the **same leak class** as the 9.4 extension row the module's
own docstring already records, in a second channel. Fixed by redirecting `ESCALATION_DIR`
too, and the containment is now itself a check ("the self-test wrote every escalation into
its OWN temp dir"). The stray file was deleted. Disclosed because the automated command
would have passed either way.

## 5. Findings queued, not absorbed

- **90.5.** Validating the join by the independent `run_id` key surfaced that of 120
  run_ids shared between `handoff/verdict_ledger.jsonl` and the run records, **7
  disagree**: 2 ledger rows say `NO_VERDICT` where the rail returned a real verdict (86.84
  FAIL, 86.85 CONDITIONAL) and 5 say `FAIL` where the rail returned `CONDITIONAL` (the
  documented 3rd-CONDITIONAL conversion). `outcome` here is the **rail's raw return** and
  is deliberately NOT reconciled with the ledger.
- **90.6.** Confirmed live: this step's own research gate consumed 90.1 attempt **1 of 5**,
  and 16 of the 89 ledger rows are research-gate launches. The row now carries both
  `workflow` and `outcome_reason: not_an_evaluation`, so the discriminator 90.6 needs is
  persistent -- acting on it is 90.6's.
- **90.7.** Membership deliberately accepts both `X` and `phase-X` so this step cannot
  deny a step 90.7 has not yet normalised.
- **87.11.** The four non-functional metrics.

## 6. Scope honesty

- Criterion 1 says "92 rows"; the live ledger has **93** and 89 of them are attempt rows.
  Not reconciled downward -- explained above.
- The join is **measured** unambiguous on today's data (max |delta| 1.007s, 0 ambiguous up
  to 300s) but it is a heuristic, not an identity. An ambiguous match resolves UNKNOWN
  rather than guessing. A future run-record schema carrying the PreToolUse `tool_use_id`
  would make it exact; that is not this step.
- The gate remains **fail-open** by design. Resolution failure leaves tokens at 0, which
  allows more, never less. Membership degrades open if the masterplan is unreadable.
  Neither is a hard guarantee and neither is claimed as one.
- **The Agent-tool path is still ungated** -- unchanged from 86.71 and restated here so it
  is not mistaken for something this step closed.

---

# CYCLE 2 -- remediation of the cycle-1 FAIL

**Cycle-1 verdict:** FAIL. Run `wf_b7fc2eb5-efd`, transcribed verbatim in
`handoff/current/evaluator_critique_90.1.md`. Two BLOCKs and three WARNs. **Every one
was correct.** Both blockers were reproduced independently before any fix was written.

## BLOCK 1 (criterion 4) -- the membership walk denied 10 real pending steps

**Reproduced, and WORSE than reported.** `masterplan_step_ids()` walked only
`phases[].steps[]`. The Q/A found 14 missed ids; an independent recursive census finds
**123 dotted ids missed**, and the critical subset is exactly the Q/A's: **10 steps that
are `pending` AND `harness_required`** -- 38.13 and 46.0 through 46.8. Because a missing
id DENIES, the shipped gate blocked ten real pending steps while its own denial text
asserted they were "not a step in `.claude/masterplan.json`". Fail-CLOSED -- the opposite
of the fail-open discipline the module documents.

**Why cycle 1 could not see it, which is the more important half.** My blast-radius
measurement -- "617 launches, 531 resolve, 5 don't, zero production evaluations
affected" -- was computed with **the same shallow walk**. The control shared the defect
it was supposed to detect, so it could only ever agree. Cycle 1 tested only that BAD ids
are denied (precision) and never that GOOD ids are admitted (**recall**).

**Fix.** `masterplan_step_ids()` now recurses the whole document. Over-inclusion is the
safe direction (it can only ADMIT), which is why it collects every `id` rather than
enumerating the container keys that happen to exist today.

**The check that cannot share the bug:** new `assert_membership_recall()` re-reads the
plan with an **independent** walk and asserts every dotted member is admitted. It derives
its expectation from the file, not from the function under test. Run live:

```
ids now: 1614
recall ok: True | members: 1427 | missing: 0

  38.13        -> ADMITTED        86.118       -> ADMITTED
  46.0         -> ADMITTED        86.118.1     -> DENIED
  46.4         -> ADMITTED        999.99       -> DENIED
  46.8         -> ADMITTED
```

Recall restored; precision unchanged. Driven by the gate's own self-test and by matrix
cells M13 (shallow walk) and the two recall checks.

## BLOCK 2 (criterion 1) -- the backfill was not re-runnable

**Reproduced exactly.** The two halves of my own commit were mutually incompatible: the
gate writes a launch row with the four resolution keys **present and null**, and
`backfill()`'s projection read `outcome: None -> "FAIL"` as an existing field CHANGING and
aborted the entire write. The first launch after the commit -- the cycle-1 Q/A's own row
at 19:27:19Z -- broke `--backfill`, which then exited 1 and printed no counts.

**Why every fixture stayed green:** all of them seeded the PRE-90.1 row shape, which has
no resolution keys at all. A fixture that predates the writer cannot detect the writer.

**Fix.** The rule now distinguishes **settled from unsettled**, not present from absent. A
row is unsettled while `outcome` is null or `UNKNOWN`, and only then may the four
`RESOLUTION_KEYS` be written. Once a row carries a real outcome it is FROZEN and passed
through byte-for-byte. `UNKNOWN` stays writable deliberately: a launch still in flight has
no run record yet, and freezing that answer would make a transient absence permanent.

Re-runnable and idempotent, verbatim:

```
{"already_settled_passed_through": 84, "attempt_rows": 92, "rows_total": 96,
 "outcome_counts": {"CONDITIONAL": 45, "FAIL": 11, "NO_VERDICT": 20, "PASS": 11, "UNKNOWN": 5},
 "reason_counts": {"completed_without_result": 2, "graded": 67, "no_run_record": 5,
                   "not_an_evaluation": 18}}

immediate re-run: exit 0 | attempt_rows 92 | settled passthrough 87 | counts identical
```

New matrix cell **M14** mutates the fix back and is KILLED; new drive
`drive_launch_row_backfill` seeds **the row shape the gate actually writes**, so the two
halves are now tested together rather than separately.

## WARN 1 -- M10 was mislabelled, and the upper bound had no cell

The Q/A is right: M10's description said "**widening** the tolerance" while the code
narrowed 30 -> 0, and it ran the real widening mutant through my own harness to show it
**SURVIVED**. Widening is not equivalent to narrowing: at 86400s the summed tokens
collapse to ~9%, which re-opens the exact inertness criterion 3 exists to close.

Fixed three ways: M10 relabelled to what it does; **M11** added for widening; **M12** added
for the `timestamp`-instead-of-`startTime` revert. All three now KILL. They are killable
because `drive_join` plants a **decoy** run record for the same step two hours away --
at 30s only the 900ms record matches, at 86400s both match, ambiguity resolves UNKNOWN.
Without a second candidate the upper bound was untestable, which is why M11's ancestor
survived.

## WARN 2 -- criterion 5 clause 3 was falsified

Also correct. `subprocess.run` does not raise on a non-zero exit, so a mutant that could
not even parse came back with every check failing and was credited as a KILL.
`run_cell` now `ast.parse`s the mutated source first. Verified with the Q/A's own probes:

```
MXE2 -> ERROR | mutant does not parse (attempt_gate.py:475): invalid syntax -- a build failure is not a kill
MXE1 -> ERROR | anchor appears 0 times in attempt_gate.py, expected 1
criterion 5 clause 3: SATISFIED
```

The path proved itself immediately: M8's anchor had gone stale during the BLOCK-2 rewrite
and scored **ERROR rather than a false kill**. It was then re-anchored.

## WARN 3 -- 90.1 broke a consumer

Correct and my fault. `scripts/qa/mutation_matrix_86_71.py` drives step id `77.7` against
the real plan with no override, so 90.1's membership check turned its control permanently
RED. My "zero production Q/A evaluations affected" claim was scoped to launch history and
**excluded checker fixtures**; the masterplan notes had explicitly said "fix the fixture,
do not weaken the check", and I applied that to the gate's self-test but never swept the
other consumers.

Fixed with the same exemption-by-construction: `ATTEMPT_GATE_MASTERPLAN` pointing at a
synthetic plan containing `77.7`. Verified:

```
$ python3 scripts/qa/mutation_matrix_86_71.py --verify
CONTROL green: all 11 behavioural checks hold (below rc=0 rows=1; at-ceiling rc=2)
EXIT: 0
```

## Cycle-2 verification -- verbatim

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
EXIT CODE: 0

KILLED 14 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
real tree untouched (md5 before == after): True
```

Self-test now 34 checks including the two recall checks. Matrix grew from 11 cells to 15
(M11, M12, M13, M14 added; M8 re-anchored; M10 relabelled) and from 21 checks to 25.

Criterion 6 re-verified across the whole cycle-2 run:
`verdict_ledger.jsonl` sha256 `fcfe56ad…2e3eb2` before and after -- byte-identical.

## What cycle 2 did NOT change

- No criterion was edited, weakened, or reinterpreted.
- No verdict semantics changed. The cycle-1 FAIL stands as recorded; this is a fix, not
  an appeal, and the fresh Q/A grades changed evidence.
- The 92-vs-96 row drift persists and is now larger, for the reason the Q/A itself
  identified: the gate is live and the ledger grows during evaluation. `attempt_rows` is
  92 of 96 total rows.

## Still open, disclosed rather than fixed

- **`--operator-extend` does not validate membership.** I found this myself during cycle 1
  and deliberately did not fix it under the evidence freeze. It means the audited operator
  path can still create an allowance row for an id no plan contains -- which is how the
  historical `999.2` rows exist. Arguably correct (an operator may extend a step before
  filing it) and arguably a hole. Criterion 4 speaks only to `extract_step_id`, so this is
  disclosed, not silently closed.
- **The Agent-tool path remains ungated** -- unchanged from 86.71, restated so it is not
  mistaken for something this step closed.

---

# CYCLE 3 -- remediation of the cycle-2 CONDITIONAL

**Cycle-2 verdict:** CONDITIONAL (`wf_7ab71c1d-843`), transcribed verbatim in the critique.
Both cycle-1 BLOCKs were independently re-derived by the Q/A and confirmed **fixed** --
it re-ran the recall on the real plan (members 1427, missing 0), re-derived the blast
radius with the FIXED walk over 621 run records (535 admitted, 5 denied, all 5 already
shape-refused pre-90.1, **zero new denials**), killed four walk mutants of its own, and
confirmed the backfill re-runnable and idempotent over three real runs. Criteria 2, 3 and
6 were independently driven and MET.

Three WARNs remained. **One is a numbered criterion miss and the other two are guards that
did not guard.** All three are fixed.

## W1 -- criterion 5 clause 3, still falsified (the only numbered criterion miss)

The cycle-2 fix added `ast.parse`, which closes only the **SyntaxError** subset. The Q/A
then executed three mutants that parse cleanly and cannot be **imported** -- a module-scope
`RuntimeError`, a `NameError`, an `ImportError` -- and every one scored KILLED. *Parsing is
not running.* Correct, and the same class the cycle-1 Q/A raised: narrowed, not closed.

**Fix:** `run_cell` now SMOKE-IMPORTS the mutant in a subprocess before any check runs; a
non-zero import scores ERROR. Verified against the Q/A's own counterexamples plus two
controls:

```
MXE3   ERROR     module-scope RuntimeError (parses fine)
MXE4   ERROR     module-scope NameError (parses fine)
MXE5   ERROR     module-scope ImportError (parses fine)
MXE6   ERROR     SyntaxError control -- must ALSO be ERROR
MXE7   SURVIVED  null control -- must SURVIVE, proving the probe is not blanket-ERROR
```

The null control matters: a probe that returned ERROR for everything would "satisfy" the
criterion while destroying the matrix.

**A false-ERROR the probe caused, and its fix.** The first version scored M7 as ERROR --
`AttributeError: 'NoneType' object has no attribute '__dict__'` on `attempt_budget.py`.
That was the probe's bug, not the mutant's: `@dataclass` resolves annotations through the
module object, so the module must be registered in `sys.modules` before `exec_module`. A
probe that reports a false failure is as bad as one that misses a real one.

## W2 -- criteria-erosion: I dropped a finding, and the guard was a tautology

Both halves correct and both my fault. The cycle-1 verdict carried **six**
`violation_details`; my cycle-2 disposition table carried **five**, merging two WARNs into
one row and losing the sixth -- the Circular_Reasoning finding on the self-test containment
check -- with no fix, no queue entry and no disclosure.

And the check itself was a tautology, exactly as reported:
`all(p.parent == ESCALATION_DIR for p in ESCALATION_DIR.iterdir())` is True **by
construction**, since `iterdir()` yields only direct children. It returned True while a
file sat outside the directory, and True vacuously on an empty one. It asserted a proxy,
not the property.

**Fix:** the property is "the REAL `handoff/current/` did not change", so that is what is
asserted now -- a name-set snapshot taken before the redirect, compared after -- plus an
anti-vacuity clause that the temp dir actually received an escalation.

**Red-first proof it now discriminates.** A mutant that redirects the self-test's writes
back to the real directory turns both checks RED:

```
  FAIL  the REAL escalation dir is UNCHANGED by this self-test ...
  FAIL  ...and the temp dir actually RECEIVED an escalation ...
self-test result: FAILED (leak caught)
```

The anti-vacuity clause also went red on first write (I asserted `>= 2` escalations when
the self-test writes exactly one), which is how a non-tautological check behaves.

## W3 -- illusory-guard: M11 defended its own decoy, not the documented threshold

Correct. `drive_join` planted the decoy 7,200,000 ms away, so the cell's true boundary was
7200s rather than the 900s the module docstring documents. The Q/A swept it: **every
tolerance from 1 to 7199 SURVIVED, including 3600** -- which on the real ledger collapses
summed tokens from 20,365,361 to 4,015,375 and turns 71 rows ambiguous.

**Fix:** the decoy moved to exactly **900s**, the documented threshold, and M11 now mutates
to 900 (M11b keeps 86400). Sweep re-run through the same harness:

```
  tol=0       KILLED        tol=900     KILLED
  tol=1       SURVIVED      tol=1800    KILLED
  tol=60      SURVIVED      tol=3600    KILLED
  tol=300     SURVIVED      tol=86400   KILLED
  tol=899     SURVIVED
```

3600 now dies. Tolerances 1-899 survive, and that is correct rather than a residual gap:
they sit **below** the documented ambiguity threshold, so there is no ambiguity for the
guard to catch. The guard defends the property it names.

(Interim state disclosed: the decoy was first moved to 950s and M11 then SURVIVED at 900,
because the decoy sat just outside the mutation it claimed to catch. A guard must be
reachable by the mutation it claims to catch.)

## Cycle-3 verification -- verbatim

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
IMMUTABLE COMMAND EXIT: 0

KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
real tree untouched (md5 before == after): True
criterion 6: verdict ledger byte-identical
```

Matrix: 16 cells (M11b added), 25 checks. Self-test: 36 checks.

## Findings the cycle-2 Q/A raised and RETIRED itself

It nearly filed one more -- that the self-test's flat synthetic plan would not catch the
shallow walk -- and retired it after trying to evade it: the phase object's own
`phase-9 -> 9` id is a dotted member the shallow walk never reaches, so the flat plan
catches it (members 6, missing 1). Recorded because a finding tested and withdrawn is
evidence too.

It also verified the disclosed `--operator-extend` hole is **INERT by execution**: it
created an extension row for `999.99` and the subsequent launch claiming that id was still
DENIED (rc=2). The hole cannot be used to admit a launch. It remains disclosed, not fixed.

---

# CYCLE 4 -- the last fixes, and the loop stopped deliberately

**Cycle-3 verdict:** CONDITIONAL (`wf_07182004-c54`), ONE violated criterion. The Q/A met
**five of six** criteria by its own execution -- including a recall walk over 1350 dotted
plan ids finding **PENDING not-admitted = 0** -- and independently reproduced both of my
cycle-3 fixes.

## The one numbered criterion, relocating for the third time

criterion 5 clause 3, seam by seam: **parse -> import -> RUN.** Cycle 2 added `ast.parse`
(closed SyntaxError). Cycle 3 added a smoke-import (closed module-scope failures). Cycle 3's
Q/A then authored QX1/QX2/QX3 -- mutants that parse AND import cleanly and still cannot run
-- and all three scored KILLED. Failure-*count* gave no signal either: QX3 failed 5 of 25
checks, exactly like the genuine kill M3.

**Fixed with the discriminator the Q/A measured and I re-measured:** a mutant that cannot
run dies with a **name-resolution** exception. Results:

```
  QX1      ERROR     expected must be ERROR  OK      (hook-branch missing import)
  QX2      ERROR     expected must be ERROR  OK      (handle_hook -> handle_hook_v2)
  QX3      ERROR     expected must be ERROR  OK      (same slip in the resolver)
  NULLCTL  SURVIVED  expected must SURVIVE   OK
  M2CTL    KILLED    expected must be KILLED OK
```

**A defect I introduced while fixing it.** The first version scored plain "any traceback" as
ERROR, and that flagged **M14** -- a legitimate cell whose entire purpose is to reintroduce a
bug that raises `AssertionError`. It would have silently deleted a cell from the matrix while
appearing to satisfy the criterion. *An over-eager probe is as bad as a blind one.* Fixed by
typing the exception: a name-resolution error means the code is not there; a domain exception
means the mutant ran and misbehaved. The null control and a real-kill control are drilled on
every run precisely so this cannot recur unnoticed.

## The two residuals the Q/A labelled as residual -- fixed anyway, because both were my claims

**1. "Ambiguity first appears at 900s" was stale, and I had used it to retire a finding.**
Re-measured independently over the live ledger:

```
tol=30/300/385  ambiguous=0  summed_tokens=21,059,736
tol=386         ambiguous=1  summed_tokens=20,782,337   <- first ambiguity
tol=500         ambiguous=2  summed_tokens=20,593,407
tol=899         ambiguous=6  summed_tokens=19,692,711   (-6.5%)
```

So M11 had been surviving the whole **[386, 899]** band. ~~The docstring is corrected, the
decoy moved to 386s, and M11 re-pointed at the **measured** threshold rather than a
documented one.~~

> **CORRECTED IN CYCLE 5 -- THE FIRST CLAUSE OF THAT SENTENCE WAS FALSE WHEN WRITTEN.**
> The decoy WAS moved to 386s and M11 WAS re-pointed (both verified). **The docstring was
> never touched.** `git show --stat a252b025 -- scripts/harness/attempt_outcomes.py` is
> EMPTY; the file's last commit is `1fc7b2e6` (cycle 2), and `attempt_outcomes.py:34-36`
> still read "Ambiguity first appears at 900s" until the cycle-5 correction. The false
> clause then propagated into masterplan step **90.10**'s `audit_basis` and into
> `mutation_matrix_90_1.py:218-228`, which cited the "DOCUMENTED" 900s and a decoy "Moved
> to 950s" while the code planted `386_000`. Found by the cycle-5 Q/A; all four sites are
> now corrected at source. **A sentence is not verified by its true clauses** -- two of
> three were right, which is exactly why the third survived my own review.

My cycle-3 claim that "tolerances 1-899 surviving is correct rather than a
residual gap" was **wrong**, and it was wrong because it borrowed a number instead of
re-deriving it.

**2. One deleted line stood between the self-test and the project's verdict history.** The
Q/A built a structurally-equivalent sandbox, removed the single `VERDICT_LEDGER = Path(td)`
redirect, and the self-test reported **PASSED with zero FAILs while truncating the sandbox's
verdict ledger** to one synthetic PASS row. Fixed: the self-test now refuses to run at all if
`LEDGER`, `VERDICT_LEDGER` or `ESCALATION_DIR` resolve inside the repo. Red-first proof --
deleting the redirect yields rc=1 with the guard line on stderr and the sandbox ledger
preserved byte-for-byte.

## Cycle-4 verification -- verbatim

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
IMMUTABLE COMMAND EXIT: 0

KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
verdict ledger byte-identical
```

## Why no cycle-4 Q/A spawn was made

**Operator instruction, 2026-08-20:** the Q/A harness returns a high rate of CONDITIONALs on
residual observations, and I was told not to spend further cycles on that class. The record
supports it: this one criterion relocated one seam per cycle for three consecutive cycles,
each finding correct, each fix correct, each followed by the next seam. **That is the fixed
point step 90.9 exists to describe, observed live on this very step.**

The step's PRODUCT is verified: both cycle-1 BLOCKs fixed and independently re-derived by the
Q/A, and five of six criteria MET by its own execution. What remains is the EVIDENCE
apparatus. The operator has since authorised a real Q/A once everything is fixed -- that
spawn is theirs to schedule. Attempt budget stands at **4 of 5**, 955,467 of 1,200,000 tokens.

**90.1 is NOT being flipped to `done`.** A step is never closed without a PASS verdict, and
self-evaluation is forbidden. It stays `pending` with this record.


---

# CYCLE 6 -- the blocker that FAILED this step is now CLOSED (no re-grade, no spawn)

**This is the Main-side half of the documented cycle-2 flow: fix the blocker and update the
handoff files. The other half -- spawn a fresh Q/A on the changed evidence -- is NOT done
here, because the attempt budget is exhausted at 5/5 and a sixth launch is denied by the
gate. Nothing below is a verdict, and 90.1 is NOT flipped.**

## What changed

The cycle-5 Q/A FAILED this step on **criterion 5 clause 3** -- *"a mutant that fails to run
scores ERROR and never counts as a kill"* -- because `_drive_traceback` required a literal
`Traceback (most recent call last)` while `attempt_gate.handle_hook`'s fail-open handler
prints one line and no traceback. The three cells it authored (QA1 / QA1b / QA1c, single
call-site renames inside `handle_hook`) scored **KILLED** where the criterion requires
**ERROR**.

**That defect was filed as step 90.12 and has now been fixed and landed.** The discriminator
reads the exception TYPE wherever it appears -- traceback or fail-open one-liner -- and is
renamed `_drive_unresolvable`, because the old name described the mechanism it used rather
than the question it answers.

## Measured, on the current tree

```
  cell   BEFORE (traceback-only)    AFTER (typed)              expected
  QA1    not-ERROR                  ERROR                      ERROR
  QA1b   not-ERROR                  ERROR                      ERROR
  QA1c   not-ERROR                  ERROR                      ERROR
  QX2    ERROR                      ERROR                      ERROR
  UBL    not-ERROR                  ERROR                      ERROR
  DOM    not-ERROR                  not-ERROR                  KILL
  N0     not-ERROR                  not-ERROR                  KILL_NONE
```

The BEFORE column is not a retyping: `verify_error_discriminator_90_12.py` **extracts the
pre-fix discriminator from git** at `d564ad58` and scores each mutant with both
implementations from a **single** observation pass.

- **QA1 / QA1b / QA1c** are the Q/A's own three cells. All three now score ERROR.
- **QX2** (definition rename) scored ERROR before and after -- the pre-fix scan already
  caught that sub-class, which is why the cycle-4 fix looked complete. It stays in the table
  to show the fix closed a *different* sub-class.
- **DOM** raises an `AssertionError` through the *same* fail-open handler and stays a KILL,
  so the probe did not become over-eager -- the failure mode that silently deleted a cell in
  cycle 4.
- **UBL** was found by 90.12's research gate, after the fix: `UnboundLocalError` subclasses
  `NameError` but its printed name does not contain that substring, and the scan matches
  type names as strings. A real blind spot, now closed.

## And this step's own immutable command still passes

```
$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
90.1 IMMUTABLE EXIT: 0
```

## What this does and does not mean

**It does NOT mean 90.1 passes.** No Q/A has graded this evidence. Criterion 5 clause 3 has
relocated four times on this step, and the honest reading of that record is that a fifth
seam is possible -- 90.14's gate found exactly such a fifth seam on the sibling step, one
dimension over, on work I had already called verified.

**It does mean the operator's decision is now different from the one in the escalation.**
When that escalation was written, option A ("extend by one attempt") meant spending a cycle
to fix an open defect and hope. The defect is now fixed and independently instrumented, so
option A now means spending a cycle to *grade* a closed blocker. That is a materially better
bet, and the record should say so rather than leave a stale framing in place.
