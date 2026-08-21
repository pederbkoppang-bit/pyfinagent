# Experiment Results -- step 90.2

**Step:** 90.2 -- route the WARN/NOTE severity the judge already emits, caller-side,
without asking it to classify anything new and without moving the verdict.
**Date:** 2026-08-21. **Contract:** `handoff/current/contract_90.2.md`.
**Research gate:** PASSED (enforced), `wf_05a76fdf-b16`,
`handoff/current/research_brief_90.2.md`.

---

## 1. What was built

| File | Change |
|---|---|
| `.claude/workflows/qa-verdict.js` | **NEW** `enforceSeverityRouting()` + two helpers (`severityTags`, `deriveSeverity`) + two constants, returned as a `severity_routing` SIBLING; the existing leak invariant EXTENDED with a third guard at the same throw-site. **No `VERDICT_SCHEMA` edit.** |
| `scripts/qa/residual_close_gate.mjs` | **NEW.** The consumer half: refuses a parent step's close while a `queue_residual` debt is unfiled or toothless. Fail-CLOSED by design. |
| `scripts/qa/verify_severity_routing_90_2.mjs` | **NEW.** The immutable checker: 66 checks over a floor of 55, a 13-cell mutation matrix with the control observed GREEN first, a behavioural drive of the leak guard, and a `--replay` mode over the real corpus. |
| `scripts/qa/fixtures/severity_routing_90_2_returns.json` | **NEW.** 24 VERBATIM real returns (6 PASS / 6 FAIL / 6 all-WARN CONDITIONAL / 6 mixed CONDITIONAL), copied unedited from run records at a stated pin. |

**Red-first baseline captured before any of it existed:** the immutable command exited
**1** with `MODULE_NOT_FOUND`. It now exits **0**.

## 2. The finding that changed the design, and it was mine to catch

The contract's §4.4 specified a **negation-aware derivation** built on the brief's
measurement that 130 of 183 BLOCK-bearing records carry a negated occurrence. I drafted
exactly that -- a proximity filter scanning the 45 characters before each token for a
negator -- and then measured it before shipping it.

**Over `violated_criteria` at the pinned corpus it moved exactly 6 runs out of
`queue_residual`, and all 6 were false positives.** Every one carried a genuine trailing
tag whose negator belonged to the finding's own prose:

```
wf_6e9d4eb1-5ff  ...relocates file paths but not the HTTP client to :8000 (WARN)
wf_555a4380-3e8  ...the phrase was never contiguous in the pre-fix source [WARN]
wf_71687e5e-c63  ...misroute guard not alias-proofed (illusory-guard #17 WARN — ...
wf_7e817466-c1c  ...FINDING (not a criterion miss), WARN -- Invalid_Precon...
wf_00a7dd53-3f5  ...load-bearing justification claim does not reproduce [WARN]
wf_7fa0e5d6-c50  ...'run_friday_promotion has no caller anywhere' (WARN)
```

**6 of 6.** So I threw the design away and measured what the field actually looks like:

```
TAG-FORM occurrence counts over every violated_criteria entry (pinned corpus):
   initial  41
   bracket  88
   paren    29
   colon    20
   dash      7
bare occurrences matching NO tag form: 12
```

**This table is ILLUSTRATIVE and rule-dependent -- see NOTE N4 in the cycle-2 section.**
It counts each occurrence once, under the first form that matches; the cycle-1 Q/A's
independent tally under a different precedence gave 41/91/37/1/5 with 2 bare. Only
`initial 41` matches across both. Nothing in the shipped code depends on this table; the
load-bearing measurement is the 41/247 replay, which is re-runnable.

Severity in `violated_criteria` is written as a **delimited tag** in ~185 of ~197
occurrences, and inspecting all 12 bare ones shows they are mostly tags with suffixes
(`(BLOCK-for-close)`, `[BLOCKING for PASS]`) plus one identifier
(`WARN_provenance_control_`) and one back-reference (`cycle-1 BLOCK 1`) that are correctly
not tags.

**The brief's kappa is not wrong; it is about a different field.** It measures re-deriving
severity from **whole-record prose**. This function reads only `violated_criteria`, where
the property is **syntactic**. Syntax is decidable; sentiment is not. The shipped matcher
reads a delimited tag, keeps a *narrow* immediate-negator rule (at most one intervening
word, so it cannot reach across a clause), and excludes identifiers via lookarounds.

**The independent corroboration that this is right:** the delimited matcher and the
**filing's own** token-anywhere matcher -- a different rule, authored by someone else at
filing time -- produce **identical run sets**, 41 and 41, zero disagreement in either
direction, at both censuses. A control built from my own walk would have proved nothing;
this one was not.

## 3. Criterion-by-criterion evidence

### Criterion 1 -- a sibling, never merged; the EXISTING invariants extended

`merged = { ...verdict, escalation, research_routing, severity_routing }`. The third guard
sits at the same throw-site as the other two:

```js
const leakedS = Object.keys(severity_routing).filter(k => k !== 'severity_routing' && k in merged)
if (leakedS.length > 0) { throw new Error('phase-90.2 invariant violated: ...') }
```

**The guard is DRIVEN, not grepped -- corrected in cycle 2.** Cycle 1 covered this
criterion with four regexes over the workflow file, and the cycle-1 Q/A showed that was
**sole-coverage vacuity**: it applied two neutering mutants in memory --
`if (false && leakedS.length > 0)`, and deleting the if/throw while leaving the invariant
message behind in a comment -- and **all four checks stayed GREEN**. The guard span is now
lifted into a callable the same way `severityTags` is extracted, and exercised:

```
[PASS] the leak guard is EXTRACTABLE and callable -- a deleted if/throw is caught here, not merely missed by a regex
[PASS] ...it does NOT throw on the correct sibling shape -- no throw
[PASS] ...it DOES throw when the routing object is FLATTENED into the verdict
[PASS] ...and when a JUDGE field collides with a routing key ("route")
[PASS] ...and it does not throw on an empty routing object (no false positive)
```

Both of the Q/A's own neutering mutants are now matrix cells **L1** and **L2**, and both
are **KILLED**. A guard that is only ever matched, never executed, is the illusory-guard
shape: the literal survives while the behaviour is stripped.

**One deliberate difference, and it makes this guard strictly stronger than its sibling.**
The `research_routing` guard carves out `research_needed` / `research_brief_spec`, because
the JUDGE authors those inside the verdict. **Nothing in `severity_routing` is
judge-authored** -- every key is invented by the caller -- so there is no carve-out.
Checked by the checker, which also asserts no routing key collides with any judge field
name (measured: none).

### Criterion 2 -- the FAIL guard is STRUCTURAL, not observational

`route` is `queue_residual` only if `verdict === 'CONDITIONAL'`. PASS and FAIL cannot
reach it.

```
[PASS] a FAIL whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
[PASS] a PASS whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
[PASS] the SAME entries under CONDITIONAL route to queue_residual -- route=queue_residual
```

The third line matters: without it the first two pass vacuously for a function that always
returns `remediate`.

**The fixture is CONSTRUCTED, and I am stating that rather than implying a natural
instance.** I searched: **0 of the 67 FAILs at the pin have every entry WARN/NOTE-tagged** (67 on the derived population; cycle 1 quoted 66 from the narrowed one).
That absence is the whole argument for a structural guard -- "never observed" is not
"cannot happen". Mutant **M1** removes the guard and is KILLED.

### Criterion 3 -- verdict byte-identity over 24 real returns

String equality, never inspection, over a fixture set of **24** real returns spanning all
three verdict values (floor: 20):

```
[PASS] the fixture set holds at least 20 REAL returns -- 24 returns, pinned 2026-08-18T12:33:57.731Z
[PASS] ...spanning all three verdict values -- CONDITIONAL,FAIL,PASS
[PASS] the judge's verdict string is byte-identical after routing, on every return -- 24/24 by string equality
[PASS] ...and the input object itself is never mutated -- 0 mutated
```

The second assertion is not decoration: a function that *replaced* the verdict with an
identical-looking copy would pass the first and fail the second.

### Criterion 4 -- the replay reproduces 41 AND 247 exactly (CORRECTED in cycle 2)

**Cycle 1 got this wrong and the correction REPLACES it.** I filtered the corpus on an
exact `workflowName === 'qa-verdict'` match, published **41 / 244**, and wrote a confident
paragraph explaining why the filed **247** "does not reproduce" -- blaming a 43-of-436
`result: null` gap, which explains the 436 -> 393 **parseable** gap, a different gap
entirely. The scope was **chosen, not derived.** Masterplan 90.2's `audit_basis` names
*"441 `qa-verdict` Workflow run records"*, and **441 is the `startsWith` count**; the exact
match is 436. The 5 excluded records run under variant names
`qa-verdict-writefirst-82-5` (x3) and `-82-7` (x2), **3 of them non-PASS** -- which is
precisely 247 - 3 = 244. Found by the cycle-1 Q/A, which re-derived the whole census
independently.

On the derived population, every filed figure reproduces:

| census | records | parseable | with_verdict | mix | non-PASS | queue_residual | remediate |
|---|---|---|---|---|---|---|---|
| PINNED @ 2026-08-18T12:33:57.731Z | **441** | **398** | **397** | PASS 109 / COND 221 / FAIL 67 | **288** | **41** | **247** |
| LIVE (no pin) | *drifts by construction -- see below* | | | | | | |

**The LIVE row is deliberately not transcribed into this document.** It is regenerated on
every `--replay` run and changes between captures: across three runs in a single session
the record count went **451 -> 452 -> 453** and the routed pair went 41/254 -> 41/255 ->
42/255, because the corpus grows every time a Q/A launches -- including the ones evaluating
this step. Cycle 2 printed one capture here and a later capture in `live_check_90.2.md`,
and a reader comparing the two same-cycle artifacts saw a contradiction that was really
just a clock. The PINNED row is the load-bearing one and is stable; whatever LIVE was at
capture time is in `live_check_90.2.md` §2, printed once, from one run.

The brief's denominators -- 441 / 436 / 398 / 397 -- were right all along. Full table with
every run id in `handoff/current/live_check_90.2.md`.

- **"41" reproduces exactly**, under both matchers, with identical run sets, at both
  censuses.
- **"247" reproduces exactly** on the derived population.
- **"32" does not reproduce** under any of four strict definitions (41 / 26 / 11 / 4),
  measured under **both** populations. The filing's strict definition is not recoverable
  from its text, and the number is not edited to match.
- **"any run mixing a WARN entry with an untagged finding must route to remediate"** --
  asserted directly and enforced by making UNTAGGED force `remediate`; mutant **M2**
  (UNTAGGED silently becomes NOTE) is KILLED.

**Every supporting count in this step is now on the derived population**, because the
narrowing had leaked into them too: `violation_details` rows carrying a judge-emitted
`severity` = **0 of 969** (was quoted as 0 of 978), and all-WARN/NOTE FAILs = **0 of 67**
(was 0 of 66). The substance -- zero in both cases -- is unchanged; the denominators are
corrected at the source rather than in a footnote.

### Criterion 5 -- the consumer that makes `queue_residual` oblige something

`scripts/qa/residual_close_gate.mjs`. Because the criterion does not define "does not
parse" and an undefined predicate is an unfalsifiable one, the module defines it: an
unparseable plan, no step referencing the parent, or a referencing step with no non-empty
`verification.command` **and** `success_criteria` -- each REFUSES. **Fail-closed**, which
is deliberately the opposite of the house's fail-open hook rule: a hook that breaks the
harness must fail open, but a close gate that cannot read its own evidence must not let
the close through.

```
[PASS] queue_residual + NO filed residual -> close REFUSED
[PASS] queue_residual + a residual that grades NOTHING -> close REFUSED -- 90.12
[PASS] queue_residual + a properly filed residual -> close ALLOWED -- filed=90.12
[PASS] remediate + nothing filed -> close ALLOWED (the gate binds only on queue_residual)
[PASS] an unparseable plan REFUSES rather than failing open -- fail-closed
[PASS] a residual filed for 90.10 does NOT satisfy a debt owed by 90.1
[PASS] ...and the parent cannot be its own residual
[PASS] driven against the REAL .claude/masterplan.json, 90.1's residual IS filed -- filed=90.3,90.6,90.8,90.9,90.10,90.11,90.12,90.13 over 1227 steps
```

The last line is driven against the **real plan of record**, not only fixtures. It also
exposes a limitation I am disclosing rather than hiding: the reference test is textual, so
any step whose `audit_basis` or `notes` mentions `90.1` counts. Eight do. The gate
therefore proves "a well-formed residual referencing this parent exists", not "the residual
for *this specific finding* exists". Tightening that needs a finding-id the verdict schema
does not carry -- which is 86.98's, behind an operator gate. Stated, not silently narrowed.

### Criterion 6 -- mutation matrix, control observed GREEN first

11 cells. **Control green before any cell ran**; null mutant SURVIVES; a real-kill control
is KILLED on the same run; a mutant that cannot resolve a name scores ERROR.

```
  ok   N0   SURVIVED  expected SURVIVED   NULL MUTANT (comment only)
  ok   M1   KILLED    expected KILLED     the verdict guard is removed
  ok   M2   KILLED    expected KILLED     an UNTAGGED finding is treated as a NOTE
  ok   M3   KILLED    expected KILLED     a reported finding is silently dropped
  ok   M4   KILLED    expected KILLED     BLOCK stops dominating
  ok   M5   KILLED    expected KILLED     the delimiter requirement is dropped
  ok   M6   KILLED    expected KILLED     absence recorded as a VALUE (disagreed false, not null)
  ok   M7   KILLED    expected KILLED     an EMPTY findings list counts as all-residual
  ok   M8   KILLED    expected KILLED     the immediate-negator check is removed
  ok   M9   KILLED    expected KILLED     the judge-emitted branch is ignored
  ok   QX   ERROR     expected ERROR      ERROR CONTROL: a call site cannot resolve a name
```

**A defect I found in my own first checker, and it is the whole lesson of this step.**
M5 and M8 **SURVIVED** the first run. Both cells looked covered -- I had a check for an
identifier (`WARN_provenance_control_x`) and one for `"no WARN fired on that path"`. Both
passed **vacuously**: the identifier is already excluded by the regex lookarounds, and the
"no WARN fired" entry is already excluded by the *delimiter* rule, so the negator rule was
never the thing doing the work. **A fixture only tests a rule if that rule is the ONLY
thing standing between the input and the outcome.** Fixed with two cells where each rule is
load-bearing (`"no WARN: nothing fired"` -- a tag position, killed only by the negator; and
`"the checker never reaches the WARN branch at all"` -- killed only by the delimiter), plus
assertions that each exclusion changes the **route**, not just the label.

**Disclosed:** the immediate-negator rule fires on **zero** entries in the pinned corpus.
It is retained as a narrow guard against a phrasing that occurs in prose generally, and its
cells are constructed fixtures. An unfired guard earns its place only if you say so.

### Explicitly NOT done (contract §5, held)

- **No `VERDICT_SCHEMA` edit.** Asserted by the checker: the schema block contains no
  `severity`. 86.98 is not pre-empted and its operator gate is not bypassed.
- **No verdict movement.** `handoff/verdict_ledger.jsonl` sha256 byte-identical before and
  after the whole run, captured both times, and the routing object cannot express a verdict
  value (`ok`/`verdict` are absent from it by construction).
- The judge is asked for nothing new; the `judge_emitted` branch is inert today
  (**0 of 969** `violation_details` rows carry a `severity` key).

## 4. Verification, verbatim

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
  checks run: 66 (floor 55)
  failed:     0
IMMUTABLE COMMAND EXIT: 0
```

Full output in `handoff/current/live_check_90.2.md`.


---

# CYCLE 2 -- what the cycle-1 FAIL changed

Verdict `wf_0e5b781a-bf9`: **FAIL**, two blockers, both correct, both reproduced by me
before I touched anything. Criteria 2, 3, 5 and 6 were MET, harness compliance clean, and
the Q/A explicitly retired two of its own candidate findings after a behavioural
differential showed they were equivalent mutants. Full verdict verbatim in
`handoff/current/evaluator_critique_90.2.md`.

**Blocker 1 -- criterion 4's 247 reproduces; my claim that it did not was wrong.** See the
corrected criterion-4 section above. The lesson is narrow and worth keeping: *I derived a
denominator I liked and then explained the discrepancy instead of questioning the
denominator.* The explanation I gave was itself checkable and did not check out -- it
described the parseable gap, not the verdict gap. **An explanation that survives because
nobody tested it is not evidence.**

**Blocker 2 -- the leak guard was source-scanned, never executed.** See the corrected
criterion-1 section above. Cells L1 and L2 are the Q/A's own mutants, now in the matrix.

## The Q/A's five NOTEs, three of which I have acted on

- **N1 -- the close gate is not wired.** DISCLOSED at source: `residual_close_gate.mjs`
  now states plainly that no close path invokes it, that its only caller today is this
  step's own checker, and that wiring it into `auto-commit-and-push.sh` changes what breaks
  when it errors and therefore needs its own step and its own red-first proof.
- **N2 -- `violation_details` content is never scored.** DISCLOSED at source in
  `qa-verdict.js`: the routing reads `violated_criteria` only, plus a judge-emitted
  `severity` key. 3 of the 41 carry detail rows with no matching tagged entry and all three
  read "SEVERITY NOTE", so there is no live counterexample -- the bound is real and is now
  stated rather than left to be discovered.
- **N5 -- supporting counts inherited the same narrowing.** FIXED at source: 0 of **969**
  detail rows, 0 of **67** FAILs.
- **N3 -- a kill-switch finding sits inside the 41** (`wf_555a4380-3e8`). Recorded for the
  operator. All three of its entries are judge-tagged `[WARN]`, so the routing is faithful
  to "severity comes from the judge"; the point is that a money-path finding CAN land in
  the residual queue, and an operator should know that before the routing obliges anything.
- **N4 -- the tag-form table publishes no reproducing command.** Acknowledged, not fixed.
  My tally was 41/88/29/20/7 with 12 bare; the Q/A's independent tally was 41/91/37/1/5
  with 2 bare. **"initial 41" matches exactly and the rest depends on an unpublished
  precedence rule** -- mine counts each occurrence once under the first form that matches,
  theirs evidently does not. The table is ILLUSTRATIVE of why severity is delimited, and
  nothing in the shipped code depends on it; the load-bearing measurement is the 41/247
  replay, which is re-runnable. Stating the discrepancy rather than quietly reprinting my
  own numbers.


---

# CYCLE 3 -- what the cycle-2 CONDITIONAL changed

Verdict `wf_546d7764-9c6`: **CONDITIONAL**. Both cycle-1 blockers confirmed fixed and
independently re-derived -- the 41/247 replay reproduced with **zero symmetric difference**
against a matcher the Q/A wrote from scratch (identical membership, not just cardinality),
and the leak guard survived four neutering shapes it applied itself. Capped by two WARNs
and three NOTEs, **all five real, all five acted on**. Full verdict verbatim in
`handoff/current/evaluator_critique_90.2.md`.

| finding | disposition |
|---|---|
| **WARN** `governing_severities` truncation survives all 66 checks | FIXED: three assertions mirrored onto it + cell **M11** at the return-literal site M3 cannot reach |
| **WARN** `allResidual` reads `governing` without requiring `comparable` | FIXED: the emitted list governs only when index-comparable; named fallback + `emitted_severities`; cell **M12** |
| **NOTE** the 969 correction reached `:186` but not `:260` | FIXED at both sites |
| **NOTE** LIVE row disagrees between the two artifacts | FIXED by removing the class: the LIVE row is no longer transcribed here at all |
| **NOTE** the negator's narrowness is unguarded and behaviour-changing | FIXED: verbatim fixture from `wf_7fa0e5d6-c50` + cell **M14** |

**The lesson that repeats, and it is the same one in three places.** A guard is only tested
where it is the ONLY thing standing between the input and the outcome. Cycle 1: my
identifier and "no WARN fired" fixtures never reached the delimiter and negator branches, so
M5 and M8 survived. Cycle 2: `derived_severities` was guarded and `governing_severities` was
not, while one matrix cell mutating their shared source made it look like both were. Cycle
2 again: the negator was pinned against *removal* but not against *widening*, and widening
was the change that moved a real run. **Coverage of a symbol is not coverage of a branch.**

**One finding of my own, reported rather than padded.** The `entries.length > 0` clause I
added to `allResidual` is provably redundant under the `comparable` gate, and I found that
by writing a mutation cell for it and watching the cell fail to kill. The clause stays as
defence-in-depth and is disclosed in the code as redundant; the cell is removed rather than
kept as a kill it cannot make. **A cell that cannot fail is not coverage, it is decoration.**

**Verification after the cycle-3 fixes:**

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
  checks run: 77 (floor 66)
  failed:     0
IMMUTABLE COMMAND EXIT: 0

16 cells: N0 SURVIVED | M1-M9, M11, M12, M14, L1, L2 KILLED | QX ERROR
PINNED replay unchanged: 41 queue_residual / 247 remediate, zero disagreement between matchers
```
