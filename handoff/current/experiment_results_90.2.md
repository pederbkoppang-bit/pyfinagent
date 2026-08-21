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
| `scripts/qa/verify_severity_routing_90_2.mjs` | **NEW.** The immutable checker: 61 checks over a floor of 50, an 11-cell mutation matrix with the control observed GREEN first, and a `--replay` mode over the real corpus. |
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
instance.** I searched: **0 of the 66 FAILs at the pin have every entry WARN/NOTE-tagged.**
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

### Criterion 4 -- the replay, with the filed counts STATED rather than resolved

Full table in `handoff/current/live_check_90.2.md`. Summary, **denominator named with
every ratio**:

| census | startsWith | exact | parseable | with_verdict | non-PASS | queue_residual | remediate |
|---|---|---|---|---|---|---|---|
| PINNED @ 2026-08-18T12:33:57.731Z | 441 | 436 | 393 | 392 | 285 | **41** | **244** |
| LIVE (no pin) | 451 | 446 | 403 | 402 | 292 | **41** | 251 |

- **"41" reproduces EXACTLY** at the pin, under both matchers, with identical run sets, and
  is stable across the 10 records the corpus gained in between.
- **"247" does not reproduce.** The filing's population was 288 = 221 CONDITIONAL + 67 FAIL
  out of 397 verdicts. The pin that reproduces 441/436 yields **392** verdicts (219 + 66 =
  285 non-PASS), 5 fewer, so the remainder is **244**. The gap lives in `parseable`: **43
  of the 436 pinned records carry `result: null`** (39 `failed`, 2 `killed`, 3
  `completed`-without-result). The number is not edited to match.
- **"32" does not reproduce under any of four strict definitions** (41 / 26 / 11 / 4). The
  filing's strict definition is not recoverable from its text.
- **"any run mixing a WARN entry with an untagged finding must route to remediate"** --
  asserted directly and enforced by making UNTAGGED force `remediate`; mutant **M2**
  (UNTAGGED silently becomes NOTE) is KILLED.

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
  (**0 of 978** `violation_details` rows carry a `severity` key).

## 4. Verification, verbatim

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
  checks run: 61 (floor 50)
  failed:     0
IMMUTABLE COMMAND EXIT: 0
```

Full output in `handoff/current/live_check_90.2.md`.
