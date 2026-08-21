# OPERATOR ESCALATION -- step 90.2, attempt budget exhausted at a FAIL

**Written 2026-08-21 by Main.** Second step today to reach the budget's designed terminal
state; the first is 90.1. **The step is not flipped.**

---

## 1. Where the step stands

| | |
|---|---|
| Step | **90.2** -- route the WARN/NOTE severity the judge already emits, caller-side |
| Status | **`pending`** |
| Verdict sequence | FAIL -> CONDITIONAL -> CONDITIONAL -> **FAIL** |
| Last verdict | `wf_01b37b7d-fd2`, 231,151 tokens, 754s |
| Budget | **5 of 5 consumed.** `attempt_gate.py --status 90.2` reports `disposition: ESCALATE` |
| Immutable command | **exits 0** -- 87 checks over a floor of 74, 19 mutation cells, control GREEN first |
| Harness compliance | clean on all 5 items, every cycle |
| **Criteria 1-5** | **MET**, independently re-derived by the cycle-3 and cycle-4 Q/A |
| Criterion 6 clause 2 | **NOT MET** |

## 2. What actually got built and verified

The product works and was re-derived by an evaluator that wrote its own matcher from
scratch:

- `enforceSeverityRouting` returns a `severity_routing` **sibling**; the judge's verdict is
  byte-identical over 24 verbatim real returns by string equality.
- The FAIL guard is **structural** -- `queue_residual` requires `verdict === 'CONDITIONAL'`.
  0 of the 67 FAILs at the pin are all-WARN/NOTE, so the fixture proving immunity is
  constructed, and that absence is the argument for the guard rather than a substitute.
- **The replay reproduces the filing exactly: 41 / 247** on the derived population (441
  records / 397 verdicts / 288 non-PASS), with **zero symmetric difference** against the
  filing's own matcher and against a third matcher the Q/A wrote independently.
- The filed "strict = 32" does not reproduce under any of four definitions (41/26/11/4),
  stated and never edited.
- `residual_close_gate.mjs` is fail-closed and correct, including against the real plan.
- **No `VERDICT_SCHEMA` edit.** 86.98 is not pre-empted and its operator gate is not bypassed.

## 3. Why it failed, and the pattern that matters more than the defect

**Criterion 6 clause 2 -- "a mutant silently dropping any reported finding from the return
must also be KILLED" -- relocated FOUR times:**

| cycle | where the drop hid | verdict |
|---|---|---|
| 1 | (covered) `derived_severities` | FAIL, for two other reasons |
| 2 | `governing_severities` | CONDITIONAL |
| 3 | `emitted_severities` -- *the field the cycle-2 fix introduced* | CONDITIONAL |
| 4 | **the probe-shape dimension underneath all three** | FAIL |

Cycle 4 generalised over the **field** dimension: enumerate every array-valued key in the
return and require the key SET to match a covered set exactly. That was the right idea and
it still failed, because every expectation was computed from **one probe input** -- 2
findings, 3 detail rows. A drop gated on arity walks through:

```
derived_severities: derived.length >= 4 ? derived.slice(0, -1) : derived,
  checks run: 87 (floor 74)   failed: 0   exit 0  -- SURVIVED
```

Not equivalent: on the checker's own fixture set it silently drops a real finding from
`wf_fc420eba-820` with the route unchanged, so every route assertion is blind to it.

**The lesson, stated once: a coverage claim is only as wide as the INPUTS it was computed
over. Enumerating the output surface while holding the input fixed moves the blind spot
instead of closing it.** My cycle-4 write-up claimed "a fifth array field fails the checker
until it is covered". **That claim is false and is retracted**, not softened.

## 4. The finding you should look at even if 90.2 never closes

**All three sibling-leak guards compute one seam upstream of what the function returns.**
`escalation` (86.78), `research_routing` (86.72) and `severity_routing` (90.2) are each
checked against `merged`, and the function then returns a *different* object -- the spread
of `merged` plus `verdict_unmodified`. A spread at that final statement bypasses all three:

```
return { ...merged, ...severity_routing, verdict_unmodified: untouched }
  checks run: 87   failed: 0   exit 0  -- SURVIVED
```

**This is inherited and it is LIVE on every Q/A spawn today**, independently of 90.2. It
would surface caller-authored fields as top-level siblings of `ok`/`verdict` in the object
Main transcribes VERBATIM -- the doer/judge blur those guards exist to prevent. The Q/A
rated it WARN because 90.2 inherited it; **I filed it P0, because "inherited" describes its
origin, not its blast radius.** Filed as **step 90.15** with its own immutable command
covering all three guards.

## 5. What I did NOT do, and why

- **I did not fix either finding.** The budget denies a sixth spawn, so nothing could grade
  the fix, and an ungraded change to apparatus a Q/A has just failed is the
  shipped-fix-that-never-ran pattern. Same call as 90.1, for the same reason.
- **I did not flip the step**, edit a criterion, or re-spawn.

## 6. What I DID do (all after the FAIL, all therefore UNGRADED)

1. Reproduced **both** findings by execution in a shadow tree before transcribing them.
2. Transcribed the verdict verbatim into `evaluator_critique_90.2.md` and **retracted** the
   false completeness claim at its source.
3. Filed **90.14** (parameterise coverage over a family of probe shapes; P1) and **90.15**
   (the inherited leak seam; P0).

## 7. The decision I need from you

Two steps now sit at the same terminal state, and the choice is the same for both:

| Option | What it means |
|---|---|
| **A. Extend by one attempt each** | `python3 scripts/harness/attempt_gate.py --operator-extend 90.2 --by 1 --reason "<reason>"`. Both remaining fixes are small and precisely specified. Risk, stated from the record: this criterion has relocated four times, and cycle 4 shows a *class-level* fix can relocate too. |
| **B. Close 90.2 on criteria 1-5, with clause 3 carried by 90.14** (my recommendation) | The product is verified and independently re-derived; what remains is the mutation apparatus, filed with its own command. Needs your explicit sign-off -- a step has never been closed without a PASS and I will not do it on my own authority. |
| **C. Leave both parked** | 90.3 stays blocked behind 90.1. |

**Independently of A/B/C: 90.15 should be scheduled soon.** It is live on every Q/A spawn
and does not depend on 90.2's disposition.
