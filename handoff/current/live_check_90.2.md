# live_check -- step 90.2

**Step:** 90.2 -- route the WARN/NOTE severity the judge already emits, caller-side.
**Date:** 2026-08-21 (CYCLE 4 -- regenerated after the cycle-3 CONDITIONAL).
**Contract:** `handoff/current/contract_90.2.md`.

The live_check the masterplan asks for: *"the verbatim 41/247 replay table over real run
ids, the strict-match 32/256 table beside it, and the FAIL-immunity cell output."*

**Two corrections carried at the top, because both changed a printed number.**

1. **Cycle 1** printed **41 / 244** and claimed the filed 247 "does not reproduce". The
   replay had narrowed the corpus to an exact `workflowName === 'qa-verdict'` match,
   dropping 5 records (3 non-PASS). Masterplan 90.2's `audit_basis` names *"441 `qa-verdict`
   Workflow run records"* -- **441 is the `startsWith` count**. On that derived population
   **41 and 247 both reproduce exactly.**
2. **The LIVE (unpinned) row drifts between captures by construction** -- 451 -> 452 -> 453
   records across three runs in one session -- because the corpus grows every time a Q/A
   launches, including the ones evaluating this step. It is printed here ONCE, from one
   run, and is deliberately not duplicated into `experiment_results_90.2.md`. **The PINNED
   row is the load-bearing one.**

---

## 1. The immutable command, unpiped

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
IMMUTABLE COMMAND EXIT: 0
```

Red-first baseline, captured BEFORE any of this code existed (2026-08-21):

```
Error: Cannot find module '.../scripts/qa/verify_severity_routing_90_2.mjs'  (MODULE_NOT_FOUND)
EXIT 1
```

## 2. The replay over real run records, verbatim (criterion 4)

`node scripts/qa/verify_severity_routing_90_2.mjs --replay`:

```

PINNED @ 2026-08-18T12:33:57.731Z
  POPULATION (DERIVED from masterplan 90.2 audit_basis, "441 qa-verdict Workflow run records"): workflowName.startsWith('qa-verdict')
  DENOMINATORS: startsWith=441 (exact-match=436, +5 under variant names ["qa-verdict-writefirst-82-7","qa-verdict-writefirst-82-5"]) parseable=398 with_verdict=397 non-PASS=288
  verdict mix: {"CONDITIONAL":221,"FAIL":67,"PASS":109}
  A. the FILING's matcher (token anywhere)  queue_residual=41  remediate=247
  B. the SHIPPED matcher (delimited tag)    queue_residual=41  remediate=247
  DISAGREEMENT  A-only=0 []  B-only=0 []
  FAILs routed to queue_residual by the shipped matcher: 0  (structurally impossible, printed to show it)

  queue_residual run ids at the pin:
    wf_1afa11f6-75a
    wf_2dd1efc9-d0c
    wf_86449fa1-bad
    wf_c5326358-53c
    wf_d0934c91-70b
    wf_555a4380-3e8
    wf_6e9d4eb1-5ff
    wf_2bdb75d1-347
    wf_d5022922-89f
    wf_46e96d67-b24
    wf_10fc5a5f-189
    wf_26b4d6d0-c33
    wf_7fa0e5d6-c50
    wf_cf6b2734-976
    wf_cf98fce9-223
    wf_0d88fe11-241
    wf_249feb74-c6d
    wf_8f83d0d5-0c9
    wf_c3dacb94-ccf
    wf_fa56f83d-814
    wf_7e817466-c1c
    wf_00a7dd53-3f5
    wf_df0a7b9e-b75
    wf_122c08a4-c3f
    wf_28cf4dbb-9aa
    wf_4f890efc-c3e
    wf_cf74d169-1e7
    wf_982cd319-493
    wf_e66ad533-e61
    wf_71687e5e-c63
    wf_0e038919-306
    wf_775cfbb1-5ee
    wf_8275f3fa-266
    wf_e7115d07-ae1
    wf_6bc4c0a4-d9c
    wf_144e1ab6-8ef
    wf_9b398d19-fa8
    wf_1ff464d6-6f1
    wf_2f31e904-f24
    wf_b184df52-3e7
    wf_b361863d-3c4

LIVE (no pin)
  POPULATION (DERIVED from masterplan 90.2 audit_basis, "441 qa-verdict Workflow run records"): workflowName.startsWith('qa-verdict')
  DENOMINATORS: startsWith=454 (exact-match=449, +5 under variant names ["qa-verdict-writefirst-82-7","qa-verdict-writefirst-82-5"]) parseable=411 with_verdict=410 non-PASS=298
  verdict mix: {"CONDITIONAL":228,"FAIL":70,"PASS":112}
  A. the FILING's matcher (token anywhere)  queue_residual=43  remediate=255
  B. the SHIPPED matcher (delimited tag)    queue_residual=43  remediate=255
  DISAGREEMENT  A-only=0 []  B-only=0 []
  FAILs routed to queue_residual by the shipped matcher: 0  (structurally impossible, printed to show it)

  THE FILED COUNTS (criterion 4):
    "41"  REPRODUCES EXACTLY at the pin, under BOTH matchers, with identical run sets.
    "247" REPRODUCES EXACTLY at the pin, on the DERIVED population. 441 records ->
          397 verdicts (PASS 109 / CONDITIONAL 221 / FAIL 67) -> 288 non-PASS ->
          41 queue_residual + 247 remediate.
          CORRECTED: the first version of this replay filtered on an exact
          workflowName match, dropping 5 records under the variant names
          qa-verdict-writefirst-82-5 (x3) and -82-7 (x2). Three of those are
          non-PASS, which is exactly 247 - 3 = 244, and I published 244 together
          with a paragraph asserting that 247 "does not reproduce" and blaming a
          43-of-436 result:null gap -- which explains the 436->393 PARSEABLE gap,
          a different gap entirely. The scope was CHOSEN, not derived; the filing
          names 441, and 441 is the startsWith count.
    "32"  (strict) DOES NOT reproduce under any of four plausible strict definitions:
          token-anywhere 41, bracketed-anywhere 26, starts-with-bare 11,
          starts-with-separator 4. Measured under BOTH populations. The filing's
          strict definition is not recoverable from its text, and that number is
          NOT edited to match either.

```

### The strict table, beside it (criterion 4)

Four plausible readings of "strict", measured at the same pin and under **both**
populations with identical results:

| definition | matcher | queue_residual |
|---|---|---|
| token anywhere (the filing's) | `'BLOCK' not in e and ('WARN' in e or 'NOTE' in e)` | **41** |
| bracketed anywhere | `[\[(](WARN|NOTE)[\])]` and no bracketed BLOCK | 26 |
| starts-with, bare | `e.startswith('WARN'|'NOTE')` | 11 |
| starts-with, with a separator | `^(WARN|NOTE)\s*[:\-—]` | 4 |

**The filed "strict = 32" reproduces under none of them.** Stated, not edited.

## 3. FAIL immunity, verbatim (criterion 2)

The fixture is CONSTRUCTED, because **0 of the 67 FAILs at the pin are all-WARN/NOTE**.

```
A. THE VERDICT GUARD IS STRUCTURAL (criterion 2)
  [PASS] a FAIL whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] a PASS whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] the SAME entries under CONDITIONAL route to queue_residual -- so the guard DISCRIMINATES rather than always denying -- route=queue_residual
```

```
  FAILs routed to queue_residual by the shipped matcher: 0  (structurally impossible, printed to show it)
```

## 4. The full self-test, verbatim (87 checks, floor 74, 19 mutation cells)

```

==========================================================================
A. THE VERDICT GUARD IS STRUCTURAL (criterion 2)
==========================================================================
  [PASS] a FAIL whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] a PASS whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] the SAME entries under CONDITIONAL route to queue_residual -- so the guard DISCRIMINATES rather than always denying -- route=queue_residual

==========================================================================
B. THE DERIVATION READS A DELIMITED TAG, NOT A TOKEN (criteria 4, 6)
==========================================================================
  [PASS] a parenthesised tag is a tag
  [PASS] a bracketed tag is a tag
  [PASS] a colon-suffixed tag is a tag
  [PASS] a dash-suffixed tag is a tag
  [PASS] an entry-initial tag is a tag
  [PASS] BLOCK dominates a co-occurring WARN
  [PASS] an untagged finding is UNTAGGED, never silently a NOTE
  [PASS] an IDENTIFIER carrying a severity substring is NOT a tag -- got UNTAGGED
  [PASS] an IMMEDIATE negator kills a tag that IS in a delimiter position ("no WARN: nothing fired") -- the negator rule is the only thing excluding it -- got UNTAGGED
  [PASS] a BARE token in prose is not a tag -- the delimiter rule is the only thing excluding it -- got UNTAGGED
  [PASS] ...and that exclusion changes the ROUTE, not just the label -- route=remediate
  [PASS] ...and so does the negator exclusion
  [PASS] a negator in the finding's own prose does NOT kill a genuine trailing tag (the measured 6-of-6 false-positive class) -- got WARN
  [PASS] ...and the same for a bracketed one

==========================================================================
C. MIXED AND EMPTY RUNS (criterion 4)
==========================================================================
  [PASS] one WARN + one UNTAGGED routes to remediate -- route=remediate
  [PASS] one WARN + one BLOCK routes to remediate
  [PASS] a CONDITIONAL with NO entries routes to remediate, never queue_residual -- route=remediate
  [PASS] ...and names the absence ABSENT rather than scoring it -- severity_source=ABSENT

==========================================================================
D. ABSENCE IS NAMED, NEVER VALUED
==========================================================================
  [PASS] disagreed is null (not false) when the judge emitted nothing -- disagreed=null
  [PASS] ...and the status says why -- nothing_emitted_to_compare
  [PASS] a judge-emitted severity GOVERNS over the caller derivation (86.98 branch) -- source=judge_emitted route=remediate
  [PASS] ...and the disagreement with the derivation is reported, not hidden
  [PASS] ...and reliability is null on that branch (nothing was derived to qualify)
  [PASS] non-index-comparable arrays yield disagreed=null with a named status -- not_index_comparable -- violation_detail

==========================================================================
E. NOTHING IS DROPPED, AND THE DERIVATION CARRIES ITS UNRELIABILITY
==========================================================================
  [PASS] every reported finding survives into derived_severities -- n=3
  [PASS] ...aligned to violated_criteria BY INDEX
  [PASS] ...with the per-entry classes intact -- WARN,UNTAGGED,NOTE
  [PASS] every reported finding survives into governing_severities too -- n=3
  [PASS] ...with its per-entry classes intact -- WARN,UNTAGGED,NOTE
  [PASS] ...and it agrees with derived_severities index-for-index when nothing was judge-emitted

==========================================================================
E1b. EVERY ARRAY IN THE RETURN IS COVERED, AND A NEW ONE FAILS LOUDLY
==========================================================================
  [PASS] the SET of array-valued keys in the return is exactly the covered set -- a NEW array field fails here until it is given a length and a content assertion -- found [derived_severities,emitted_severities,governing_severities] covered [derived_severities,emitted_severities,governing_severities]
  [PASS] array `derived_severities` carries every element it was given -- length -- n=2 want 2
  [PASS] array `derived_severities` carries every element it was given -- content in order -- WARN,WARN
  [PASS] array `governing_severities` carries every element it was given -- length -- n=2 want 2
  [PASS] array `governing_severities` carries every element it was given -- content in order -- WARN,WARN
  [PASS] array `emitted_severities` carries every element it was given -- length -- n=3 want 3
  [PASS] array `emitted_severities` carries every element it was given -- content in order -- WARN,NOTE,BLOCK
  [PASS] ...and the probe fixture makes the three arrays DIFFERENT lengths, so no truncation can hide behind a length-1 case -- 2 vs 3

==========================================================================
E2. THE 86.98 BRANCH CANNOT FILE AN UNCLASSIFIED FINDING AWAY
==========================================================================
  [PASS] a judge-emitted list that does NOT line up with the findings cannot file two untagged blockers away as residual -- route=remediate
  [PASS] ...and the fallback to the derivation is NAMED, not silent -- judge_emitted_not_index_comparable_falling_back_to_derivation
  [PASS] ...and the judge-emitted list is still REPORTED rather than discarded
  [PASS] an EMPTY findings list cannot reach queue_residual on the emitted branch either -- no findings is never a residual -- route=remediate
  [PASS] a judge-emitted list that DOES line up still governs (86.98 is satisfied, not pre-empted)

==========================================================================
E3. THE NEGATOR IS NARROW BY MEASUREMENT, AND THE NARROWNESS IS PINNED
==========================================================================
  [PASS] a negator three words back does NOT kill a genuine trailing tag (verbatim from wf_7fa0e5d6-c50, the run a 45-char window moves) -- got WARN
  [PASS] ...while an IMMEDIATE negator still does, so the rule is narrow, not absent
  [PASS] an empty findings list cannot reach queue_residual under ANY branch -- derived, emitted-comparable, or emitted-mismatched

==========================================================================
E4. RELIABILITY TRAVELS WITH THE DERIVATION
==========================================================================
  [PASS] reliability travels with the derivation on the NOT-index-comparable branch too, where the derivation is what governs -- object
  [PASS] ...and is null ONLY when the judge-emitted list actually governs
  [PASS] the derivation is labelled NON-authoritative
  [PASS] ...and carries the brief's figures attributed to the BRIEF, not to this step
  [PASS] queue_residual carries the FILE-don't-fix instruction
  [PASS] ...and remediate carries null rather than an empty string

==========================================================================
F. VERDICT BYTE-IDENTITY OVER >= 20 REAL RETURNS (criterion 3)
==========================================================================
  [PASS] the fixture set holds at least 20 REAL returns -- 24 returns, pinned 2026-08-18T12:33:57.731Z
  [PASS] ...spanning all three verdict values -- PASS,FAIL,CONDITIONAL
  [PASS] the judge's verdict string is byte-identical after routing, on every return -- 24/24 by string equality
  [PASS] ...and the input object itself is never mutated -- 0 mutated

==========================================================================
G. THE FIXTURE BUCKETS REPRODUCE (criterion 4, in miniature)
==========================================================================
  [PASS] every fixture return routes as its recorded bucket predicts -- 24/24
  [PASS] no real FAIL in the fixture set routes to queue_residual -- 6 FAILs

==========================================================================
H. THE queue_residual CONSUMER REFUSES A PARENT CLOSE (criterion 5)
==========================================================================
  [PASS] queue_residual + NO filed residual -> close REFUSED
  [PASS] queue_residual + a residual that grades NOTHING -> close REFUSED -- 90.12
  [PASS] queue_residual + a properly filed residual -> close ALLOWED -- filed=90.12
  [PASS] remediate + nothing filed -> close ALLOWED (the gate binds only on queue_residual)
  [PASS] an unparseable plan REFUSES rather than failing open -- fail-closed
  [PASS] a residual filed for 90.10 does NOT satisfy a debt owed by 90.1
  [PASS] ...and the parent cannot be its own residual
  [PASS] driven against the REAL .claude/masterplan.json, 90.1's residual IS filed -- filed=90.3,90.6,90.8,90.9,90.10,90.11,90.12,90.13 over 1227 steps

==========================================================================
I. THE SIBLING INVARIANT IS EXTENDED, NOT REINVENTED (criterion 1)
==========================================================================
  [PASS] severity_routing is spread as a NAMED SIBLING of the judge object
  [PASS] ...and never flattened into it
  [PASS] a phase-90.2 leak guard exists at the same throw-site as its two siblings
  [PASS] ...and it has NO carve-out, unlike the research_routing guard (strictly stronger)
  [PASS] the leak guard is EXTRACTABLE and callable -- a deleted if/throw is caught here, not merely missed by a regex -- function
  [PASS] ...it does NOT throw on the correct sibling shape -- no throw
  [PASS] ...it DOES throw when the routing object is FLATTENED into the verdict -- phase-90.2 invariant violated: severity_routing fields leaked into the
  [PASS] ...and when a JUDGE field collides with a routing key ("route") -- phase-90.2 invariant violated: severity_routing fields leaked into the
  [PASS] ...and it does not throw on an empty routing object (no false positive) -- no throw
  [PASS] no routing key collides with any judge field name -- none

==========================================================================
J. VERDICT SEMANTICS FROZEN
==========================================================================
  [PASS] the routing object cannot express a verdict value
  [PASS] no VERDICT_SCHEMA edit was made (86.98 is not pre-empted)

==========================================================================
K. MUTATION MATRIX (criterion 6) -- control observed GREEN first
==========================================================================
  ok   N0   SURVIVED  expected SURVIVED
         NULL MUTANT (comment only). If this scores KILLED the harness is broken and every other kill this run is meaningless.
  ok   M1   KILLED    expected KILLED
         criterion 2, NAMED: the verdict guard is removed, so a FAIL whose findings are all WARN-tagged can be filed away instead of fixed
  ok   M2   KILLED    expected KILLED
         criterion 6, NAMED: an UNTAGGED finding is treated as a NOTE, so a run mixing a WARN with an unclassified defect is filed as residual
  ok   M3   KILLED    expected KILLED
         criterion 6, NAMED: a reported finding is silently dropped from the return
  ok   M4   KILLED    expected KILLED
         BLOCK stops dominating, so a real blocker co-occurring with a WARN is filed away
  ok   M5   KILLED    expected KILLED
         the delimiter requirement is dropped -- the naive token-anywhere matcher the measurement rejected, which also fires on identifiers
  ok   M6   KILLED    expected KILLED
         absence is recorded as a VALUE: disagreed becomes false rather than null when there is nothing to compare
  ok   M7   KILLED    expected KILLED
         an EMPTY findings list counts as all-residual, so a CONDITIONAL with no findings is filed away
  ok   M8   KILLED    expected KILLED
         the immediate-negator check is removed, so "no WARN fired" reads as a WARN tag
  ok   M9   KILLED    expected KILLED
         the judge-emitted branch is ignored, so 86.98's "severity comes from the judge" is silently unimplementable
  ok   M11  KILLED    expected KILLED
         criterion 6 clause 2, NAMED: a reported finding is silently dropped from `governing_severities` IN THE RETURN LITERAL -- the cycle-2 survivor. M3 mutates the shared source array and cannot reach this site.
  ok   M12  KILLED    expected KILLED
         the judge-emitted list governs even when it does not line up with the findings, so untagged blockers and an empty findings list reach queue_residual on the 86.98 branch
  ok   M15  KILLED    expected KILLED
         criterion 6 clause 2, THIRD RELOCATION: a reported finding is dropped from `emitted_severities` in the return literal (drop-first shape) -- the field the cycle-2 fix introduced
  ok   M16  KILLED    expected KILLED
         the same drop from the OTHER end (drop-last), which removes a judge-emitted BLOCK -- a length-1 fixture cannot see either shape
  ok   M17  KILLED    expected KILLED
         the cycle-3 reliability gate reverts to the pre-cycle-3 `anyEmitted`, so a DERIVED route ships with reliability=null and no unreliability label
  ok   M14  KILLED    expected KILLED
         IMMEDIATE_NEGATOR is widened back to the 45-char proximity window this step measured and discarded -- survived cycle 2 while moving one real run out of queue_residual
  ok   L1   KILLED    expected KILLED
         criterion 1, NAMED: the leak guard is made unreachable (`if (false && ...)`) while its literal text survives -- the illusory-guard shape the source scans could not see
  ok   L2   KILLED    expected KILLED
         criterion 1, NAMED: the if/throw is DELETED and the invariant message left behind in a comment, so every regex still matches
  ok   QX   ERROR     expected ERROR
         ERROR CONTROL: a call site is renamed, so the code cannot RESOLVE A NAME and never runs. It must score ERROR, never a kill.
  [PASS] null mutant SURVIVED (the harness is not scoring everything as a kill) -- SURVIVED
  [PASS] a real-kill control was KILLED on the same run (the harness is not scoring everything as a survivor) -- KILLED
  [PASS] a mutant that cannot RESOLVE A NAME scored ERROR, never a kill -- ERROR
  [PASS] every mutation cell scored as expected -- 0 unexpected
  [PASS] no cell was a NO-OP (a mutation that changed nothing tests nothing) -- none

==========================================================================
L. THE RUN CHANGED NOTHING IT SHOULD NOT HAVE
==========================================================================
  [PASS] handoff/verdict_ledger.jsonl sha256 byte-identical before and after -- cddc78f43062bdc8 -> cddc78f43062bdc8

==========================================================================
SUMMARY
==========================================================================
  checks run: 87 (floor 74)
  failed:     0

```
