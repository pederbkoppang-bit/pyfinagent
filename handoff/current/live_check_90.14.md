# live_check -- step 90.14

**Step:** 90.14 -- a completeness check bound to ONE probe input shape is not completeness.
**Date:** 2026-08-21. **Research gate:** PASSED (enforced), `wf_85906afa-8c5`,
`handoff/current/research_brief_90.14.md` -- 10 sources read in full, 31 URLs, recency scan
performed, `self_report_disagreed: false`, `violations: []`.

> **BUILT AND VERIFIED, NOT EVALUATED.** No Q/A spawned. See
> `handoff/current/experiment_results_90.14.md`.

## The immutable command, unpiped

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
EXIT 0
```

## The full checker, verbatim

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
E1b. COVERAGE OVER A DECLARED INPUT MODEL, AT A STATED STRENGTH (90.14)
==========================================================================
  [PASS] CONSERVATION holds on every shape: every returned array carries exactly as many elements as the INPUT array it summarises. Stated against the input, never re-derived from the routing rule -- a control built from the same walk as the code shares its bug -- all 12 shapes
  [PASS] every ARRAY-CAPABLE key is exercised as an array by at least one shape, and NO undeclared key is ever an array -- this is what catches a field that is an array only on a branch one probe never reaches -- seen [derived_severities,emitted_severities,governing_severities]
  [PASS] the family is 2-way COMPLETE over the declared input model {arity, details, verdict} -- every pair of factor values appears in at least one shape. This is the ONLY completeness this step claims; no source supports an unconditional one -- 12 shapes
  [PASS] ...and the VERDICT is one of the declared factors, which the provisional four-shape family pinned to CONDITIONAL -- the fifth relocation the research gate measured before it shipped -- CONDITIONAL,FAIL,PASS
  [PASS] ...as are arity and details-shape, so the model is genuinely 3-factor

==========================================================================
E1c. LEAVE-ONE-OUT: EVERY FACTOR VALUE IS NECESSARY (90.14 criterion 4)
==========================================================================
  factor value           mutant that escapes without it
  arity=0               A4 empty-only fabrication
  [PASS] leave-one-out: dropping every shape with arity=0 lets a mutant escape, so that factor value is NECESSARY -- A4 empty-only fabrication
  arity=2               NOTHING
  arity=5               A1 arity-gated drop (>=4)
  [PASS] leave-one-out: dropping every shape with arity=5 lets a mutant escape, so that factor value is NECESSARY -- A1 arity-gated drop (>=4)
  details=none          NOTHING
  details=aligned       A2 aligned-only drop
  [PASS] leave-one-out: dropping every shape with details=aligned lets a mutant escape, so that factor value is NECESSARY -- A2 aligned-only drop
  details=mismatched    A3 mismatched-only drop
  [PASS] leave-one-out: dropping every shape with details=mismatched lets a mutant escape, so that factor value is NECESSARY -- A3 mismatched-only drop
  verdict=CONDITIONAL   A0c CONDITIONAL-gated drop
  [PASS] leave-one-out: dropping every shape with verdict=CONDITIONAL lets a mutant escape, so that factor value is NECESSARY -- A0c CONDITIONAL-gated drop
  verdict=FAIL          A0 FAIL-gated drop
  [PASS] leave-one-out: dropping every shape with verdict=FAIL lets a mutant escape, so that factor value is NECESSARY -- A0 FAIL-gated drop
  verdict=PASS          A0b PASS-gated drop
  [PASS] leave-one-out: dropping every shape with verdict=PASS lets a mutant escape, so that factor value is NECESSARY -- A0b PASS-gated drop
  [PASS] the set of REDUNDANT factor values is exactly the declared one -- redundancy is reported, not asserted away, and cannot grow silently -- measured [arity=2,details=none] declared [arity=2,details=none]
  [PASS] every declared FACTOR has at least one necessary value, so no factor is carried without earning its place -- 7 of 9 values necessary
  [PASS] ...and every attribution mutant is caught by SOMETHING, so the table is not measuring a set of no-ops -- all caught
  [PASS] ...and the UNGATED control is caught by every shape, which is what shows the gated ones survive because of their GATE and not because the field is unguarded -- ungated drop caught by all non-empty shapes

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
  [PASS] driven against the REAL .claude/masterplan.json, 90.1's residual IS filed -- filed=90.3,90.6,90.8,90.9,90.10,90.11,90.12,90.13 over 1234 steps

==========================================================================
I. THE SIBLING INVARIANT IS EXTENDED, NOT REINVENTED (criterion 1)
==========================================================================
  [PASS] severity_routing is spread as a NAMED SIBLING of the judge object
  [PASS] ...and never flattened into it
  [PASS] a phase-90.2 leak guard exists at the same throw-site as its two siblings
  [PASS] ...and it has NO carve-out, unlike the research_routing guard (strictly stronger)
  [PASS] the rail returns the guarded object UNCHANGED -- no second construction step -- return returned
  [PASS] the leak guard is EXTRACTABLE and callable -- a deleted if/throw is caught here, not merely missed by a regex -- function
  [PASS] ...it does NOT throw on the correct sibling shape -- no throw
  [PASS] ...it DOES throw when the routing object is FLATTENED into the verdict -- phase-90.2 invariant violated: severity_routing fields leaked into the
  [PASS] ...and when a JUDGE field collides with a routing key ("route") -- phase-90.2 invariant violated: severity_routing fields leaked into the
  [PASS] ...and it does not throw on an empty routing object (no false positive) -- no throw
  [PASS] the phase-90.15 completeness guard is EXTRACTABLE and callable -- function
  [PASS] ...it does NOT throw on the exact shape the rail returns -- no throw
  [PASS] ...it DOES throw on a caller key that belongs to NO named sibling -- the case none of the three per-object filters can see -- phase-90.15 invariant violated: the returned object carries 
  [PASS] ...and on a flattened routing object -- phase-90.15 invariant violated: the returned object carries 
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
  ok   M18  KILLED    expected KILLED
         phase-90.15: the routing object is FLATTENED at the FINAL return, one construction step past where the guards used to run -- the mutant that survived the whole checker before the seam was removed
  ok   M19  KILLED    expected KILLED
         phase-90.15: a caller key belonging to NO named sibling is spread into the return -- the case none of the three per-object filters can see, and the only cell the positive-completeness guard alone can kill
  ok   M20  KILLED    expected KILLED
         90.14 criterion 2: an ARITY-GATED drop from derived_severities -- invisible to any probe with fewer than 4 findings, and the mutant that FAILED step 90.2 at cycle 4
  ok   M21  KILLED    expected KILLED
         90.14 criterion 2: a BRANCH-GATED drop (drop-last on the comparable branch only)
  ok   M22  KILLED    expected KILLED
         90.14 criterion 2: the other branch-gated shape (drop-first on the comparable branch only)
  ok   M23  KILLED    expected KILLED
         90.14 criterion 3: a NEW array-valued field that is an array only on a branch a single probe never reaches -- the case that defeated the cycle-4 set-equality check
  ok   V1   KILLED    expected KILLED
         90.14 THE FIFTH RELOCATION, found by the research gate before it shipped: a drop gated on verdict===FAIL. It SURVIVED all 105 checks of the provisional four-shape family, which pinned verdict to CONDITIONAL, and is non-equivalent on 6 of 24 real fixture returns
  ok   V2   KILLED    expected KILLED
         90.14: the same gate on PASS -- the other verdict the provisional family never produced
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
  [PASS] handoff/verdict_ledger.jsonl sha256 byte-identical before and after -- ee58607b406fb7fd -> ee58607b406fb7fd

==========================================================================
SUMMARY
==========================================================================
  checks run: 100 (floor 100)
  failed:     0

```
