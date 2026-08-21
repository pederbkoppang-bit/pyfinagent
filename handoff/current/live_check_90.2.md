# live_check -- step 90.2

**Step:** 90.2 -- route the WARN/NOTE severity the judge already emits, caller-side.
**Date:** 2026-08-21. **Contract:** `handoff/current/contract_90.2.md`.

The live_check the masterplan asks for: *"the verbatim 41/247 replay table over real run
ids, the strict-match 32/256 table beside it, and the FAIL-immunity cell output."*
All three are below, verbatim, together with the two filed numbers that do not reproduce.

---

## 1. The immutable command, unpiped

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
90.2 IMMUTABLE COMMAND EXIT: 0
```

Red-first baseline, captured BEFORE any of this code existed (2026-08-21):

```
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
Error: Cannot find module '.../scripts/qa/verify_severity_routing_90_2.mjs'  (MODULE_NOT_FOUND)
EXIT 1
```

## 2. The replay over real run records, verbatim (criterion 4)

`node scripts/qa/verify_severity_routing_90_2.mjs --replay`:

```

PINNED @ 2026-08-18T12:33:57.731Z
  DENOMINATORS: startsWith=441 exact=436 parseable=393 with_verdict=392 non-PASS=285
  A. the FILING's matcher (token anywhere)  queue_residual=41  remediate=244
  B. the SHIPPED matcher (delimited tag)    queue_residual=41  remediate=244
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
  DENOMINATORS: startsWith=451 exact=446 parseable=403 with_verdict=402 non-PASS=292
  A. the FILING's matcher (token anywhere)  queue_residual=41  remediate=251
  B. the SHIPPED matcher (delimited tag)    queue_residual=41  remediate=251
  DISAGREEMENT  A-only=0 []  B-only=0 []
  FAILs routed to queue_residual by the shipped matcher: 0  (structurally impossible, printed to show it)

  THE FILED COUNTS, STATED RATHER THAN RESOLVED (criterion 4):
    "41"  REPRODUCES EXACTLY at the pin, under BOTH matchers, with identical run sets.
    "247" DOES NOT reproduce. At the pin the non-PASS population is 285 (219 CONDITIONAL
          + 66 FAIL) and the remainder is 244. The filing's 288 = 221 + 67 comes from a
          census of 397 verdicts; the pin that reproduces 441 startsWith / 436 exact
          yields 392 verdicts, 5 fewer. 43 of the 436 pinned records carry result=null
          (39 failed, 2 killed, 3 completed-without-result), which is where the parseable
          gap lives. The number is NOT edited to match.
    "32"  (strict) DOES NOT reproduce under any of four plausible strict definitions:
          token-anywhere 41, bracketed-anywhere 26, starts-with-bare 11,
          starts-with-separator 4. The filing's strict definition is not recoverable
          from its text. The number is NOT edited to match.

```

### The strict table, beside it (criterion 4)

Four plausible readings of "strict", each measured at the same pin over the same
285 non-PASS runs:

| definition | matcher | queue_residual |
|---|---|---|
| token anywhere (the filing's) | `'BLOCK' not in e and ('WARN' in e or 'NOTE' in e)` | **41** |
| bracketed anywhere | `[\[(](WARN|NOTE)[\])]` and no bracketed BLOCK | 26 |
| starts-with, bare | `e.startswith('WARN'|'NOTE')` | 11 |
| starts-with, with a separator | `^(WARN|NOTE)\s*[:\-—]` | 4 |

**The filed "strict = 32" reproduces under none of them.** It is stated, not edited.
The filed pair was "32 vs 41"; 41 reproduces exactly and 32 does not.

## 3. FAIL immunity, verbatim (criterion 2)

From the self-test, section A -- and note the fixture is CONSTRUCTED, because
**0 of the 66 FAILs at the pin are all-WARN/NOTE**. That is exactly why the guard has
to be structural: "never observed" is not "cannot happen".

```
A. THE VERDICT GUARD IS STRUCTURAL (criterion 2)
  [PASS] a FAIL whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] a PASS whose every entry is WARN/NOTE-tagged routes to remediate -- route=remediate
  [PASS] the SAME entries under CONDITIONAL route to queue_residual -- so the guard DISCRIMINATES rather than always denying -- route=queue_residual
```

And over the live corpus, printed on every replay run:

```
  FAILs routed to queue_residual by the shipped matcher: 0  (structurally impossible, printed to show it)
```

## 4. The full self-test, verbatim

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
  [PASS] non-index-comparable arrays yield disagreed=null with a named status

==========================================================================
E. NOTHING IS DROPPED, AND THE DERIVATION CARRIES ITS UNRELIABILITY
==========================================================================
  [PASS] every reported finding survives into derived_severities -- n=3
  [PASS] ...aligned to violated_criteria BY INDEX
  [PASS] ...with the per-entry classes intact -- WARN,UNTAGGED,NOTE
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
  checks run: 61 (floor 50)
  failed:     0

```
