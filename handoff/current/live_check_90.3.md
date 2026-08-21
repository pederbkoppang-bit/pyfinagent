# live_check -- step 90.3

**Step:** 90.3 -- progress-gated retry, corrected. **Date:** 2026-08-21.
**Research gate:** PASSED (enforced), `wf_8f0a6091-2d0`, 10 sources read in full, 34 URLs.

> **BUILT AND VERIFIED, NOT EVALUATED. NOT CLOSEABLE.** No Q/A spawned. The mechanism ships
> **DEFAULT-OFF** (`ATTEMPT_GATE_PROGRESS_DIGEST`), so it is dark on the live rail until an
> operator enables it — an ungraded deny-capable gate that misfires blocks the whole harness.

## The immutable command, unpiped

```
$ bash -c 'python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_3.py --verify'
EXIT 0
```

Red-first baseline, before either half existed: **EXIT 2** (`mutation_matrix_90_3.py` did
not exist; captured unpiped, because a piped capture returns the pipe's status).

## The mutation matrix, verbatim

```
==========================================================================
CONTROL (the real, unmutated gate)
==========================================================================
  exit=0  FAIL lines=0
  CONTROL GREEN

==========================================================================
CELLS
==========================================================================
  ok   N0   SURVIVED  expected SURVIVED  
         NULL MUTANT (comment only). If this scores KILLED the harness is broken and every other kill in this run is meaningless.
  ok   D1   KILLED    expected KILLED    
         THE CELL THIS STEP EXISTS FOR: handoff/audit/ is removed from the exclusions, so the gate's OWN audit stream re-enters the digest and it advances by construction -- 89.1's defect through a different door, measured by the 90.3 research gate before any code existed.
  ok   D2   KILLED    expected KILLED    
         criterion 1: the digest mixes in mtime, so os.utime() on an unchanged file moves it and a touched-but-identical relaunch is admitted.
  ok   D3   KILLED    expected KILLED    
         criterion 1: the digest becomes a constant, so nothing ever differs and every relaunch after the first is denied.
  ok   D4   KILLED    expected KILLED    
         criterion 4: a MISSING declared input becomes a silent skip instead of a DENY, so a digest computed over a SUBSET masquerades as the digest of the whole set.
  ok   D5   KILLED    expected KILLED    
         criterion 2: the NO_VERDICT exemption is removed, so a byte-identical relaunch after a dropped rail is DENIED -- the doctrine-mandated retry 89.1 would have blocked on 14 of 16 real drops.
  ok   D6   KILLED    expected KILLED    
         criterion 3: comparison narrows to the PREVIOUS digest only, so an A->B->A revert oscillates forever instead of denying on the third launch.
  ok   D7   KILLED    expected KILLED    
         criterion 3: another step's digests leak into this step's comparison, so an unrelated step can deny this one.
  ok   QX   ERROR     expected ERROR     NameError
         ERROR CONTROL: a call site is renamed, so the code parses, imports, and then cannot RESOLVE A NAME at run time. It must score ERROR, never a kill (phase-90.12).

==========================================================================
CONTAINMENT (criteria 6, 7)
==========================================================================
  scripts/harness/attempt_gate.py md5   231b992e8733f2fa5fc4cfbd6a8dcc33 -> 231b992e8733f2fa5fc4cfbd6a8dcc33
  handoff/verdict_ledger.jsonl sha256   ee58607b406fb7fd51e2cbb93467194c -> ee58607b406fb7fd51e2cbb93467194c
  real tree untouched: True   (a denial is NOT a verdict, so the verdict ledger must not move)

==========================================================================
KILLED 7 | SURVIVED 0 (excl. N0) | ERROR 1 | null mutant survived: True | real-kill control killed: True
==========================================================================

```

## The self-test, verbatim

```
--reason is required: an unexplained extension is exactly the silent act this gate exists to prevent.
[attempt-gate] verdict-ledger read failed for step 9.1: IsADirectoryError: [Errno 21] Is a directory: '/var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/tmp1k55bcja/verdict_isadir' -- proceeding WITHOUT the PASS exception (fail-closed: this can only deny more, never allow more)
[attempt-gate] verdict-ledger read failed for step 9.1: IsADirectoryError: [Errno 21] Is a directory: '/var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/tmp1k55bcja/verdict_isadir' -- proceeding WITHOUT the PASS exception (fail-closed: this can only deny more, never allow more)
[attempt-gate] verdict-ledger read failed for step 9.1: IsADirectoryError: [Errno 21] Is a directory: '/var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/tmp1k55bcja/verdict_isadir' -- proceeding WITHOUT the PASS exception (fail-closed: this can only deny more, never allow more)
[attempt-gate] verdict-ledger read failed for step 9.1: IsADirectoryError: [Errno 21] Is a directory: '/var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/tmp1k55bcja/verdict_isadir' -- proceeding WITHOUT the PASS exception (fail-closed: this can only deny more, never allow more)
  ok    fresh step -> allow
  ok    at ceiling (5) -> deny
  ok    count survives re-read from disk
  ok    operator extension re-opens exactly one attempt
  ok    extension consumed -> deny again
  ok    verdict-ledger PASS -> allow (re-grades never budget-blocked)
  ok    corrupt row counts as an attempt (over-count is the safe direction)
  ok    deny path emits no verdict artifact (no such key exists)
  ok    no step_id -> not attributed
  ok    string args attribute correctly
  ok    malformed string args salvage the step id
  ok    hostile step id refused
  ok    operator extension WITHOUT --reason is refused
  ok    refused extension appends NO row
extension recorded: step 9.4 allowance +1
  ok    operator extension WITH a reason appends its labelled row
  ok    verdict-ledger read error is LOUD on stderr (V1)
  ok    read error grants NO PASS exception -- at ceiling stays deny (V2)
  ok    a claimed step id ABSENT from the plan of record is refused (90.1 c4: '9.9' is well-formed but names no step)
  ok    appending '.1' to a real id no longer mints an allowance (90.1 c4: '9.1.1' refused)
  ok    a digit-appended near-miss is refused (90.1 c4: '9.10')
  ok    a REAL step id is still admitted (90.1 c4: the check denies only what the plan does not contain)
  ok    the CLAIM survives validation so the denial can name it
  ok    a hostile claim cannot escape the escalation dir (90.1: the denial path is REACHED by ids the shape regex refused)
  ok    write_escalation REFUSES to forge '# BUDGET EXHAUSTED' for a step that is not exhausted (90.1 c2)
  ok    the refusal wrote NO file at all
  ok    a non-exhaustion denial writes its OWN reason-named path (90.1 c2)
  ok    and its body says what actually happened, not 'BUDGET EXHAUSTED'
  ok    ONE attempt costing 1,200,001 tokens is DENIED on the TOKEN ceiling with 4 of 5 attempts still unused (90.1 c3)
  ok    and one token UNDER the ceiling is still allowed -- so the check discriminates rather than always denying
  ok    a NO_VERDICT row is recorded as a DROP, not as a verdict (90.1 c1/c5: dropped=1, verdicts_seen=0)
  ok    and a graded row IS counted as a verdict -- the probe discriminates (dropped=0, verdicts_seen=1)
  ok    every dotted id the plan of record contains is ADMITTED (90.1 c4 RECALL, checked against the file not the function)
  ok    a step id nested under subphases[] is ADMITTED -- the plan is NOT uniformly phases[].steps[] (the cycle-1 BLOCK)
  ok    the REAL escalation dir is UNCHANGED by this self-test -- compared name-set before vs after, not the tautology of asking a temp dir about its own children (the 9.4 lesson)
  ok    ...and the temp dir actually RECEIVED an escalation, so the check above cannot pass by writing nothing at all
  ok    the digest is computed and reports ok over readable inputs
  ok    os.utime on EVERY input does not change the digest -- content only, so a touched-but-unchanged relaunch still DENIES (c1)
  ok    ...but changing one BYTE does move it, so the digest is not a constant
  ok    a MISSING declared input is inputs_incomplete with the file named -- not a silent skip over a subset (c4)
  ok    handoff/audit/ is EXCLUDED -- the gate writes it on every launch, so including it would make the digest advance by construction (the 90.3 gate's finding)
  ok    ...as is .claude/agent-memory/, which the agents write themselves
  ok    ...and the exclusions are not vacuous: a path under an excluded root is dropped even though its root is also allowlisted
  ok    prior_digests returns EVERY digest for the step, not just the last -- which is what makes an A->B->A revert deny on the third launch (c3)
  ok    ...and it does not leak another step's digests
  ok    a most-recent NO_VERDICT row EXEMPTS the check, so a byte-identical relaunch after a dropped rail is ADMITTED (c2)
  ok    ...while a graded CONDITIONAL that POSTDATES the last attempt does NOT exempt it -- the exemption is for drops, not for verdicts (c2)
  ok    ...and when no verdict postdates the previous attempt the check is skipped, because the prior launch produced nothing to respond to
  ok    hook drive 1: allowed, and the digest is SKIPPED because no verdict row exists yet (c2)
  ok    hook drive 2: a graded verdict now postdates the attempt, so the digest is COMPUTED and this first sighting is allowed
  ok    hook drive 3: the SAME evidence is now DENIED with reason no_new_evidence -- the wiring is exercised, not just the functions (c8)
  ok    ...and a row is appended on the DENY as well as on the allows, which is what makes an A->B->A revert deny on the third launch (c3)
  ok    ...and the denial did not touch the verdict ledger -- a denial is not a verdict (c7)
  ok    ...and with the flag OFF (the shipped default) the same launch is ALLOWED -- the mechanism is dark on the live rail until enabled
SELF-TEST PASSED

```
