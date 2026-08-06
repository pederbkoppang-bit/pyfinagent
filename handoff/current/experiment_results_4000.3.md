# Experiment results -- 4000.3: the operator-gated live smoke window

Date: 2026-08-06. Author: Main. Contract: contract_4000.3.md (research
wf_b7081c09-538 < contract < window < artifacts, mtime-ordered).

## What happened -- the headline

The window ran ONE AAPL analysis through the real backend on the live rail
and the smoke returned an HONEST FAIL: the analysis died in production at the
pipeline's Step-2 data-fetch (`[RuntimeError] Step 'quant': QuantAgent failed
for AAPL: 'NoneType' object has no attribute 'get'`, orchestrator.py:1792)
BEFORE the first LLM call. E4 red; E1/E2 red via the positive-control design
(zero rows without a rail-row control is a FAIL, never evidence of metered
darkness); E3/E5/E6 legs green. Exit 1 (see the live_check's exit-code
caveat). Per the operator's pre-stated end-state rule this is a valid closable
outcome: rail STAYS ON (post-window GET capture confirms True), no retry, the
defect queued as step 4000.10.

THE STEP's own criteria concern the window being run correctly and the
evidence being complete -- not the smoke passing -- and all eight are met.

## Artifacts

| Artifact | Path |
|---|---|
| Research brief | handoff/current/research_brief_4000.3.md |
| Contract | handoff/current/contract_4000.3.md |
| Live evidence (all captures verbatim) | handoff/current/live_check_4000.3.md |
| Check script (criterion 1) | scripts/qa/verify_phase_4000_3_live_smoke.sh |
| Raw window log + probe envelope | session scratchpad (window_4000_3_output.log, probe_envelope_4000_3.json; verbatim copies embedded in the live_check) |
| This file | handoff/current/experiment_results_4000.3.md |

Produced-file set: the four suffixed handoff files + the check script + the
researcher auto-memory file from the 4000.3 spawn. git diff scope: handoff/ +
scripts/qa/ only.

## Verification command output, verbatim

```
$ bash scripts/qa/verify_phase_4000_3_live_smoke.sh
...
ok   [lc: queued step]
RESULT: PASS
exit=0
```

Mutation demo (criterion-1 content-marker proof): the FAILed-checks section
deleted from a COPY ->

```
FAIL [lc: queued step]: missing marker: masterplan step 4000.10 (added in the same edit
RESULT: FAIL (1)
exit=1
```

(One check-script iteration is itself disclosed: the first marker spanned a
wrapped line in the artifact and the real run FAILED until the marker was
re-anchored to a single-line substring -- the gate was demonstrated red-then-
green on the real artifact, which is the honest direction to find it.)

## Findings of record (full detail in live_check_4000.3.md)

1. QUANT CRASH = the queued defect (4000.10): pre-LLM, so it kills EVERY full
   analysis on this path; the newest successful AAPL analysis_results row is
   2026-03-07 -- five months old -- independent corroboration of the phase's
   throughput thesis. Same defect class as the phase-27.6.2 sibling fix at
   orchestrator.py:1809-1811.
2. The E6 per-analysis rail-call number REMAINS UNMEASURED (the window never
   reached the rail); its measurement moves to 4000.10's live_check (re-run
   window on the fixed pipeline under a fresh operator token).
3. Duplicate-canonicalModel modelUsage shape observed live AGAIN (third
   sighting; envelope embedded in the live_check) -- 4000.7's trigger.
4. End state: flag True before, during and after; the restore path executed;
   backend/.env untouched by the window beyond the PUT round-trip.

## Deviations / notes for Q/A

- The complete CLI envelope in the live_check is a POST-window probe capture
  (the in-window probe printed only its summary line); disclosed inline.
- The zsh pipestatus display bug ate the literal exit-code echo; the exit is
  established by the proven all_ok->exit mapping + the summary line
  (disclosed in the live_check).
- 4000.10 is named in the live_check and will be ADDED in the same masterplan
  edit as this step's flip (criterion 8's correspondence requirement).

## Follow-up (cycle 2) -- Q/A cycle-1 findings fixed, 2026-08-06

Cycle-1 verdict: CONDITIONAL (evaluator_critique_4000.3.md, verbatim), four
findings, none requiring a window re-run. All fixed; evidence CHANGED:

1. End-state guard vacuity (criterion 5): the check script now extracts the
   line(s) immediately after the POST-WINDOW header and asserts the true
   value THERE -- a capture flipped to false is killed with the flipped value
   echoed in the failure message. Demonstrated: real artifact PASS exit 0;
   flipped-capture mutant FAIL exit 1 on exactly the end-state check.
2. False process provenance (criterion 7): the live_check section is REWRITTEN
   from re-derived facts -- the stale 4000.1 pid claim is retracted inline; a
   restart DID occur at 11:46:15Z (parent 89530 + child 89533, one start
   instant, both workers cycled), post-dating the RAIL TIERS record
   (09:59:27Z) and pre-dating this step's first artifact (14:21:03Z), so the
   no-restart-inside-the-step conclusion survives on honest evidence.
3. Brackets (criterion 6): prose summaries replaced by the as-run facts plus a
   FRESH full verbatim git add -An capture (14:41:58Z, 17 paths) with every
   path reconciled into this-step / shared-streams / concurrent-session sets;
   the falsified forward-looking claim is retracted inline; the flip protocol
   now waits if the concurrent session's research_brief_82.58.md is still
   advancing (mtime-stability check) per feedback_uncommitted_is_not_protected.
4. 4000.10 (criterion 8): added to .claude/masterplan.json EARLY, ahead of the
   flip (P1; fix the pre-LLM quant crash; its live_check is the re-run window
   under a fresh operator token, which is also where E6 finally gets measured).

Check-script marker re-anchors after the rewrites (two line-wrapped markers
found red on the real artifact and fixed -- same red-then-green discipline as
cycle 1's anchor fix):

```
$ bash scripts/qa/verify_phase_4000_3_live_smoke.sh
...
ok   [lc: queued step]
RESULT: PASS
exit=0
$ bash scripts/qa/verify_phase_4000_3_live_smoke.sh $SC/mutant_endstate_false.md
FAIL [lc: end state]: the post-window capture block does not carry the true value ({"paper_use_claude_code_route": false})
RESULT: FAIL (1)
exit=1
```

## Follow-up (cycle 3) -- Q/A cycle-2 findings fixed, 2026-08-06

Cycle-2 verdict: CONDITIONAL (evaluator_critique_4000.3.md cycle-2 block).
The auto-FAIL rule is in force for this spawn. All findings closed:

1. Criterion 6a (brackets): the live_check now states EXPLICITLY that the
   full bracket path-lists were never retained and are unrecoverable -- the
   evaluator's own named fix -- and pastes the AS-RUN filtered captures
   verbatim (count + filter pipeline + output, both brackets). The 9->11
   delta is honestly declared partially unreconcilable (the rename is
   net-zero; the cycle-1 claim that it explained the delta is retracted; the
   named known movement is the 85.1 memory file; >=1 POST path is unnamed).
   What the as-run filters DO establish is stated: no backend/frontend/
   settings path in either bracket's residue.
2. Criterion 6b (wrong sentinel + foreign production code): the flip protocol
   is REPLACED -- the gate is now the FULL derived foreign set: two
   `git add -An` captures >=3 minutes apart must be IDENTICAL and contain
   ZERO backend/, frontend/, or .claude/settings paths not this step's own,
   else WAIT or let the 82.58 session commit first; the passing capture is
   pasted into the live_check's 'Pre-flip capture' addendum before the flip.
3. Criterion 2 (single-writer leg): the check script now greps the
   single-writer confirmation from harness_log.md itself (plus the
   live_check's own reference separately).
4. Criterion 5 decoy header (NOTE): the script asserts exactly ONE
   POST-WINDOW header before the after-header value check.

Mutation matrix, verbatim (real + three mutants):

```
== real ==                          RESULT: PASS   exit=0
== M1 end-state flipped ==          exit=1 | FAIL [lc: end state]: ... does not carry the true value ({"paper_use_claude_code_route": false})
== M2 decoy earlier header ==       exit=1 | FAIL [lc: end state]: expected exactly 1 POST-WINDOW header, found 2
== M3 fake-repo log, single-writer deleted == exit=1 | FAIL [log: single-writer]: missing marker: single-writer window
```
