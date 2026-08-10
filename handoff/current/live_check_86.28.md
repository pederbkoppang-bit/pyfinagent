# live_check -- step 86.28

Generated 2026-08-10 by Main (session pyfinagent-06). Every block below is
verbatim command output or a verbatim workflow return value.

## 1. Immutable verification command -- BEFORE any edit (baseline)

```
$ node scripts/qa/verify_research_gate_workflow.mjs   # captured before the first edit
ALL GREEN: 40 passed, 0 failed
```

## 2. Immutable verification command -- AFTER (re-run just now)

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 61 passed, 0 failed
```

## 3. New checks, verbatim (phase-86.28 sections)

```
[6b] phase-86.28 -- an UNSUPPORTED tier fails closed; an ABSENT tier does not
  ok   UNSUPPORTED tier => gate_passed false (refuses to certify a standard never applied)
  ok   the violation names the requested tier and what actually ran
  ok   refusal path (env=null, unsupported tier) reports the TIER as the cause
  ok   refusal path does NOT claim an agent returned null (no agent was asked to run)
  ok   refusal path still fails closed
  ok   empty return on a supported tier still reports empty_or_errored_return
  ok   ABSENT tier still PASSES (defaulting is legitimate when the caller named nothing)
  ok   a SUPPORTED tier passes
  ok   absent opts.tier behaves exactly as before (pre-86.28 call sites unaffected)

[6c] phase-86.28 -- the two formerly-uncorroborated self-reports are checked against the brief
  ok   recency_scan_performed=true with NO recency section in the brief => gate_passed false
  ok   recency corroboration PASSES when the brief carries the section
  ok   recency_scan_performed=false still fails via the original check, not the corroboration
  ok   urls_collected over-claim (99 claimed, 25 in the brief) => gate_passed false
  ok   urls_collected within what the brief carries PASSES
  ok   absent verification still fails via fail-closed ONLY (corroboration does not double-fire)

[7] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it
```

## 4. Mutation output -- every mutant, verbatim (criterion 5)

```
[7] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it
  ok   mutant "FLOOR_SOURCES 5 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "FLOOR_URLS 10 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "audit-class dry check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "over-claim check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "fail-closed on absent verification removed" is KILLED [threw: Cannot read properties of null (reading 'brief_exists')]
  ok   mutant "tier_unsupported check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency corroboration removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "urls corroboration removed" is KILLED [let-a-bad-envelope-through]

[8] structural -- no stripped schema keywords, no forbidden runtime imports, riders intact
```

## 5. Structural + ordering assertions (criterion 8 riders intact)

```
[8] structural -- no stripped schema keywords, no forbidden runtime imports, riders intact
  ok   no `minimum:` in the schema (stripped on the wire -- would be false assurance)
  ok   no `minItems:` in the schema (capped at 1 on the wire)
  ok   gate_passed is NOT const:true (honest failure must be representable)
  ok   additionalProperties:false on the envelope
  ok   NO static imports of ANY form (found 0) -- the Workflow runtime parses only dynamic import()
  ok   agentType is 'researcher' (needs Write for write-first)
  ok   model is 'opus' (rider-trap R4)
  ok   no Monitor/watchdog (rider-trap R11)
  ok   exactly ONE export (`export const meta`) -- a trailing export list is unlaunchable
  ok   driver REFUSES TO SPAWN on an unsupported tier
  ok   the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  ok   the refusal path returns gate_passed:false
  ok   enforceGate is pure -- no fs/process use in its body

ALL GREEN: 61 passed, 0 failed
```
