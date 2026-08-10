# Contract -- step 86.28

**Step**: `86.28` (phase-86, P2, `harness_required: true`)
**Phase**: PLAN
**Date**: 2026-08-10
**Cycle driver**: Main (session `pyfinagent-06`)

---

## 1. Research gate summary

**Gate result: PASSED** (run `wf_60de95f7-5dc`, Workflow rail
`.claude/workflows/research-gate.js`).

| Field | Value |
|---|---|
| `gate_passed` (enforced) | `true` |
| `agent_self_reported_gate_passed` | `true` |
| `self_report_disagreed` | `false` |
| sources read in full | 7 (floor 5) |
| URLs collected | 34 (floor 10) |
| snippet-only sources | 27 |
| internal files inspected | 6 |
| recency scan | performed |
| brief | `handoff/current/research_brief_86.28.md`, 41,652 chars, independently read |
| artifact cross-check | all 7 claimed sources present in the brief, 0 missing |

Brief: `handoff/current/research_brief_86.28.md`.

### Findings that decide the design

**F1 -- Silent substitution is endorsed by NO source.** Protocol design
allows exactly two dispositions when a caller names a capability the
implementation does not provide:

- **Fail closed** -- TLS `inappropriate_fallback` (RFC 7507 §3), HTTP
  `Expect`/417, LDAP control criticality.
- **Proceed with a machine-readable signal IN THE RESPONSE** -- RFC 7240
  `Prefer` is ignorable *only because* `Preference-Applied` exists (§3).

**The deciding variable is whether the caller can detect the substitution
from the response** -- not the severity of the difference. RFC 9413 §6:
hiding consequences conceals bugs; §4.1: entrenchment.

**F2 -- "A note in the agent prompt is payload, not response."** This is
precisely the shipped defect: `tierDefaulted` (`research-gate.js:150`) is
surfaced only inside the agent PROMPT at `:173` and never appears in the
return value at `:490-504`. The caller cannot detect the substitution.

**F3 -- ABSENT and UNSUPPORTED are different in kind.** `:150` collapses
them into a single `tierDefaulted` flag, which additionally makes the
`:173` prompt string *factually false* for `tier:'deep'` (it says
"NOT passed by the caller" when the caller did pass one).

**F4 -- A SECOND invisibility, not previously identified.**
`ENVELOPE_SCHEMA.tier` at `:214` derives its `enum` from `VALID_TIERS`,
and **`enum` is NOT stripped on the wire** (unlike `minimum`/`minItems`).
An honest `'deep'` is therefore unrepresentable in the envelope -- the
same `const: true` trap the header warns about at `:25-27`, in a
different place.

**F5 -- EviBound**: prompt-level self-reflection yields 100% false
completion claims; a post-hoc artifact gate yields 0%, at +8.3% runtime.
Direct support for corroborating self-reported fields against the brief.

**F6 -- EBTE / Proof-or-Stop**: demote an unverifiable PROCESS claim to
*non-authorizing* rather than faking a proxy for it. **Structural is not
semantic.** This is a caution against over-reading the checks added here.

---

## 2. Hypothesis

The gate's own thesis (`research-gate.js:29-31`) is that a self-report is
recorded and the script RECOMPUTES the real value against the artifact.
Three places violate that thesis:

1. A caller-named tier the rail does not implement is silently replaced,
   so the gate certifies at MODERATE standards while the caller believes
   DEEP standards (>=20 sources, >=1 adversarial source, multi-pass
   structure -- `researcher.md:206-273`) were applied.
2. `urls_collected` (`:307`) and `recency_scan_performed` (`:310`) are
   compared against nothing but themselves.
3. The docs describe a tool surface the code does not ship.

Fixing 1 and 2 makes the gate's behaviour match its stated contract.
Fixing 3 is free.

**Disposition chosen for the unsupported tier: FAIL CLOSED, plus report
the fields.** Justification from F1: the deciding variable is caller
detectability, and both dispositions are legitimate -- but `tier` is not
an ignorable hint here, it *defines what "passed" means* (deep = 20
sources, not 5). Certifying `gate_passed: true` against a standard that
was never applied is an over-claim by the gate itself. Fail-closed breaks
no existing caller: **zero callers pass `deep` today** (`grep -c deep
research-gate.js` = 0). ABSENT tier keeps today's behaviour exactly.

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. the checker is GREEN with strictly MORE checks than the pre-change baseline of 40 passed / 0 failed (measured 2026-08-10 before any edit); the new total is stated and no pre-existing check is deleted or weakened to accommodate a new one
2. an UNSUPPORTED tier is reported rather than silently applied: a caller passing a tier this rail does not implement gets that fact in the RETURN VALUE, not only in the agent prompt, and the gate does not certify as though the requested standard had been met. An ABSENT tier still defaults to moderate as today -- the two cases must be distinguishable in the output
3. 'deep' is NOT added to VALID_TIERS and no producer fan-out is implemented; the divergence between researcher.md's documented deep tier and the rail that does not implement it is DISCLOSED in the handoff for an operator decision rather than resolved unilaterally
4. recency_scan_performed and urls_collected are corroborated against the brief on disk via the EXISTING stage-2 verifier, and an over-claim on either produces a violation. Fail-closed behaviour on an absent or non-object verification is preserved unchanged
5. each new check has its own MUTANT in scripts/qa/verify_research_gate_workflow.mjs proving it can FAIL: the mutant must be shown KILLED, and the mutation output is recorded verbatim. A check whose mutant is not demonstrated is not counted as delivered
6. coverage.dry and opts.floors are left untouched, with the reason recorded in the handoff -- dry is not establishable from a file, and floors has no consumer until the tier decision is made
7. the doc drift is fixed so .claude/agents/researcher.md, CLAUDE.md and the shipped code agree on agentType:'researcher', including the self-contradiction inside CLAUDE.md's own section
8. no rider trap is broken: model stays 'opus', no Monitor/watchdog, no internal research-to-re-grade loop, zero static imports, exactly one export, no `minimum`/`minItems` in the schema, gate_passed never const:true -- all still asserted GREEN by the checker's structural section
9. a LIVE spawn of the gate is exercised at least once after the change (the checker's own header records that `node --check` passes on scripts that cannot run), and the live result is recorded in the live_check

**Verification command** (immutable):
`node scripts/qa/verify_research_gate_workflow.mjs`

**live_check**: `live_check_86.28.md` containing the verbatim pre-change
baseline and post-change checker output, verbatim mutation output showing
each NEW mutant KILLED, the return value of a live gate spawn showing the
tier fields, and the deep-tier divergence disclosure.

---

## 4. Plan

**P1 -- Split ABSENT from UNSUPPORTED** (`research-gate.js:147-150`).
Replace the single `tierDefaulted` with an explicit three-way
classification mirroring the file's own args-boundary idiom at `:77-96`:

- `tier_absent` -- no tier passed. Default to `moderate`. **No violation**
  (today's behaviour, preserved).
- `tier_unsupported` -- a tier was named that `VALID_TIERS` does not
  contain. Default to `moderate` for the run, **and raise a violation** so
  the gate cannot certify the unrequested standard.
- otherwise supported.

**P2 -- Report it in the RESPONSE** (`:490-504`). Add `tier_requested`,
`tier_applied`, `tier_supported` to the returned object, so a caller can
detect the substitution without reading a brief. This is the
`Preference-Applied` pattern from F1. Fix the `:173` prompt string so it
no longer asserts "NOT passed by the caller" when a tier WAS passed.

**P3 -- Corroborate the two checkable self-reports** via the EXISTING
stage-2 verifier (no new agent, no new spawn):

- Extend `BRIEF_VERIFICATION_SCHEMA` with `recency_section_present`
  (boolean) and `distinct_urls_in_brief` (integer).
- Extend the stage-2 prompt to report both, factually.
- In `enforceGate`: `recency_scan_performed === true` while the brief
  carries no recency section is an over-claim -> violation.
  `urls_collected` exceeding the distinct URLs observable in the brief is
  an over-claim -> violation.
- **Naming discipline per F6**: these checks are STRUCTURAL. The check
  names and messages must say what they actually establish (a section
  heading exists; a URL count is not contradicted), never imply the scan
  was substantively performed.

**P4 -- Preserve fail-closed.** When `verification` is absent or not a
plain object, the new checks must not run and must not soften the
existing fail-closed violation at `:336-340`.

**P5 -- Mutants** in `scripts/qa/verify_research_gate_workflow.mjs`: one
per new check, each demonstrated KILLED. Per project doctrine a guard
that has not been observed failing is not a guard.

**P6 -- Docs**: `researcher.md:75` and `CLAUDE.md:272` ->
`agentType:'researcher'`, resolving CLAUDE.md's internal contradiction.

**P7 -- Live spawn** after the change (criterion 9) -- `node --check`
green does not mean launchable (measured 2026-08-09).

### Explicitly NOT doing

- **Not** adding `'deep'` to `VALID_TIERS`. `researcher.md:248-263`
  defines deep's fourth requirement as a multi-subagent producer fork
  ("2-3 parallel deep-tier researcher subagents", "~1 Claude Max 5-hour
  rolling window per subagent"). Enabling it ships producer fan-out onto
  an N=1 artifact rail and pre-empts an unresolved operator decision.
- **Not** touching `coverage.dry` (`:315`). Per F6 and the adversarial
  finding behind this step: "dry" is K consecutive executed search rounds
  with no new findings -- a property of executed discovery, not of a file.
  No read-only file check can establish it. Faking a proxy is the exact
  anti-pattern EBTE warns against.
- **Not** wiring `opts.floors` (`:269`). Zero callers pass it; its only
  consumer would be tier-aware floors, which depend on the deep decision.
- **Not** changing the envelope `tier` enum (F4). The agent reports the
  tier it ACTUALLY operated at, which is always a supported value, so the
  enum is not lying. The requested-vs-applied distinction is the SCRIPT's
  to report, and P2 puts it there. F4 is disclosed, not actioned.

---

## 5. References

- `handoff/current/research_brief_86.28.md` (this step's brief, gate PASSED)
- RFC 9413 (§4.1 entrenchment, §6 hiding consequences); RFC 7240 §3
  (`Prefer` / `Preference-Applied`); RFC 7507 §3 (`inappropriate_fallback`)
- EviBound (artifact gate 0% vs prompt self-reflection 100% false completion)
- EBTE / Proof-or-Stop (demote unverifiable process claims; structural != semantic)
- `.claude/workflows/research-gate.js` (`:77-96` args-boundary idiom,
  `:147-150` tier handling, `:268-363` enforceGate, `:424-467` stage 2)
- `.claude/agents/researcher.md:204,206-273` (deep tier; `:248-263` fork)
- `.claude/rules/research-gate.md` (mandatory recency-scan section)
- `scripts/qa/verify_research_gate_workflow.mjs` (40-check baseline)
- Audit run `wf_d61fef3b-25c` (origin of the defect list; report-only)
