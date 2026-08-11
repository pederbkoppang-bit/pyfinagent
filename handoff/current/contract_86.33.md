# Contract -- step 86.33

**Step**: `86.33` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~16:0x CEST, read from `date`) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** No production file is modified at this moment.

---

## 1. Research gate

**PASSED** -- `wf_883e1c4a-56a`, tier `moderate`. Script-enforced and recomputed:
**12 sources read in full** (floor 5), **22 URLs** (floor 10), recency scan present,
**all 12 claimed URLs verified in the brief on disk**, `brief_status: COMPLETE`,
`rail_dropped: null`, 9 internal files inspected.

Sources read in full: Anderson 1972 (reference monitor), Saltzer & Schroeder 1975,
NIST SP 800-162 (ABAC), Claude Code hooks / sub-agents / security docs, SPIFFE spec
+ SPIRE concepts, Kubernetes admission controllers + service accounts, RFC 9700,
AWS IAM Roles Anywhere.

## 2. THE CENTRAL QUESTION (criterion 2) IS ANSWERED: **NO**

**On this installed Claude Code version, the PreToolUse payload cannot distinguish
the subagent TYPE from the caller-chosen NAME.**

The docs say `agent_type` **is** the definition's `name` field -- *"identity comes
only from the name frontmatter field"*. But **only 2 agent definitions exist against
72 distinct logged `agent_type` values**, so invocation labels occupy the same field
as definition names. There is no second field carrying the type.

`agent_id` is uniformly present on real subagent writes (**63/63**, and **0/77** of
Main's) but it is an **instance IDENTIFIER, not a role ATTRIBUTE**: 17 chars,
`a`+16hex, 18 distinct values, none shared across roles, joining to nothing
authoritative. It can say *"a subagent did this"*; it cannot say *"which role"*.

The literature is unanimous that this settles it. Anderson's reference monitor must
be **tamper-proof**; Saltzer & Schroeder require **fail-safe defaults**; NIST 800-162
requires attributes issued by **an authority**; SPIFFE/SPIRE **attest process
properties**; Kubernetes **mints** ServiceAccount tokens; AWS uses a **trust
anchor**. In every case the identity is minted by the platform, never asserted by
the caller.

### And I demonstrated the bypass on myself, twice, by accident

`agent_type` is simply whatever the caller passes as `agentType`. My own scripts pin
`'qa'` and `'researcher'`; `qa-verdict.js:207` still records *"agentType 'qa' (was
'general-purpose')"*, so **`general-purpose` -- one of the two values this step was
filed about -- was MY OWN FORMER PIN**, and it wrote `evaluator_critique_82.5/.7`.

More pointedly: **my own prover
(`scripts/qa/prove_qa_write_separation_86_31.py`) fabricates
`agent_type: "qa"` at will.** It produced 24 log rows I then mis-counted as real
subagent traffic, twice -- see §5. A field a test script sets freely is precisely a
field that cannot carry authorization.

## 3. I CONTRADICT THE GATE ON ONE SEVERITY CLAIM

The gate reports **"32 SyntaxError traces prove the fail-open arm fired"**. I
measured the log myself and that overstates it:

```
lines matching 'SyntaxError'  : 5
lines matching 'NameError'    : 1
lines matching 'Traceback'    : 1
```

All six sit in two clusters timestamped **2026-08-10T20:16 and T20:29** -- yesterday
-- immediately adjacent to `/T/qa_guard_probe_z02c0bak/g.sh` and `pythonNOPE3:
command not found`, a deliberately broken interpreter name.

**These are artifacts of a deliberate liveness probe, not evidence that the guard
silently broke in production.** The vulnerability is real and was demonstrated on
purpose; it was not an unnoticed outage. The guard parses clean now
(`bash -n` -> YES).

**This does NOT weaken criterion 6** -- the fail-open quoting trap is still a live
design hazard and still needs protecting. It changes the *evidence*, not the
requirement.

## 4. Immutable success criteria -- VERBATIM

Copied from `verification.success_criteria`; carried here in the contract because
the 86.32 cycle-1 Q/A raised their absence as a harness-compliance finding.

> 1. The agent_type population is RE-DERIVED from handoff/logs/qa_write_guard.log by a committed script, not transcribed from this step text, and the derivation is re-runnable
> 2. Whether the PreToolUse payload can distinguish the subagent TYPE from the spawn NAME on the INSTALLED Claude Code version is MEASURED by a probe that drives the real hook, not inferred from documentation
> 3. The researcher rail's write-first is proven UNBROKEN by the change: drive the guard with the real researcher identities from the log (researcher, research-*, res-*) and show every one still writes
> 4. The fix is MUTATION-TESTED in scripts/qa/mutation_matrix_86_31.py or a successor: reverting it must turn a NAMED assertion red, and the matrix must run control-green-first
> 5. If the chosen direction makes the hook fail-CLOSED for any input it currently allows, that is recorded as a numbered operator ask and NOT shipped in this step
> 6. The guard's embedded python is protected against the FAIL-OPEN quoting trap: the hook body lives inside a bash single-quoted block, and one apostrophe silently disables the whole guard

## 5. TWO MEASUREMENT ERRORS OF MINE, ALREADY CORRECTED IN THE ARTIFACTS

Recorded here because both are about *this step's* evidence and both were mine.

**(a) "24 of 68 post-P0 `qa` rows lack `agent_id`"** -- wrong. Those 24 target
`/tmp/evil.md`, `../../../etc/x`, `backend/main.py`, `qa/MEMORY.md`: **my own
prover**, not real spawns. The gate's 63/63 is correct.

**(b) "23 of 33 role-typed rows (70%)"** -- same artifact, same filter, and I left
it standing two paragraphs below the correction for (a) until a second pass caught
it. All 10 of its "missing" rows are prover rows too. **The correct figure is
100%.**

**Root cause, single and reused:** I filtered on `agent_type == 'qa'` without asking
whether a row came from a real spawn or from my own harness -- which is *this step's
own central finding* about that field, committed against myself while measuring it.

## 6. Plan

**P1 -- CRITERION 1: a COMMITTED, RE-RUNNABLE derivation script.** Not a one-off
command in an artifact. It must emit the full distinct `agent_type` set with counts
and split on the P0 boundary (`8a9a4293`, 2026-08-11T07:55:14 UTC), **and it must
separate real spawns from prover rows**, because conflating them is exactly the
error in §5.

**P2 -- CRITERION 2: a PROBE THAT DRIVES THE REAL HOOK.** The criterion says
measured, not inferred from docs. The gate's answer is documentary plus log-shaped;
I will drive the installed hook and record what the payload actually contains.

**P3 -- CRITERION 3 BEFORE ANY CHANGE: prove the researcher rail unbroken.** Derived
population, rule stated: `startswith('research')` gives **826 events / 31
spellings**; adding `res-*` gives **872 / 34**. The gate reported 821/31 -- close
but not identical, and I will not quote either without its rule.

**P4 -- CRITERION 5 IS THE LIKELY OUTCOME, AND I WILL NOT SHIP PAST IT.** Every
sound fix in the literature keys on an attribute the caller cannot choose, and
**this payload has none**. So the honest direction is fail-closed, which by
criterion 5 must become **a numbered operator ask rather than a shipped change** --
particularly because the gate found `qa.md`'s `memory: project` **re-injects Write
by design**, so a platform `disallowedTools` fix fights the agent definition itself.

**P5 -- CRITERION 6: protect the quoting trap, and mutation-test it.** A single
apostrophe inside the bash single-quoted block disables the entire guard silently.
That is the failure mode with the worst blast radius here.

### Explicitly NOT doing

- **Not** shipping a fail-closed change (criterion 5 forbids it without the ask).
- **Not** widening the prefix match -- the gate shows it cannot work
  (`quality-auditor`, `general-purpose` already evade it).
- **Not** editing `.claude/agents/qa.md` or its `memory: project` injection.

### Risk

The subject is the guard in front of every subagent filesystem write. **A
fail-closed mistake breaks the researcher rail's write-first**, which is the
mechanism that saved a 76KB brief and three Q/A evaluations today. That is why
criterion 5 exists and why P4 routes to an ask.

## 7. References

- `handoff/current/research_brief_86.33.md` (gate `wf_883e1c4a-56a`)
- `handoff/current/measurement_86.33_agent_type_is_caller_chosen.md`,
  `measurement_86.33_agent_id_runtime.md` (both corrected)
- Anderson 1972; Saltzer & Schroeder 1975; NIST SP 800-162; SPIFFE/SPIRE;
  Kubernetes admission + service accounts; RFC 9700; AWS Roles Anywhere;
  Claude Code hooks/sub-agents/security docs
- `.claude/hooks/qa-write-guard.sh`; `.claude/workflows/qa-verdict.js:207,232`;
  `.claude/workflows/research-gate.js:670`
