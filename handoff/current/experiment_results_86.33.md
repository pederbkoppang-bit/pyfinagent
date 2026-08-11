# Experiment results -- step 86.33

**Step**: `86.33` (phase-86, P2) | **Phase**: GENERATE | **Date**: 2026-08-11
**Driver**: Main (`pyfinagent-06`) | **Contract**: `10a703db` (written BEFORE any code)

**NO BEHAVIOURAL CHANGE WAS SHIPPED.** The only edit to the guard is log-only, and
the reason is criterion 5 -- see §6.

---

## 1. Criterion 1 -- population DERIVED by a committed script

`scripts/qa/derive_agent_type_population_86_33.py`, re-runnable, reads
`handoff/logs/qa_write_guard.log`.

**72 distinct `agent_type` values.** A qa-role prefix match covers 34 of them and
misses 37, including `general-purpose` (24 rows), `quality-auditor` (11),
`workflow-subagent` (82) and 34 researcher spellings.

### The script's first revision was wrong in this step's own characteristic way

It carried a hand-written path list to separate synthetic rows from real ones, under
a docstring claiming the list was *"derived by reading the prover, not guessed"*.
**That claim was false** -- written from memory of a console output -- and the list
was incomplete, so it reported **85%** where the gate measured **100%**.

Deriving the list properly made it worse. The prover's CASES table targets **8**
paths and **none** is `/tmp/evil.md` or `../../../etc/x`, which also appear in the
log (from `/T/qa_guard_probe_*/g.sh`, 2026-08-10). Several prover targets are paths
real agents legitimately write.

**So the synthetic/real split cannot be derived from this log at all**, because a
harness fabricates every field the log records -- including `agent_type`. That is
this step's central finding one level up: the field cannot carry authorization
because the caller chooses it, and for the same reason the log cannot certify its
own population. The script now reports coverage **unfiltered** and names what it
cannot establish.

**Three measurements went wrong here before it settled**: mine at 24/68, mine at
23/33 (70%), and the script's own 85%. Each quoted a rate whose population rule was
unsound.

## 2. Criterion 2 -- the question is now ANSWERABLE; the real answer is PENDING

The guard read four fields and logged only those, so **the log could never say
whether the payload carries any other identity field**. Every identity conclusion in
86.31 and 86.33 rested on `agent_type` alone because nothing else was recorded.

`.claude/hooks/qa-write-guard.sh` now records `sorted(payload.keys())` --
**keys only, never values**, since a value could carry file content or prompt text
and this log is committed. **LOG-ONLY**, exactly like the P0's `agent_id` leg.

Driving the real hook confirms the mechanism:

```
payload_keys: ['agent_id','agent_type','cwd','hook_event_name',
               'permission_mode','session_id','tool_input','tool_name']
```

> **THIS IS NOT YET THE ANSWER, AND I WILL NOT PRESENT IT AS ONE.** That key set is
> what **I** put in a synthetic payload -- it reports my own input back to me. The
> criterion demands the INSTALLED platform's payload, which only a **real spawn**
> produces. The Q/A spawned to grade this step supplies the first genuine key set,
> and the verdict cycle should read it from the log rather than from here.

The gate's documentary answer is **NO**: the docs say `agent_type` **is** the
definition's `name`, but **2 definitions exist against 72 logged values**, so
invocation labels occupy the same field. `agent_id` is uniformly present on real
subagent writes but is an instance IDENTIFIER, not a role ATTRIBUTE.

## 3. Criterion 3 -- the researcher rail is UNBROKEN, driven not reasoned

`scripts/qa/prove_researcher_rail_unbroken_86_33.py`: **34 spellings derived from
the log, all ALLOW**, plus a control (`qa` -> `backend/main.py`) that must BLOCK and
does (`rc=2`).

**The control is load-bearing**: without it the entire run would pass against a
guard replaced by `exit 0`. Both population rules are reported because they disagree
(`startswith('research')` = 31; adding `res-*` = 34) and the **wider** one is used
for the assertion -- if a spelling is a researcher and the guard blocks it, the rail
is broken regardless of prefix convention.

## 4. Criterion 4 + 6 -- mutation matrix, 3 cells, ALL KILLED

```
[control]  guard parses (bash -n)        GREEN
           qa write-separation prover    GREEN
           researcher rail prover        GREEN
           payload key set recorded      GREEN

M1 revert the 86.31 widening to exact 'qa'   -> KILLED  (separation RED)
M2 stop recording the payload key set        -> KILLED  (keyset RED)
M3 inject ONE apostrophe into the body       -> KILLED  (ALL FOUR RED)

[restore] byte-identical: True
```

**M3 is criterion 6 made concrete.** One apostrophe terminates the bash
single-quoted block and the guard degrades to **allow-everything, silently**. My own
addition contains zero apostrophes and says so in a comment aimed at the next editor.

### The control-green-first rule caught MY OWN broken check

The first `keyset_recorded()` read subprocess stderr -- but the hook routes stderr
into the log file (`2>>"$GUARD_LOG"` at `:132`), so it was **RED against a healthy
guard**. The matrix aborted. **Had it not, all three cells would have scored against
a broken instrument and I would have reported a clean 3/3 built on nothing.** My M1
anchor was also a guess (`_is_qa_role`); the real predicate is
`is_qa_role(agent_type) and tool_name in ("Write", "Edit")` at `:124`, and the
anchor-miss guard reported it rather than silently mutating nothing.

## 5. Criterion 6 -- the quoting trap, protected and proven

The guard parses (`bash -n` -> `guard-parses`, the step's immutable command). My
edit adds zero apostrophes inside the single-quoted body and carries an explicit
warning for future editors. M3 proves the trap is **detected**, not merely
documented.

**One correction to the gate, with evidence.** It reported *"32 SyntaxError traces
prove the fail-open arm fired"*. Measured: **5 `SyntaxError` + 1 `NameError` + 1
`Traceback`**, all in two clusters timestamped **2026-08-10T20:16 and T20:29**,
adjacent to `/T/qa_guard_probe_z02c0bak/g.sh` and `pythonNOPE3: command not found`
-- a deliberately broken interpreter name. **These are artifacts of a deliberate
probe, not evidence of silent production breakage.** The vulnerability is real and
was demonstrated on purpose. That changes the evidence, not the requirement.

## 6. Criterion 5 -- **OPERATOR ASK #6. NO FAIL-CLOSED CHANGE SHIPPED.**

The criterion: *"If the chosen direction makes the hook fail-CLOSED for any input it
currently allows, that is recorded as a numbered operator ask and NOT shipped in
this step."*

**Every sound fix in the literature keys on an attribute the caller cannot choose,
and this payload has none.** Anderson requires a tamper-proof reference monitor;
Saltzer & Schroeder require fail-safe defaults; NIST 800-162 requires attributes
from an authority; SPIFFE/SPIRE attest process properties; Kubernetes **mints**
tokens; AWS uses a trust anchor. In every case identity is minted by the platform,
never asserted by the caller.

**Widening the prefix cannot work** and this is measured, not argued:
`general-purpose` already evades it and **wrote `evaluator_critique_82.5/.7`** --
and it was **this repo's own former pin** (`qa-verdict.js:207` still records
*"agentType 'qa' (was 'general-purpose')"*). Renaming the bypass is not closing it.

**ASK #6 -- choose the direction, because both have a real cost:**

1. **Fail-closed on unknown `agent_type`** -- deny Write/Edit unless the identity is
   on a known-good list. **Cost: it will break any new legitimate spawn shape
   silently, and the researcher rail is what preserved a 76KB brief and three Q/A
   evaluations today.**
2. **Platform-enforced `disallowedTools` on the qa agent** -- the strongest fix, but
   the gate found `qa.md`'s `memory: project` **re-injects Write by design**, so
   this fights the agent definition itself.
3. **Accept that the guard is a CONVENTION CHECK, not a boundary** -- document it as
   such, keep it fail-open, and rely on it for accident-prevention only.

**I am not choosing.** Option 1 changes behaviour on the live write path, option 2
needs an agent-definition change I am separated-of-duties from, and option 3 is a
downgrade of a stated security property. All three are yours.

## 7. What is NOT claimed

- **Not** that the guard is now secure. It is not; `agent_type` remains
  caller-chosen and I demonstrated the bypass on myself twice.
- **Not** that criterion 2 is fully answered -- the real key set needs a real spawn.
- **Not** that the fail-open arm fired in production; the traces are probe artifacts.
