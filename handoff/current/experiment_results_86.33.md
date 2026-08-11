# Experiment results -- step 86.33

**Step**: `86.33` (phase-86, P2) | **Phase**: GENERATE | **Date**: 2026-08-11
**Driver**: Main (`pyfinagent-06`) | **Contract**: `10a703db` (written BEFORE any code)

**NO BEHAVIOURAL CHANGE WAS SHIPPED.** The only edit to the guard is log-only, and
the reason is criterion 5 -- see §6.

---

## 1. Criterion 1 -- population DERIVED by a committed script

**TWO scripts cover this criterion, and pointing at only one was the cycle-2
blocker.**

* `scripts/qa/census_qa_write_guard_log_86_31.py --before <cutoff>` -- supplies the
  parts criterion 1 explicitly names: the **--before cutoff with its excluded-row
  count**, the **per-identity Write/Edit counts targeting paths outside
  `.claude/agent-memory/qa/`**, and the **breach recall validated against the
  derived class** (20 events across 10 identities). Output transcribed verbatim in
  `live_check_86.33.md` §0.
* `scripts/qa/derive_agent_type_population_86_33.py` -- the full distribution and
  the guard-predicate partition. It produces **none** of the three elements above.

**I wrote the second without checking that the first already existed.** The
covering evidence was in the repo the whole time and my handoff record cited the
wrong script -- the same shape as the cycle-1 finding, where criterion 2's answer
lived only in the log and the verdict.

**AS OF 2026-08-11T18:1x CEST: 78 distinct `agent_type` values**, partitioned by
the guard's own predicate:

```
total distinct                    : 78
  matched by the guard predicate  : 36
  NOT matched                     : 42
  -> 36 + 42 = 78   (EMPTY is counted, not dropped)
```

> **CORRECTED after the cycle-1 Q/A.** An earlier revision of this line read
> *"72 distinct ... covers 34 ... misses 37"* -- and **34 + 37 = 71, not 72.** The
> script's `if t` silently dropped the **EMPTY** `agent_type` from the "NOT matched"
> side while the headline total counted it. EMPTY is the **largest single bucket**
> (2,151 rows: Main-shaped writes), so the partition omitted its biggest member
> without saying so.
>
> **The script also REIMPLEMENTED the guard's predicate** as
> `startswith(("qa-","qa_","QA-","QA_"))` where the guard lowercases first
> (`qa-write-guard.sh:120-121`). They **diverge on `Qa-Mixed`**: the guard MATCHES
> it, my script reported it as evading. Now fixed to use the guard's own form.
>
> **EVERY COUNT FROM THIS LOG IS PERISHABLE.** The log is live and gitignored.
> `quality-auditor` read **11** when I wrote the first draft, **21** when the Q/A
> checked, and **97** when I re-measured an hour later. A frozen figure from a
> growing log is stale on arrival, so counts here are **date-stamped**, and the
> re-runnable script -- not this file -- is the source of truth.

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

## 2. Criterion 2 -- **ANSWERED: NO.** Measured on the installed platform

The guard read four fields and logged only those, so the log could never say whether
the payload carries any other identity field. It now records
`sorted(payload.keys())` -- **keys only, never values** -- **LOG-ONLY**, exactly like
the P0's `agent_id` leg.

**THE MEASUREMENT, produced by a REAL spawn.** My own probe could only echo a
synthetic payload back at me, so I refused to present it as the answer. The cycle-1
Q/A's own `Write` drove the real hook and supplied it (rows
`2026-08-11T12:59:29.967328Z` and `13:06:06.080792Z` in
`handoff/logs/qa_write_guard.log`):

```
REAL SUBAGENT WRITE -- 12 keys:
  agent_id, agent_type, cwd, effort, hook_event_name, permission_mode,
  prompt_id, session_id, tool_input, tool_name, tool_use_id, transcript_path
  with agent_type='qa'  agent_id='afd21026f4056c9e0'

MAIN-SHAPED WRITE -- 10 keys:
  agent_type and agent_id BOTH ABSENT
```

**THE ANSWER IS NO.** The payload carries exactly **one caller-chosen role field**
(`agent_type`) plus **one opaque instance id** (`agent_id`). Nothing separates the
subagent TYPE from the spawn NAME. The four keys my synthetic probe lacked --
`effort`, `prompt_id`, `tool_use_id`, `transcript_path` -- are **none of them role
attributes**.

This corroborates the documentary finding by measurement: the docs say `agent_type`
**is** the definition's `name`, and **2 definitions exist against 65 distinct values in the clean
pre-contamination slice** (the unfiltered figure is 78; **13** distinct values appear ONLY at or after the
2026-08-10T09:30:00Z cutoff, i.e. are prover- or probe-introduced -- and per §1 the
synthetic/real split cannot be soundly derived from this log at all, so no
contamination count here should be leaned on), so invocation labels occupy the same
field.

> **Recorded because it matters procedurally:** the instrument that answered this
> criterion is the Q/A's own write-first record -- the mechanism phase-86.31 shipped
> this morning. The evaluation and the measurement were the same act.

## 3. Criterion 3 -- the researcher rail is UNBROKEN, driven not reasoned

`scripts/qa/prove_researcher_rail_unbroken_86_33.py`: **34 spellings derived from
the log AS OF this run, all ALLOW**, plus a control (`qa` -> `backend/main.py`) that must BLOCK and
does (`rc=2`).

**The control is load-bearing**: without it the entire run would pass against a
guard replaced by `exit 0`. Both population rules are reported because they disagree
(the two rules disagree; the wider gives 34 at the time of this run -- and the log GROWS, so the script re-derives rather than trusting this number) and the **wider** one is used
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
- Criterion 2 **is** now answered (§2) -- the cycle-1 Q/A's own write supplied the
  real payload. An earlier revision of this line said it was not; superseded.
- **Not** that the fail-open arm fired in production; the traces are probe artifacts.
