# live_check — 86.84: the Layer-3 rail drop is TURN-BUDGET EXHAUSTION

**REWRITTEN 2026-08-14 after the cycle-2 Q/A (findings V-1 and V-2).** The prior
revision was written before the remediation and was never brought forward, so it
asserted "No agent `.md` was edited. No cap was changed", quoted `qa.md:6 —
maxTurns: 30` in the present tense, said the gate was still in flight, and
carried three figures that no longer reproduced. All of that is now false and
patching eight fragments would have left a ninth. This file is regenerated from
the live measurement instead.

Re-runnable, both of them:
- `python3 scripts/qa/rail_turn_cap.py --verify` — diagnosis **and** remediation
- `python3 scripts/qa/mutate_rail_turn_cap.py --verify` — the mutation matrix

---

## 0. The question the operator asked

> did the agent run out of turns, end with text instead of the tool, or hit a
> tool-availability problem?

**It ran out of turns** — on every observed drop, with no exceptions in either
role.

- **"ended with text instead of the tool" — REFUTED.** 0 of 48 dropped
  transcripts end on an assistant text turn; all 48 end on a `tool_result`.
  **The tail shape is not diagnostic on its own** and I nearly misread it as
  such: **343 of 343** completed qa/researcher spawns end on a `tool_result`
  too. The difference is only *which* tool — `StructuredOutput` in a success,
  and in a drop Bash (37), Edit (4), Write (2), Read (2), WebFetch (1),
  WebSearch (1). **That is 47 of the 48**; the 48th (`wf_d4e2e794-567`) had
  `StructuredOutput` as its last `tool_use` and is the lone counterexample to
  this sentence's own contrast.
  *(This figure has been wrong twice and the second time is the more
  instructive. It first read **393 of 394** — a mis-scoped enumeration that
  globbed every `agent-*.jsonl` per run directory and swept in the stage-2
  `Explore` spawns `research-gate.js` launches beside the researcher. I then
  corrected it to **347 of 347**, which was still wrong: 347 is the
  `not dropped` population, and that includes 4 spawns from `killed` runs which
  do **not** end on a `tool_result` — they end on a `user` line. So 347/347 is
  false under either reading. The reproducible figure is **343 of 343**
  completed spawns. This is the very `killed`-is-a-third-status defect I had
  already fixed **inside the script** as F4, then carried forward **in prose
  without re-deriving it** — into a file whose own header says it was
  regenerated from the live measurement. Fixing a defect in code does not fix
  the sentences you wrote from the old code.)*
- **"tool-availability problem" — REFUTED.** `StructuredOutput` is emitted by
  **1257 of 1267** completed spawns against **1 of 48** dropped. The tool is
  present and callable; the agent never reaches the turn that would call it.

## 1. The measurement

Population rule, stated beside the ratios: one row per `workflowProgress` entry
with `type == "workflow_agent"` across every `*/workflows/wf_*.json` record under
this project's `~/.claude/projects/` tree — **572 records, 1325 spawns, 0
transcripts missing**. **Turns = distinct `requestId` over `type == "assistant"`
lines** of that spawn's transcript; one requestId is one API round-trip.
**Dropped = the run's named `status` field == `failed`**, never a blob scan (the
record embeds the workflow source, which quotes the drop string — the trap that
produced 38 phantom drops out of 81 in `f88f8190`). **Cap = the `maxTurns` in
force WHEN THE RUN RAN**, not what the file says today; see §4.

```
  agentType           cap     n  drop  @cap  >cap  ok p50  ok max  ok@cap
  Explore               -   263     0     0     0       7      56       0
  None                  -   414     0     0     0       9      93       0
  claude-code-guide     -     1     0     0     0       7       7       0
  general-purpose       -   252     0     0     0      12      63       0
  qa                   30   302    39    39     0      18      30       6
  researcher           40    93     9     9     0      24      40       3

Turn counts observed on dropped spawns, per role:
  qa                 cap=30  observed=[30]
  researcher         cap=40  observed=[40]
```

`observed=` is the **set** of distinct turn counts on dropped spawns: `{30}` for
all 39 qa drops, `{40}` for all 9 researcher drops. Not near the cap — at it.

## 2. The previously-reported MODEL split is confounded

```
  claude-opus-4-8[1m]   Explore=0/24, general-purpose=0/223, qa=0/9, researcher=0/2
  claude-opus-5[1m]     Explore=0/109, None=0/289, general-purpose=0/19,
                        qa=39/290, researcher=8/89
```

**223 of the 258 `claude-opus-4-8[1m]` spawns were uncapped `general-purpose`** —
a type that has never dropped on any model. Its clean 0.0% measures what it ran.
Holding the model fixed at `claude-opus-5[1m]`: **47/379 capped vs 0/417
uncapped**. The `opus-4-8[1m] × qa` cell is 0/9 and proves nothing about the
model; I am not claiming it does.

The four hypotheses refuted by earlier work — prompt size, wall-clock, effort,
preamble-suppression — **stay refuted**, and each is *consistent* with turn
exhaustion. Prompt size does not change how many turns an investigation needs,
which is exactly why the operator's lean-prompt run still dropped. Eight
byte-identical scripts producing both outcomes is what a cap near the workload
median looks like. A retry works because it is a fresh turn budget.

## 3. Controls

- **C1, turn counter alive:** 1325/1325 spawns return a positive turn count; **0**
  transcripts have assistant lines but zero counted turns.
- **C2, the cap is a real ceiling, not an artefact of my counter:** **0** capped
  spawns of any outcome exceed their cap. Derived from the completed population,
  about which the hypothesis predicts nothing.
- **Detector positive control:** 1257/1267 completed vs 1/48 dropped.
- **C3, negative control** (contributed by the cycle-1 Q/A, now computed by the
  script): the 10 spawns in `killed` runs sit at turns
  `[1, 1, 2, 2, 2, 3, 4, 5, 6, 16]` — **0 at a cap**. A termination that is not
  exhaustion lands nowhere near one, which is what stops "at cap" from being a
  generic property of long runs.
- **Cardinality floors with no opt-out** (≥200 spawns, ≥5 drops, ≥1 capped type,
  ≥10 at-risk uncapped spawns).

**At-risk denominator, not the flattering one:** the raw "0 drops in 930 uncapped
spawns" is inflated — only **50** of those 930 ever exceeded 30 turns. The honest
comparison is **0/50 at-risk against a 12.2% capped rate**, and the script prints
that instruction in its own output.

**The at-cap non-emitter population is 49, not 48.** 57 spawns sit at a cap; 49
never emitted `StructuredOutput`; **2 of those are inside runs that COMPLETED**
(`wf_078f4125-57a`, `wf_a6ea31e7-9b9`) because the phase-86.81 retry absorbed
them. Run status is a proxy for the mechanism, and it understates it.
*(The cycle-1 Q/A said 50. It added its 2 to the 48 dropped spawns without
subtracting the one drop it had itself identified as an emitter: (48−1)+2 = 49.
The cycle-2 Q/A independently re-derived 49 and confirmed the correction. Taking
an evaluator's arithmetic on trust is the same failure as taking my own.)*

## 4. The remedy: BOTH CAPS REMOVED

`maxTurns` is **gone** from `.claude/agents/qa.md` (was 30) and
`.claude/agents/researcher.md` (was 40).

**Why removed rather than raised** — phase-59.1 already raised these same caps
for this same failure (`qa.md`: *"the old 12 cap caused mid-evaluation stalls…
30 gives headroom"*; `researcher.md`: *"complex briefs hit the old 30 cap
mid-write"*) and it recurred:

1. **`maxTurns` counts tool-use turns only, and `StructuredOutput` is itself a
   tool call.** The budget must be `work_turns + 1` — a cap sized to the work
   cannot terminate.
2. **Right-censoring.** A run that used exactly N turns under a cap of N proves
   the requirement was **≥N**, never that N sufficed. Both 12→30 and 30→40 were
   fit to a distribution the previous cap had created. The only uncensored
   evidence is the uncapped types, at **63 and 93 turns — both above 40**.
3. **No per-call turn budget exists** in Workflow `agent()` opts, and forcing the
   schema call was requested and **closed as not planned** (#20625) — so
   "reserve the last turn" is not expressible today.
4. **Raising is exposed to #41143** (`maxTurns` silently *not enforced* on the
   Agent-tool path, closed as not planned); removing the key is immune.
5. **`agentType: 'qa'` is unchanged.** Cap and agentType are independent;
   `general-purpose` would re-expand Edit/Write/Bash plus the deferred MCP
   surface phase-75.20 pinned away.

Research gate: `handoff/current/research_brief_86.84.md`, **`brief_status:
COMPLETE`, 11 sources read in full, 19 URLs, recency scan performed,
`gate_passed: true`.** Plan: `handoff/current/contract_86.84.md`.

**The removal broke the verifier first, and that was the right outcome.** The
script had been reading *today's* frontmatter to score *historical* runs, so
removing the caps turned it red ("nothing to test"; 48 drops reclassified as
uncapped). Each run is now scored against the cap in force when it ran, via a
two-entry timeline (`HISTORICAL_CAPS` = qa 30 / researcher 40, `CAP_REMOVED_AT` *(cycle-5 note: constant since renamed `CAP_EDIT_AT` and the boundary is now derived per run from the owning session directory's birth time -- the F-E fix, commit d59cf424)*).
The cycle-2 Q/A corroborated those constants independently **from git history**
(`git rev-list -1 --before=<d>` then parsing the frontmatter: 2026-06-12 onward
reads 30/40 across the whole corpus window), so the hardcode is correct — a
maintainability note, not an honesty problem.

Per **V-7**, `CAP_REMOVED_AT` *(now `CAP_EDIT_AT`; see the cycle-5 note above)* is the **first session after the edit**, not the
file-edit instant: the Agent-tool roster snapshots at session start, so a cap
removed at 17:35Z is still in force for spawns of the session already running.
Using the edit instant would score those as uncapped and, if one exhausted, the
verifier would go red blaming the *diagnosis* rather than the boundary.

**The same command checks the remediation.** Restoring any pin turns
`--verify` red — on **both** Python interpreters on this machine, which is not
where that started. Cycle-3 finding **F-C**: bare `python3` here resolves to
`/usr/bin/python3`, which has **no PyYAML**, so the shipped verification command
was silently taking a regex fallback that read `!!int 30`, `&anchor 30` and
`0x1e` as *uncapped*. My "parse the YAML" fix was not executing under the
command the masterplan freezes. The fallback no longer interprets the value at
all: **any top-level `maxTurns` key with a non-null value is a pin**, because
over-detection can only make the check redder. The output now prints which
parser it used, so the guard's strength is never an undisclosed property of the
interpreter. Verified across 13 shapes on the no-PyYAML path and 22 mutation
cells on both.

**F-E — FIXED 2026-08-14 (cycle-4).** *(The paragraph this replaces described
`CAP_REMOVED_AT = "2026-08-15T00:00:00Z"` as open and unfixed. It is now removed
from the code, so the description is replaced rather than annotated.)*

The retired constant was a **prediction about a future event**, and it was wrong in
**both** directions — the earlier write-up named only one:

* a spawn of the **pre-removal** session running past midnight is still capped by
  its roster snapshot but would score `cap=None` — a drop there reddens `--verify`
  against the **diagnosis** when the real fault is the boundary (the disclosed half);
* a spawn of the **post-removal** session **before** midnight is genuinely uncapped
  but would score against the phase-59.1 pins — so the uncensored sample the whole
  removal exists to produce would be read back as censored evidence. **This half was
  live from 19:27Z tonight and was not previously stated.**

**The fix is structural, not a better constant.** The cap a spawn ran under is a
property of **its session**, not of the wall clock — and sessions overlap, so no
single instant can separate them. `effective_cap()` now takes the run's
**session** rather than a timestamp, and `session_is_post_removal()` decides it
from the **birth time of the session directory owning the run record**. The only
remaining constant, `CAP_EDIT_AT = 2026-08-14T17:37:50Z`, is the commit instant of
`85127353` — a fact that has already happened, not a forecast.

**It populated itself, with no hand-edit** (which is the point — the old design
required someone to remember to "bump this"):

```
before this session's first spawn:
  first uncapped  : NONE ON DISK YET -- ... The realised uncapped turn
                    distribution is NOT YET MEASURABLE.
after it:
  caps removed at : 2026-08-14T17:37:50Z  (commit 85127353)
  first uncapped  : 2026-08-14T19:35:25.339Z  (2 spawn(s) past the boundary)
```

Verified: `rail_turn_cap.py --verify` exit **0** and
`mutate_rail_turn_cap.py --verify` exit **0**, on **both** the venv (PyYAML) and
the bare `python3` (fallback) interpreters — the interpreter trap this file
already records.

**The mutation matrix was retargeted, and that was load-bearing.** Cells M11/M11b/M14
mutated `CAP_REMOVED_AT`. Left alone, `setattr` would have created an attribute
**nothing reads** — three cells silently INERT, the exact "operation that cannot fail
loudly" class. They now mutate `CAP_EDIT_AT` and are **KILLED**. A new cell **M21**
forces `session_is_post_removal` to return `True`; it is **KILLED**, which is what
proves the *derivation* — not merely the constant — is load-bearing. Matrix is now
**22 cells, 0 real survivors**, 3 known/equivalent (M14, M6, M6b) *(cycle-4 capture -- SUPERSEDED. The M14 equivalence in this sentence was adjudicated FALSE by the cycle-4 Q/A once post-removal spawns existed; M14 now KILLS (C2 fires). Current run, cycle 6: **33 cells, 0 real survivors, 2 known survivors BY OUTCOME (M6, M6b)**, kills labelled by mode -- see the cycle-6 block in section 11 and re-derive with `python3 scripts/qa/mutate_rail_turn_cap.py --verify`.)*

## 4b. The re-measurement — status AS OF 2026-08-15, **SUPERSEDED**

*(cycle-5 correction, 2026-08-17: this section's claim is no longer true and the
heading said so in the wrong tense. The uncensored sample is now **n = 47**
post-removal qa/researcher spawns — qa n=36, p50=40, 32 past the old cap, 0
drops, 0 non-emitters — printed re-runnably by `rail_turn_cap.py` itself and
guarded by verify() floors; see §10. The paragraphs below are kept as the
2026-08-15 state of knowledge, and their "n = 2" and "none has occurred yet"
read against that date only.)*

The uncensored sample the removal exists to produce now exists, and it is **n = 2**:

| agent | turns | cap | StructuredOutput | status |
|---|---:|---|---|---|
| `researcher` | **15** | None | emitted | completed |
| `Explore` | **3** | None | emitted | completed |

**0 drops.** Against the right-censored pre-removal contrast:

| role | n | cap | p50 | max | at cap | drops |
|---|---:|---:|---:|---:|---:|---:|
| `qa` | 302 | 30 | 20 | **30** | 45 | 39 |
| `researcher` | 93 | 40 | 25 | **40** | 12 | 9 |

**This does not yet verify the fix, and must not be read as if it does.** A run that
used **15** turns would not have exhausted a **40** cap either — it carries no
information about the cap. Two completed spawns with zero drops is consistent with
the fix working and equally consistent with two lucky short runs; the pre-removal
`researcher` p50 is already 25. `rail_drop_rate.py` cannot help yet either: its
split is on the **retry** commit (10:15:17Z), a different boundary, and it prints
its own refusal — *"only 9 run(s) have LAUNCHED since the fix — too few to call a
rate."*

**What would verify it:** post-boundary spawns that run **past 30 (qa) / 40
(researcher)** turns and still emit `StructuredOutput`. That is the observation the
censored corpus could never contain, and none has occurred yet. Until then the fix
is *reasoned and now correctly instrumented*, not *verified*.

## 5. Mutation matrix — `python3 scripts/qa/mutate_rail_turn_cap.py --verify`

**V-1: this used to exist only as three lines of commit-message prose.** It is
now executable code with a recorded control, per-cell results, and a
byte-identical-restore proof. **22 cells** *(cycle-4 count; 33 as of cycle 6 -- S1-S11 source/injection cells added; run the matrix for the current figure)* (15 at cycle-3; +6 pin-shape cells,
+M21 for the F-E derivation), **control observed GREEN first**,
**0 real survivors**, real tree md5-unchanged (mutations run against a temp
mirror of `.claude/agents`).

```
  M4r   qa pin restored, bare `maxTurns: 30`                          KILLED
  M5r   qa pin restored at a different value, 60                      KILLED
  M9    researcher pin restored alone, 40                             KILLED
  M8    no space after the colon, `maxTurns:30`                       KILLED
  M7c   space before the colon, `maxTurns : 30`                       KILLED
  M7b   pin with a trailing YAML comment, `maxTurns: 30  # restored`  KILLED
  M7    quoted scalar, `maxTurns: "30"`                               KILLED
  M11   CAP_REMOVED_AT moved before the corpus (2026-01-01)           KILLED
  M11b  CAP_REMOVED_AT moved mid-corpus (2026-08-01)                  KILLED
  M12   HISTORICAL_CAPS qa 30 -> 31                                   KILLED
  M12b  HISTORICAL_CAPS qa 30 -> 29                                   KILLED
  M13   HISTORICAL_CAPS researcher 40 -> 41                           KILLED
  M14   CAP_REMOVED_AT moved far future (2027)                        SURVIVED (equivalent)   <- cycle-4 capture; FALSE since post-removal spawns landed. M14 KILLS now (C2: capped spawns exceed their cap); annotation corrected in the matrix at cycle 5
  M6    qa.md deleted entirely                                        SURVIVED (known gap)
  M6b   both agent files deleted                                      SURVIVED (known gap)
```

**M7b and M7 were REAL survivors found by the cycle-2 Q/A (V-5), and they are
fixed.** My first guard matched `^\s*maxTurns\s*:\s*(\d+)\s*$`, so
`maxTurns: 30  # restored` — a live integer pin — read as "all pins removed:
True". That shape is not exotic: every other line of those frontmatter blocks is
a `#` comment, so "restore the pin with a note" is the most likely way it would
come back. The guard now **parses the YAML** rather than pattern-matching the
line, and coerces quoted scalars, because over-detecting a pin can only make the
check redder. My own cycle-1 matrix had two cells and reported 0 survivors; it
was too narrow, not sound.

*(cycle-6 correction: the paragraph below is the cycle-4 reading and its M14
half is WRONG -- exactly the claim the cycle-4 verdict adjudicated false and
the cycle-5 verdict found still standing here. M14 is NOT equivalent any more:
59+ post-removal spawns sit past the 2027 boundary, so moving CAP_EDIT_AT
reclassifies them as capped-era and C2 fires. Only M6/M6b remain known
survivors, by OUTCOME, in the current run.)*

The three surviving cells are labelled rather than dropped: **M14** is
behaviourally equivalent (the whole corpus already precedes any later boundary),
and **M6/M6b** are an accepted absent-subject gap (a vanished `qa.md` breaks the
roster loudly elsewhere).

The harness itself avoids two probe defects, both measured: it re-runs
`collect()` per cell (the cap is resolved at collect time — the cycle-2 Q/A's
first pass reused one cached corpus and *every* timeline cell falsely survived),
and it scores a KILL only on `verify()==False` **with** a problem string, so a
cell that merely errors is recorded as ERROR rather than counted as a kill.

## 6. What is NOT established

- **That the cap is the whole cause.** Exhaustion is proven **necessary** on
  every observed drop (48/48) and no uncapped spawn has dropped in 930 tries. A
  second mechanism that only fires at the cap is not excluded.
- **That the uncapped qa/researcher distribution will look like the uncapped
  distribution measured here.** The 930 uncapped spawns are *different roles with
  different workloads*, so "uncapped agents self-terminate" is **empirical, not
  structural** — the cycle-2 Q/A's caveat, and I am adopting it.

  **Cost tail, corrected after cycle-3 finding F-B.** Turns: p50 9, p90 23,
  p95 32, p99 53, **max 93**. Tokens across the 808 uncapped spawns carrying a
  token figure: p50 80,499, p95 191,887, p99 259,390, **max 416,614**. An
  earlier revision called "~93 turns / ~259K tokens" the *ceiling*; 259,390 is
  the **p99**, and quoting it as a ceiling understated the disclosed tail by
  1.6× — inside the very paragraph that states the residual risk of removing
  the cap. The honest ceiling is **416,614 tokens**. The trade is paying that
  tail on the ~13% of evaluations needing >30 turns instead of losing 100% of
  those tokens, which is still clearly favourable — but it should be argued
  against the real maximum, not a percentile that flatters it.
- **Anything behavioural about the uncapped rail from this session.** The roster
  snapshots at session start, so the removal is committed but **NOT IN FORCE**
  until the next session. Verify with `scripts/qa/verify_qa_roster_live.sh`.
- **The right cap, if anyone ever reinstates one.** Same censoring argument.

**Re-measure next:** the realised turn distribution once uncapped is the
uncensored sample nobody has ever had, and it is what turns this from a reasoned
fix into a verified one.

## 7. Retraction — corrected twice, and the second correction stands

I originally wrote that the Agent-tool path probably degrades gracefully at
maxTurns while the schema path returns nothing, labelled as a hypothesis. I then
**retracted it on the docs** (`error_max_turns` → "result field available? No").
**That retraction was wrong in scope** and is itself withdrawn: the research
gate's reading of the installed 2.1.232 runtime shows the workflow **schema**
branch throws while the **non-schema** branch returns its text unconditionally,
so degradation does exist off the schema path. The doc I cited describes a
different surface.

End state on the degradation question: **degradation exists off the schema path
in the Workflow rail; whether the Agent tool behaves identically at its own cap
is still not directly measured.** That rests on a peer's decompilation of the
installed binary, not on documentation, and should be re-verified before it is
load-bearing.

**But the explanation I attached to it was wrong, and is now measured.** I wrote
that "rail 0-for-4, Agent-tool 3-for-3" was "adequately explained by those
spawns finishing inside 30 turns." **It is not — they did not finish inside 30.**

| dispatch path | role | n | max turns | over the nominal cap |
|---|---|---:|---:|---:|
| **Agent-tool** | qa (cap 30) | 39 | **67** | **24 of 39** |
| **Agent-tool** | researcher (cap 40) | 24 | **59** | 1 of 24 |
| **Workflow** | qa (cap 30) | 302 | 30 | **0 of 302** |
| **Workflow** | researcher (cap 40) | 93 | 40 | **0 of 93** |

**`maxTurns` is enforced on the Workflow rail and is NOT enforced on the
Agent-tool dispatch path.** 24 of 39 Agent-tool `qa` spawns ran past 30 turns,
one reaching 67; not one of 302 Workflow `qa` spawns ever did. This reproduces
anthropics/claude-code **#41143** (closed as not planned) on this machine, and
it is the real reason the Agent-tool fallback keeps working while the rail
drops: **the cap never bound there.**

Three consequences, and the third is the uncomfortable one:

1. It **sharpens the diagnosis** — the cap binds exactly and only where the 48
   drops occurred.
2. It **confirms the remediation is aimed correctly** — removing the pin changes
   behaviour on the Workflow rail and changes nothing on the Agent-tool path,
   which never honoured it.
3. It means **every Agent-tool Q/A in this project has been running uncapped all
   along**, including the three that evaluated this very step (67, 34+, 17
   turns). So the "uncapped agents self-terminate" claim has more support than
   §6 credits — but it is support from a path whose cap was *never* enforced,
   which is not the same as evidence that removing an *enforced* cap is safe.
   §6's caveat stands.

## 8. Scope

Agent files: only the `maxTurns` pin removed — `qa.md`'s body is byte-identical
(45,398 == 45,398, verified by the cycle-2 Q/A), `researcher.md` lost only the
pin line. Both workflow-file diffs are **comment-only**. No threshold, no gate,
no verdict semantics touched. `verify_research_gate_workflow.mjs` 124/0 and
`verify_rail_retry.mjs` 38/0 both green — the latter's section **[F]** is the
executed evidence for criterion 4 (an exhausted retry yields no value, rethrows,
still RECOMPUTES `gate_passed`, and assigns no verdict field), cited here per
**V-10** because it was previously proven but unmapped. None of the concurrent
peer session's files are in any commit.
## Post-boundary verification of the maxTurns removal -- 2026-08-15

**The gap this closes.** The prior evidence was a sample of n=2 at 15 and 3 turns
with 0 drops, which carried NO information: a 15-turn run could never have exhausted
a 40 cap, so it cannot distinguish "the removal worked" from "the cap was never
reached". The verification needed was a post-boundary spawn running PAST the removed
cap that still emits StructuredOutput.

**It exists now, and it was produced as a by-product of grading 86.74.**

```
run          : wf_8c3730a1-32e   (86.74 cycle-7 Q/A)
agentType    : qa                (the role whose cap was 30)
realised     : 61 tool-use turns
last tool    : StructuredOutput  -> the schema call was emitted
tokens       : 236,421   duration 771s
```

**Why this is decisive rather than suggestive.** 61 > 30 by 31 turns. Under the
removed cap this run dies at turn 30 with full token cost and nothing returned --
which is precisely the drop signature 86.84 diagnosed. The run is therefore only
OBSERVABLE because the cap is gone, and it is the first uncensored point of the
realised turn distribution that the cap made unmeasurable.

**Boundary check -- the removal was in force.** The removal landed in
`85127353` at 2026-08-14T19:37:50+02:00; this session started 2026-08-15, after it.
The agent roster snapshots at session start, so it was live for this spawn. Note the
BEHAVIOUR is the stronger proof: a capped spawn cannot exceed its cap, so a 61-turn
`qa` run is itself evidence the cap is absent, independent of reading the file.

**What this does NOT establish.**
1. **n=1 above the boundary.** One uncensored observation proves the cap was the
   binding constraint; it does not characterise the distribution's tail.
2. **It does not discharge 86.84's other criteria** -- in particular the criterion
   that a turn-exhausted spawn must yield NO VERDICT and never a PASS, which needs
   its own executed test and was not run today. *(cycle-5 correction, 2026-08-17:
   discharged since -- `verify_rail_retry.mjs` section [F] 38/38 and
   `verify_research_gate_workflow.mjs` 124/124, both exit 0, run fresh today; §10.)*
3. **It says nothing about `researcher`** (cap was 40). No post-boundary researcher
   spawn ran today.

**Recommendation on "may 86.84 close unverified?" -- still NO, but the reason has
narrowed.** The specific "the removal is not verified" blocker is now DISCHARGED by
the run above. 86.84 should still not close until its remaining criteria are graded
by a fresh Q/A, which has not returned.

---

## §10 — Cycle-4 live evidence (2026-08-17): verification command green on both interpreters, re-measurement re-runnable, matrix green post-edit

Every block below was executed 2026-08-17 in the operator-attended session; commands shown beside their output.

**The step's verification command (both interpreter paths):**

```
$ source .venv/bin/activate && python3 scripts/qa/rail_turn_cap.py --verify
VERIFY: PASS -- controls green, turn-exhaustion claim holds.
(exit 0)
$ /usr/bin/python3 scripts/qa/rail_turn_cap.py --verify      # bare-python3 fallback parser path
VERIFY: PASS -- controls green, turn-exhaustion claim holds.
(exit 0)
```

Pre-fix state the same morning, for the record (external-audit finding D1, root
cause and fix in `experiment_results_86.84.md` §Cycle 4):

```
VERIFY: FAIL
  - no agent type carries a maxTurns cap; nothing to test
(exit 1)
```

**Corrected per-role table (rows were misreporting `cap=-` under the group[0] defect):**

```
  agentType           cap     n  drop  @cap  >cap  ok p50  ok max  ok@cap
  Explore               -   274     0     0     0       7      56       0
  None                  -   415     0     0     0       9      93       0
  general-purpose       -   144     0     0     0      17      63       0
  qa                   30   338    39    39     0      20      60       6
  researcher           40   104     9     9     0      23      40       3
run records read: 580 · agent spawns analysed: 1275 · transcripts missing: 0
```

**The committed re-measurement, now printed by the script itself on every run
(REMEDIATION block; population = post-removal spawns, i.e. runs owned by a
session directory born after CAP_EDIT_AT, per role; percentile rule stated in
the output):**

```
REALISED UNCAPPED TURN DISTRIBUTION (the committed re-measurement --
the uncensored sample; percentile rule: sorted[int(frac*(n-1))]):
  qa           n= 36  dropped=0  non-emitters=0  p50=40  p90=54  max=60  >old-cap(30)=32
  researcher   n= 11  dropped=0  non-emitters=0  p50=19  p90=36  max=38  >old-cap(40)=0
```

**Mutation matrix after the D1 edit (control green first, byte-identical restore):**

```
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify
BYTE-IDENTICAL RESTORE (md5 before == after, real tree never written):
  ok scripts/qa/rail_turn_cap.py  baed6162861ff2d1265eacc40370fb2a
  ok .claude/agents/qa.md  4c9faa6d7eb14aba70eea2fc7f804727
  ok .claude/agents/researcher.md  a9592ee0950e55d24fc3e1bb65d5c26f
cells=22  real survivors=0  known/equivalent survivors=3   [cycle-4 capture; current: cells=33, 2 known BY OUTCOME, kills by mode {VERIFY 27, ORACLE 1, INJECTED_TRUTH 2, MUST_STAY_GREEN 1} -- section 11]
VERIFY: PASS -- control green, 0 real survivors, tree unchanged.
(exit 0)
```

**Criterion-4 executed evidence, fresh run:**

```
$ node scripts/qa/verify_rail_retry.mjs
ALL GREEN: 38 passed, 0 failed
(exit 0)
```

**What §9's "still NO" recommendation was waiting for, resolved:** the specific
remaining ask was "its remaining criteria are graded by a fresh Q/A". A cycle-4
Q/A was spawned on this changed evidence and its verdict is transcribed in
`evaluator_critique_86.84.md` §9. *(cycle-5 edit: this paragraph previously
stated which outcomes were "admissible" -- consequence framing inside graded
evidence, removed on the cycle-4 Q/A's flag.)*


---

## 11. Cycle-6 live evidence (2026-08-17): the cycle-5 FAIL's findings closed

```
$ python3 scripts/qa/rail_turn_cap.py --verify
VERIFY: PASS -- controls green, turn-exhaustion claim holds.        (exit 0, both interpreters)

$ python3 scripts/qa/mutate_rail_turn_cap.py --verify
kills by mode (never pooled): {'VERIFY': 27, 'ORACLE': 1, 'INJECTED_TRUTH': 2, 'MUST_STAY_GREEN': 1}
cells=33  real survivors=0  known/equivalent survivors (BY OUTCOME)=2  errors=0
VERIFY: PASS -- control green, 0 real survivors, outcomes match annotations, tree unchanged.  (exit 0)
```

**What changed since cycle 5, each answering a named cycle-5 finding:**

1. **The killed-status conflation is out of the floor** (Invalid_Precondition):
   `post_removal_turns` rows now carry `killed_n` NAMED, and `non_emitters`
   counts only spawns that ran to completion without emitting -- an operator
   abort can no longer redden the immutable command with a false
   new-loss-mechanism message. Pinned by cell **S11** (killed-run injection,
   source unmutated, verify MUST stay green -- a MUST_STAY_GREEN negative
   control) and the fixed floor still fires on a genuine completed non-emitter
   (S6).
2. **past_old_cap and the per-role sample are now GUARDED, not just reported**
   (illusory-guard): verify() cross-checks the report against an INDEPENDENT
   second derivation computed in the by-type grouping
   (`uncapped_past_hist_cap`, and `n == uncapped_n`), so a single-site
   inversion, a role-filter break, a truncated qa sample, or an emptied
   post-removal set all disagree visibly. Cells: S4 (role break) now dies at
   the cross-check as well as the floor; the qa-only variants the evaluator
   drove are caught by the same disagreement.
3. **The monotone floor has cells** (S8 percentile-reversed, S9 median-as-max
   -- the two mutations the evaluator confirmed would kill).
4. **The third hiding shape has a cell** (S10, non-emitter narrowed to
   dropped-only, caught by the injected-truth assertion).
5. **Kill modes are never pooled** (Overgeneralization): every kill prints its
   mode inline and the summary counts per mode, so an ORACLE detection can
   never read as a shipped-guard kill.
6. **This file's stale matrix claims are corrected at the site** (the
   "report survivors" clause): the 22-cell/M14-equivalent statements above now
   carry their capture cycle and the adjudicated correction.
