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
  such: **347 of 347** completed qa/researcher spawns end on a `tool_result`
  too. The difference is only *which* tool — `StructuredOutput` in a success,
  and in a drop Bash (37), Edit (4), Write (2), Read (2), WebFetch (1),
  WebSearch (1). **That is 47 of the 48**; the 48th (`wf_d4e2e794-567`) had
  `StructuredOutput` as its last `tool_use` and is the lone counterexample to
  this sentence's own contrast.
  *(An earlier revision said "393 of 394". That was a mis-scoped enumeration —
  my script globbed every `agent-*.jsonl` per run directory and swept in the
  stage-2 `Explore` spawns `research-gate.js` launches beside the researcher.
  347/347 is the reproducible figure and it makes the argument stronger.)*
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
two-entry timeline (`HISTORICAL_CAPS` = qa 30 / researcher 40, `CAP_REMOVED_AT`).
The cycle-2 Q/A corroborated those constants independently **from git history**
(`git rev-list -1 --before=<d>` then parsing the frontmatter: 2026-06-12 onward
reads 30/40 across the whole corpus window), so the hardcode is correct — a
maintainability note, not an honesty problem.

Per **V-7**, `CAP_REMOVED_AT` is the **first session after the edit**, not the
file-edit instant: the Agent-tool roster snapshots at session start, so a cap
removed at 17:35Z is still in force for spawns of the session already running.
Using the edit instant would score those as uncapped and, if one exhausted, the
verifier would go red blaming the *diagnosis* rather than the boundary.

**The same command checks the remediation.** Restoring any pin turns
`--verify` red.

## 5. Mutation matrix — `python3 scripts/qa/mutate_rail_turn_cap.py --verify`

**V-1: this used to exist only as three lines of commit-message prose.** It is
now executable code with a recorded control, per-cell results, and a
byte-identical-restore proof. 15 cells, **control observed GREEN first**,
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
  M14   CAP_REMOVED_AT moved far future (2027)                        SURVIVED (equivalent)
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
  structural** — the cycle-2 Q/A's caveat, and I am adopting it. The observed
  uncapped ceiling is ~93 turns / ~259K tokens (p50 9, p90 23, p95 32, p99 53).
  The trade is paying up to ~2–3× tokens on the ~13% of evaluations needing >30
  turns instead of losing 100% of those tokens — favourable, but the qa/researcher
  uncapped tail is genuinely unobserved until it runs.
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

End state: **degradation exists off the schema path in the Workflow rail;
whether the Agent tool behaves identically at its own cap is still not directly
measured**, and "rail 0-for-4, Agent-tool 3-for-3" remains adequately explained
by those spawns finishing inside 30 turns. This rests on a peer's decompilation
of the installed binary, not on documentation, and should be re-verified before
it is load-bearing.

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
