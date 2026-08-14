# live_check — 86.84: the Layer-3 rail drop is TURN-BUDGET EXHAUSTION

Measured 2026-08-14, 17:00–17:05Z (19:00–19:05 CEST), by Main, on this machine.
Re-runnable form of every number below: `python3 scripts/qa/rail_turn_cap.py --verify`.

## 0. The question the operator asked

> Read the 46 records (`~/.claude/projects/*/workflows/wf_*.json`, `error` field):
> did the agent run out of turns, end with text instead of the tool, or hit a
> tool-availability problem?

**Answer: it ran out of turns.** Not sometimes — on every single observed drop,
with no exceptions in either role.

The other two hypotheses are refuted by the same data:

- **"ended with text instead of the tool" — REFUTED.** 0 of 48 dropped
  transcripts end on an assistant text turn. All 48 end on a `tool_result`.
  **This tail shape is NOT by itself diagnostic** and I nearly misread it as
  one: 393 of 394 *successful* qa/researcher transcripts end on a `tool_result`
  too — the difference is only *which* tool. In a success the last `tool_use` is
  `StructuredOutput`; in a drop it is Bash (37), Edit (4), Write (2), Read (2),
  WebFetch (1), WebSearch (1).
- **"tool-availability problem" — REFUTED.** The same `agentType` emits the tool
  fine: `StructuredOutput` appears as a `tool_use` block in **1257 of 1277**
  completed spawns against **1 of 48** dropped ones. The tool is present and
  callable; the agent never reaches the turn in which it would be called.

## 1. The measurement

Population rule, stated beside the ratios rather than left implicit: one row per
`workflowProgress` entry with `type == "workflow_agent"` across every
`*/workflows/wf_*.json` record under this project's `~/.claude/projects/` tree
(572 records, 1325 spawns, 0 transcripts missing). **Turns = distinct
`requestId` over `type == "assistant"` lines** of that spawn's
`subagents/workflows/<runId>/agent-<agentId>.jsonl` — one requestId is one API
round-trip, which is what a turn is. **Dropped = the run's named `status` field
== `failed`**, never a scan of the record (a run record embeds the dispatched
workflow SOURCE, and both workflow files quote the drop string in comments, so a
blob predicate matches itself — that trap produced 38 phantom drops out of 81
once, commit `f88f8190`, and is not reopened here). **Cap = `maxTurns:` parsed
from the YAML frontmatter block of `.claude/agents/<type>.md` only** — qa.md
also discusses maxTurns in body prose and that line must not be read as a pin.

```
  agentType           cap     n  drop  @cap  >cap  ok p50  ok max  ok@cap
  Explore               -   263     0     0     0       7      56       0
  None                  -   414     0     0     0       9      93       0
  claude-code-guide     -     1     0     0     0       7       7       0
  general-purpose       -   252     0     0     0      12      63       0
  qa                   30   302    39    39     0      18      30       6
  researcher           40    93     9     9     0      23      40       3

Turn counts observed on dropped spawns, per role:
  qa                 cap=30  observed=[30]
  researcher         cap=40  observed=[40]
```

Read the `observed=` line carefully: it is the **set** of distinct turn counts
seen on dropped spawns. For `qa` that set is `{30}` — all 39 drops, no other
value. For `researcher` it is `{40}` — all 9. Not "near the cap", not "mostly".
Exactly at it, every time.

- `.claude/agents/qa.md:6` — `maxTurns: 30`
- `.claude/agents/researcher.md:6` — `maxTurns: 40`
- `general-purpose`, `Explore` and the default workflow subagent (`None`) carry
  **no** `maxTurns` frontmatter. They reach 63, 56 and **93** turns respectively
  and have dropped **0 times in 930 spawns**.

The mechanism: the subagent spends its final permitted turn on ordinary work.
The runtime's in-conversation nudge fires, but there is no turn left in which
the schema call could be emitted, so the run dies with the tokens spent and
nothing returned.

## 2. The previously-reported MODEL split is confounded, and this is why it held up

`scripts/qa/rail_drop_rate.py` and the twin comment blocks in both workflow
files report the rate splitting by model — `claude-opus-5[1m]` 11.4%,
`claude-fable-5` 3.0%, `claude-opus-4-8[1m]` 0.0% — and conclude "the mechanism
is UNPROVEN: size, wall-clock, effort and the documented preamble-suppression
trigger were each tested and refuted."

Those refutations were all correct. The model attribution was not, because model
and agentType are near-collinear in this corpus:

```
  claude-opus-4-8[1m]   Explore=0/24, general-purpose=0/223, qa=0/9, researcher=0/2
  claude-opus-5[1m]     Explore=0/109, None=0/289, general-purpose=0/19,
                        qa=39/290, researcher=8/89
```

**223 of the 258 `claude-opus-4-8[1m]` spawns were uncapped `general-purpose`** —
a type that has never dropped on any model. Its clean 0.0% measures what it ran,
not what it is. Holding the model fixed at `claude-opus-5[1m]`, the separation is
total: 47/379 on the two capped roles, **0/417** on the three uncapped ones.

The `claude-opus-4-8[1m]` qa cell is 0/9 — too small to say anything about the
model, and I am not claiming it does.

Every other refuted hypothesis is also *consistent* with turn exhaustion, which
is why none of them pointed here:

- **Prompt size — correctly refuted.** A lean prompt does not reduce how many
  turns an investigation needs. The operator re-confirmed this on 2026-08-14
  with a lean prompt that still dropped; that observation stands and this
  diagnosis explains it.
- **Stochastic across byte-identical scripts — explained.** phase-86.81 found
  eight byte-identical script versions producing both outcomes, the largest
  dropping 17 and completing 179. Whether a given evaluation happens to need 28
  or 31 turns is exactly the kind of thing that varies run to run at a fixed
  script.
- **Retry helps — explained.** A retry is a fresh turn budget.

## 3. Controls — because a separation this clean is also what a broken probe produces

- **C1, turn counter is alive:** 1325 of 1325 spawns return a positive turn
  count; **0** transcripts have assistant lines but zero counted turns.
- **C2, the cap is a real ceiling and not an artefact of my counter:** **0**
  capped spawns of *any* outcome exceed their cap. Derived from the completed
  population, about which the hypothesis predicts nothing — so it is not a
  control built from the pattern it tests. If the counter were inflating turns,
  successful spawns would breach the cap too.
- **Detector positive control:** `StructuredOutput` emitted by **1257/1277**
  completed spawns vs **1/48** dropped. The zero side is not vacuous.
- **Cardinality floors with no opt-out** in `--verify` (≥200 spawns, ≥5 drops,
  ≥1 capped type), so the check cannot pass over an empty or truncated corpus.

## 4. This is a RECURRENCE of a defect phase-59.1 already fixed once

- `.claude/agents/qa.md:15` — "maxTurns 30 (phase-59.1): the old 12 cap caused
  mid-evaluation stalls (20-26 tool-uses per evaluation); 30 gives headroom."
- `.claude/agents/researcher.md:16` — "maxTurns 40 (phase-59.1): complex briefs
  hit the old 30 cap mid-write; 40 gives headroom."

Same failure, same roles, answered by raising the cap to a number the workload
has since outgrown. And `qa.md` actively pushes the agent *into* the cap: its
"Verification budget" bullet says "your real bound is maxTurns … Depth is the
point — do not truncate verification to chase a clock."

**So a fix that only raises the number again is on a clock.** Worse, the number
cannot be sized from this data: the capped roles' turn distribution is
**right-censored** at the cap. The observed `qa` median of 18 is a censored
median; the tail beyond 30 was never observed, only truncated. Choosing 45 or 60
from these percentiles repeats exactly the inference that produced 30.

## 5. What is NOT established

- That the cap is the **whole** cause. Proven: exhaustion is **necessary** on
  every observed drop (48/48) and no uncapped spawn has dropped in 930 tries. A
  second mechanism that only fires at the cap is not excluded.
- The right replacement cap, for the censoring reason above. Sizing needs an
  uncensored sample (raise it, re-measure) or a mechanism that does not depend
  on guessing the tail — e.g. reserving a terminal turn for the schema call, or
  moving these roles onto the uncapped default workflow subagent. Which of those
  Claude Code actually supports is the open question the 86.84 research gate is
  running against; **it was still in flight at session freeze and no remedy has
  been chosen or applied.**
- ~~Whether the Agent-tool path degrades gracefully at the same cap.~~
  **RETRACTED before this artifact was ever acted on.** I had written that a
  non-schema spawn hitting maxTurns probably still returns its partial text
  while a schema spawn returns nothing, labelled as a hypothesis. The 86.84
  research gate **refuted it from the docs the same hour**: the agent-loop
  result-subtype table gives `error_max_turns` → *"`result` field available?
  **No**"*, and the one documented partial-return path applies when "a rate
  limit, overload, or server error cuts off a subagent that already produced
  text output" — an **API error, not a turn-limit stop**. So there is no partial
  output to salvage at the cap on **either** path, and the operator's measured
  "rail 0-for-4, Agent-tool 3-for-3" must have another explanation (most likely
  that the Agent-tool spawns simply finished inside 30 turns). The replacement
  claim, not the original, is what stands.

Three findings from that gate change the REMEDY and are recorded here because
they arrived before freeze (`handoff/current/research_brief_86.84.md`,
`brief_status: INCOMPLETE`, 7 sources read in full, gate NOT passed):

- **`maxTurns` counts tool-use turns only, and `StructuredOutput` is itself a
  tool call** — so emitting the schema *costs a turn*. The budget must be
  `work_turns + 1`. **A cap sized to the work is a cap that cannot terminate.**
  This makes the remedy arithmetic rather than a guess, and it is the strongest
  argument yet against simply picking a bigger number.
- **The documented default for an absent `maxTurns` is "No limit"** — a real
  absence of a cap, not a high default. That corroborates the 0/930 measured
  above from the vendor side.
- **The throw may not be catchable at script level** (issue #65500, OPEN), which
  is adversarial to the phase-86.81 retry loop already shipped in both workflow
  files. Not yet verified against the version in use here.

## 6. Scope discipline

No agent `.md` was edited. No cap was changed. No gate was loosened. The only
file added is the read-only measurement script
`scripts/qa/rail_turn_cap.py`, which writes nothing and reads only Claude Code's
own run records and transcripts.
