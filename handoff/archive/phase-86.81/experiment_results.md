# experiment_results — step 86.81

**Step:** the StructuredOutput drop retry has never once executed, and the metric built
to measure it counts its own source code.
**Research gate:** PASSED (`wf_e03da94d-d14`, 12 checks, 0 violations). Cycle 1 FAILED
on an over-claim and was re-run, not overridden.
**Verification command:** `node scripts/qa/verify_rail_retry.mjs` → **ALL GREEN: 38
passed, 0 failed**, exit 0.

---

## What the problem actually was

Three distinct things, and only the first is what the operator asked about.

1. **The retry was shipped and had never run.** Commit `6b4df8f9`
   (2026-08-14T10:15:17Z) added `agentRetryingDrops` to `qa-verdict.js` and retry loops
   to both `research-gate.js` stages. Measured across all 566 run records:
   `agentRetryingDrops` appears in **0** dispatched `script` fields, and **zero
   `qa-verdict` runs had launched** since the fix. It parsed; nothing else was known.
   A repo-wide search returned no checker for it.
2. **The metric built to prove it worked was contaminated in both directions.**
   `rail_drop_rate.py:62` counted retries with `blob.count(RETRY_LOG)` over
   `json.dumps(record)` — and a run record **embeds the workflow's own source**, which
   contains that literal once, in the log call itself. So every future `qa-verdict` run
   would have reported a phantom retry. In the other direction, `research-gate`'s two
   retry log shapes were never matched at all.
3. **A launch-form trap could have made the fix unreachable.** `Workflow({name})`
   dispatches a session snapshot of the script.

## How the cause was found

The diagnosis was **not** a new theory of why models drop the call — that mechanism
remains unproven and four hypotheses are already refuted. What was established is that
the failure is **stochastic**, which is what makes a retry the right remedy:

- Population: every `*/workflows/*.json` run record carrying a `runId` (n=565),
  classified as dropped **only** from the record's `error` field → **44/565 = 7.8%**.
  By model: `claude-opus-5[1m]` 40/351 = 11.4%, `claude-fable-5` 4/135 = 3.0%,
  `claude-opus-4-8[1m]` **0/73**.
- An **independent channel** agrees: over 1,271 schema-bearing subagent transcripts
  (dropped iff no `tool_use` block named `StructuredOutput` was ever emitted), 43 of the
  44 error-field drops contain such an agent, and all 44 have a transcript on disk.
- **Stochasticity, measured:** grouping by sha1 of the embedded `script`, **8 distinct
  byte-identical script versions produced BOTH outcomes**; the largest dropped 17 times
  and completed 179. Same bytes, same caller, both results.
- Drops concentrate on the two gate roles: by `agentType`, `qa` 19/280 = 6.8% and
  `researcher` 2/80 = 2.5%, while `Explore` is **0/377** and `general-purpose` 0/245.
  The detector was control-tested on an `Explore` transcript (returns `called=True`, and
  `False` once the `tool_use` name is mutated away), so that zero is measured, not blind.

**Two timing errors were made and corrected during this step, both of the same class.**
I first compared a local-time commit stamp (`+0200`) against UTC run timestamps and
concluded the drops predated the fix. The research gate then made the mirror error,
reading the record's `timestamp` (which is the **completion** instant) as the launch
instant and concluding `scriptPath` had delivered stale code. Re-measured from
`startTime`: both drops **started** at 10:10:26Z / 10:10:48Z, ~5 minutes *before* the
fix existed, so they carried pre-retry code correctly; and the first run to *start*
after the fix picked it up 102 seconds later. The two fields can even disagree on order
— the drop that ended later started earlier.

## What was built

| file | change |
|---|---|
| `scripts/qa/verify_rail_retry.mjs` | **new** — the verification command. 38 checks in 6 sections. |
| `scripts/qa/mutation_matrix_86_81.mjs` | **new** — 6 mutants, control-green-first, sha256-pinned subject. |
| `scripts/qa/gen_live_retry_probe.mjs` | **new** — generates the live forced-drop probe, embedding the retry span verbatim. |
| `scripts/qa/rail_drop_rate.py` | corrected predicates, launch-instant split, `--json`, retracted figures replaced. |
| `.claude/workflows/qa-verdict.js` | comment-only: retracted figures replaced; snapshot semantics corrected. |
| `.claude/workflows/research-gate.js` | comment-only: retracted figures replaced with a pointer to the single retraction notice. |
| `scripts/qa/verify_research_gate_workflow.mjs` | comment-only: retracted figures replaced. |
| `CLAUDE.md` | harness-protocol section: `scriptPath` not `name`; the two snapshot classes separated. |
| `.claude/workflows/qa-verdict.js.export.mjs` | `git rm --cached` — see below. |

**No change was made to any retry's logic.** The retry was correct; it was unproven and
unmeasurable. The only behavioural change in this step is to the reader.

## Criterion-by-criterion

- **C1 — retry proven to execute.** `agentRetryingDrops` is brace-matched out of the
  shipped file (skipping the parameter list, which `maxAttempts = 2` would otherwise
  break, and walking back over `async`) and imported through a factory injecting its two
  free variables. All three behaviours shown: recovery returns the value on attempt 2
  (`A1`, `A1b` calls==2), exhaustion throws (`A2`, `A2c` calls==maxAttempts), a non-drop
  error surfaces on attempt 1 with no retry (`A3`, `A3b` calls==1). `maxAttempts` is
  **read off the shipped source**, not assumed.
- **C2 — both research-gate loops.** Stage 2 had **no** coverage and now has four cells
  (`C1`–`C4`), driven by wrapping the whole workflow body in an async function — the
  house technique. Stage 1 was **already covered** by
  `verify_research_gate_workflow.mjs`; that coverage is **asserted by running it**
  (`D1`, 124/0) rather than duplicated.
- **C3 — live drive.** Forced drop reproduced on the real rail: run `wf_9f387ad8-b5c`,
  `agentCount: 2`, one transcript with no `StructuredOutput` call, run record logs
  `'qa-verdict: StructuredOutput DROP on attempt 1/2 -- retrying'`, `status: completed`,
  marker flipped to `SECOND`. The **first** live attempt was invalid (the agent complied
  instead of dropping) and is reported in full in `live_check_86.81.md` §1, along with a
  false-negative defect in my own probe that the same run exposed.
- **C4 — mutation matrix.** 6/6 killed, control observed GREEN first, subject sha256
  identical before and after. Includes the three the criterion names plus two TSD cells
  chosen from EMSE's survival ranking; the nastiest (`M4`, silently retrying a real bug)
  was killed by a named assertion.
- **C5 — RETRIED corrected and contamination demonstrated.** On one fixture with ground
  truth `retried=1, exhausted=0`: pre-fix reader reports **4 / 1**, corrected reader
  reports **1 / 0**. All three retry log shapes are matched, and `E0` pins each literal
  to both the workflow that emits it and the reader that counts it.
- **C6 — error-field only, launch-instant split.** Done. **The criterion's parenthetical
  numbers had already moved before the work started, and that is my error to own:** I
  froze "0 post-fix runs where it currently shows 18" into an immutable criterion, and
  **both numbers are monotone-increasing counters** — every workflow run this session,
  including this step's own research-gate and Q/A runs, launches after the fix instant
  and increments them. So no point value stated here can stay true; the Q/A read "3" as
  already-stale and measured 5 minutes later. **The invariant that does not rot is
  `post_fix_exhausted == 0`**, and the live source is
  `python3 scripts/qa/rail_drop_rate.py --json`. The criterion's substance —
  error-field-only classification, split on the launch instant `2026-08-14T10:15:17Z`
  rather than the date — is satisfied and demonstrated exactly. The criterion was not
  edited.
- **C7 — retracted figures.** Swept; the only surviving hit is the single retraction
  notice that names them in order to forbid them. `research-gate.js` no longer restates
  them. **This sweep found a real defect:** `.claude/workflows/qa-verdict.js.export.mjs`
  — a duplicate `name: 'qa-verdict'` inside the dispatch directory, carrying all three
  retracted figures — was **committed by the very commit (`f237bb8d`) whose subject says
  it stopped it being committable**. Untracked; left on disk as another session's file.
- **C8 — reachability.** Settled by measurement: `{name}` dispatches ran up to **8h36m**
  stale; `scriptPath` picked up fresh commits in 88 s and 102 s, with a 62-second A/B
  isolating the fault to NAME dispatch. `CLAUDE.md` and the shipped comment were both
  saying the opposite and now say what was measured, separating the **script** snapshot
  (fixed by `scriptPath`) from the **agent-definition** snapshot (a `qa.md` *deletion*
  still needs a restart).
- **C9 — no verdict semantics move.** `F1`–`F4`: exhaustion yields no value at all,
  rethrows the original error, `enforceGate` still recomputes `gate_passed` after the
  loop, and the retry body assigns no verdict or gate field. `M5` proves the checker
  notices if exhaustion is made to return `undefined` instead of throwing.

## What this does NOT establish — read before quoting any of it

- **No effectiveness rate is claimed, and none can be.** Retry math assumes
  independence, which ReliabilityBench refutes in *both* directions (Gemini 2.0 Flash
  measures below independence, GPT-4o essentially at it). Every `p²`/`p³` figure is an
  **upper bound**, never a forecast. The honest number is a measured conditional rate
  `P(drop on attempt 2 | drop on attempt 1)` and this repo has **zero** real second
  attempts on real drops. The retracted arithmetic was removed rather than recomputed.
- **Proving the mechanism fires is `L_f`, not `S_f`** (MAS-FIRE): a 100% local recovery
  rate coexisted with >39% task loss in that paper. "The retry works" must not be read
  as "drops are solved".
- **The cause is still unproven**, deliberately and per the step's scope.
- **Exhausted-run blindness is disclosed, not fixed.** `logs` is empty on **44 of 44**
  dropped runs, so the attempts a lost run burned are not observable. The reader prints
  this rather than implying a zero it cannot see.
- **The live drop was injected by instruction**, not sampled from the wild. It proves
  the runtime raises the error, the wrapper catches it, a second agent is spawned, and
  the value returns. It does not reproduce whatever makes real drops happen.

## Defects found but NOT fixed here — queued

1. **Nested retry amplification (I-6).** Claude Code's runtime already retries stalled
   agents (`retrying (1/5)`) and `agentRetryingDrops` multiplies on top — up to **5 × 2
   = 10** attempts on one evaluation at ~175–195K tokens each, against Google SRE's
   explicit warning about multi-level retries. Newly measured; needs a budget, not a
   proof.
2. **A duplicate `name: 'qa-verdict'` remains on disk** in `.claude/workflows/`.
   Untracked, but deleting another session's working file is the operator's call.
3. **The `research-gate.js` "10.3% on the low-effort Explore path" comment does not
   reproduce.** By `agentType`, `Explore` is **0/377** on the transcript channel. The
   claim is load-bearing twice — it justifies `STAGE2_MAX_ATTEMPTS = 2` *and* it is the
   stated reason the effort hypothesis was rejected. Confound stated honestly: role is
   not effort, since `qa`/`researcher` also carry far larger prompts. Left in place
   pending its own step rather than edited on a confounded reading.
