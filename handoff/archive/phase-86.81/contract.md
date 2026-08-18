# Contract — step 86.81

**Step:** 86.81 — *the StructuredOutput drop retry has never once executed, and the
metric built to measure it counts its own source code*
**Priority:** P1 · **Harness required:** true
**Research gate:** PASSED — `wf_e03da94d-d14`, 12 checks, 0 violations, 6 sources read
in full (floor 5), 30 URLs (floor 10), recency scan present, brief
`handoff/current/research_brief_86.81.md` (43,262 chars) declares `brief_status:
COMPLETE`. Cycle 1 (`wf_c6f105c8-52f`) FAILED on a single enforced over-claim
(`urls_collected=26` vs 25 distinct URLs in the brief) and was re-run, not overridden.

---

## Research-gate summary

Cited per claim; full brief at `handoff/current/research_brief_86.81.md`.

**External (6 read in full):**

- **R1 — the vendor guarantee does not cover this failure.** Constrained decoding
  guarantees the *shape* of emitted output, not that emission happens: *"Claude may
  call the tool first (tool_use) or respond with JSON (text)"*
  (https://platform.claude.com/docs/en/build-with-claude/structured-outputs). Non-emission
  is undocumented and carries no vendor retry guidance. **So a retry is the caller's
  job, and this step is not working around a bug that Anthropic will fix for us.**
- **R2 — retry math must not assume independence, in either direction.**
  ReliabilityBench (https://arxiv.org/html/2601.06112): *"Under independence,
  pass^k=(pass¹)^k, but stochastic coupling often causes deviations"* — Gemini 2.0
  Flash pass² 91.04% vs 93.86% predicted, while GPT-4o lands essentially *at*
  independence. **Therefore every `p²`/`p³` figure in the shipped comments is an upper
  bound on benefit, never a forecast**, and the only honest effectiveness number is a
  measured conditional rate `P(drop on attempt 2 | drop on attempt 1)` — of which this
  repo currently has **zero** observations.
- **R3 — fault injection must be deterministic and must target the shipped path.**
  AgentChaos (https://arxiv.org/html/2608.06790): *"All modification functions are
  deterministic given same configuration and response"*.
- **R4 — trigger verification is mandatory.** AgentChaos §4.4 filters tasks where the
  fault never fired, so an unfaulted run cannot be scored as a recovery. Direct
  analogue of this repo's own `feedback_mutation_probe_must_discriminate`.
- **R5 — recovery is four rates, not one.** MAS-FIRE (https://arxiv.org/html/2602.19843v1)
  measures `L_f = 100%` (local fix worked) alongside `S_f > 61%` (task succeeded) on the
  same faults. **A perfect retry can coexist with a large task-level loss**, so "the
  retry fired" must never be reported as "the harness is fixed".
- **R6 — error-recovery code is the least-tested code, and mutation testing is the
  proof.** EMSE 2021 (https://ar5iv.labs.arxiv.org/html/2105.00500), 12,331 mutants:
  survival is highest for operators that *delete or divert a throw* — CRE ~85%,
  CBD ~84%, **TSD ~75%**, FBD 59% — versus CBR 100% for replacing a catch body.
  **Mapped onto `qa-verdict.js`, the highest-risk mutant is TSD on the `throw e` in the
  non-drop branch: the mutant that silently retries a real bug.** That cell is
  prioritised and must be killed by a NAMED assertion.
- **R7 — bound the retry and never nest them unbudgeted.** Google SRE
  (https://sre.google/sre-book/addressing-cascading-failures/): *"Avoid amplifying
  retries by issuing retries at multiple levels: a single request at the highest layer
  may produce a number of attempts as large as the product of the number of attempts at
  each layer."*

**Internal (9 files inspected) — the load-bearing measurements:**

- **I-1 / I-1c — the retry has provably never run.** Every run record embeds the
  dispatched source in its `script` field, so this is measured, not inferred:
  `agentRetryingDrops` is absent from **0/566** dispatched scripts, and **zero
  `qa-verdict` runs have STARTED** since the fix landed at `10:15:17Z`.
- **I-1b — the stale-code class is `Workflow({name})`, not `scriptPath`.** Three NAME
  dispatches carried a byte-identical 18,321-char script matching commit `2e40e8c7`
  (up to **8h36m** stale), while a `scriptPath` dispatch **62 seconds after** one of
  them carried 22,961 chars. `scriptPath` picked up a fresh commit in **88 s** in one
  case and **102 s** in another. **This corrects a cycle-1 error in the brief and an
  error of my own**: both were caused by reading the record's `timestamp` (completion)
  as the launch instant. The two drops *started* `10:10:26Z`/`10:10:48Z`, ~5 minutes
  **before** the fix, so they carried pre-retry code correctly.
- **I-2 — the RETRIED metric is blind exactly where it is read.** `logs` is empty on
  **44 of 44** dropped runs; the channel is otherwise alive (`'gate passed ('` in 43
  records). So a **recovered** run's retry is observable and an **exhausted** run's
  attempts are not — a limitation that must be printed, not hidden.
- **I-3 — the before/after split is wrong twice**: wrong granularity (date, not
  instant) *and* wrong field (`timestamp`, not `startTime`).
- **I-4 — a latent self-match survives** in the `exhausted` predicate's second
  disjunct; the DROP string occurs in the `script` field of 31 records.
- **I-5 — the proving technique already exists in-repo** at
  `verify_escalation_86_78.mjs:52-76` and needs no invention.
- **I-6 — two nested retry levels already exist, unbudgeted.** Claude Code's runtime
  already retries stalled agents (`retrying (1/5)`), and `agentRetryingDrops` multiplies
  on top: up to **5 × 2 = 10** attempts on one evaluation at ~175–195K tokens each.

---

## Hypothesis

Falsifiable, and testable by the step's verification command:

> The retry logic shipped in `agentRetryingDrops` and in the two `research-gate` stage
> loops is **correct but unexercised and unmeasurable**. Therefore: (a) driving the real
> functions with deterministic fault injection will show all three required behaviours
> without any change to their logic; (b) the shipped `rail_drop_rate.py` will report a
> retry for a run that never retried, and zero for a run that did; and (c) after the
> fix, the reader will report retries only from the `logs` channel and split on
> `startTime`, with the exhausted-run blindness printed rather than concealed.

If (a) fails, the retry has a real defect and this step becomes a fix rather than a
proof. If (b) does not reproduce, my reading of `:62` is wrong and C5 cannot be claimed.

---

## Success criteria — verbatim from `.claude/masterplan.json`

Copied without edit; each annotated with the research-backed threshold it will be held
to. **The criteria themselves are immutable and are not restated in my own words.**

1. *"the retry is proven to EXECUTE rather than merely to exist: drive the REAL
   agentRetryingDrops extracted byte-for-byte from the shipped
   .claude/workflows/qa-verdict.js -- never a hand-copy that can drift -- with an
   injected agent that throws, and show all three behaviours: a drop followed by a
   success RETURNS the recovered value, a drop on every attempt THROWS, and a NON-drop
   error surfaces on the first attempt with no retry"*
   → threshold from **R3** (deterministic injection) and **I-5** (extraction technique
   already in-repo). Extraction is brace-matching over the shipped file, with the
   parameter-list skip that `opts = {}` broke on its first use.
2. *"both research-gate stage loops are covered by the same driven proof, or the step
   states plainly and specifically which loop is not covered and why -- the two
   workflows use different retry shapes (a function in one, inline for-loops in the
   other) and a checker that silently covers only the easy one is the defect this step
   exists to prevent"*
   → stage 1 is **already covered** by `verify_research_gate_workflow.mjs`
   (`dropsOnceThenSucceeds`, three named cells; control observed GREEN at 124/0 before
   any change this session). Stage 2 has **no** cell today and gets one. Prior art is
   extended, not rebuilt.
3. *"a LIVE drive on the real Workflow rail forces a drop and shows the retry firing:
   quote the run record's own logs array showing the retry log line, and a recovered
   return on a later attempt. If a forced drop cannot be produced on the live rail, say
   so plainly, state exactly what was attempted, and do not present the deterministic
   proof as if it were the live one"*
   → threshold from **R4**: the live drive must prove the fault actually fired, and an
   unfaulted run may not be scored as a recovery.
4. *"every check is mutation-tested with its control observed GREEN first and a
   byte-identical restore afterwards, and the matrix includes at minimum: deleting the
   msg.includes drop-string guard, reducing maxAttempts to 1, and replacing the retry
   call with a bare agent() call -- each must turn a NAMED check red, and a mutant that
   survives is reported rather than quietly dropped"*
   → threshold from **R6**: the operator ranking says TSD-on-`throw` is the most
   survival-prone mutant, so it is added **beyond** the three the criterion names.
5. *"the RETRIED metric is corrected and the contamination is DEMONSTRATED, not merely
   described: show the current reader reporting a retry for a run that never retried,
   then show the corrected reader reporting zero for that same run. The corrected count
   must read the record's logs array only, never the embedded script, and must match
   BOTH workflows' retry log shapes"*
   → **three** log shapes exist, not two: `qa-verdict: StructuredOutput DROP on
   attempt`, `STAGE-1 RAIL DROPPED (attempt`, `STAGE-2 brief-verify failed (attempt`.
   Matching "both workflows" therefore means matching all three sites.
6. *"the exhausted predicate is classified from the error field alone, and the
   before/after comparison splits on the fix TIMESTAMP rather than the date -- with the
   corrected reader showing 0 post-fix runs where it currently shows 18"*
   → **DISCLOSURE, and it is mine to own:** the parenthetical counts in this criterion
   were a snapshot taken when I filed the step, and I froze a moving number — the exact
   `feedback_immutable_criteria_must_be_green_able` trap. The shipped reader now shows
   **19**, not 18, and the true post-fix count is no longer 0 because *this step's own
   research-gate runs started after the fix*. The criterion's **substance** — error
   field only, split on the fix instant — is satisfiable and will be demonstrated
   exactly. Its **numbers** will be reported as measured today alongside the values that
   held when it was written, with the drift named. I am not editing the criterion.
7. *"the retracted figures do not survive anywhere in the shipped tree: sweep for
   '21.8', '4.8%', '53.4' and '4x amplification', report the hit list with file:line,
   and show it empty afterwards -- a correction must REPLACE the superseded text, not
   sit beside it"*
   → the sweep must **not** delete the two legitimate *retraction statements*
   (`qa-verdict.js:371`, `research-gate.js:679`), which quote the retracted figures in
   order to forbid them. Distinguishing those from survivals is part of the criterion.
8. *"reachability is settled from evidence rather than from documentation: state whether
   a production launch carries the retry, name the launch form that guarantees it, and
   make the docs and the shipped comment agree with what was measured -- ending with two
   contradictory claims live is a fail"*
   → answered by **I-1b** with a 62-second A/B and three commit-dated pickups.
9. *"no verdict semantics move: show that an exhausted retry still produces NO VERDICT
   on the Q/A rail and gate_passed false on the research rail, and that no combination
   of retry outcomes can manufacture a PASS"*
   → threshold from **R5**: `L_f` (the retry worked) must never be reported as `S_f`
   (the step is fine).

---

## Design

Absolute paths.

**New — `/Users/ford/.openclaw/workspace/pyfinagent/scripts/qa/verify_rail_retry.mjs`**
The step's verification command. House style from `verify_escalation_86_78.mjs`: named
`check()` assertions, an expected-count cardinality floor with no opt-out, and a
`PYFIN_*_OVERRIDE` env seam so the mutation matrix never writes the tracked file.
Sections:
- **A** — drives the REAL `agentRetryingDrops`, brace-matched out of the shipped
  `qa-verdict.js` and imported through a temp module that injects its two free
  variables (`agent`, `log`) via a factory wrapper, so the function source stays
  byte-identical. Cells: recovery, exhaustion, non-drop-error passthrough, happy path,
  attempt counting, and `maxAttempts` **read off the shipped source** rather than
  guessed.
- **B** — asserts the call site actually *uses* the wrapper (a correct function that
  nothing calls is the `feedback_guards_stop_one_seam_short` failure).
- **C** — drives `research-gate.js` **stage 2** by wrapping the whole workflow body in
  `export async function __drive(args, phase, log, agent)`, the same technique
  `verify_research_gate_workflow.mjs:78-91` already uses, with an agent stub keyed on
  `agentType === 'Explore'`.
- **D** — delegates stage 1 to the existing `verify_research_gate_workflow.mjs` and
  requires exit 0, so prior art is asserted rather than duplicated.
- **E** — metric cells: a synthetic record whose `script` contains the retry literal and
  whose `logs` are empty must count **0** retries; a record with a real retry log line
  in `logs` must count **1**; a failed record whose `script` quotes the DROP string but
  whose `error` does not must **not** be classified exhausted.

**New — `/Users/ford/.openclaw/workspace/pyfinagent/scripts/qa/mutation_matrix_86_81.mjs`**
Cells M1 delete the `msg.includes` guard, M2 `maxAttempts = 1`, M3 replace the retry
call with a bare `agent()`, M4 **TSD on `throw e`** (R6's highest-survival operator),
M5 TSD on the exhaustion `throw lastErr`. Green control first; sha256 of the tracked
file compared before and after; anchor uniqueness checked; each kill must name the
assertion that caught it.

**Modified — `/Users/ford/.openclaw/workspace/pyfinagent/scripts/qa/rail_drop_rate.py`**
`exhausted` from the `error` field alone (drop I-4's latent disjunct); `retries` from
`logs` only, matching all three shapes; split on `startTime` against the fix **instant**
`2026-08-14T10:15:17Z`; print the exhausted-run blindness (I-2) and the R2 caveat that
`p²` is an upper bound; remove the retracted `21.8% -> 4.8%` NOTE and the retracted
`~1 run in 5` / `39 times and completed 34` docstring figures.

**Modified — the retracted-figure survivals** at `research-gate.js:667`,
`qa-verdict.js:387`, `verify_research_gate_workflow.mjs:497`.

**Modified — the reachability doc contradiction** in `qa-verdict.js` (the
snapshot-semantics comment) and `CLAUDE.md`'s harness-protocol section, to match I-1b.

**Live drive** — a probe workflow written to the **scratchpad**, never to
`.claude/workflows/` (a stray copy in the dispatch directory was the subject of commit
`f237bb8d`), launched by `scriptPath`, carrying `agentRetryingDrops` **byte-identical**
via a generator that copies it out of the shipped file.

---

## Anti-patterns guarded

1. **A probe that matches its own source or documentation** (repo: commit `f88f8190`,
   38 phantom drops; memory `feedback_a_probe_can_match_its_own_documentation`). Guarded
   by cell **E**: the metric must read `logs`, and a record whose `script` contains the
   literal while `logs` is empty must count zero.
2. **A cell that survives because the control and the mutant's fail-safe answer
   coincide** (R4; memory `feedback_mutation_probe_must_discriminate`). Guarded by
   requiring each kill to name the assertion that caught it, and by observing the
   control GREEN first.
3. **A guard that stops one seam short** (memory `feedback_guards_stop_one_seam_short`).
   Guarded by cell **B**: the function being correct is not the property; the call site
   using it is.
4. **Reporting `L_f` as `S_f`** (R5). Guarded by C9's cells and by reporting the retry's
   yield as a *recovery* count, never as a claim that drops are solved.
5. **A vacuous checker** (memory `feedback_zero_assertion_guard_passes_vacuously`).
   Guarded by a cardinality floor with no opt-out.

---

## Out of scope

- **The CAUSE of the drop.** The mechanism is unproven and four hypotheses are already
  refuted. A fifth guess is not what this step buys.
- **Pinning the gate roles to `claude-opus-4-8`** to take the rate to zero — explicitly
  rejected by the operator on 2026-08-14 and recorded as a rider-trap in the workflow.
- **I-6's nested-retry budget.** Real (up to 10 attempts × ~185K tokens) and newly
  measured, but it is a *budget* change, not a proof that the retry works. It will be
  **filed as its own step**, per `feedback_queue_discovered_defects_in_masterplan`.
- **86.79's attempt-counter interaction** with a retried attempt writing a second
  write-first record. Named as an open question in the step notes; not settled here.
- Any change to `.claude/agents/qa.md` (separation of duties — the review queue is
  already 6+ Main-authored edits deep).

---

## Risk — what can still go wrong after this step passes

- **The live drive may not be able to force a drop.** Forcing an agent to end without
  emitting the tool is an instruction the runtime nudges against. If it cannot be
  forced, C3's escape clause applies and I report that plainly rather than dressing the
  deterministic proof as a live one.
- **A proven retry is not a measured retry.** Per R2 and R5, the *effectiveness* number
  still requires real second attempts on real drops, of which there are zero. This step
  can prove the mechanism fires; it **cannot** report a recovery rate, and claiming one
  would be the exact overreach R5 warns about.
- **Exhausted-run blindness is not fixed, only disclosed.** I-2 is a property of what
  the runtime persists; the reader can print the limitation but cannot see through it.
- **The retry increases token spend on the failing subset** and interacts with an
  already-unbudgeted runtime retry (I-6). Bounded here only by `maxAttempts`.
