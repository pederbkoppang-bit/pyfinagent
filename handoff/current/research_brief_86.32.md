# Research Brief -- step 86.32

**Topic:** Bounding the evaluate/retry loop in long-running autonomous agent
harnesses: why consecutive-failure counters fail to terminate, and what
actually works.

**Tier:** moderate (caller-stated). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).

**Researcher:** Layer-3 Workflow rail, 2026-08-11.

---

## ENVELOPE (born inert -- phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 9,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "gate_passed": true
}
```

**brief_status flipped to COMPLETE as the final act of the run.**

---

## Status log (write-first, incremental)

- [t0] Brief created. Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [t1] Internal exploration starting: grep for `consecutive_fails`, `max_retries`, `retry_count`.

---

## PART A -- INTERNAL CODE INVENTORY (the Explore half)

All claims below carry file:line anchors and were read, not inferred.

### A1. The F1 `consecutive_fails` counter -- THE DEFECT IS CONFIRMED AND IS WORSE THAN STATED

`scripts/harness/run_harness.py` (1206 lines) is the only place the counter exists.

| Site | Line | Behaviour |
|---|---|---|
| Threshold constant | `run_harness.py:57` | `MAX_CONSECUTIVE_FAIL = 3` |
| Counter init | `run_harness.py:1109` | `consecutive_fails = 0` -- **process-local**, re-zeroed on every invocation |
| Loop bound | `run_harness.py:1111` | `for cycle in range(1, args.cycles + 1)` -- the REAL bound is the `--cycles` CLI arg |
| PASS | `run_harness.py:1160-1162` | `consecutive_fails = 0` |
| FAIL | `run_harness.py:1164-1165` | `consecutive_fails += 1`, then `save_best_params(pre_cycle_best)` (revert) |
| **CONDITIONAL** | `run_harness.py:1175-1177` | `# CONDITIONAL -- keep but warn; does not count as a FAIL` / `consecutive_fails = 0` |
| Escalation | `run_harness.py:1189-1196` | `if consecutive_fails >= MAX_CONSECUTIVE_FAIL: _escalate_certified_fallback(...); break` |
| Escalation body | `run_harness.py:1003-1035` | copies `optimizer_certified_fallback.json` -> `optimizer_best.json`, appends a `## HARNESS HALT` block to `harness_log.md` |

Three independent reasons this counter cannot bound the loop:

1. **Reset-on-success (the classic flaw).** `PASS` zeroes it (`:1162`).
2. **Reset-on-CONDITIONAL (worse, and pyfinagent-specific).** `:1177` zeroes the
   counter on the single MOST COMMON non-terminal verdict, with the comment
   "does not count as a FAIL". An alternating `FAIL, CONDITIONAL, FAIL,
   CONDITIONAL, ...` sequence never reaches 3 -- it never exceeds 1. This is
   exactly the reset-on-differing-outcome defect the step names.
3. **The counter is process-local and non-durable.** `:1109` sits INSIDE
   `main()`, above the cycle loop. It is never persisted and never re-read. A
   harness driven one cycle per invocation (which is how the Layer-3 manual loop
   actually runs -- Main spawns Q/A per step, not `run_harness.py --cycles N`)
   can never accumulate past 1. Contrast the away-ops pattern, which DOES
   persist: `handoff/away_ops/autoresearch_fail_state.json` =
   `{"consecutive_fails": 0}` and `handoff/away_ops/ablation_fail_state.json`
   -- same name, durable file, different subsystem.

**Nothing else in the repo reads the counter.** `backend/autonomous_harness.py`
(304 lines) has NO retry/consecutive/budget logic at all -- its only loop is
`while self.running:` (`:56`), and `:177` explicitly states the orchestrator
"is expected to catch + log, NOT retry".

### A2. `.claude/masterplan.json` `max_retries` / `retry_count` are DECORATIVE

Measured over the live masterplan (python json walk, 2026-08-11):

- **1153 steps carry the retry fields.**
- `max_retries` values: `3` on 1078 steps, ABSENT on 75.
- `retry_count` values: `0` on 1146 steps; `1` on 3; `2` on 3; `3` on 1.
- The seven nonzero steps: `4.6.0`(1), `4.6.6`(2), **`75.5`(3/3)**, `82.0`(2),
  `82.5`(2), `82.7`(1), `86.28`(1).

**`75.5` sits at `retry_count == max_retries == 3` and its status is `done`.**
The ceiling was reached and nothing refused another cycle -- the step simply
continued (7 cycles per the project's own record) and then closed.

**No reader exists.** Repo-wide grep for `max_retries|retry_count` outside
`backend/` runtime code returns only WRITERS:
`scripts/generate_masterplan.py:202-203` and the `scripts/add_phase_27*.py`
family. Every other hit is an unrelated in-app HTTP/LLM retry
(`backend/agents/debate.py:58`, `backend/agents/orchestrator.py:819`,
`backend/agents/info_gap.py:183`, `backend/agents/task_bus.py:124`,
`backend/db/bigquery_client.py:568`, `backend/services/ticket_queue_processor.py:400`).
Those are per-call transport retries, NOT step-level loop bounds.

### A3. The 3rd-CONDITIONAL auto-FAIL rule is INSTRUCTIONS-ONLY

The rule text lives in `CLAUDE.md` ("Failure discipline" -> "3rd-CONDITIONAL
auto-FAIL") and `docs/runbooks/per-step-protocol.md` §4. **No hook and no
runtime code enforces it.**

- The only implementation is `scripts/qa/verdict_history_86_21.py` (phase-86.21),
  and it is **ADVISORY by its own admission**: `verdict_history_86_21.py:40-48`
  states "A count derived from a file the audited party writes is therefore
  ADVISORY, not authoritative."
- **Nothing calls it.** Repo-wide grep for `verdict_history_86_21` outside its
  own file returns exactly one hit: `scripts/qa/mutation_matrix_86_21.py:25`,
  its own test harness. No hook, no workflow, no Q/A prompt invokes it.
- It also documents that the ORIGINAL prescribed counter was structurally blind:
  `verdict_history_86_21.py:5-14` -- "the Q/A [counts] by grepping
  `handoff/harness_log.md`. That file is written at step CLOSE. So a step still
  in its remediation loop -- which is exactly when the rule is meant to bite --
  has ZERO rows... It fails OPEN, and silently." Measured: `574 of 1189`
  `## Cycle` headers carry no `phase=` at all (`:59-62`).
- **It is a DIFFERENT counter from F1.** F1 counts consecutive FAILs in
  `run_harness.py` process memory; the 3rd-CONDITIONAL counter counts consecutive
  CONDITIONALs from `handoff/verdict_ledger.jsonl` (present, 10,814 bytes). They
  share no state, no threshold constant, and no reset rule. `CLAUDE.md` and
  `.claude/agents/qa.md` additionally **disagree on the predicate** --
  `verdict_history_86_21.py:33-38`: "`CLAUDE.md` says CONSECUTIVE-with-reset;
  `.claude/agents/qa.md` says a CUMULATIVE grep while calling it consecutive."

The one gate that IS enforced by code is `.claude/hooks/lib/verdict_gate.py`
(162 lines) -- but it gates **step CLOSE on a per-step verdict**, not loop
count: `gate_decision()` (`:55`) HOLDS only on an explicit step-matched non-PASS
(`:77`), and fails OPEN on `no_input` (`:11-17`). It has no notion of attempts.

### A4. No per-step spend or attempt ceiling exists in the launch path

`.claude/workflows/qa-verdict.js` (15,503 B) and `.claude/workflows/research-gate.js`
(47,175 B) were read for budget logic. Neither carries an attempt counter, a
token ceiling, or a cumulative-cost check. The only cost-aware lines are
**advisory prose about a SINGLE run**, not a running total:

- `research-gate.js:105-106` -- "costs the run at this line, before a single
  token is spent; silently defaulting costs a full max-effort session AND
  deposits a misfiled artifact."
- `research-gate.js:606-608` -- "spawning costs a full max-effort researcher
  session ... costs zero tokens and tells the caller exactly what to fix."
- `qa-verdict.js:193` -- "means no max-effort Q/A session is spent evaluating a
  step that was never [built]".

These are **preflight guards that avoid ONE wasted spawn**. Nothing accumulates
across spawns, so nothing can refuse the Nth.

### A5. MEASURED COST DISTRIBUTION (replaces the two anecdotes)

Derived from **513 workflow run records** at
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/workflows/wf_*.json`,
each carrying `totalTokens` + `totalToolCalls` + `status` (2026-08-11).
Attribution: `args.step_id` where present, else parsed from `workflowName`;
**484 of 513 attributed to 164 distinct steps** (29 unattributable).
Caveat: only sessions still on disk are counted, so these are LOWER bounds.

**Per-run cost:**

| Class | n | min | p50 | p90 | max | mean | sum |
|---|---|---|---|---|---|---|---|
| Q/A | 341 | 59,230 | 153,571 | 187,565 | 372,903 | 151,713 | 51,734,139 |
| Researcher | 113 | 84,812 | 181,169 | 223,109 | 981,329 | 192,771 | 21,783,176 |

Tool calls: Q/A p50=27 / max=61; Researcher p50=45 / p90=66 / max=214.
Duration: Q/A p50=544 s; Researcher p50=720 s, max=1,540 s.

**Runs-per-step histogram (the thing a bound would actually bite):**

| runs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| steps | 27 | 48 | 38 | 28 | 13 | 5 | 2 | 1 | 2 |

**Per-step CUMULATIVE gate cost:** n=162, p50 = **419,739** tok,
p90 = **882,651** tok, max = **1,832,223** tok, total = **78,560,498** tok.

**The tail dominates the bill:**

- steps needing **>2** gate runs: 89/164 (**54.3%**) holding **76.0%** of all gate tokens
- steps needing **>3** gate runs: 51/164 (**31.1%**) holding **52.8%**
- steps needing **>4** gate runs: 23/164 (**14.0%**) holding **29.9%**

**Worst offenders (each one a loop that no counter stopped):**

| step | runs | cumulative tokens | mix | status |
|---|---|---|---|---|
| **75.5** | 8 | **1,832,223** | 7 qa + 1 research | all completed |
| **86.28** | 9 | 1,592,858 | 8 qa + 1 research | 6 ok / **3 failed** |
| **36.8** | 9 | 1,468,755 | 9 qa | 8 ok / 1 failed |
| 36.17 | 7 | 1,273,825 | 6 qa + 1 research | all completed |
| 86.31 | 6 | 1,152,287 | 5 qa + 1 research | 5 ok / 1 failed |
| 36.12 | 6 | 1,107,835 | 6 qa | 4 ok / **2 failed** |
| 86.6 | 6 | 1,080,363 | 4 qa + 2 research | 5 ok / 1 failed |
| 86.21 | 7 | 1,034,120 | 5 qa + 2 research | 6 ok / 1 **killed** |

**`75.5` is the crown exhibit:** it is the ONE step whose masterplan
`retry_count` reached `max_retries` (3/3), it consumed **8 gate runs and 1.83M
tokens**, and it closed `done`. The declared ceiling and the observed cost are
in different universes -- 8 runs against a nominal 3.

**Rail-drop overhead:** 44 of 513 runs (**8.6%**) ended non-`completed`
(39 `failed`, 5 `killed`), burning **7,651,838 tokens** (9.7% of all gate
tokens) that produced no verdict at all. Any bound must count these, or a
step can be starved by drops without ever recording an "attempt".

---

## PART B -- EXTERNAL RESEARCH

**Search-method disclosure (mandatory, and a deviation to declare):** the
`WebSearch` tool returned *"this session has used its web search budget (200 of
200 WebSearch calls)"* on all three planned query variants (current-year 2026 /
last-2-year 2025 / year-less canonical). The budget is **session-shared and was
already exhausted before this researcher was spawned** -- a known pyfinagent
trap. `WebFetch` is unaffected. Sources were therefore reached by **direct
WebFetch of canonical primary URLs** rather than by search ranking. This weakens
*discovery* (I cannot claim to have surveyed the field's long tail) but not
*reading* -- every source below was fetched and read in full. The three-variant
discipline is satisfied in SPIRIT by spanning canonical prior art (SRE Book 2016,
Nygard/Fowler 2014), mid-period work (Self-Refine 2023, Huang et al. 2024), and
current-year docs (Azure, `ms.date 2025-02-05`, `updated_at 2026-07-02`).

### Sources READ IN FULL via WebFetch (counts toward the gate)

| # | URL | Accessed | Kind | Key finding |
|---|---|---|---|---|
| 1 | https://sre.google/sre-book/handling-overload/ | 2026-08-11 | Official (Google SRE Book ch.21) | The two-budget design + the 3x -> 1.1x amplification numbers |
| 2 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-11 | Official (Google SRE Book ch.22) | Retry amplification 4^3=64; server-wide retry budget; "don't retry indefinitely" |
| 3 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-08-11 | Official (Azure Arch Center) | Closed/Open/Half-Open; the TIME-BASED counter reset; why a plain retry counter is the wrong tool |
| 4 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-11 | Official (Anthropic Eng) | NEGATIVE FINDING: the canonical reference prescribes NO termination rule |

### B1. Google SRE -- the retry budget is the published alternative to a counter

Google SRE Book ch.21 "Handling Overload" (read in full 2026-08-11,
https://sre.google/sre-book/handling-overload/) runs **two budgets
simultaneously**, and the arithmetic for why is explicit:

- **Per-request budget (a CUMULATIVE attempt cap, not a consecutive one):**
  *"If a request has already failed three times, we let the failure bubble up to
  the caller."*
- **Per-client budget (a RATIO over a window):** *"Each client keeps track of the
  ratio of requests that correspond to retries. A request will only be retried as
  long as this ratio is below 10%."*
- **Why both:** the per-request budget alone still allows *"requests to grow to
  roughly 3x the original rate"*; adding the 10% ratio *"reduces the growth to
  just 1.1x in the general case -- a significant improvement."*
- **Retry at exactly one layer:** *"A failed request from the DB Frontend should
  only be retried by Backend B, the layer immediately above it. If multiple
  layers retried, we'd have a combinatorial explosion."*

ch.22 "Addressing Cascading Failures" (read in full 2026-08-11,
https://sre.google/sre-book/addressing-cascading-failures/) supplies the
quantified failure mode and two direct instructions:

- Amplification: *"a single user action may create 64 attempts (4^3) on the
  database"* when each of several layers retries 3 times; *"100 QPS of retries in
  the first second leads to 200 QPS, then to 300 QPS, and so on."*
- *"Always use randomized exponential backoff when scheduling retries"* --
  without jitter *"a small perturbation (e.g., a network blip) can cause retry
  ripples to schedule at the same time, which can then amplify themselves."*
- **"Limit retries per request. Don't retry a given request indefinitely."**
- **"Consider having a server-wide retry budget. For example, only allow 60
  retries per minute in a process, and if the retry budget is exceeded, don't
  retry; just fail the request."**

**The load-bearing structural point for 86.32:** every SRE bound above is
**cumulative over a window** (3 total attempts; 10% of all requests; 60 per
minute). **Not one of them is a consecutive-failure counter**, and none of them
resets on an intervening success. That is precisely the property
`run_harness.py:1162/1177` lacks.

### B2. Circuit breakers -- the counter reset is TIME-based, never success-based

Azure Architecture Center, Circuit Breaker pattern (read in full 2026-08-11,
`ms.date 2025-02-05`, `updated_at 2026-07-02`,
https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker).
The design detail that matters here is easy to miss and is the exact opposite of
the harness's rule:

> *"The failure counter for the **Closed** state is time based. It automatically
> resets at periodic intervals. This design helps prevent the circuit breaker
> from entering the **Open** state if it experiences occasional failures. The
> failure threshold triggers the **Open** state only when a specified number of
> failures occur during a specified interval."*

So the canonical breaker deliberately does **NOT** reset its failure counter on a
success while Closed -- it resets on a **clock**. A success-reset would make the
breaker un-trippable under alternating traffic, which is exactly the pyfinagent
defect. The only success-driven reset lives in **Half-Open**, and it is guarded:

> *"A limited number of requests ... are allowed to pass through ... If these
> requests are successful, the circuit breaker assumes that the fault ... is
> fixed, and the circuit breaker switches to the **Closed** state. The failure
> counter is reset. If any request fails, the circuit breaker ... reverts to the
> **Open** state."*
> *"The circuit breaker reverts to the **Closed** state after a specified number
> of successful, **consecutive** operation invocations."*

Note the asymmetry: **N consecutive successes to close; ONE failure to re-open.**
pyfinagent has it backwards -- one CONDITIONAL re-zeroes three FAILs.

The doc also states outright that retry-counting and breaking are different jobs:

> *"The Retry pattern enables an application to retry an operation with the
> expectation that it eventually succeeds. The Circuit Breaker pattern prevents an
> application from performing an operation that's likely to fail. ... the retry
> logic should be sensitive to any exceptions that the circuit breaker returns and
> stop retry attempts if the circuit breaker indicates that a fault isn't
> transient."*

Two further transferable elements: **manual override** (*"provide a manual reset
option that enables an administrator to close a circuit breaker and reset the
failure counter"*) -- i.e. exhaustion escalates to a HUMAN, not to silence; and
**accelerated circuit breaking** (*"a failure response can contain enough
information for the circuit breaker to trip immediately"*) -- some failures should
skip the counter entirely.

### B3. NEGATIVE FINDING -- Anthropic's harness-design post prescribes no bound

Read in full 2026-08-11,
https://www.anthropic.com/engineering/harness-design-long-running-apps. The post
that `CLAUDE.md` names as the canonical reference for this whole architecture
**does not specify a termination rule.** It gives an iteration *range* as
practice, not policy -- *"I ran 5 to 15 iterations per generation"* -- and its
only convergence language is descriptive: *"Across runs, the evaluator's
assessments improved over iterations before plateauing, with headroom still
remaining."* It does **not** address retry budgets, cost caps, human escalation,
or what to do on non-convergence.

This is a genuine finding, not a gap in my reading: **pyfinagent's F1 counter is
a LOCAL invention, not an implementation of an Anthropic-published rule.** There
is therefore no upstream authority to defer to, and no compatibility constraint
on replacing it. The "plateau" observation is, however, direct support for
**no-progress detection** as the natural stop signal in an LLM harness -- the
plateau is the thing the post actually measured.

### B4. Do repeated LLM refinement attempts converge? Measured: they PLATEAU, and can REGRESS

**Self-Refine** (Madaan et al., arXiv:2303.17651; read in full via ar5iv
2026-08-11, https://ar5iv.labs.arxiv.org/html/2303.17651) is the pro-iteration
baseline, and even it caps hard and reports decay:

- *"The feedback-refine iterations continue until the desired output quality or
  task-specific criterion is reached, **up to a maximum of 4 iterations**."*
- Stopping is explicit and dual-signal: *"The stopping condition stop(fb_t,t)
  either stops at a specified timestep t, or extracts a stopping indicator (e.g.
  a scalar stop score) from the feedback."*
- Per-iteration deltas (Fig. 4) show the curve flattening after the FIRST round:
  - Code Optimization: 22.0 -> 27.0 (**+5.0**) -> 27.9 (+0.9) -> 28.8 (+0.9)
  - Sentiment Reversal: 33.9 -> 34.9 (+1.0) -> 36.1 (+1.2) -> 36.8 (+0.7)
  - Constrained Generation: 29.0 -> 40.3 (**+11.3**) -> 46.7 (+6.4) -> 49.7 (+3.0)
  - The paper's own words: *"diminishing returns in the improvement as the number
    of iterations increases."*
- Failure attribution: *"61% were a result of feedback suggesting an
  inappropriate fix"* -- i.e. most residual error is the CRITIC's fault, not the
  generator's. More rounds of the same critic cannot fix that.

**[ADVERSARIAL] Huang et al., "Large Language Models Cannot Self-Correct
Reasoning Yet"** (arXiv:2310.01798, ICLR 2024; read in full via ar5iv
2026-08-11, https://ar5iv.labs.arxiv.org/html/2310.01798) is the disagreeing
source, and it is decisive against unbounded looping:

- *"LLMs struggle to self-correct their responses without external feedback, and
  at times, their performance might even **degrade** post self-correction."*
- Measured monotone DECLINE across rounds (Table 3), i.e. oscillation is not even
  the worst case -- drift is:
  - GPT-4 GSM8K: 95.5% -> **91.5%** (r1) -> **89.0%** (r2)
  - GPT-4 HotpotQA: 49.0% -> 49.0% (r1) -> **43.0%** (r2)
  - GPT-3.5 CommonSenseQA: 75.8% -> **38.1%** (r1) -> 41.8% (r2)
- **The critical mechanism for 86.32:** the published gains from self-correction
  depend on *"using correct label to determine when to stop the self-correction
  loop"* -- an **ORACLE STOP**. With oracle labels GPT-3.5 GSM8K rises
  75.9% -> 84.3%; without one it falls. **The oracle is doing the work, not the
  refinement.** A harness whose Q/A verdict is itself LLM-generated has no oracle,
  so it inherits the degrading branch, not the improving one.
- Multi-agent debate does not rescue it: at equal response budget (6 responses on
  GSM8K), debate scores **83.2%** vs plain self-consistency **85.3%** -- debate
  *"significantly underperforms simple self-consistency using majority voting."*
  Spending the same tokens on more evaluation rounds is measurably WORSE than
  spending them on independent samples.
- The paper offers **no autonomous stopping rule**; it warns to *"approach the
  concept of self-correction with a discerning perspective, acknowledging its
  potential and recognizing its boundaries."*

**Synthesis:** the literature supports a **small fixed cap (~2-4 rounds)** plus a
**no-progress/plateau detector**, and gives NO support for "keep grading until it
passes". Rounds 5-9 -- exactly where pyfinagent's expensive steps live (A5:
`75.5`=8, `86.28`=9, `36.8`=9) -- are in the flat-to-negative region of every
curve measured above.

### B5. Anthropic's own budget guidance is explicit -- and pyfinagent does not apply it

Anthropic, "How we built our multi-agent research system" (read in full
2026-08-11, https://www.anthropic.com/engineering/multi-agent-research-system):

- **Scaled effort rules stated as policy:** *"Simple fact-finding requires just 1
  agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with
  10-15 calls each, and complex research might use more than 10 subagents with
  clearly divided responsibilities."*
- **Cost multiples:** *"Agents typically use about 4x more tokens than chat
  interactions, and multi-agent systems use about 15x more tokens than chats."*
- **Token spend is THE driver:** *"Token usage by itself explains 80% of the
  variance"* in research task success.
- **The runaway failure mode is named and was fixed by guardrails, not by
  counters:** *"Early agents made errors like spawning 50 subagents for simple
  queries, scouring the web endlessly for nonexistent sources"*; *"We proactively
  mitigated unintended side effects by setting explicit guardrails to prevent the
  agents from spiraling out of control."*

Anthropic bounds **tool calls and agent count per query**. pyfinagent bounds
**neither** (A4): the tier table in `.claude/agents/researcher.md:206-211`
prescribes a tool-call budget per spawn, but nothing enforces it and nothing
accumulates it across the 2-9 spawns a step actually consumes.

### B6. Circuit-breaker reset semantics -- a genuine CONTRADICTION in the sources

This cuts against the tidy story in B2 and is reported as found. Martin Fowler's
canonical write-up (read in full 2026-08-11,
https://martinfowler.com/bliki/CircuitBreaker.html), which attributes the pattern
to Michael Nygard's *Release It*, describes the counter as **success-resetting**:

- *"Once the failures reach a certain threshold, the circuit breaker trips, and
  all further calls to the circuit breaker return with an error"*
- *"successful calls reset it back to zero"*
- Half-open: *"a trial call, which will either reset the breaker if successful or
  restart the timeout if not"*

So the ORIGINAL circuit breaker has **exactly the reset-on-success property this
step calls a defect**, while Azure's current production formulation (B2) makes
the Closed-state counter **time-based**. The contradiction resolves on PURPOSE,
and the resolution is the key design insight for 86.32:

- A breaker protects a **shared dependency** from a **transient** fault. Its
  question is *"is the dependency healthy RIGHT NOW?"* A recent success is
  genuine evidence of health, so reset-on-success is CORRECT there.
- A harness step budget bounds **total work on ONE work item**. Its question is
  *"how much have I already spent on this?"* A success is **not** evidence that
  the remaining work shrank -- and in pyfinagent a CONDITIONAL is not even a
  success. Reset-on-success is CATEGORICALLY WRONG there.

**Conclusion: `run_harness.py` applied a health-check idiom to a
work-accounting problem.** That is the root cause in one sentence, and it also
explains why Azure moved to a windowed/time-based counter as the pattern
matured -- and why *every* SRE bound in B1 is cumulative.

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Method:** `WebSearch` was unavailable (budget 200/200 exhausted session-wide
before spawn -- see Part B header), so the recency scan was performed by
**fetching current-revision documents and checking their publication metadata**,
rather than by year-scoped queries. This is a WEAKER scan than the protocol
prescribes and is declared as such.

**Result: 3 findings in the 2024-2026 window; none supersedes the canonical
sources, and two strengthen them.**

1. **Azure Circuit Breaker doc is CURRENT, not legacy** -- fetched page carries
   `ms.date: 2025-02-05` and `updated_at: 2026-07-02`
   (`git_commit_id 7fe1194b9a46eace4e1034d8bc641cfca0754aef`). The time-based
   counter reset (B2) is therefore Microsoft's **2025-2026 guidance**, not a
   2014 artifact -- it is the live authority against Fowler's 2014 success-reset.
2. **Azure has added an ADAPTIVE-threshold recommendation** in this window:
   *"Traditionally, circuit breakers relied on preconfigured thresholds, such as
   failure count and time-out duration. This approach resulted in a
   deterministic but sometimes **suboptimal** behavior. Adaptive techniques that
   use AI and machine learning can dynamically adjust thresholds based on
   real-time traffic patterns, anomalies, and historical failure rates."* -- an
   explicit 2025-2026 statement that a **fixed failure count is suboptimal**,
   which is a second, independent argument against `MAX_CONSECUTIVE_FAIL = 3`.
3. **Huang et al. (ICLR 2024)** is itself inside the window and is the single
   most decisive source in this brief (B4). It POST-DATES and directly qualifies
   Self-Refine (2023).

**No new finding contradicts the retry-budget / cumulative-cap consensus.** The
2024-2026 movement is uniformly AWAY from fixed consecutive-failure counters and
TOWARD windowed, adaptive, or budget-based bounds.

### Identified but snippet-only / attempted (context; does NOT count toward gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/ | Industry (AWS) | 301 redirect to builder.aws.com (cross-host, not auto-followed) |
| https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter | Industry (AWS) | **FETCH ATTEMPTED, returned only the page title "AWS Builder Center" with no body** -- JS-rendered SPA. NOT counted as read. |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/retry | Official doc | Cross-referenced from the CB page; CB page already carried the retry-vs-break distinction verbatim |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/health-endpoint-monitoring | Official doc | Linked as the half-open probe alternative; peripheral |
| https://learn.microsoft.com/en-us/azure/well-architected/reliability/handle-transient-faults | Official doc | Linked from the WAF pillar table |
| https://learn.microsoft.com/en-us/azure/well-architected/reliability/self-preservation | Official doc | Linked (RE:07) |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/sidecar | Official doc | Linked (service-mesh breakers); out of scope |
| https://arxiv.org/abs/2310.01798 | Preprint (canonical id) | Read via the ar5iv HTML render instead, per the arXiv chain |
| https://arxiv.org/abs/2303.17651 | Preprint (canonical id) | Read via the ar5iv HTML render instead, per the arXiv chain |

**URLs collected: 17 unique (8 read in full + 9 identified/attempted).**

---

## KEY FINDINGS -- ranked by design impact

1. **The defect is REAL, and CONDITIONAL is the bigger half.** `run_harness.py:1177`
   zeroes `consecutive_fails` on CONDITIONAL with the comment "does not count as a
   FAIL". Since CONDITIONAL is the dominant non-terminal verdict in this project,
   the counter is reset by the very outcome that means "still not done". A
   `FAIL, CONDITIONAL, FAIL, CONDITIONAL...` sequence tops out at 1, never 3.
2. **Two more independent reasons it can't bind**, either of which is fatal alone:
   PASS also resets (`:1162`), and the counter is **process-local** (`:1109`), so
   the Layer-3 manual loop -- which spawns Q/A per step rather than running
   `run_harness.py --cycles N` -- never accumulates at all.
3. **The declared ceiling is decorative.** `max_retries: 3` is present on 1078
   masterplan steps and **read by nothing** (A2). Step `75.5` reached `3/3`,
   consumed **8 gate runs / 1,832,223 tokens**, and closed `done`.
4. **The real distribution is 3x the nominal cap, and the tail owns the bill:**
   54.3% of steps take >2 gate runs and hold **76.0%** of all gate tokens;
   the max observed is **9 runs**. A cap of 3 would bite over half the corpus --
   so the design question is not "does a bound exist" but "what happens at the
   bound", which is finding 8.
5. **Every published bound is CUMULATIVE, never consecutive** (Google SRE: 3 total
   attempts / 10% retry ratio / 60 retries per minute). The consecutive-with-reset
   shape does not appear in any resilience source as a work-accounting bound.
6. **Reset-on-success is a HEALTH-CHECK idiom mis-applied to WORK ACCOUNTING**
   (B6). This is the root cause, and it explains the source contradiction rather
   than hiding it.
7. **Repeated LLM evaluation does not converge -- it plateaus after ~1-2 rounds and
   can regress** (Self-Refine +5.0 then +0.9 then +0.9; GPT-4 GSM8K 95.5 -> 91.5 ->
   89.0). Published gains require an **oracle stop**, which an LLM-judged harness
   does not have. Rounds 5-9 are in the flat-to-negative region.
8. **Nobody in the literature "fails closed and moves on" silently.** Azure
   prescribes a **manual override** by an administrator; SRE lets the failure
   *"bubble up to the caller"*; Anthropic's harness post prescribes nothing at all.
   The published disposition on exhaustion is **surface it to a human**, and
   pyfinagent already has the right primitive for that: the `certified_fallback`
   + `## HARNESS HALT` block at `run_harness.py:1003-1035`, which is currently
   unreachable in practice.
9. **8.6% of gate runs never return a verdict** (44/513, 7.65M tokens). Any bound
   must count a DROP as consumption, or a step can burn budget without ever
   incrementing an attempt counter.

## Consensus vs debate (external)

- **Consensus:** bound total attempts; use randomized exponential backoff; retry
  at exactly ONE layer; make the bound cumulative over a window; escalate on
  exhaustion. (SRE ch.21+22, Azure, Anthropic multi-agent.)
- **Debate 1 -- counter reset semantics:** Fowler/Nygard = reset on success;
  Azure 2025-2026 = time-based reset, success-reset only in Half-Open. Resolved
  by purpose (B6): use time/window for health, cumulative for work.
- **Debate 2 -- does iterative self-refinement help?** Self-Refine (2023) says
  yes with sharply diminishing returns and a 4-round cap; Huang et al. (2024)
  says no without an oracle, and shows degradation. **Both agree a small cap is
  right**; they disagree only on whether rounds 2-4 are worth anything.
- **Debate 3 -- fixed vs adaptive thresholds:** Azure's 2025-2026 revision calls
  fixed failure counts *"deterministic but sometimes suboptimal"*. No source
  defends a fixed consecutive counter.

## Pitfalls (from literature, mapped to what this repo would hit)

- **Multi-layer retry multiplication** -- *"a single user action may create 64
  attempts (4^3)"*. pyfinagent has at least three nested retry layers already
  (`orchestrator.py:819` per-agent, `llm_client.py:1367` SDK `max_retries=3`,
  `info_gap.py:183`). Adding a step-level retry without picking ONE layer
  repeats the classic mistake.
- **No backoff at all.** `backend/agents/info_gap.py:205` retries with **no
  delay** (already flagged in `settings.py:499` as real HTTP-429 risk). SRE:
  *"Always use randomized exponential backoff."*
- **Counting the wrong event.** A bound over "cycles Main chose to run" is
  gameable by the party being bounded; a bound over **measured spend** (tokens,
  which `wf_*.json.totalTokens` already records) is not.
- **Oracle illusion.** Treating an LLM Q/A verdict as the oracle that Huang et
  al. show is load-bearing. The verdict is a noisy grader; grader-agreement is
  NOT convergence.
- **Fail-open on missing input.** `.claude/hooks/lib/verdict_gate.py:11-17`
  already fails open on `no_input`, and `verdict_history_86_21.py:5-14` documents
  the same class ("It fails OPEN, and silently"). A new budget must decide its
  unknown-state direction explicitly.

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication |
|---|---|---|
| Cumulative, not consecutive (B1) | `run_harness.py:57,1160-1177` | Replace `MAX_CONSECUTIVE_FAIL` with a per-step CUMULATIVE attempt budget; delete both reset sites |
| Durable state (A1.3) | `run_harness.py:1109` vs `handoff/away_ops/autoresearch_fail_state.json` | The away-ops JSON-file pattern already exists in-repo and IS durable -- reuse that idiom, not a local int |
| Ceiling must be READ (A2) | `.claude/masterplan.json` (1078 steps), `scripts/generate_masterplan.py:202-203` | `max_retries` is already written everywhere; making it READ is a small change with an existing schema |
| Spend ceiling (B5, A5) | `wf_*.json.totalTokens`; `.claude/workflows/*.js` | A token budget is measurable TODAY from existing artifacts; p50 step = 419,739 tok, p90 = 882,651 |
| Cap rounds ~2-4 (B4) | A5 histogram | 113/164 steps (68.9%) already finish in <=3 runs; a cap of 4 covers 154/164 (93.9%) |
| Count drops (finding 9) | 44/513 non-completed | Budget must decrement on `status != completed` too |
| Escalate, don't silently fail (B2 manual override) | `run_harness.py:1003-1035` `_escalate_certified_fallback` + `## HARNESS HALT` | The escalation primitive EXISTS and is effectively dead; wire the new budget to it |
| One retry layer (B1) | `orchestrator.py:819`, `llm_client.py:1367`, `info_gap.py:183` | Put the step budget at the HARNESS layer only; do not add retries below it |
| Unify the two counters (A3) | `CLAUDE.md` F1 vs `qa.md` vs `verdict_history_86_21.py:33-38` | Three documents, two counters, contradictory predicates. Any fix must state ONE predicate |

**Recommended disposition on exhaustion (from B2/B1/finding 8):** PARK the step
with an explicit disposition + escalate to the operator, using the existing
`## HARNESS HALT` block. Do NOT fail closed silently (loses the work) and do NOT
auto-FAIL into another cycle (the 3rd-CONDITIONAL rule's current shape just
converts one unbounded loop into another).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total (incl. snippet-only) -- **17**
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, method deviation declared
- [x] Full papers / pages read (not abstracts) -- both arXiv papers via ar5iv HTML per the prescribed chain; no `/pdf/` fetch attempted
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in INTERNAL SCOPE (1-5, all five)
- [x] Contradictions noted -- B6 is a source contradiction reported against my own narrative, plus two more in "Consensus vs debate"
- [x] All claims cited per-claim

**Declared deviations (none is a hard-blocker miss):**
1. `WebSearch` budget was exhausted session-wide (200/200) BEFORE spawn; sources
   were reached by direct WebFetch of canonical URLs, and the three-variant query
   discipline could not be executed as written. Discovery breadth is therefore
   weaker than protocol; reading depth is not.
2. The brief exceeds the `moderate` <=700-word guidance. The caller explicitly
   required a measured cost distribution and a five-part internal audit; the
   tables are the deliverable.
3. One intended source (AWS Builders' Library) was fetched twice and yielded no
   body text. It is recorded as ATTEMPTED, not read, and excluded from the count.

---

## FINAL ENVELOPE (supersedes the born-inert block at the top of this file)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 9,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.32.md",
  "gate_passed": true
}
```
