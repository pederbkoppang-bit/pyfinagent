# Contract -- step 86.32

**Step**: `86.32` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~12:0x CEST -- clock read from `date`, not assumed)
**Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** No production file is modified at this moment.

---

## 1. Research gate

**PASSED** -- `wf_ae89a734-9cd`, tier `moderate`, brief
`handoff/current/research_brief_86.32.md` (37,931 chars). Script-enforced and
**recomputed, not trusted**: **8 sources read in full** (floor 5), **17 URLs**
(floor 10), recency scan present, all 8 claimed URLs verified present in the brief
on disk, `brief_status: COMPLETE`, `rail_dropped: null`, self-report agreed with
the enforced result. 14 internal files inspected.

Sources read in full: Google SRE (handling overload; addressing cascading
failures), Azure Circuit Breaker pattern, Fowler *CircuitBreaker*, Anthropic
harness-design + multi-agent-research, arXiv 2310.01798 (LLMs cannot self-correct
reasoning yet), arXiv 2303.17651 (Self-Refine).

## 2. THE DEFECT IS REAL, AND WORSE THAN THE STEP TEXT SAYS

The step says a consecutive counter means "an intervening FAIL resets the counter".
**The dominant reset is not FAIL -- it is CONDITIONAL**, the most common
non-terminal verdict this harness produces.

**I verified every load-bearing claim myself rather than inheriting it.**

### 2a. `consecutive_fails` is ZEROED on CONDITIONAL (verified, verbatim)

`scripts/harness/run_harness.py:1174-1177`:

```python
        else:
            # CONDITIONAL -- keep but warn; does not count as a FAIL
            logger.info("CONDITIONAL -- keeping result with warnings")
            consecutive_fails = 0
```

PASS also resets it (`:1162`). So the sequence `FAIL, CONDITIONAL, FAIL,
CONDITIONAL, ...` **tops out at 1 forever** and `MAX_CONSECUTIVE_FAIL` can never be
reached. The counter is also **process-local** (`:1109`), so nothing accumulates
across sessions -- and the Layer-3 per-step loop I actually run is a *different*
loop from this one.

### 2b. `max_retries` in the masterplan is DECORATIVE (verified precisely)

`grep max_retries` returns 80+ hits, and **almost all are unrelated** -- per-call
retry loops in `llm_client.py`, `debate.py`, `info_gap.py`, `bigquery_client.py`,
`ticket_queue_processor.py`. Quoting that count as evidence would be the
count-the-class-not-your-list defect.

The precise question is whether anything reads **the masterplan step's** field.
Every file that touches `masterplan.json` AND mentions `max_retries` writes it as a
literal `"max_retries": 3` -- `generate_masterplan.py:203` and eight
`add_phase_*.py` scripts. **All writers. Zero readers.**

**Decisive evidence:** step **75.5 carries `retry_count: 3, max_retries: 3,
status: done`**. It reached its ceiling and closed anyway. If any code enforced the
field, that state is impossible. 1,078 steps carry the field; exactly one has ever
reached it, and it was not stopped.

### 2c. The 3rd-CONDITIONAL rule is instructions-only

Per the gate: `verdict_history_86_21.py` is ADVISORY, **called by nothing**, and
implements a **different counter with a contradictory predicate** from the one
CLAUDE.md describes. So the rule that would have bounded my own loop today exists
only as prose.

### 2d. No spend or attempt ceiling exists in either Workflow launch path

Neither `qa-verdict.js` nor `research-gate.js` caps attempts or tokens per step.

## 3. THE COST, MEASURED -- and I am an instance

Gate measurement over **513 `wf_*.json` runs / 164 steps**: runs-per-step
`{1:27, 2:48, 3:38, 4:28, 5:13, 6:5, 7:2, 8:1, 9:2}`; per-step **p50 419,739
tokens**, **max 1,832,223** (step 75.5, 8 runs); **54.3% of steps take >2 runs and
hold 76.0% of the tokens**; **8.6% of runs drop with no verdict (7.65M tokens)**.

**86.41, closed by me an hour ago, is a third independent instance**: 3 graded
cycles, ~528K subagent tokens, 1 drop. Nothing stopped it; nothing would have.

### A DENOMINATOR DISAGREEMENT I AM NOT PAPERING OVER

I measured **29.2% of runs dropped today** (7 of 24 completed). The gate measured
**8.6% over 513 runs**. **These are not the same measurement and I will not quote
one number:**

- **Different windows** -- mine is 2026-08-11 only; the gate's is all-time.
- **Different sources** -- I counted `journal.jsonl` files with zero `result`
  lines; the gate counted `wf_*.json` run summaries.

GENERATE must reconcile these or report both with their rules. **This matters for
the design**, not just for bookkeeping: see §4 P2.

### My own drop detector was vacuous first, and I am recording it

My first probe reported a **0.0% drop rate**. It looked for a `result` line whose
value was empty -- but a dropped run has **no result line at all**, so the check
could never fire. Zero was a property of the query. Same shape as the UTC/local cut
error I made earlier today. Any instrument this step ships must be mutation-tested
against a synthetic drop.

## 4. Plan

**P1 -- FIX THE COUNTER, OR REPLACE IT.** Research is unambiguous: **every SRE
bound is cumulative, never consecutive** (Google SRE retry budgets; Azure and
Fowler both keep circuit-breaker failure counts over a rolling window, not a
consecutive streak). Reset-on-success is a *health-check* idiom; applying it to
*work accounting* is the root cause. The fix is a cumulative per-step attempt
count that no verdict resets.

**P2 -- A CEILING MUST COUNT DROPS, AND THIS IS THE SUBTLE PART.** A dropped run
costs full tokens and returns **no verdict**, so it advances no verdict-based
counter. A ceiling that counts verdicts is blind to between 8.6% and 29.2% of
spend -- which is exactly why the denominator question above is load-bearing rather
than pedantic. **The counter must increment on ATTEMPT, not on OUTCOME.**

**P3 -- MAKE `max_retries` REAL OR DELETE IT.** A field on 1,078 steps that no code
reads is worse than absent: it reads as a guarantee. Either enforce it or remove
it. I will not leave a third state.

**P4 -- WHAT HAPPENS AT THE CEILING.** Per the research: park with a disposition
and escalate to a human; do not silently continue. This project already has that
vocabulary (PARKED + disposition), so the terminal state should reuse it rather
than invent one.

**P5 -- MUTATION-TEST THE CEILING, INCLUDING THE DROP CASE.** A cell that reverts
the cumulative counter to consecutive must go red, and a cell that makes a drop
invisible must go red. Green control captured first.

### Explicitly NOT doing

- **Not** changing any verdict semantics, or what PASS/CONDITIONAL/FAIL mean.
- **Not** touching the 86.21 counter (peer-parked) -- but the contradiction between
  it and CLAUDE.md is recorded here for whoever closes it.
- **Not** retroactively re-grading any closed step.
- **Not** loosening any gate: a ceiling makes the harness stop *sooner*, never
  accept more.

### Risk

`run_harness.py` is the scheduled optimization driver. A ceiling that is too tight
stops legitimate work; the failure mode of getting this wrong is **halted progress,
not bad trades**, and it cannot reach the book. The book runs at 20:00 CEST and
this step touches nothing on the analysis path.

## 5. References

- `handoff/current/research_brief_86.32.md` (gate `wf_ae89a734-9cd`)
- Google SRE: handling overload, addressing cascading failures; Azure Circuit
  Breaker; Fowler *CircuitBreaker*; Anthropic harness-design + multi-agent-research;
  arXiv 2310.01798; arXiv 2303.17651
- `scripts/harness/run_harness.py:1109,1160-1177`; `scripts/generate_masterplan.py:203`;
  masterplan step 75.5
