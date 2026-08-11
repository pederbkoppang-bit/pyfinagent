# Contract -- step 86.38

**Step**: `86.38` (phase-86, P1, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 | **Driver**: Main (`pyfinagent-51`), Opus 5 / effort max
**Written BEFORE any production code.** The only 86.38 artifacts on disk at this
moment are the research brief and `scripts/qa/derive_lite_fallback_census_86_38.py`,
which is a PRE-CONTRACT MEASUREMENT INSTRUMENT committed deliberately at
`c116e63a` so the research -> plan -> generate ordering is provable from git
rather than collapsing into one commit. `git diff -- backend/` is empty.

---

## 1. Research gate

**PASSED** -- `handoff/current/research_brief_86.38.md`, run `wf_f4c36719-23b`.
Envelope: `brief_status COMPLETE`, `gate_passed true`, 7 sources read in full
(floor 5), 28 URLs (floor 10), recency scan performed, 8 internal files
inspected. The script recomputed the gate and cross-checked all 7 claimed URLs
against the brief on disk; `self_report_disagreed: false`.

**The gate did not confirm my framing -- it partly REFUTED it, and that is the
most valuable thing it produced.** See section 2(a).

## 2. What the step said was NOT ESTABLISHED, now MEASURED

The step text names three unknowns. All three are answered below, before
planning, so the plan responds to the system as it is rather than as the step
text assumed.

### (a) Is the 429 a per-minute limit, a daily quota, or a billing state?

**THE QUESTION CONTAINS A FALSE PREMISE, and the step text should be corrected
rather than answered on its own terms.** Per the Vertex AI quotas documentation
read in full: **Vertex generative AI has NO per-day quota.** Every enforced
dimension is per-minute (TPM/RPM); there are zero "per day" rows in the quotas
page. Per-day limits belong to the **AI Studio free tier**, a different product.
Under **Dynamic Shared Quota** (Gemini 2.0+, and this book runs
`gemini-2.5-flash`) there is no project-level number at all, which is the
documented explanation for a 429 arriving while measured usage looks trivial.

The real trichotomy is **rate limit / DSQ capacity / model-or-billing state**.

**And the step's second premise is also wrong**: it says "the error body was
truncated in the log line and was not read". It was neither. Captured verbatim
from `backend.log`:

```
Full orchestrator failed for HUM: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429,
'message': 'Resource exhausted. Please try again later. Please refer to
https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more
details.', 'status': 'RESOURCE_EXHAUSTED'}} -- falling back to lite Claude analyzer
```

The body is **complete and simply carries no discriminator**. Google's own
guidance is that the HTTP code never carries the cause (the same condition
surfaces as 429 REST, `ResourceExhausted` gRPC, 403 on GCE, and 5XX under
Provisioned Throughput) and that classification must be done **out of band** from
`serviceruntime.googleapis.com/quota/rate/net_usage`, never by parsing the error
string. **So no amount of better logging at the call site can answer this
question** -- that is a design finding, not a defect to fix.

### (b) How many prior cycles degraded, and does it correlate with the drought?

Measured with `scripts/qa/derive_lite_fallback_census_86_38.py`, whose parser
**asserts its own coverage** because a first version silently dropped 416 events
from the June-era log (an older non-JSON ANSI format) and reported a zero that
was a property of the parser, not the system:

| date | full | lite | causes |
|---|---|---|---|
| 2026-07-24 | 3 | 1 | QuantAgent NoneType |
| 2026-07-30 | 3 | 0 | |
| 2026-07-31 | 4 | 3 | QuantAgent NoneType x3 |
| 2026-08-03 | 8 | 0 | |
| 2026-08-04 | 11 | 0 | |
| 2026-08-05 | 6 | 2 | QuantAgent NoneType x2 |
| 2026-08-06 | 11 | 0 | |
| 2026-08-07 | 10 | 0 | |
| 2026-08-09 | 8 | 0 | |
| 2026-08-10 | 3 | 3 | **429 RESOURCE_EXHAUSTED x3** |
| **TOTAL** | **67** | **9** | **11.8% lite** |

**This REFUTES the step text's own inference.** The step reasoned from
`llm_call_log` row counts that "the deep path has been mostly dead" while
correctly flagging that the inference was unconfirmed. Confirmed against the
orchestrator's own accounting: the deep path completed **67 of 76 (88.2%)** over
ten days. The 429 is a **single-day event**, and **6 of the 9 fallbacks are a
different defect entirely** -- `QuantAgent failed for X: 'NoneType' object has no
attribute ...`, a code bug, not quota.

**The drought does not correlate with degradation.** The last trade was
2026-07-31; 2026-08-03 through 2026-08-09 ran 54 full-pipeline analyses with
**zero** fallbacks and still produced no trades. Whatever stopped the book
trading, it is not the fallback.

### (c) Is the lite fallback silent to the operator?

**Partly. It is RECORDED but not PAGED, and the distinction matters.**

RECORDED: the lite analysers stamp `_path: "lite"`, the fallback site stamps
`_fallback_reason`, and `_persist_analysis` writes both into the persisted
`full_report` JSON. A BQ reader can distinguish a lite row from a full one.

NOT PAGED: `_fallback_rate_check` fires only when `n_fallback / n_total >
threshold` **strictly**, with threshold 0.5. On 2026-08-10 the cycle produced
**3 full-pipeline Critic verdicts and 3 fallbacks**; `3/6 = 0.500`, which does
**not** exceed 0.500. The alarm **evaluated and did not fire** -- `grep -c
"Fallback-rate alarm fired"` = 0 and `grep -c "Fallback-rate alarm errored"` = 0,
with the pattern proven matchable by injection, so the zero is a real
non-firing and not a never-evaluated block.

**Honest limit on that measurement**: the predicate's denominator is
`len(candidate_analyses) + len(holding_analyses)`, which I did **not** measure
directly. I measured six *analyses* (3 + 3). n_fallback = 3 and the alarm did not
fire, which is consistent with n_total >= 6. The exact denominator is
**unconfirmed**, and the claim "it missed by exactly one ticker" is therefore
**not established** -- only that it did not fire.

Two further findings from the gate's internal sweep, both re-measured by me:

- **`_intended_path` is write-only** -- one write at `autonomous_loop.py:2189`,
  **zero reads** repo-wide (recall-tested: the same search shape returns 13 hits
  for `_fallback_reason`). It is also **redundant** with `_fallback_reason`,
  which already marks exactly the intended-full-but-landed-lite set.
- **The alarm payload fields are dropped downstream.** `fallback_rate`,
  `fallback_reasons`, `degraded` and `degraded_analyses` are set on `summary` but
  filtered at three whitelists, so **the P1 page is the only operator channel**.

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

The step's criteria are expressed through its `live_check` requirement:

> `live_check_86.38.md` with: the verbatim 429 body or an explicit statement that
> it could not be captured; the per-cycle full-versus-lite table over >=10 cycles
> with the command that produced it; and the measured date of the last
> paper_trades row.

**Verification command** (immutable):
`bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'`
-- a PARSE CHECK only. It proves the module is syntactically valid and nothing
else. Stated here so no reader mistakes a green command for a green step.

## 4. Plan

**P1 -- Correct the step text's two false premises IN the step record.** The
per-day quota does not exist for this product and the 429 body was never
truncated. A step that answers a mis-stated question and leaves the mis-statement
standing has taught the next reader the wrong thing.

**P2 -- Make the fallback visible without touching any gate.** The defect worth
fixing is (c): a degraded cycle that pages nobody. Options, to be decided in
GENERATE against measured behaviour, NOT guessed:
 (i) surface `fallback_rate` / `degraded` through the whitelists that currently
     drop them, so the existing operator surfaces can show it; or
 (ii) leave the threshold alone and add a *separate, lower-severity* signal for
     any non-zero fallback rate.
**The threshold itself is NOT changed in this step** -- see NON-SCOPE.

**P3 -- Remove `_intended_path`, do not wire it.** It is write-only AND redundant
with `_fallback_reason`. Wiring a redundant field would add a second source of
truth for one fact. Removal must be proven safe by a repo-wide search with a
recall test, and by a mutation cell.

**P4 -- File the QuantAgent `NoneType` defect as its own step.** It is 6 of the 9
fallbacks over ten days -- more of the degradation than the 429 -- and it is
squarely out of this step's scope. Per the standing queue-discovered-defects
rule it gets its own research-gated step, not a prose mention.

**P5 -- Record the drought/degradation NON-correlation.** The step was filed on
the premise that the zero-trade outcome was explained by the degraded path. It is
not. That finding must land where the next reader of the trade-drought question
will see it, or the same wrong inference gets made again.

**P6 -- Mutation-test every new guard**, including reverting each change at its
call site. A guard whose mutant survives does not count.

### Explicitly NOT doing

- **NOT touching the RiskJudge, any risk threshold, position sizing, or any
  gate.** The step text is emphatic and correct: the gate behaved correctly on
  the evidence it was given. `paper_risk_judge_*` settings are untouched.
- **NOT changing the `>` to `>=` in `_fallback_rate_check`.** It is already
  pinned by `backend/tests/test_phase_60_1_deep_pipeline.py`, changing it would
  require amending that test, and the research found no external best practice
  favouring either (Google's SRE Workbook uses both). An alarm-sensitivity change
  is an operator decision, and it is raised as an ask rather than taken.
- **NOT enabling a paid Gemini tier**, per the step text and the standing $0
  metered constraint.
- **NOT trying to classify the 429 from the error string.** Google's guidance is
  explicit that this is impossible; the honest output is to say so.
- **NOT fixing the QuantAgent NoneType defect here** (P4 queues it).

### Risk

`autonomous_loop.py` is the live trading loop. The book runs at 20:00 CEST daily
and the backend is NOT restarted mid-session, so any change here is **committed
but NOT in force** until the session-end restart. Every claim about live
behaviour must be read from the RUNNING process, not from the file.

## 5. References

- `handoff/current/research_brief_86.38.md` (gate PASSED, `wf_f4c36719-23b`)
- `scripts/qa/derive_lite_fallback_census_86_38.py` (pre-contract, `c116e63a`)
- `backend/services/autonomous_loop.py` -- fallback site, `_fallback_rate_check`
  and its single call site
- Vertex AI quotas + error-code-429 docs; Google SRE Workbook alerting chapter
