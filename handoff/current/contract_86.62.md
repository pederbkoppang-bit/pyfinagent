# Contract — step 86.62

**Step:** 86.62 — the 2026-08-11 cycle logged three distinct degradations and not one of them is triaged anywhere
**Priority:** P2  |  **Status at contract time:** pending
**Date:** 2026-08-14 (~01:40 CEST)

---

## Research gate — PASSED

Run `wf_07a0d6c8-b7c`, tier `simple`. `gate_passed: true` recomputed by the script,
**no rail drop**. 6 sources read in full (floor 5), 19 URLs (floor 10), corroborated
19 <= 19 distinct in the brief, all 6 claimed sources present, recency section
present, `brief_status: COMPLETE`.
Brief: `handoff/current/research_brief_86.62.md` (39,586 chars).

**Tier `simple` sets depth of analysis only — the floors are unchanged and were met.**

---

## Hypothesis

All three degradations are **chronic, not transient**, and the reason none was triaged
is that **the cycle reports itself clean**. The reporting channel, not the operator's
attention, is the defect.

---

## What the gate established (to be RE-DERIVED in GENERATE, not inherited)

| Degradation | Measured rate | Transient? |
|---|---|---|
| promoted-strategy 404 | **19/19 cycles (100%)** | No |
| p95 breach (6267ms vs 500ms) | 10 of 14 MetaCoordinator decisions | No |
| AV social rate limit | 27 events on 14 of 21 days | No |

~~`quant_opt` — the action the p95 breach is supposed to trigger — fired 0 times in
21 days.~~ **CORRECTED (cycle 2): this was wrong and is retained struck-through so the
error is visible rather than erased.** The p95 branch returns **`perf_opt`**, which
fired **10 of 10**; `quant_opt` is the unrelated Priority-2 low-Sharpe action, and the
bare string occurs 17 times. The real finding: Priority 1's early return **STARVES**
Priorities 2 and 3 on 10 of 14 decisions.

**Criterion 4 — CORRECTED (cycle 2):** the codebase does **BOTH**, decided by whether
`fallback_articles` was passed. Consumer `backend/tasks/analysis.py:251`
`.get("avg_sentiment")`; producer `social_sentiment.py:73-81` returns `0.0` on the
fallback branch (**ZEROES**) and a `NO_DATA` dict with no such key on the other
(**OMITS** → `None`). Saying "the production path ZEROES" flattened a two-branch answer
on the exact dichotomy the criterion exists to resolve. `backend/tools/social_sentiment.py::_keyword_score` ends
`if total == 0: return 0.0`, so a rate-limited fetch that falls back to keyword-scoring
headlines yields **exactly 0.0 — inside the NEUTRAL band**. The provenance that would
reveal it (`yfinance_fallback`) is produced and then **dropped**: `save_report` has no
social-provenance column. **"No data" and "genuinely neutral" are the same number.**

**The 404 is a MISSING TABLE, not a permission** — reason `notFound`, dataset IS `US`,
and the request carried a Job ID proving `jobs.create`. A permission failure returns
403. **Consequence: NIL** — `best_params` sets two summary fields plus the heartbeat,
and `decide_trades` (`portfolio_manager.py:164-172`) does not consume it. So the cycle
**should** have proceeded on fallback parameters.

**F7, verified by me:** `cycle_history` entry `86667da7` carries `degradation: None`
and `error_count: 0` on **both** its `started` and `completed` rows, while all three
degradations were firing.

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. each of the three degradations is traced to a cause and reported separately -- 'transient' is a permitted conclusion only when supported by a measured recurrence rate across cycles, not asserted
2. the p95 latency figure is RE-DERIVED from the source that produced it, and the population it is a p95 OF is stated -- a latency percentile with no stated population is not actionable
3. the promoted-strategy 404 is resolved to a specific missing BQ object or a specific permission, and whether the cycle SHOULD have run on fallback parameters is stated as a yes/no with its consequence for that cycle's decisions
4. whether the social-sentiment rate limit silently zeroes a signal versus omitting it is determined by reading the consumer, since a zeroed signal and an absent signal are different inputs to a score
5. the causal links to 86.47 (trade drought) and 86.60 (blind overlays) are either demonstrated or explicitly ruled out -- speculation in either direction is recorded as untested
6. no threshold is loosened to make a breach disappear; if 500ms is the wrong threshold that is a separate, argued change with its own evidence

Immutable verification command:
```
bash -c 'test -f backend.log && grep -c "Paper trading cycle complete" backend.log'
```
Required live_check: `live_check_86.62.md quoting each degradation verbatim with its timestamp, plus the measured recurrence of each across the available cycles`

---

## Plan — GENERATE is measurement and reporting ONLY. No production code changes.

1. **Re-derive each of the three rates independently** over a stated population, with
   the command beside each number and a positive control paired to every zero. Report
   any disagreement with the gate's figures rather than adopting them.
2. **Criterion 2:** re-derive the p95 from the source that produced it and **state the
   population it is a p95 OF**. A percentile with no stated population is not
   actionable and does not satisfy the criterion.
3. **Criterion 3:** state the 404's specific missing BQ object, and answer
   *should the cycle have run on fallback params?* as an explicit **yes/no** with its
   consequence for that cycle's decisions.
4. **Criterion 4:** quote `_keyword_score` verbatim and show the zero reaches the score.
5. **Criterion 5:** demonstrate **or explicitly rule out** the causal links to 86.47 and
   86.60. **Speculation in either direction is recorded as UNTESTED** — that is what the
   criterion asks for, and an untested link stated as untested is a pass, not a gap.
6. **Criterion 6: change NO threshold.** If 500ms is wrong, that is a separate argued
   change with its own evidence — file it, do not make it here.
7. Write `live_check_86.62.md` quoting each degradation **verbatim with its timestamp**
   plus the measured recurrence.

---

## Scope honesty, declared up front

- **No file under `backend/` will be modified by this step.** The deliverable is a
  triage report. Criterion 6 forbids the only code change that would be tempting.
- **Known limit to carry:** the 19/19 figure spans the rotated archives. The live
  `backend.log` alone holds 4 Step-1 cycles, so it cannot be reproduced from the live
  log in isolation. State the population; do not claim more.
- The `F7` self-clean finding is **adjacent to, and not owned by, any of the six
  criteria.** Record it; do not let it masquerade as a criterion deliverable.

---

## References

- `handoff/current/research_brief_86.62.md` — gate PASSED
- `backend/tools/social_sentiment.py::_keyword_score` — `if total == 0: return 0.0`
- `backend/services/portfolio_manager.py:164-172` — `decide_trades` does not take `best_params`
- `handoff/cycle_history.jsonl` — entry `86667da7`, the self-clean cycle
- `handoff/current/q1_binding_constraint_86.59.md` — the 86.69 empty-HOLD class this repeats
