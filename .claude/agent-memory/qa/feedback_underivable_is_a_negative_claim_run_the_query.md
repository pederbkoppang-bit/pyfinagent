---
name: underivable-is-a-negative-claim-run-the-query
description: "'The signal is underivable' is an unverified negative -- run the alternative route yourself; 86.47's refusal funnel was live at 93.1% coverage behind a JSON_VALUE-on-object trap"
metadata:
  type: feedback
---

A step may report a measurement as **UNDERIVABLE**. That is a claim about the world, not a
scope decision, and it is the one claim shape that *looks* like honest restraint while
hiding an unrun query. **Run the alternative route yourself before accepting it.**

**Why:** step 86.47 (2026-08-18) reported the risk-gate refusal funnel underivable because
`analysis_results.risk_judge_decision` is 3.1% populated -- true, and correctly proven. But
the funnel was live the whole time at `full_report_json.final_synthesis.risk_assessment.judge`:
382/526 rows since 2026-05-01, **256/275 (93.1%) post-break**, and **13/13** in the exact
13-analysis silence window the step was filed on, carrying
`{"decision": REJECT|APPROVE_REDUCED|APPROVE_HEDGED, "reasoning": "<full stated reason>",
"recommended_position_pct": N}`. Eight of those 13 were `REJECT` at `pct=0` on `path=full`,
which `portfolio_manager.py` treats as no-buy unconditionally since phase-86.74. The step's
headline -- "no gate is at fault, a gate cannot refuse a recommendation that was never
produced" -- was contradicted by data it never queried.

**The mechanism of the miss, and it will recur:** `judge` is a JSON **object**, and
`JSON_VALUE(blob,'$...judge')` returns NULL for objects. A coverage probe built on
`JSON_VALUE` reports **0/526 populated** -- indistinguishable from a genuinely empty field.
`JSON_QUERY` returns 382. I made the identical mistake mid-evaluation and only caught it by
dumping one row's key list (`risk_assessment keys: [aggressive, conservative, judge, ...]`)
and noticing `judge` was there while my probe said it was not.

**How to apply:**
- Treat "underivable" / "not populated" / "no such signal" as a **finding to falsify**, at the
  same bar as a positive claim. Two queries is the whole cost.
- Before believing an absence, **dump the container's keys** (`TO_JSON_STRING(JSON_QUERY(...))`
  on one row) rather than trusting a scalar extractor's NULL.
- Check whether the step's own **research brief already named the alternative route**. 86.47's
  brief said verbatim: "use `paper_trades.risk_judge_decision` *and the JSON blob*". The route
  was documented and simply not taken -- a research-gate-compliance gap, not just a data gap.
- An escape hatch scoped to one column (criterion 3: "any funnel number keyed on
  risk_judge_decision") does **not** waive a separate criterion that asks for the same counts
  by any means (criterion 2). Watch for a narrow permission being spent on a broad omission.

Related: [[feedback-queued-is-a-claim-that-must-reproduce]],
[[feedback-unwired-is-a-claim-with-an-expiry]],
[[feedback-suspect-the-clean-check]].
