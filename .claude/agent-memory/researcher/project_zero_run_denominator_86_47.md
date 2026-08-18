---
name: zero-run-denominator-86-47
description: A trade "drought" is a DENOMINATOR question first; analysis_results/signals_log live in financial_reports NOT pyfinagent_data; the parse-fail fallback fabricates a verdict and its fingerprint goes NULL not False
metadata:
  type: project
---

Measured 2026-08-18 for step 86.47 (whether the zero-trade run was anomalous).

**The answer was "no", and it was arithmetic, not argument.** Last trade 2026-08-13.
Naive framing = "5 calendar days". Real opportunity count = **2 analysis days / 13
analyses** -- 08-15/08-16 are weekend and 08-18 had not run yet (heartbeat said last cycle
ended 2026-08-17T19:47Z; `paper_trading_hour = 10` ET). Post-break BUY rate 8/262 = 0.0305
=> P(0 | 13 opportunities) = **0.672**; rule-of-three bound 3/13 = 0.231 contains it. You
need **~97 analyses (~16 days)** for p<0.05, and `paper_analyze_top_n = 5` caps opportunities
per cycle -- so this system is *structurally* incapable of evidencing a drought quickly.

**Why:** two prior steps (86.38, 86.41) were filed on drought theories their own research
gates refuted, both by reasoning from a degradation they had just measured to the drought.

**How to apply:** for any "X stopped happening" step, compute the base rate and the
opportunity denominator BEFORE looking for a cause. Quote the normalisation rule with the
rate. See [[project_86_69_empty_hold_regression]]-adjacent break: the real event was
**2026-06-11 (first contaminated day) -> 06-15 (complete)**, BUY 46/64 -> 0/7; re-derive it,
06-12 is the wrong start date.

## Data-surface corrections (the step spec was wrong, verified)

- `analysis_results` (580 rows, **91** columns) and `signals_log` (119) are in
  **`financial_reports`** (us-central1), NOT `pyfinagent_data`. Both 404 in `pyfinagent_data`.
  `llm_call_log` (7248) IS in `pyfinagent_data` but has **no `created_at`** column.
- `paper_trades` = **66 rows ever**; `created_at` is a **STRING**; the action column is
  `action` not `side`; there is no `executed_at`.
- `pyfinagent_data.risk_intervention_log` = **0 rows** -- the refusal event stream is empty.
- `outcome_tracking` = 3 rows, `paper_positions` = 2.

## Three traps that will recur

1. **`risk_judge_decision` on `analysis_results` is populated on 18/580 = 3.1%** (0 in
   2026-05, 0 in 2026-07). It became populated on **2026-08-14** when the phase-86.74 writer
   fix landed. Never key a funnel on it without reporting the population rate first.
2. **`_judge_parse_fail_fallback` (`backend/agents/risk_debate.py:158`, sole call `:375`)
   fabricates a verdict** on unparseable judge output: `APPROVE_REDUCED / pos 3 / MODERATE /
   rac 0.5` (flag `paper_risk_judge_reject_binding`-adjacent flag
   `paper_risk_judge_parse_fail_reject`, `settings.py:346`, default False). Two rows on
   2026-06-11 carry the full fingerprint. **On the 2026-08 rows the 4-part fingerprint
   evaluates to NULL, not False**, because `risk_adjusted_confidence` is unpopulated -- a
   `COUNTIF` scores the unknown row as clean. Report UNRESOLVED.
3. **The lite/full split IS derivable**: `JSON_VALUE(full_report_json, '$._path')` is
   populated on 100% of rows (values `lite`/`full`), stamped by `_persist_analysis`
   (`autonomous_loop.py:3561`). `_fallback_reason` marks intended-full-landed-lite.
   `total_tokens` / `decision_trace_count` / `debate_rounds_count` are NULL on 100% of rows
   since 2026-07-25 -- do not use those to characterise the pipeline.

## Collinearity trap on 2026-08-14

Four things changed in the same 2-day window: `standard_model` sonnet-4-6 -> **sonnet-5**;
risk columns empty -> populated 13/13; zero-scores -> 0; `recommendation` -> `Hold` on 13/13.
Zero degrees of freedom -- no attribution is identifiable. Flag it as non-identifiability
rather than picking a cause.

Also re-derived: **exactly 3 `paper_trades` rows are `action='BUY'` AND
`risk_judge_decision='REJECT'`** (066570.KS 06-09, DELL 06-03, HPE 06-02) -- the gate at
`portfolio_manager.py:383-400` is default-OFF, so a stored REJECT is not evidence the gate
fired. `paper_risk_judge_reject_binding` is **not among the 45 keys** `GET /api/settings/`
exposes, so its running value is unreadable from that surface.

Recommendation vocabulary, whole table: `HOLD` 284, `Hold` 137, `BUY` 94, `Buy` 40, `Sell` 18,
`Strong Buy` 5 (none since 2026-05-22), `N/A` 2. The last pre-drought BUY is spelled `Buy`.
