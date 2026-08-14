---
name: risk-gate-veto-86-74
description: Step 86.74 falsy-zero risk veto -- the `or 10.0` idiom is at FOUR sites and the approved fix guards ONE; three default-OFF flags whose conjunction is the safety property, and the "fail-safe" one alone makes failure WORSE (3%->10%)
metadata:
  type: project
---

Research for step 86.74 (risk judge returns 0% REJECT, code converts it to a
10%-of-NAV BUY). Three findings that were NOT visible from the incident memo or
the caller's stated scope.

**1. The `or 10.0` idiom is at FOUR sites; the approved fix guards ONE.**
`backend/services/portfolio_manager.py` `:507` (main loop, guarded by
`paper_risk_judge_shape_fix_enabled`), `:800`, `:853`, `:878` -- the last three
UNGUARDED by any flag. `:878` is the swap path's real `buy_amount` sizing. The
phase-57.1 comment at `:350-357` records that the away week's 3 REJECT BUYs were
"all via the swap path", so the unguarded sites are exactly the ones with a
measured history of executing REJECTs. Promoting the 66.2 flag alone ships a fix
that does not bind where the failure was observed.

**2. Three default-OFF flags, and the "fail-safe" one alone makes it WORSE.**
`settings.py:342` `paper_risk_judge_reject_binding`, `:346`
`paper_risk_judge_parse_fail_reject`, `:350` `paper_risk_judge_shape_fix_enabled`.
The safety property is their CONJUNCTION. With `parse_fail_reject=ON` and
`shape_fix=OFF`: `risk_debate.py:152` writes the fail-safe `0` into the nested
`judge` dict (`:345`), the top-level read misses it, `None or 10.0` -> **10%**.
So the flag named "fail-safe" escalates a parse failure from 3% NAV to 10% NAV.
Promotion must be atomic.

**3. The submission seam is a recorder, not a gate.**
`paper_trader.py` takes `risk_judge_decision` (`:243`) and
`risk_judge_position_pct` (`:245`) and only persists them (`:432`, `:489`,
`:513`, `:677`). `grep REJECT backend/services/paper_trader.py` returns NOTHING.
Zero enforcement at execution. A chokepoint there is invariant to the number of
upstream sizing sites -- which is the point, given finding 1.

**Why:** the defect class is `Optional[float]` carrying three domain states
(judge-said-0 / judge-said-N / no-judge). `_extract_position_pct:939-955` is
lossy BEFORE `or 10.0` runs, which is why the 66.2 fix needed a second patch at
`:324-330` to re-read the raw value. The residual: `:949`'s
`analysis["risk_judge_position_pct"]` keeps its falsy-zero check under EVERY
flag setting.

**How to apply:** when a step says "fix the falsy-zero check at line X", grep the
whole file for the defaulting idiom before scoping -- the count is the finding.
And check whether a safety fix is behind a flag whose partial promotion is a
regression. See [[project_research_gate_discipline]],
[[project_dead_sell_rule_86_58]] (same shape: canonicaliser guarded the READ
only).

**External anchors that settle the design question:** Saltzer & Schroeder
fail-safe defaults ("a mistake in an exclusion-based mechanism fails dangerously
by granting access"); 17 CFR 240.15c3-5(c)(2)(i) is written in permission form
("Prevent the entry of orders **unless** there has been compliance");
arXiv:2605.14744 measures governance-task decoupling (advisory governance leaves
27.3% of deferrals vacuous vs 7.4% mechanical). ADVERSARIAL note: AgentSpec
(arXiv:2503.18666, ICSE'26) is itself fail-OPEN on an indeterminate predicate --
adopting an enforcement framework does not buy fail-closed semantics.
