---
name: conditional-collapse-87-6
description: Step 87.6 -- the CONDITIONAL/FAIL surge is confounded by a 61% evaluator-spec growth in the same window; 98.7% of findings are apparatus not product; apparatus:product line ratio moved 0.16 -> 2.73
metadata:
  type: project
---

The PASS-rate collapse (95.4% -> 39.3% graded-only, at the 2026-08-11 break) is **NOT
established to be a code-quality decline**, and four measurements say the dominant
mechanism is elsewhere.

**Why:** the caller's premise ("Main introduces more defects") was the natural reading, and
it is the one 87.1-87.5 were scoped against. It does not survive measurement.

1. **The confound is real and large.** `.claude/agents/qa.md` went **556 -> 897 lines
   (+61.3%)** between 2026-08-09 and 08-17 -- inside the collapse window. In the SAME window
   the `maxTurns` cap that was truncating Q/A evaluations was removed (`85127353`), and the
   `verdict_wip_*` corpus (per-spawn observability) was CREATED (`d23a981e` 08-10). Three
   simultaneous changes to the JUDGE. arXiv:2606.15474: a naive drift read false-alarms on
   **75%** of drift-free streams; the fix is a **frozen anchor set** -- re-run today's qa.md
   against UNCHANGED pre-08-09 evidence.
2. **The defects are in the APPARATUS, not the product.** 154 finding-blocks since 08-14
   across 27 steps: **2 (1.3%) are product-code-only**; 94% touch evidence and/or guards.
3. **Because that is where the code now is.** `scripts/qa/` : `backend|frontend`
   added-lines-per-day went **0.16:1 -> 2.73:1** (17x). `scripts/qa/` is **102 files /
   34,082 lines**, up from **6 files on 2026-07-06**. The apparatus has no verifier but the
   Q/A, so every defect in it lands as a CONDITIONAL.
4. **The loop cannot converge:** cross-cycle finding novelty is **86%** (145/169), **83%** at
   cycle 3+. Median remediation commit = **286 inserted lines** (mean 465; 45% >=300; 10
   >=1000, n=80). Each fix enlarges the surface. Also: **24 of 32 CONDITIONAL critiques
   contain ZERO `NOT MET`** -- 75% of the signal driving a ~465-line remediation reports no
   unmet criterion.
5. **Criteria inflation is a mechanical driver needing no quality change:** mean
   success_criteria went phase-79 **2.58 / 397 chars** -> phase-86.120+ **7.43 / 1,741**. A
   conjunctive verdict at a stipulated 90%/criterion drops 78% -> 46% on that alone
   (ILLUSTRATIVE bound, not a measurement -- criteria are not independent).

**How to apply:** never accept "the model got worse" on a metric whose JUDGE changed in the
same window -- demand the frozen-anchor replay first. When a harness's findings cluster in
one artifact class, check the line-volume ratio before theorising about capability. And note
the class 87.1-87.5 MISS: **silent failure** (a guard that CAN fail, runs, goes green, and
covers none of the live population -- `M10 survives ... ZERO coverage`, `the fix is INERT on
the live corpus`, `MARKER-based, not outcome-based`, `whole-file BYTE-PRESENCE pins`). 87.4
targets vacuity (guard text); inertness is a property of the guard's INTERSECTION WITH THE
LIVE POPULATION. NOVA (arXiv:2606.27243v1) makes it a first-class metric, `SFR`, with
`EPR = LPR x (1-SFR)`.

Two literatures that appear to contradict actually agree once you condition on **whether the
feedback signal is externally verifiable**: iteration against executable tests converges
(arXiv:2604.10508, 76-95% of gains in 2 rounds, no regressions); iteration against
model-generated critique degrades (arXiv:2604.22273v2 EIR threshold ~0.5%;
arXiv:2604.01029 content effect -3.1..-7.9 pp on code). pyfinagent's product cycles have an
oracle; its evidence cycles do not.

Stale-doc correction found: CLAUDE.md F1b still says `attempt_budget.py` has "no runtime
caller". It DOES -- `scripts/harness/attempt_gate.py:84`.

Related: [[project_third_conditional_rule_parks_converging_steps]],
[[project_verdict_population_86_98]], [[project_harness_external_audit_2026_08_17]].
