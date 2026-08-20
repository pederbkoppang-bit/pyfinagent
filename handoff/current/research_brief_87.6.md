# Research Brief -- step 87.6

**Topic:** Why Layer-3 harness Q/A cycles increasingly return CONDITIONAL/FAIL on
defects Main itself introduces within the same step (self-inflicted, not
pre-existing production bugs), and what internal + literature-grounded techniques
reduce first-pass GENERATE defect rates in agentic coding harnesses.

**Tier:** complex (caller-stated; not self-selected)
**Audit-class:** YES (caller-set). Loop-until-dry, K_required = 2.
**Brief path:** `handoff/current/research_brief_87.6.md`
**Researcher:** Layer-3 researcher, Workflow rail (`.claude/workflows/research-gate.js`)
**Started:** 2026-08-18

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 20,
  "snippet_only_sources": 18,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 24,
  "coverage": {
    "audit_class": true,
    "rounds": 10,
    "dry_rounds": 3,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "gate_passed": true
}
```

**COMPLETE. Gate PASSED: 20 sources read in full (floor 5), 38 URLs (floor 10), recency scan performed, audit-class loop dry after 3 consecutive dry rounds (K=2).**

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md` §Search-query composition)

| # | Variant | Query |
|---|---|---|
| Q1 | current-year frontier | `LLM agent code generation first-pass defect rate self-introduced bugs evaluator feedback loop 2026` |
| Q2 | last-2-year window | (see Recency scan section) |
| Q3 | year-less canonical | (see Recency scan section) |

---

## ROUND 1 -- INTERNAL: the PASS-rate collapse, MEASURED

**Normalization rule stated with the ratio (memory: `feedback_normalization_rule_must_be_stated_with_the_ratio`).**
Source = `handoff/harness_log.md` `## Cycle` headers ONLY (1,289 headers). A header counts
if it carries a parseable `YYYY-MM-DD` AND a `result=` token. `PASS%` denominator =
GRADED verdicts only = `PASS + CONDITIONAL + FAIL`; `PARKED`, `NO_VERDICT`, `BLOCKED`,
`GATE-FAILED` and free-text results are EXCLUDED from the denominator (they are not Q/A
judgments of code). Prefix-normalised: `PASS_WITH_FINDINGS`/`PASS_AFTER_RETRY` -> PASS,
`NO-VERDICT`/`NO_VERDICT` -> NO_VERDICT.

| Window | headers | graded | PASS | CONDITIONAL | FAIL | PASS% |
|---|---|---|---|---|---|---|
| ALL-TIME | 1,271 | 696 | 630 | 43 | 23 | **90.5%** |
| before 2026-08-11 | 1,191 | 635 | 606 | 21 | 8 | **95.4%** |
| 2026-08-11 onward | 80 | 61 | 24 | 22 | 15 | **39.3%** |

**The caller's framing ("77.2% -> 33.3%") and mine ("95.4% -> 39.3%") are the SAME
phenomenon under different denominators** -- theirs evidently counts PARKED/NO_VERDICT
against the numerator. I am NOT reconciling them into one number; I state my rule and
report both (memory: `feedback_a_borrowed_number_becomes_your_claim`). **Under EITHER
rule the collapse is real and large: a ~56-point / ~44-point drop.**

Per-day since the break (graded only):

| Day | PASS | COND | FAIL | PARKED | NO_VERDICT | PASS% |
|---|---|---|---|---|---|---|
| 08-11 | 7 | 5 | 3 | 0 | 1 | 47% |
| 08-13 | 1 | 0 | 0 | 0 | 0 | 100% |
| 08-14 | 3 | 8 | 0 | 0 | 3 | 27% |
| 08-15 | 1 | 1 | 2 | 0 | 0 | 25% |
| 08-16 | 1 | 0 | 5 | 0 | 0 | 17% |
| 08-17 | 9 | 7 | 1 | 3 | 0 | 53% |
| 08-18 | 2 | 1 | 4 | 5 | 1 | 29% |

**Step-level iteration counts since 08-11** (`## Cycle` headers per `phase=`):
86.74 = 7, 86.79 = 5, 86.59 = 4, 86.78 = 4, 86.85 = 4, 86.9 = 3, 86.84 = 3.

**Q/A spawn corpus** -- `.claude/agent-memory/qa/verdicts/` holds 180 files, 173 of which
match `verdict_wip_<step>__<ts>.md` (7 older files use the un-timestamped
`verdict_wip_<step>.md` form and are the phase-86.24/.25/.29/.30/.31/.34/.37 originals).
173 timestamped spawns across **44 distinct steps** -- i.e. **3.9 Q/A spawns per step on
average**, and the tail is far worse: 86.85 = 12, 86.84 = 11, 86.74 = 10, 86.78 = 8.
Spawns/day: 08-11 = 31, 08-13 = 8, 08-14 = 25, 08-15 = 6, 08-16 = 23, 08-17 = 56,
08-18 = 24. **08-17 alone spent 56 Q/A spawns.**

## ROUND 1 -- INTERNAL: the CONFOUND the caller asked about is REAL and LARGE

The caller asked whether Q/A's rigor bar "recently tightened, which would itself explain
part of the CONDITIONAL/FAIL rise independent of any change in Main's code quality."
**It did, and the timing is near-exact.** `git log --follow -- .claude/agents/qa.md`,
line count of `git show <sha>:.claude/agents/qa.md`:

| Date | commit | qa.md lines | numstat +/- |
|---|---|---|---|
| 2026-07-31 | `c3286524` | 532 | +5/-4 |
| 2026-08-09 | `6381b000` | 556 | +24/-0 |
| 2026-08-10 | `d23a981e` | 621 | +67/-2 |
| 2026-08-11 | `5595055c` | 639 | +26/-8 |
| 2026-08-13 | `9a59a4fa` | 691 | +67/-15 |
| 2026-08-14 | `89e254fc` | 748 | +63/-6 |
| 2026-08-14 | `2e40e8c7` | 764 | +25/-9 |
| 2026-08-14 | `9b4d5281` | 835 | +116/-45 |
| 2026-08-14 | `85127353` | 871 | +39/-3 |
| 2026-08-17 | `77f15b4d` | 888 | +17/-0 |
| 2026-08-17 | `1777cc8d` | 890 | +4/-2 |
| 2026-08-17 | `2dbe09d4` | 897 | +8/-1 |

**+61.3% (556 -> 897 lines) between 2026-08-09 and 2026-08-17** -- i.e. the evaluator
specification grew by nearly two thirds *inside the collapse window*. Obligation-language
density (`grep -icE 'MUST|MANDATORY|NON-NEGOTIABLE|FAIL if|auto-FAIL|forbidden|REQUIRED'`)
went **25 -> 33 lines (+32%)** over the same span, with **9 such lines added** in the diff
`6381b000..HEAD`.

**Therefore any claim of the form "Main's code got worse" is CONFOUNDED and must not be
made without a design that separates the two.** The judge's specification and the
subject's output changed *simultaneously*. This is the collinearity trap from
`feedback_check_collinearity_before_crediting_an_attribution`.

**Two structural changes make the confound worse than a simple bar-raise:**
1. `2026-08-14 85127353` REMOVED the `maxTurns` caps from BOTH Layer-3 agent files
   (`.claude/agents/qa.md`, `.claude/agents/researcher.md`). Prior to that, a capped Q/A
   was being CUT OFF mid-evaluation -- the rail-drop class -- so pre-cap-removal verdicts
   are drawn from a *truncated* evaluation and post-removal ones from a complete one. A
   Q/A that now runs to completion finds strictly more.
2. `d23a981e` (86.31) introduced **Q/A write-first**, and `5595055c` (86.36) made the WIP
   records **run-stamped**. Before 08-10 a Q/A left no durable per-spawn record; after,
   every spawn writes one. **The 173-file `verdict_wip_*` corpus therefore does not exist
   before the collapse window at all** -- observability itself was created here. Some of
   the apparent rise is *measurement coming online*, not defect rate rising.

## ROUND 2 -- INTERNAL: WHAT the Q/A is actually finding (defect-class census)

**Rule stated first.** Population = the 173 timestamped `verdict_wip_*` files; restricted to
spawns with timestamp `>= 20260814` (post-`maxTurns`-removal, so every spawn ran to
completion and the corpus is comparable). Unit = a markdown heading matching
`^#+\s*(FINDING|BLOCKER|VIOLATION|DEFECT)`. **n = 154 finding-blocks across 27 steps.**
Classifier = keyword presence in the 900 chars following the heading; classes are NOT
mutually exclusive, so a block can carry two labels. **This is a PROXY and I am labelling it
one** (`feedback_assert_the_property_not_a_proxy`): I validated it by reading a random
n=30 sample of the headings, which is reported below.

| Class combination | blocks |
|---|---|
| EVIDENCE/PROSE + GUARD/TEST | 60 |
| EVIDENCE/PROSE only | 34 |
| GUARD/TEST only | 20 |
| EVIDENCE/PROSE + GUARD/TEST + PRODUCT-CODE | 16 |
| unclassified | 10 |
| GUARD/TEST + PRODUCT-CODE | 8 |
| EVIDENCE/PROSE + PRODUCT-CODE | 4 |
| **PRODUCT-CODE only** | **2** |

**PRODUCT-CODE-only = 2 of 154 = 1.3%.** Even the maximal reading (any block mentioning
product code at all) is 30/154 = 19.5%, and 28 of those 30 co-mention evidence or guards.

Random n=30 sample of headings, verbatim, confirms the direction and shows the mechanism:

- `[86.109] ### FINDINGS (all EVIDENCE-class; ZERO product defects found)`
- `[86.110] ### FINDING F3 (WARN, NEW RISK INTRODUCED BY THIS DIFF) -- the guard's repair can`
- `[86.110] ### FINDING F2 (evidence staleness) -- the FULL-SUITE block predates the new tests`
- `[86.116] ### FINDING F3 (WARN, guard vacuity -- EXECUTED, tree untouched)`
- `[86.71] ## FINDING F1 (mutation-proven, WARN) -- 1 of the 3 NEW self-test checks is a TAUTOLOGY`
- `[86.71] ## FINDING F2 (NOTE/WARN) -- the new ERROR-on-import guard is MARKER-based, not outcome-based`
- `[86.108] ### FINDING P1 (guard coverage): the AST guard is defeated by any non-literal`
- `[86.79] ### FINDING F2 (WARN): the new 4b/4c pins are whole-file BYTE-PRESENCE pins`
- `[86.84] ### FINDING B -- the fix is INERT on the live corpus (Unjustified_Inference)`
- `[86.94] ## FINDING C (REAL, MY OWN CELLS) -- the provenance check IS circular`
- `[86.120] ### FINDING 1 (BLOCKING) -- M10 survives: criterion 2's production wiring has ZERO coverage`
- `[86.85] ### FINDING 2 (BLOCKING): a REAL surviving mutant on a LIVE branch -- \`_dedup_key\``
- `[86.78] ### FINDING F4 (WARN): \`escalation.override\` is structurally unsettable`
- `[86.116] ### FINDING F2 (MATERIAL, criterion 6): the quantified gate mechanism credits a`
- `[86.108] ### FINDING E4 (disclosure): an UNWIRED emit site listed as equivalent`
- `[86.71] ### FINDING 5 (scope honesty, WARN) -- "every Layer-3 run originates" is false`

**THE ANSWER TO THE CALLER'S PRIMARY QUESTION.** The defects Main introduces within a step
are overwhelmingly **NOT product bugs**. They are defects in the *apparatus Main builds to
PROVE the step* -- the guard, the mutation matrix, the evidence capture, the disclosure
prose. The recurring shapes are: a guard that is vacuous/tautological/marker-based rather
than outcome-based; a pin that is byte-presence rather than semantic; a fix that is INERT
on the live corpus; a capture whose prose figure has drifted from its own command output;
a disclosure that overstates scope.

## ROUND 2 -- INTERNAL: a large share of CONDITIONALs are NOT defect reports at all

Population = every `handoff/current/evaluator_critique_*.md` whose LAST `verdict:` token
is `CONDITIONAL` (n = **32**). Counting rule: a critique "grants every criterion" if the
literal string `NOT MET` occurs **zero** times in it.

**24 of 32 (75%) CONDITIONAL critiques contain ZERO `NOT MET` occurrences** -- i.e. the
judge granted every immutable criterion and returned CONDITIONAL anyway, on residual
quality observations. Examples with (MET, NOT-MET) counts: `86.108` (11, 0), `86.110`
(9, 0), `86.29` (12, 0), `85.5.1` (15, 0), `82.10` (12, 0), `85.4` (12, 0).

This corroborates and slightly sharpens the standing `86.98` record (57% on a different
population). **Consequence: the CONDITIONAL RATE IS NOT A MEASUREMENT OF MAIN'S DEFECT
RATE.** Three quarters of the CONDITIONAL signal that drives the fix-fix-fix loop carries
no unmet criterion at all -- yet the harness treats CONDITIONAL as non-terminal and
re-spawns, and the 3rd-CONDITIONAL rule then converts a converging step into a PARKED one.

## ROUND 2 -- INTERNAL: THE APPARATUS TAX (the single strongest measurement in this brief)

**Rule.** `git log --since=2026-07-01 --numstat`; for every commit, sum ADDED lines by path
prefix into three buckets: `handoff/` (evidence artifacts), `scripts/qa/` (verification
apparatus), `backend/|frontend/` (production code). A "day" counts only if it has at least
one commit touching any bucket. Binary/`-` numstat rows skipped.

| Window | active days | `handoff/` added/day | `scripts/qa/` added/day | `backend/`+`frontend/` added/day | apparatus : product |
|---|---|---|---|---|---|
| 2026-07-01 .. 08-10 | 25 | 19,051 | **430** | **2,658** | **0.16 : 1** |
| 2026-08-11 .. 08-18 | 8 | 22,177 | **2,958** | **1,082** | **2.73 : 1** |

**The ratio of verification-apparatus code to production code moved from 0.16 to 2.73 --
a 17x shift -- in exactly the window where the PASS rate collapsed.** Production output
per day FELL 59% (2,658 -> 1,082) while checker output per day ROSE 588% (430 -> 2,958).
Raw totals: `scripts/qa/` 10,769 lines added across the 25 pre-days vs **23,669 across the
8 post-days** -- more than double the absolute volume in less than a third of the time.

**This closes the loop with the defect-class census.** 98.7% of Q/A findings land on
evidence + guards, and 73% of all new non-`handoff` code IS evidence + guards. The
harness's defect surface MIGRATED. Main is not writing worse product code -- Main is
writing 2.7x more *verification apparatus* than product per day, apparatus is net-new
code authored under the same time pressure, and **the apparatus has no verifier of its own
except the Q/A.** Every defect in it therefore lands on the Q/A's desk as a CONDITIONAL.

**And the masterplan MANDATES more of it.** Every one of steps 87.1-87.5 carries the
success criterion *"mutation-test every new guard: control observed GREEN first ... and a
byte-identical SHA-256-verified restore"* (`.claude/masterplan.json`, 87.1-87.5
`verification.success_criteria`, last element of each). So each step must now ship: a fix,
a guard for the fix, a mutation matrix proving the guard is falsifiable, a control run, a
verified restore, and prose quoting all of it. **Five of those six deliverables are
apparatus.** This is a positive feedback loop, not a steady state.

## ROUND 3 -- INTERNAL: does iterating FIND NEW THINGS, or RE-FIND the same things?

**First, a probe I am REPORTING AS UNSOUND rather than quoting.** I ran a lexical probe for
findings that blame the current cycle's own diff (`NEW RISK INTRODUCED BY THIS DIFF`,
`introduced by this fix|cycle|change`, `the fix is INERT|vacuous|tautological|circular`,
`inherited the defect it replaced`). It returned **4 of 173 spawns (2%)**. Reading the
excerpts shows the probe is **matching DENIALS as often as affirmations** -- e.g.
`[86.88] "...pre-existing, out of scope, NOT introduced by this cycle"` and
`[86.90] "...the container guard is PRE-EXISTING, not a regression introduced here"`.
A regex that cannot tell an assertion from its negation is not a measurement
(`feedback_suspect_the_clean_check`). **I do NOT claim a 2% self-inflicted rate.** The
sound measurement is the cross-cycle novelty test below.

**THE DECISIVE INTERNAL MEASUREMENT.** Rule: for each step with `>=3` timestamped WIP
spawns (n = 33 steps), take every `FINDING|BLOCKER|VIOLATION|DEFECT` heading, strip the
finding-id token, normalise to lowercase alpha, and call a heading NOVEL if its
`difflib.SequenceMatcher` ratio against every heading seen in ALL PRIOR spawns of that same
step is `<= 0.72`.

- **169 finding-headings; 145 NOVEL (86%); only 24 (14%) re-found.**
- **Restricted to cycle 3 and later: 100 headings, 83 NOVEL (83%).**

**This REFUTES the "the Q/A is just re-litigating / verdict-shopping" hypothesis, and it
refutes "the step is stuck".** By cycle 5, 7, 10 the Q/A is still surfacing *new* material
at the same rate as at cycle 1. Worked examples:
`86.84` (11 spawns) raises 5 new at cycle 4, 1 at c6, 5 at c7, 6 at c8, 1 at c9, 1 at c10.
`86.85` (12 spawns) raises new findings at c1, c3, c6, c7, c10, c11.
`86.78` (8 spawns) raises 6 new at c4, 2 at c6, 1 at c7.

**The loop does not converge because each REMEDIATION ADDS NEW UNVERIFIED SURFACE.** That
is the mechanism, and it is the internal instantiation of the Error-Introduction-Rate model
in `arXiv:2604.22273v2` (below) -- except the errors are being introduced into the
*verification apparatus*, not the product.

## ROUND 3 -- INTERNAL: remediation cycles are NOT small targeted fixes

`git show --shortstat` on every commit whose subject greps `phase-<sid>`, for the 8
highest-cycle steps. Representative per-cycle insertion counts:

- `86.85`: 1213, 995, 807, 683, 569, 372, 326, 163, 28
- `86.84`: 1389, 971, 558, 390, 183, 85, 71
- `86.78`: 1245, 699, 306, 295, 187, 155, 137, 130, 117, 106, 65, 13
- `86.74`: 291, 281, 261, 253, 237, 118, 81, 81, 78, 61, 25

**A "remediation" of a single CONDITIONAL routinely lands 300-1400 inserted lines across
4-13 files.** That is not a targeted fix; it is a fresh feature-sized body of code and
prose, written to close observations that in 75% of cases came attached to *no unmet
criterion*. Anthropic's harness-design post prescribes the opposite -- *"the
one-feature-at-a-time approach ... worked well for scope management"* and
*"decomposing the build into tractable chunks"*
(https://www.anthropic.com/engineering/harness-design-long-running-apps, accessed
2026-08-18). **Every one of those 300-1400 lines is new surface for the next Q/A.** The
86% cross-cycle novelty rate is the direct consequence, and the two measurements are
mutually confirming.

## ROUND 4 -- INTERNAL: what actually TERMINATES a step now

`grep -E 'PARKED \((3rd-CONDITIONAL|attempt budget|budget exhausted)' handoff/harness_log.md`:

```
Cycle 1244 -- 2026-08-17 -- phase=75.11.4 result=PARKED (budget exhausted)
Cycle 1247 -- 2026-08-17 -- phase=86.108  result=PARKED (3rd-CONDITIONAL rule)
Cycle 1249 -- 2026-08-18 -- phase=86.110  result=PARKED (3rd-CONDITIONAL rule)
Cycle 1250 -- 2026-08-18 -- phase=86.47   result=PARKED (attempt budget exhausted 5/5)
Cycle 1251 -- 2026-08-18 -- phase=86.59   result=PARKED (3rd-CONDITIONAL rule)
Cycle 1252 -- 2026-08-18 -- phase=86.116  result=PARKED (3rd-CONDITIONAL rule)
Cycle 1254 -- 2026-08-18 -- phase=86.120  result=PARKED (3rd-CONDITIONAL rule)
```

**7 PARKs in 2 days; 5 of them by the 3rd-CONDITIONAL rule, 2 by the attempt budget. ZERO
by a FAIL on an unmet criterion.** And three steps closed only because a human overrode the
loop: `86.85` (*"CLOSED by operator authorization -- residuals queued as 86.107"*), `86.90`,
`86.75` (*"CLOSED by operator authorization -- all 8 criteria MET, cap
unrepairable-history"*). **The termination rules are now the binding constraint on step
closure, not verdict quality** -- steps whose every criterion is MET are being parked by a
counter. This corroborates the standing record
`project_third_conditional_rule_parks_converging_steps`.

**A probe I ran and am REPORTING AS UNSOUND:** counting Workflow run records
(`~/.claude/projects/.../workflows/wf_*.json`, n=635) that contain the string
`without calling StructuredOutput` returns 32/33 on 08-16, 67/67 on 08-17 and 29/29 on
08-18. A 100% drop rate is implausible on its face, and the agent-file text now quoted into
every prompt itself contains that phrase, so the probe is matching its own input
(`feedback_a_probe_can_match_its_own_documentation`). **No rail-drop rate is claimed from
it.** The run-count column IS sound: **232 Workflow runs since 2026-08-11**, 67 on 08-17
alone.

---

## EXTERNAL SOURCES -- READ IN FULL via WebFetch (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2604.22273v2 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Self-correction as feedback control. Defines **EIR** = P(correct -> incorrect after refinement) and **ECR** = P(incorrect -> correct). Theorem 1 equilibrium: `ECR(k)/EIR(k) = Acc(k)/(1-Acc(k))`. At 96% baseline accuracy you need **ECR/EIR ~24x just to break even**. Empirically only models with **`EIR <~ 0.5%`** benefit; 5 of 8 degraded. GPT-4o-mini at 2% EIR lost **-6.2 pp**. A verify-first prompt drove EIR 2% -> 0% and converted -6.2 pp into +0.2 pp (p<1e-4) -- **EIR is the causal control variable.** |
| 2 | https://arxiv.org/html/2604.01029 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Decomposes second-pass gain into re-solving + scaffold + **content**. On code (LiveCodeBench) the **content effect is significantly NEGATIVE**: -3.1 pp (Pair 1), **-7.9 pp** (Pair 2), worsening with difficulty (easy -0.6, medium -3.4, **hard -5.1**). *"the null scaffold outperforms the standard revision pipeline, and the content effect is significantly negative."* Mechanism named: **"artifact-level anchoring"** -- a weak draft traps the reviser in a suboptimal trajectory. |
| 3 | https://arxiv.org/html/2406.01297v3 | 2026-08-18 | peer-reviewed (TACL) | WebFetch (arXiv HTML) | *"no prior work demonstrates successful self-correction with feedback from prompted LLMs, except ... tasks exceptionally suited for self-correction."* Intrinsic self-correction *"does not improve or even degrade the performance"* on code generation. Self-correction works only with **external, verifiable feedback** (code interpreters, search, symbolic reasoners) or 100K+ fine-tuning instances. |
| 4 | https://arxiv.org/html/2606.15474 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **[ADVERSARIAL / METHOD-CRITICAL to this very brief.]** *"a silent version bump or scoring-prompt update changes how it scores -- so every drift alarm is ambiguous between a worse product and a changed judge."* Requires a **frozen anchor set** re-scored by the current judge; *"only a change in the judge can move it."* A strict-prompt regrade was correctly attributed to the JUDGE on 110/120 contaminated runs; a naive rolling z-test **false-alarms on 75% of drift-free streams**. |
| 5 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-18 | official vendor doc | WebFetch | *"Separating the agent doing the work from the agent judging it proves to be a strong lever."* But also: *"the evaluator is still an LLM that is inclined to be generous"* -> tuning it skeptical is the lever. Scope: *"the one-feature-at-a-time approach ... worked well for scope management"*, *"decomposing the build into tractable chunks"*. And: *"Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing ... they can quickly go stale."* |
| 6 | https://www.anthropic.com/engineering/multi-agent-research-system | 2026-08-18 | official vendor doc | WebFetch | Effort must be **scaled to complexity**: *"Simple fact-finding requires just 1 agent with 3-10 tool calls ... complex research might use more than 10 subagents."* Names **"overinvestment in simple queries"** as *"a common failure mode."* Cost: *"multi-agent systems use about 15x more tokens than chats"*; *"token usage by itself explains 80% of the variance."* |
| 7 | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 2026-08-18 | official vendor doc | WebFetch | **Directly indicts a 897-line evaluator prompt.** *"context rot: as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases."* Warns against *"hardcoding complex, brittle logic in their prompts"* which *"creates fragility and increases maintenance complexity"*, and against *"a laundry list of edge cases"*. Prescribes the **"right altitude"** and *"the minimal set of information that fully outlines your expected behavior."* |
| 8 | https://arxiv.org/html/2603.11078v1 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | CR-Bench. Adding a reflexion/critic loop raised recall **27.01% -> 32.76%** but collapsed signal-to-noise **5.11 -> 1.95** (and to **0.91** on a smaller model). *"While the Reflexion paradigm significantly enhances discovery Recall, it simultaneously incurs a steep cost in signal integrity."* **Strictness buys recall by paying in noise.** |
| 9 | https://arxiv.org/html/2604.10508 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **[ADVERSARIAL -- disagrees with #1 and #2.]** *"two repair rounds capture the majority (76-95%) of achievable gains"*; *"R4 yields no additional improvement over R3"*; and explicitly *"Self-repair improves pass rates for every model tested ... No evidence of later iterations introducing performance regressions."* Self-repair also **cheaper than resampling** (11-54% token saving). |
| 10 | https://arxiv.org/html/2606.29718 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Context rot in long-horizon agents, measured. Baseline ReAct on BrowseComp: 35.0% accuracy with a **53.4% premature-termination rate**. Turn-triggered compaction: 46.6% accuracy, premature termination **53.4% -> 1.8%**. **Context isolation via sub-agents: 54.0% (+19 pts)** -- the strongest mitigation for strong models. |
| 11 | https://arxiv.org/html/2411.10213v2 | 2026-08-18 | peer-reviewed (ICSE 2026) | WebFetch (arXiv HTML) | Six agentic bug-fixers on SWE-bench Verified (500). **19.2% (96/500) resolved by NO system.** Names the validation gap explicitly: *"F2P tests may not be comprehensive, allowing a patch to pass F2P and be deemed correct without fully addressing the user's issue."* No measurement of introduced regressions -- a stated gap in the literature. |
| 12 | https://arxiv.org/html/2508.00083v1 | 2026-08-18 | preprint (survey) | WebFetch (arXiv HTML) | *"code generated by agents often contains logical defects, performance pitfalls, or security vulnerabilities that are difficult to cover with unit tests."* Names **trajectory efficiency** as a first-class metric: *"An efficient agent should reach the goal through a minimal and effective sequence of actions, avoiding redundant operations."* Context loss in multi-step workflows: *"key information from earlier steps can be easily lost."* |
| 13 | https://arxiv.org/html/2510.22249 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Self-admitted technical debt in TEST code across 50 repos: 2,779 test vs 14,987 production SATD instances. Test-originated issues (348) dominated by **"Test Completeness" (137, of which 120 = "incomplete or unimplemented tests")** and **"workarounds" (53)**. Test methods carrying SATD have higher cyclomatic complexity (1.72 vs 1.29) and more code smells. **The single most common debt in test code is a test that does not actually test.** |

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 14 | https://arxiv.org/html/2603.24755v1 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **SlopCodeBench -- the single closest external analogue to what this harness is doing.** Structural erosion *"rises in 80% of trajectories"*, mean high-complexity function count **4.1 -> 37.0**; verbosity *"rises in 89.8%"*. Agent erosion **0.68+/-0.20 vs human 0.31+/-0.12 (2.2x worse)**; verbosity **0.32+/-0.11 vs 0.11+/-0.07 (3x worse)**. *"Cost grows 2.9x across checkpoints while correctness declines"* -- *"additional spending does not improve correctness."* **And prompting does not fix it:** anti-slop prompts cut initial verbosity 34.5% but *"the accumulation of issues persists regardless of prompt"* with degradation slopes *"largely parallel"* and *"no difference in any pass-rate subtype."* Human repos *"plateau"*; agent code *"climb[s] monotonically."* |
| 15 | https://arxiv.org/html/2603.25773 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **The Correlated Error Hypothesis.** When an AI reviewer grades AI-written code with no external specification, both *"reason from the same artefact"* and *"share the same training distribution"*: *"They are not two independent estimators. They are two samples from the same prior."* Without ground truth *"the review checks code against itself, not against intent."* Also **Residual Defect Category E**: *"No verification pipeline catches Category E defects because the pipeline verifies conformance to the specification."* Measured blind spots: ICD-10-CM rule missed **0/20** across four model families. |

## Identified but SNIPPET-ONLY (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/html/2601.00509 | preprint | RAG+multi-tool feedback for secure codegen; security-specific (58.55% -> 22.19% defect rate) -- adjacent, not on the CONDITIONAL/FAIL question |
| https://conf.researchr.org/details/icse-2026/icse-2026-research-track/227/ | conference listing | Landing page for source #11, which was read in full at arXiv |
| https://arxiv.org/pdf/2501.19204 | preprint | Legacy web-app upgrades via MAS; different task shape |
| https://openreview.net/pdf?id=6RmpFMEeOX | workshop paper | ICLR 2026 Agents-in-the-Wild; PDF, superseded by #12's survey coverage |
| https://arxiv.org/html/2608.05643 | preprint | Test-time self-correction by refining over resampling; overlaps #9 |
| https://arxiv.org/pdf/2602.04288 | preprint | "Contextual Drag" -- errors in context degrade reasoning; mechanism covered by #7 + #10 |
| https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/ | peer-reviewed | TACL landing page for source #3 (read in full at arXiv) |
| https://sback.it/publications/icsme2018a.pdf | peer-reviewed (ICSME'18) | **PDF -- deliberately NOT WebFetched** per the standing measurement that WebFetch PDF summaries fabricate quotes. Snippet: smelly tests carry **81% higher defect risk** and **47% higher change-proneness**. Recorded as snippet-only, not counted. |
| https://arxiv.org/pdf/2107.13902 | preprint | Developer perception of test-smell severity; PDF form, covered by #13 |
| https://arxiv.org/pdf/2606.22082 | preprint | CodeTeam repo-level multi-agent codegen; architecture, not defect dynamics |
| https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf | vendor report | PDF; same no-PDF-WebFetch rule |
| https://deepchecks.com/llm-judge-calibration-automated-issues/ | community/vendor blog | Tier-5; superseded by #4 which is the rigorous treatment |
| https://galileo.ai/blog/calibrate-llm-judge-human-annotations | vendor blog | Tier-5; same |
| https://arxiv.org/html/2602.08672v1 | preprint | LLMs designing + applying rubrics; adjacent to #4 |
| https://www.morphllm.com/context-rot | vendor blog | Tier-5 restatement of #7/#10 |
| https://arxiv.org/pdf/2401.13407 | preprint | Returns of highly maintainable code; PDF, background only |
| https://www.augmentcode.com/guides/agent-run-development-loop | vendor guide | Tier-5 spec-driven-development marketing |
| https://tryzeroshot.com/blog/spec-driven-development-with-ai-coding-agents | community blog | Tier-5; the specify/plan/task/implement framing is already in #5 |

**URL count: 15 read in full + 18 snippet-only = 33 unique URLs.**

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Queries run, showing the three-variant discipline:

1. **Current-year frontier (2026):** `LLM agent code generation first-pass defect rate self-introduced bugs evaluator feedback loop 2026`; `context rot long prompts degrade instruction following LLM agents 2026 attention dilution many instructions`; `reduce agent code iteration cycles plan first specification contract before implementation empirical 2025 2026`.
2. **Last-2-year window (2024-2025):** `self-refine iterative repair LLM degrades performance overcorrection more iterations worse code` (surfaced the 2024 TACL survey #3 and the 2026 follow-ups); `LLM-as-judge rubric strictness drift evaluator calibration length of judge prompt degrades agreement`.
3. **Year-less canonical:** `test code defect density higher than production code test smells maintenance burden empirical study` (surfaced ICSME'18 test-smell prior art, 2018-2021 canon).

**RESULT: 5 findings from the last-2-year window that SUPERSEDE the older canon, and they
change the conclusion.**

1. **The 2024 canonical position was "self-correction fails without external feedback"**
   (#3, TACL). **The 2026 refinement is quantitative and sharper**: it is not that
   self-correction fails, it is that it obeys a *stability threshold* -- benefit requires
   `EIR <~ 0.5%` and `ECR/EIR` above `Acc/(1-Acc)` (#1). This converts a qualitative
   warning into a design target you can measure.
2. **NEW (2026): the "content" of a prior draft is actively HARMFUL on code tasks** --
   -3.1 to -7.9 pp, worsening with difficulty, via *artifact-level anchoring* (#2). No
   2024 source states this; it directly explains why cycle-N artifacts make cycle-N+1
   worse rather than better.
3. **NEW (2026): SlopCodeBench measures long-horizon iterative degradation directly** and
   finds prompt interventions do NOT change the degradation slope (#14). This is the
   single most important recency finding for this step, because the harness's current
   response to every CONDITIONAL is *more prose instruction*.
4. **NEW (2026): judge-vs-system drift attribution is a solved methodological problem**
   requiring a frozen anchor set (#4). A naive trend test false-alarms on **75%** of
   drift-free streams. Directly applicable, and directly indicts a naive reading of this
   harness's own PASS-rate collapse.
5. **NEW (2026): the Correlated Error Hypothesis** (#15) -- an AI judge grading AI code
   without an external spec is not an independent estimator. This is a 2026 formalisation
   of the exact failure this harness is exhibiting.

**Superseded:** the "just add more evaluator rigor" intuition implicit in the 2018-2021
code-review canon is superseded by #8 (recall up, SNR **5.11 -> 1.95**) and #14 (prompting
does not change the slope).

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 16 | https://arxiv.org/html/2511.05524 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **[ADVERSARIAL -- the strongest case AGAINST cutting the evidence apparatus.]** EviBound's dual approval+verification gates take hallucinated completion claims from **100% (8/8) -> 25% -> 0% (0/8)** at *"~8.3% total execution time"* overhead: *"8.3% time overhead -> 100pp hallucination reduction."* *"Research integrity is an architectural property -- achieved through governance gates rather than emergent from model scale."* Self-reflection prompting alone gave **100% hallucination**. |
| 17 | https://arxiv.org/html/2510.10931 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Proof-of-Use / tool-call hacking. Defines the Goodhart failure precisely: the policy *"attains high training rewards by satisfying tool-call format requirements ... while its stepwise reasoning and action behavior does not causally depend on the content returned by the tools."* The mitigation is **causal dependence** (perturb the cited evidence and require the answer to move), not more format rules. |

**Updated URL count: 17 read in full + 18 snippet-only = 35 unique URLs.**

---

## KEY FINDINGS

**F1. The primary question rests on a confounded comparison, and the confound is
measurable.** The PASS rate fell 95.4% -> 39.3% (graded-only denominator) in the same
window in which the evaluator specification `.claude/agents/qa.md` grew **556 -> 897 lines
(+61.3%)**, its obligation-language density rose **25 -> 33 lines (+32%)**, the `maxTurns`
cap that had been truncating Q/A evaluations was REMOVED (`85127353`, 2026-08-14), and
per-spawn Q/A observability (`verdict_wip_*`) was created for the first time (`d23a981e`,
2026-08-10). *"every drift alarm is ambiguous between a worse product and a changed judge"*
(https://arxiv.org/html/2606.15474). A naive trend read false-alarms on **75%** of
drift-free streams in that paper's own control. **No claim that "Main's code quality
declined" is supportable from the current evidence.**

**F2. The defects Main introduces in-step are almost entirely in the VERIFICATION
APPARATUS, not the product.** Of 154 finding-blocks across 27 steps since 2026-08-14, only
**2 (1.3%)** are product-code-only; 94% touch evidence and/or guards. The recurring shapes
are a guard that is vacuous/tautological/marker-based, a byte-presence pin standing in for
a semantic one, a fix that is INERT on the live corpus, a capture whose prose figure has
drifted from its own output, and a disclosure that overstates scope.

**F3. THE MECHANISM: the harness's defect surface migrated into the apparatus, because that
is where the code now is.** `scripts/qa/` : `backend/`+`frontend/` added-lines-per-day went
from **0.16:1 to 2.73:1** across the break -- a **17x shift**. Production output per day
fell 59%; checker output rose 588%. **The apparatus is now the majority of what Main writes,
it is net-new code written under the same time pressure, and it has NO verifier of its own
except the Q/A.** Every defect in it therefore surfaces as a CONDITIONAL.

**F4. The loop does not converge because each remediation ENLARGES the surface.**
Cross-cycle novelty is **86%** (145/169 finding-headings novel vs all prior spawns of the
same step); **83%** even restricting to cycle 3+. Meanwhile a single remediation commit has
a **median of 286 and a mean of 465 inserted lines** across the 8 highest-cycle steps
(n=80 commits; **45% are >=300 lines; 10 are >=1000**). This is the Error-Introduction-Rate
regime of https://arxiv.org/html/2604.22273v2 applied to evidence: with `EIR` above the
`~0.5%` threshold, *"for high-accuracy models, pi* can be far below baseline, making all
self-correction harmful."* It is also the *artifact-level anchoring* of
https://arxiv.org/html/2604.01029 -- content effect **-3.1 to -7.9 pp on code**, worse on
harder problems.

**F5. Three quarters of CONDITIONALs report NO unmet criterion.** 24 of 32 CONDITIONAL
critiques in `handoff/current/` contain **zero** `NOT MET` strings. Yet CONDITIONAL is
non-terminal, so it triggers a ~465-line remediation and a fresh Q/A. **The signal driving
the most expensive action in the harness is, three times out of four, not a criterion
failure at all.** This is exactly CR-Bench's finding: reflexion lifted recall
27.01% -> 32.76% while SNR fell **5.11 -> 1.95**
(https://arxiv.org/html/2603.11078v1) -- *"a steep cost in signal integrity."*

**F6. The harness's standard response to a CONDITIONAL -- add more instruction prose -- is
externally measured NOT to work.** SlopCodeBench: quality-aware prompting cut initial
verbosity 34.5% but *"the accumulation of issues persists regardless of prompt"*, slopes
*"largely parallel"*, and *"no difference in any pass-rate subtype"*
(https://arxiv.org/html/2603.24755v1). Independently, a 897-line evaluator prompt is
squarely in the regime Anthropic warns about: *"hardcoding complex, brittle logic in their
prompts ... creates fragility"*, *"a laundry list of edge cases"*, *"context rot"*
(https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

**F7. Main and Q/A are not independent estimators on evidence-class questions.** *"They are
not two independent estimators. They are two samples from the same prior ... the review
checks code against itself, not against intent"*
(https://arxiv.org/html/2603.25773). For PRODUCT questions pyfinagent has external ground
truth (tests, BQ rows, a live book) and the separation holds. For EVIDENCE questions --
"is this prose figure right", "is this guard vacuous" -- the only referent is Main's own
artifact, so the pair degenerates into one estimator sampled twice. **That is why 94% of
findings are evidence/guard class and why they never run out.**

**F8. The termination rules, not verdict quality, now close steps.** 7 PARKs in 2 days: 5
by the 3rd-CONDITIONAL rule, 2 by the attempt budget, **0 by a FAIL on an unmet criterion**;
plus 3 steps closed only by explicit operator override (86.85, 86.90, 86.75 -- the last
recorded as *"all 8 criteria MET"*).

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 18 | https://arxiv.org/html/2606.10106v1 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | Necessary+sufficient conditions for a harness: agent loop, tool interface, **context management**, and *"control mechanisms, that is, limits, verification, and deterministic actions."* The prescription: *"detect the divergence between what the agent claims and the real state, verify that state deterministically, and run the sensitive parts with ordinary code rather than trust the model's word."* And the open tension, stated as an open question: *"How much of the harness is reusable across domains and how much must be bespoke?"* with *"the evidence suggests that effective control is problem-specific."* |
| 19 | https://arxiv.org/html/2606.27243v1 | 2026-08-18 | preprint (industrial) | WebFetch (arXiv HTML) | **NOVA -- the closest working answer to "how do I cut the iteration count".** A *"silent-failure-aware multi-stage verification cascade"* using **transferable semantic gates** rather than bespoke per-task validation. Introduces **Silent Failure Rate (SFR)** = runnable-but-ineffective candidates -- pyfinagent's "guard that cannot fail" / "fix is INERT" class, made a first-class METRIC. Local Pass Rate **99.0% vs 33.3%** (OpenHands) on L2; **86.7% vs 27.3%** on L3. Effective Pass Rate = `LPR x (1-SFR)`: **60.0% vs 31.1% (human loop) vs 10.2%** on L3. And the anti-re-finding mechanism: *"When a candidate is rejected, its failure pattern is stored in trajectory memory H and used as a forbidden direction in later rounds ... reducing repeated semantic failures."* Human-attended time **54u -> 4u (13.5x)**. Ablation: removing the design stage collapses EPR **60.0% -> 18.2%** -- **planning is the highest-leverage stage, not more checking.** |

**Updated URL count: 19 read in full + 18 snippet-only = 37 unique URLs.**

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 20 | https://arxiv.org/html/2607.03691 | 2026-08-18 | preprint | WebFetch (arXiv HTML) | **[DECISIVE -- and it is about harnesses, not models.]** *"Don't Blame the Large Language Model."* Holding the LLM CONSTANT across **35 sequential harness releases** (Mar 2025 - Feb 2026): *"despite continuous development activity and growing codebase complexity of the agent harnesses, there is no statistically significant improvement in SWE-bench benchmark score"* (**Spearman rho=0.208, p=0.231**). Meanwhile token use **+70% (391K -> 668K per task)**, tool calls **6.9 -> 14.3**, system prompt **+8%**, conversation turns **+18%** -- *"without corresponding gains in task resolution."* Root cause named: *"the absence of Agentic Quality Assurance"* -- QA of NON-functional properties (token efficiency, tool overhead), because the regressions *"passed all existing automated checks."* |

**Final URL count: 20 read in full + 18 snippet-only = 38 unique URLs.**

---

## IS 87.6's PATTERN THE SAME CLASS AS 87.1-87.5, OR DISTINCT? (the caller's explicit question)

Mapping the 154 post-08-14 finding-blocks onto what each existing step actually covers
(keyword buckets, NON-exclusive, proxy -- labelled as such):

| Existing step | scope | finding-blocks it would cover |
|---|---|---|
| 87.4 guardlib / vacuous guard | `vacuous, tautolog, cannot fail, guard, mutation, known-bad, falsif, _ok(` | **70 (45%)** |
| 87.1 prose<->capture drift | `stale figure, prose, drift, hand-authored, typed, figure, capture, reproduce, quote` | **67 (44%)** |
| 87.3 verdict-ledger write | `ledger, verdict_history, no_rows_for_step` | **64 (42%)** |
| 87.2 pre-spawn class sweep | `sweep, pre-spawn, pre_spawn, class sweep` | 11 (7%) |
| 87.5 moot / wrong steps | `moot, nonexistent step, re-scope, disposition` | 3 (2%) |
| **none of the above** | -- | **15 (10%)** |

**ANSWER: MOSTLY THE SAME CLASS, WITH ONE GENUINELY DISTINCT RESIDUAL AND ONE MISSING
SYSTEMIC LAYER.**

**(a) Same class.** ~90% of what the Q/A finds is covered in principle by 87.1 (prose/capture
drift) and 87.4 (guard vacuity). 87.6 is NOT a new defect taxonomy.

**(b) The DISTINCT residual: the SILENT-FAILURE class.** Residual vocabulary the 87.1-87.5
scopes do not name: `survives` (12), `inert` (9), `surviving mutant` (7), `undisclosed` (7),
`false-negative` (6), `dead` (5), `unwired` (4), `not wired` (2), `zero coverage` (2),
`structurally unsettable` (2). Verbatim examples:

- `### FINDING 1 (BLOCKING) -- M10 survives: criterion 2's production wiring has ZERO coverage`
- `### FINDING B -- the fix is INERT on the live corpus (Unjustified_Inference)`
- `## FINDING F2 (NOTE/WARN) -- the new ERROR-on-import guard is MARKER-based, not outcome-based`
- `### FINDING F2 (WARN): the new 4b/4c pins are whole-file BYTE-PRESENCE pins`
- `### FINDING 1 (BLOCKING, criterion 8) -- the mutation matrix is NON-DISCRIMINATING`
- `### FINDING E4 (disclosure): an UNWIRED emit site listed as equivalent`

**87.4 targets a guard that CANNOT FAIL (vacuity). This residual is a guard that CAN fail,
runs, goes green -- and covers none of the real population (inertness).** Those are
different defects with different detectors. **This is exactly NOVA's `Silent Failure Rate`:
"runnable but AUC-negative candidates"** (https://arxiv.org/html/2606.27243v1), which NOVA
elevates to a first-class metric alongside pass rate precisely because pass rate cannot see
it. pyfinagent measures no analogue.

**(c) The MISSING SYSTEMIC LAYER, which no 87.x step touches.** None of 87.1-87.5 addresses
(i) the **apparatus:product ratio** (0.16 -> 2.73), (ii) the **judge-vs-system drift
confound** (qa.md +61.3% in the same window), or (iii) a **non-functional QA** of the
harness itself -- the very gap arXiv:2607.03691 names as the root cause of harness churn
producing *"no statistically significant improvement"* at **+70% token cost**. **87.1-87.5
are five more pieces of apparatus.** Adopted as written -- each ending in *"mutation-test
every new guard: control observed GREEN first ... byte-identical SHA-256-verified
restore"* -- they add to the ratio that is generating the findings.

## ROUND 5 -- INTERNAL: CRITERIA INFLATION -- a mechanical PASS-rate driver requiring NO change in code quality

Rule: for every masterplan step id matching `^\d+(\.\d+)+$` that carries a
`verification.success_criteria` list, count list length and total characters; group by
top-level phase; report only phases with `>=4` such steps.

| phase | steps | mean #criteria | mean chars |
|---|---|---|---|
| 4 | 136 | 3.26 | 104 |
| 73 | 21 | 3.00 | 433 |
| 79 | 55 | **2.58** | 397 |
| 80 | 47 | 4.43 | 621 |
| 85 | 4 | 5.50 | 1,212 |
| **86** | **125** | **6.39** | **1,383** |
| 87 | 5 | 5.20 | 1,030 |
| **88** | 4 | **7.00** | **1,692** |

Within phase-86, monotone in the step number:

| sub-range | n | mean #criteria | mean chars |
|---|---|---|---|
| 86.0-19 | 18 | 6.28 | 1,334 |
| 86.20-39 | 20 | 6.30 | 1,494 |
| 86.40-59 | 20 | 5.70 | 1,157 |
| 86.60-79 | 20 | 6.65 | 1,369 |
| 86.80-99 | 20 | 7.30 | 1,574 |
| 86.100-119 | 20 | 5.75 | 1,238 |
| **86.120-139** | 7 | **7.43** | **1,741** |

**From phase-79 (2.58 criteria / 397 chars) to phase-86.120+ (7.43 / 1,741) is +188% in
criteria COUNT and +338% in criteria TEXT.** A verdict is CONJUNCTIVE over criteria.
Holding per-criterion quality perfectly constant at an illustrative 90%, `P(all pass)` moves
from `0.90^2.58 = 78%` to `0.90^7.43 = 46%`. **That is a ~32-point PASS-rate drop produced
entirely by contract inflation.** I am labelling this an ILLUSTRATIVE ARITHMETIC BOUND, not
a measurement: criteria are not independent and 90% is a stipulated figure
(`feedback_a_labelled_inference_still_argues`). But it establishes that a large fraction of
the observed collapse is **arithmetically expected from the contract alone**, with Main's
per-criterion quality unchanged.

**Every instructed-reading file inflated in the same window** (`git show <sha>:<file> | wc -l`):
`CLAUDE.md` 378 -> **508** (+34%), `docs/runbooks/per-step-protocol.md` 435 -> **483**,
`.claude/rules/research-gate.md` 292 -> **337**, `.claude/agents/researcher.md` 371 -> **421**,
`.claude/agents/qa.md` 556 -> **897**. **The whole specification surface grew ~30-60% in
three weeks.** Anthropic: *"context rot: as the number of tokens in the context window
increases, the model's ability to accurately recall information from that context
decreases"*
(https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
arXiv:2607.03691 measured a harness system prompt growing only **8%** alongside **+70%**
token burn and **zero** benchmark gain; pyfinagent's evaluator prompt grew **61%** in nine
days.

---

## INTERNAL CODE INVENTORY (every claim above has a file anchor here)

| File | Size now | Role | Status / finding |
|---|---|---|---|
| `handoff/harness_log.md` | 36,833 L / 1,289 `## Cycle` headers | cycle ledger, feeds the Harness tab | The PASS-rate source. 696 graded verdicts all-time; 61 since 08-11. |
| `.claude/agent-memory/qa/verdicts/` | 180 files (173 timestamped) | per-spawn Q/A write-first records | **Created 2026-08-10 (`d23a981e`) / run-stamped 08-11 (`5595055c`)** -- so this corpus does not exist before the collapse window. 44 steps, mean 3.9 spawns/step, max 12. |
| `.claude/agents/qa.md` | **897 L** (was 556 on 08-09) | the evaluator specification | **+61.3% in 9 days**; obligation-language lines 25 -> 33. §`4b` Claim auditing (`:398`), §`4c` Guard-vacuity (`:441`), §`4a` adversarial N-lens (`:505`). |
| `.claude/agents/researcher.md` | 421 L (was 371 on 07-31) | research-gate spec | +13.5%; `maxTurns` removed 08-14 (`85127353`). |
| `.claude/rules/research-gate.md` | 337 L (was 292) | authoritative gate floors | +15% |
| `docs/runbooks/per-step-protocol.md` | 483 L (was 435) | operator runbook | +11% |
| `CLAUDE.md` | **508 L** (was 378) | always-loaded project instructions | **+34%** |
| `scripts/qa/` | **102 files, 34,082 lines** | the verification apparatus | **6 files on 2026-07-06 -> 8 (07-24) -> 9 (08-04) -> 17 (08-09) -> 102 today.** ~17x file growth, and 6x of it inside the collapse window. |
| `scripts/harness/attempt_budget.py` | 337 L | cumulative attempt budget | **NOW WIRED** via `scripts/harness/attempt_gate.py:84` (`from attempt_budget import ...`), a PreToolUse gate (phase-86.71). CLAUDE.md's "NOT YET WIRED" paragraph is STALE and should be corrected by whoever owns the next doc pass. |
| `scripts/harness/attempt_gate.py` | -- | PreToolUse attempt gate | Writes `handoff/audit/attempt_budget_audit.jsonl:89`; consumed by `scripts/harness/research_router.py:50`. |
| `scripts/qa/qa_wip.py` | 635 L | prior-attempt counter | The LIVE per-step bound; `qa.md:655` invokes it. |
| `scripts/qa/pre_spawn_gate.py` | 484 L | 87.2's target | **EXISTS already** -- 87.2 is a WIRING step, not a build step. |
| `scripts/qa/guardlib_selftest.py` | 1,035 L | 87.4's target | **EXISTS already** -- 87.4 is an ADOPTION step. |
| `handoff/verdict_ledger.jsonl` | 138 rows | 87.3's target | Exists and parses. |
| `.claude/workflows/research-gate.js` | 1,118 L | this gate's rail | `agentType:'researcher'` pin + `enforceGate()` recompute. |
| `.claude/workflows/qa-verdict.js` | 858 L | Q/A rail | Launch by `scriptPath`, never `name` (named dispatch is a session snapshot). |
| `.claude/masterplan.json` 87.1-87.5 | 5 steps, mean 5.20 criteria / 1,030 chars | the queued remediation | Each ends in *"mutation-test every new guard: control observed GREEN first ... byte-identical SHA-256-verified restore"*. |

## CONSENSUS vs DEBATE (external)

**Consensus (4+ independent sources):**
- Separating generator from evaluator is necessary but NOT sufficient; the evaluator must be
  tuned skeptical and given external ground truth (Anthropic harness-design; arXiv 2603.25773;
  arXiv 2406.01297v3).
- Intrinsic self-correction without external, verifiable feedback does not reliably help and
  can hurt (2406.01297v3; 2604.22273v2; 2604.01029).
- Scope/decomposition discipline dominates: one feature at a time, small chunks (Anthropic
  harness-design; Anthropic multi-agent-research on "overinvestment in simple queries"; NOVA's
  ablation showing Solution Design is the single highest-leverage stage, EPR 60.0% -> 18.2%
  without it).
- Long, rule-dense prompts degrade instruction-following (Anthropic context-engineering;
  2606.29718; 2607.03691).

**Genuine debate -- and it is directly load-bearing here:**
- **Does iteration help or hurt?** `arXiv:2604.10508` [ADVERSARIAL] measured *"Self-repair
  improves pass rates for every model tested ... No evidence of later iterations introducing
  performance regressions"*, with 76-95% of gains in the first two rounds. This DIRECTLY
  contradicts `2604.22273v2` (5 of 8 models degraded; -6.2 pp) and `2604.01029` (content
  effect -3.1 to -7.9 pp on code). **Reconciliation:** 2604.10508 iterates against
  EXECUTABLE test feedback on HumanEval/MBPP -- an external oracle. 2604.22273 and 2604.01029
  iterate against MODEL-generated critique. **The two literatures agree once you condition on
  whether the feedback signal is externally verifiable.** That is precisely pyfinagent's
  split: product-code cycles (tests, BQ, the live book) converge; evidence/guard cycles (the
  only referent is Main's own artifact) do not.
- **Is the evidence apparatus worth its cost?** `arXiv:2511.05524` [ADVERSARIAL] says
  emphatically yes: **100% -> 0% hallucinated claims for ~8.3% overhead**. Against
  `arXiv:2607.03691`: harness churn produced **zero** benchmark gain at **+70%** token cost.
  **Reconciliation:** EviBound's gates are GENERIC, CHEAP and REUSABLE (query a run_id, check
  artifacts exist, check a metric range). pyfinagent's are **BESPOKE PER STEP** -- a new
  guard, a new mutation matrix, a new capture, per step. **The disagreement is not about
  whether to verify; it is about whether the verifier is amortised or rebuilt each time.**
  `arXiv:2606.10106v1` leaves this open explicitly: *"How much of the harness is reusable
  across domains and how much must be bespoke?"*

**Literature GAP found (a finding, not a failure):** round-10 searches for research on
conjunctive acceptance-criteria proliferation depressing pass rates returned nothing; the
search engine itself reported the topic is *"not... well-documented in publicly available
sources."* The criteria-inflation arithmetic in ROUND 5 above therefore has no external
corroboration and is offered as an internal, clearly-labelled bound only.

## PITFALLS (from the literature, mapped to what this harness is about to do)

1. **Adding evaluator rigor buys recall and pays in noise.** CR-Bench: recall
   27.01% -> 32.76%, SNR **5.11 -> 1.95** (2603.11078v1). pyfinagent's own analogue: 75% of
   CONDITIONALs carry no unmet criterion.
2. **Prompt-level quality instructions do NOT change the degradation slope.** SlopCodeBench:
   anti-slop prompting cut initial verbosity 34.5% but *"the accumulation of issues persists
   regardless of prompt"* (2603.24755v1). **Writing a longer qa.md is measured not to work.**
3. **Naive drift readings false-alarm on 75% of drift-free streams** (2606.15474). Do not
   attribute the PASS collapse to Main without a frozen anchor set.
4. **An AI judge grading AI artifacts with no external spec is one estimator, not two**
   (2603.25773). Evidence-class findings will never run out.
5. **Category-E defects are invisible to any conformance pipeline** -- *"No verification
   pipeline catches Category E defects because the pipeline verifies conformance to the
   specification"* (2603.25773). A defect IN the criteria cannot be found by checking against
   the criteria.
6. **Goodhart on evidence:** rewarding format-compliance produces behaviour that *"does not
   causally depend on"* the underlying reality (2510.10931). A byte-presence pin and a
   marker-based guard are exactly this shape -- and both were found in pyfinagent verbatim.
7. **Over-investment on simple work is a named failure mode** (Anthropic multi-agent-research).

## APPLICATION TO PYFINAGENT (external findings -> internal anchors)

**A1. Before ANY remediation, settle the confound -- otherwise 87.x optimises the wrong
system.** Build the frozen-anchor design from `arXiv:2606.15474`: take N completed steps
whose verdicts predate 2026-08-09, re-run TODAY's `.claude/agents/qa.md` (897 L) against
their UNCHANGED evidence, and compare to the recorded verdict. *"Only a change in the judge
can move it."* If today's Q/A returns CONDITIONAL on evidence that passed under the 556-line
qa.md, the collapse is judge-side and no amount of 87.1-87.5 will move the PASS rate.
**This is cheap** -- the artifacts are all in `handoff/archive/phase-*/` and
`handoff/current/evaluator_critique_*.md` -- **and it is the only experiment that separates
the two hypotheses.** Without it every 87.x outcome is uninterpretable.

**A2. The distinct class 87.1-87.5 do NOT cover is SILENT FAILURE -- adopt NOVA's SFR.**
`arXiv:2606.27243v1` splits outcomes into Local Pass Rate and **Silent Failure Rate**
(runnable but ineffective), with `EPR = LPR x (1-SFR)`. pyfinagent's residual findings are
verbatim SFR: `M10 survives ... ZERO coverage`, `the fix is INERT on the live corpus`,
`MARKER-based, not outcome-based`, `whole-file BYTE-PRESENCE pins`, `the mutation matrix is
NON-DISCRIMINATING`, `an UNWIRED emit site`. **87.4 detects a guard that CANNOT fail;
nothing detects a guard that CAN fail, runs green, and covers nothing.** These need
different detectors: vacuity is a property of the guard's text, inertness is a property of
its INTERSECTION WITH THE LIVE POPULATION.

**A3. Cap the apparatus, not the rigor.** The lever with external support is NOT weakening
`qa.md` -- Anthropic is explicit that a skeptical evaluator is the right design, and
EviBound measured 100% hallucination WITHOUT gates. The lever is **amortising** the
apparatus: `scripts/qa/` is **102 files / 34,082 lines**, up from **6 files on 2026-07-06**,
and each step still ships a bespoke `verify_*_86_NN.py` + `mutation_matrix_86_NN.py`.
EviBound got 100%->0% for **8.3%** overhead because its gates are GENERIC. NOVA reuses
rejected failure patterns as *"forbidden directions"* in trajectory memory, *"reducing
repeated semantic failures."* pyfinagent's `scripts/qa/pre_spawn_gate.py` (484 L, EXISTS)
is precisely that mechanism unwired -- which is what **87.2** already is. **87.2 is the
highest-leverage of the five and should be sequenced first**, not because it is easiest but
because it is the only one that converts a per-step cost into a one-time cost.

**A4. Cap the remediation diff.** Median remediation is 286 inserted lines, mean 465, 45%
over 300, 10 over 1,000 (n=80 across the 8 highest-cycle steps). Against Anthropic's
*"one-feature-at-a-time"* and *"tractable chunks"*, and against 2604.01029's measured
`-3.1 .. -7.9 pp` content harm that WORSENS with difficulty. **A CONDITIONAL that reports no
unmet criterion should not authorise a 465-line commit.** Concretely: split the CONDITIONAL
response into (i) criterion-unmet -> fix now, (ii) residual observation -> queue as its own
step, which is already the standing `feedback_queue_discovered_defects_in_masterplan`
doctrine and the operator's 2026-08-17 PRODUCT-vs-EVIDENCE ruling
(`feedback_product_fix_vs_evidence_churn`). **75% of CONDITIONALs are category (ii).**

**A5. Attack first-pass defect rate at PLAN, not at EVALUATE.** NOVA's ablation is
unambiguous: removing Solution Design collapses EPR **60.0% -> 18.2%** -- a bigger effect
than removing verification feedback (60.0 -> 37.5). Anthropic harness-design puts the same
weight on the sprint contract negotiated *"before any code was written."* pyfinagent's
contract already exists (`handoff/current/contract.md`) but the criteria it copies have
inflated to **7.43 criteria / 1,741 chars** at 86.120+. **Fewer, sharper, independently
falsifiable criteria is a first-pass-defect intervention; more evaluator prose is not**
(SlopCodeBench: slopes *"largely parallel"*).

**A6. Add non-functional QA of the harness itself.** `arXiv:2607.03691` names the root cause
of exactly this pattern: *"the absence of Agentic Quality Assurance"* -- regressions
*"passed all existing automated checks"* because the checks only verified patch correctness,
never token efficiency or tool overhead. pyfinagent tracks PASS/CONDITIONAL/FAIL and nothing
else. **Minimum viable instruments: spawns-per-step (now 3.9 mean, 12 max), cross-cycle
novelty rate (now 86%), apparatus:product line ratio (now 2.73:1), and
CONDITIONALs-with-zero-NOT-MET (now 75%).** All four are computable from artifacts that
already exist, and all four are currently unmeasured.

**A7. Correct a stale instruction.** `CLAUDE.md` F1b still says
`scripts/harness/attempt_budget.py` has *"no runtime caller"*. **It does now**:
`scripts/harness/attempt_gate.py:84` does `from attempt_budget import (...)` as a PreToolUse
gate, writing `handoff/audit/attempt_budget_audit.jsonl` (`:89`), consumed by
`scripts/harness/research_router.py:50`. Left uncorrected this misdirects the next reader
about what actually terminates a loop -- and termination is the subject of this step.

---

## RESEARCH GATE CHECKLIST

**Hard blockers:**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **20**, all via
      `arxiv.org/html/` or vendor HTML; **zero** `arxiv.org/pdf/` fetches (the one PDF-only
      candidate, `sback.it/.../icsme2018a.pdf`, is recorded snippet-only and NOT counted).
- [x] 10+ unique URLs total -- **38** (20 full + 18 snippet-only).
- [x] Recency scan (last 2 years) performed + reported -- dedicated section; **5 findings
      that supersede the older canon**, listed.
- [x] Full pages read (not abstracts) for the read-in-full set -- every row carries an
      extracted quote or a numeric result from the body, not an abstract.
- [x] file:line anchors for every internal claim -- see Internal Code Inventory + inline
      anchors (`qa.md:398/:441/:505`, `attempt_gate.py:84/:89`, `research_router.py:50`).

**Soft checks:**
- [x] Internal exploration covered every module the caller scoped: `handoff/harness_log.md`,
      `.claude/agent-memory/qa/verdicts/*`, the archived critiques for phase-36.17 / 86.79 /
      86.85 (**note: `handoff/archive/phase-86.74/` and `phase-86.9/` contain NO
      `evaluator_critique.md`** -- those steps' critiques live in `handoff/current/` and were
      read there instead), masterplan 87.1-87.5, `qa.md`, `researcher.md`.
- [x] Contradictions / consensus noted -- two live debates reconciled, both with the
      reconciling variable named.
- [x] All claims cited per-claim with URL + access date (2026-08-18) or file:line.
- [x] Three-variant search discipline visible (current-year / last-2-year / year-less).
- [x] Audit-class loop-until-dry: **10 rounds, last 3 dry** (rounds 8, 9, 10 produced ZERO
      new read-in-full findings beyond de-dup). `K_required = 2`, so `coverage.dry = true`.
- [x] Adversarial sourcing (not required at `complex`, done anyway): **3 tagged
      [ADVERSARIAL]** -- `2604.10508` (iteration does NOT regress), `2511.05524` (evidence
      gates are cheap and worth it), `2606.15474` (this brief's own attribution is
      confounded).

