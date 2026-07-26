# Evaluator Critique — phase-80.5

**Evaluator:** Layer-3 Q/A subagent `qa-80-5` (Agent-tool, `model: opus`). **Date:** 2026-07-26.
**Verdict: CONDITIONAL** — all 5 immutable criteria independently verified MET; withheld
from PASS on three named one-line items the Q/A measured and the artifacts did not mention.

Transcribed VERBATIM. Main records the verdict and never authors it.

---

Verification complete. Everything restored byte-identically; operator's `:8000` (pid 70791) and `:3000` untouched and healthy.

## Q/A verdict — phase-80.5

**Harness compliance (5-item audit, all PASS):** researcher ran first (`research_brief_80.5.md`, `gate_passed: true`; I re-derived `urls_collected` → **38 unique URLs, matches exactly**; 10+28=38 consistent; recency section at :82) → contract → generate → live_check. mtime chain confirms order: research 03:25 → contract 03:27 → capture 03:32 → live_check 03:33 → results 03:34. No `harness_log` entry yet (log-last, correct). No prior 80.5 verdicts, so the 3rd-CONDITIONAL rule is N/A. Contract criteria block is **byte-identical** to `masterplan.json` (programmatic diff, 5/5).

**Deterministic, verbatim:** `npm test -- PortfolioAllocationDonut` → `Tests 20 passed (20)`, exit **0**. `npx tsc --noEmit` exit **0**. `npx eslint` on the two changed files exit **0** (1 warning, `react-hooks/immutability` on `acc` at :149 — untouched by this diff, pre-existing). No `.py` in the diff, so the ruff gate is N/A. `md5 d41b053dc7601cf20050cdaf19739ca6` reproduces exactly.

**Mutation matrix re-run — all 6 of Main's rows reproduce exactly**, including the killing test names. M1 (class-only) kills 1; **M1b (faithful in-flow restore to the card root) kills 2**, with `expected 4 to be 3` — the structural guard is genuinely non-vacuous, so criterion 5's mutation requirement is really satisfied. M2/M3/M4/M5 each kill 1.

**Independent evidence I derived myself:** the raw MCP snapshot still on disk (`.playwright-mcp/page-2026-07-26T01-31-56-072Z.yml` — tool output, not Main's prose) confirms **1 tooltip node** (`ref=e646`, as claimed), **0 `title` nodes**, centre `5.6%` == legend `5.6%`, and the tooltip DOM-nested inside `region "Allocation"`. §B's arithmetic re-derives exactly (Δ-from-tabpanel 0/231/248 identical in both states; heights 598/215/113/20 identical; uniform 130px scroll), and the §C tooltip box reconciles to the pixel against the card box (1045+1+16=1062 x; 363−2×17=329 w; 493−17=476 bottom). The scroll-adjustment reasoning is sound, not an explain-away — and the PNG's KPI-row position independently corroborates the 130px scroll.

### Findings (none violate an immutable criterion)

1. **Measured regression, undisclosed — focus/grace-timer race.** `onFocus={() => setHoverIdx(i)}` (`PortfolioAllocationDonut.tsx:303`) never calls `cancelClose()`, so a pending 200ms close wipes a tooltip that keyboard focus just opened. My probe **fails on the unmutated shipped code**: `expected null not to be null`. Pre-fix this worked (no timer existed), so it is new. WCAG 1.4.13 *Persistent*. One-line fix: `onFocus={() => openAt(i)}`.
2. **Unguarded escape hatch.** Removing `relative` from the chart row leaves **all 20 tests green**. I verified statically that no ancestor creates a containing block (the one `backdrop-blur-lg` at `positions/page.tsx:185` is a sibling), so the tooltip would pin to the **viewport** — the operator-rejected escaping behaviour, with a green suite.
3. **Escape/Dismissible is untested.** Removing the document listener survives all 20. My probe confirms the mechanism *does* work — true claim, no guard.
4. Main's matrix had **no mutation for criterion 3**. I ran it (restore `<title>`): KILLED, `expected 2 to be +0`. Guard sound; gap closed.
5. Prose: the "`container.textContent` passes vacuously" rationale is half-right — measured, the positive assertion passes unscoped, but the negative `not.toContain("56%")` still fails. Shipped test is correctly scoped regardless.
6. Prose: "at-rest figures reproduce that baseline **exactly**" — height 215 and heading y=656 do; card x/width do not (1038/359 → 1045/363).

**On the disclosed gap (missing unhovered screenshot): acceptable partial, does not block on its own.** Criterion 1 names its own evidence shape and that is complete for both states; the hovered PNG covers the state where the defect manifests; and it shows the ring painting correctly under live `fill="none"`, which corroborates the "no visual change" claim the unhovered shot would have tested. The honest disclosure isn't what earns this — the evidence is.

**Required §1c disclosure:** I could not take my own capture — the `:3100`/`:8001` rig was already torn down (`curl :3100` → 000, no listeners) and server lifecycle is Main's. My UI judgment rests on Main-produced boxes plus the on-disk raw snapshot I parsed independently. Degraded path.

**Also for Main:** `handoff/current/done_definition_evidence_2026-07-26.md` became modified during my run — not by me. Check `git add -An` before any flip. The two frontend files carry mtime 03:42 from my post-mutation `cp` restore; content is byte-identical (md5 verified).

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 5 immutable criteria independently verified MET and every numeric claim in the artifacts reproduced exactly (md5, 20/20 tests, tsc/eslint exit 0, all 6 mutation rows, the 38-URL gate envelope, and the box arithmetic). M1b confirms the headline structural guard is non-vacuous. Withheld from PASS on three named, one-line items I measured and the artifacts do not mention: (1) a NEW regression -- onFocus at :303 does not cancelClose, so a pending 200ms close wipes a tooltip keyboard focus just opened (probe fails on unmutated shipped code; pre-fix this worked), violating WCAG 1.4.13 Persistent in the exact mechanism this step claims to fix; (2) removing `relative` from the chart row keeps all 20 tests green while the tooltip would pin to the viewport -- no ancestor creates a containing block, verified statically -- i.e. the operator-rejected escaping behaviour is unguarded; (3) the document-level Escape listener works (probe-confirmed) but no shipped test guards it. The missing unhovered screenshot was judged an acceptable partial and is NOT a basis for this verdict.",
  "violated_criteria": [
    "Regression_Introduced: focus/grace-timer race (onFocus lacks cancelClose)",
    "Unguarded_Regression_Path: `relative` removal escapes the card with a green suite",
    "Unguarded_Behaviour: document-level Escape (WCAG Dismissible) has no test"
  ],
  "violation_details": [
    {
      "violation_type": "Overgeneralization",
      "action": "onFocus={() => setHoverIdx(i)} left unconverted at PortfolioAllocationDonut.tsx:303 while onMouseEnter/onMouseLeave were converted to openAt/scheduleClose",
      "state": "Evaluator probe on the UNMUTATED shipped tree: mouseEnter(A) -> mouseLeave(A) [schedules 200ms close] -> focus(B) -> advanceTimersByTime(300) => tooltip is null. 'AssertionError: focus trigger still present -- WCAG Persistent: expected null not to be null'. Pre-fix the same sequence held (no timer existed), so this is new in this diff. Undisclosed in all four artifacts, whose WCAG section is presented as a strict accessibility improvement.",
      "constraint": "WCAG 2.2 SC 1.4.13 Persistent: 'The additional content remains visible until the hover or focus trigger is removed...'; contract Do-no-harm HARD STOP: 'any change making the component less accessible than today'"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Mutation MQ1: drop `relative` from the chart-row wrapper (:262), leaving the tooltip `absolute inset-x-0 bottom-0 z-10`",
      "state": "SURVIVED -- 20 passed (20). Behavioural differential confirmed statically: no ancestor of the card creates a containing block (no relative/absolute/fixed/sticky/transform/filter on the grid, page shell or root layout; the sole backdrop-blur-lg at positions/page.tsx:185 is a SIBLING card), so the tooltip resolves against the initial containing block and pins to the viewport bottom at full viewport width. The anti-portal test asserts DOM containment (card.contains(tip)) which stays true, and the computed-position test asserts only `absolute`. Criterion 2 is MET for the shipped code (proven by the live snapshot + PNG); the mechanism that keeps it true is what is unguarded.",
      "constraint": "Criterion 2 -- 'the rejected portaled/escaping-tooltip behaviour is NOT reintroduced'; qa.md 4c -- a guard that cannot fail when its subject is broken does not count"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Mutation MQ3: delete the document-level keydown listener (:217-218), the shipped WCAG Dismissible mechanism",
      "state": "SURVIVED -- 20 passed (20). Evaluator probe PROBE-ESC passes on the shipped code and fails under MQ3, so the mechanism genuinely works and the experiment_results 5 claim is TRUE -- but zero shipped tests cover it, while experiment_results argues the overlay REVOKES the 'does not obscure' exception and therefore makes a real dismiss mechanism REQUIRED. Visually confirmed in the hovered PNG: the tooltip does obscure the legend rows.",
      "constraint": "WCAG 2.2 SC 1.4.13 Dismissible; experiment_results_80.5.md 5 asserts the mechanism as delivered work"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "criteria_byte_diff_vs_masterplan",
    "verification_command_exit_code",
    "tsc_noEmit",
    "eslint_changed_files",
    "md5_baseline_reproduction",
    "mutation_matrix_rerun_M1_M1b_M2_M3_M4_M5",
    "evaluator_mutations_MQ1_MQ2_MQ3_MQ4_MQ5_portal_title_restore",
    "fixture_probes_escape_unscoped_rounding_focus_race",
    "raw_playwright_snapshot_independent_parse",
    "box_arithmetic_rederivation",
    "screenshot_inspection",
    "research_gate_envelope_rederivation",
    "artifact_mtime_ordering",
    "harness_log_conditional_count",
    "service_health_8000_3000"
  ]
}
```

---

# Follow-up — Main's remediation (2026-07-26)

All six findings addressed. Test count 20 → **23**; mutation matrix 6 → **10**.

## Finding 1 — focus/grace-timer race — **A REAL REGRESSION I INTRODUCED. FIXED.**

This is the important one, and the Q/A is right that it was **new**: pre-fix no timer
existed, so the bug could not have occurred. My own WCAG *Hoverable* fix created it.

`onFocus={() => setHoverIdx(i)}` (`:303`) did not cancel a pending close, so a grace timer
scheduled by a prior `mouseleave` fired ~200ms later and wiped a tooltip that **keyboard
focus had just opened** — breaking SC 1.4.13 *Persistent* for exactly the users the WCAG
work was meant to serve. The Q/A's probe failed against my shipped, unmutated code.

Fixed: `onFocus={() => openAt(i)}` (`openAt` cancels then opens). Guarded by a new test
that mouse-enters, mouse-leaves, focuses a *different* slice mid-window, advances 3× the
grace period, and asserts the tooltip survives. **Mutation M6** (revert to `setHoverIdx`)
→ KILLED.

## Finding 2 — unguarded escape hatch — FIXED

Removing `relative` from the chart row left **all 20 tests green**, and with no positioned
ancestor the tooltip would resolve against the **viewport** — the operator-rejected
escaping behaviour, with a green suite. New test asserts the tooltip's parent establishes a
containing block. **Mutation M8** → KILLED.

## Finding 3 — Escape/Dismissible untested — FIXED

The mechanism worked (the Q/A verified it), but nothing guarded it: deleting the
document-level listener survived all 20. Since the overlay tooltip now *obscures* content,
SC 1.4.13 *Dismissible* is genuinely required rather than vacuously satisfied — so an
unguarded implementation is a live risk. New test fires `keyDown Escape` on `document`.
**Mutation M7** → KILLED.

## Finding 4 — criterion 3 had no mutation — FIXED

My matrix had no mutation for the "exactly one tooltip" criterion. Added **M9** (restore the
SVG `<title>`) → KILLED, `expected 2 to be +0`, reproducing the Q/A's own independent run
exactly.

## Finding 5 — the vacuity rationale was half-right — CORRECTED

I claimed an unscoped `container.textContent` assertion "passes vacuously". The Q/A
**measured** it: the *positive* assertion passes unscoped (the legend supplies the string),
but the *negative* `not.toContain("56%")` still fails — so an unscoped test would not be
fully vacuous. The shipped test is correctly scoped either way; the stated rationale was
too strong. Corrected in `experiment_results_80.5.md`.

## Finding 6 — "exactly" overstated — CORRECTED

I wrote that the at-rest boxes reproduce the masterplan baseline *"exactly"*. Height (215)
and heading-y (656) match; **x and width do not** (`1038/359` → `1045/363`). Horizontal
only, and criterion 1 is a vertical-shift criterion — but the word was wrong. Corrected in
`live_check_80.5.md` §B.

## Also actioned

- Test counts were stale after the additions. **Re-measured rather than computed:** HEAD
  had 14 `it(` blocks, the file now has 23, and vitest reports 23. Updated everywhere.
- The Q/A noted `done_definition_evidence_2026-07-26.md` changed during its run — that was
  Main recording done-definition item 6, unrelated to 80.5. `git add -An` will be run before
  the flip, per `feedback_audit_the_commit_not_the_diff`.
- The Q/A's §1c degraded-path disclosure (it could not take its own capture because the rig
  was already torn down) is accurate and is left standing in the record.

## On the disclosed screenshot gap

The Q/A judged it an acceptable partial that does not block on its own, and explicitly said
the honest disclosure is *not* what earns that — the evidence is. Recorded without
argument; the unhovered PNG remains uncaptured.


---

# CYCLE 2 — fresh Q/A

**Evaluator:** `qa-80-5-cycle2` (Agent-tool, `model: opus`). **Date:** 2026-07-26.
**Verdict: CONDITIONAL** (2nd on this step-id; no intervening PASS or FAIL).

Transcribed VERBATIM. Main records the verdict, never authors it.

---

Verification complete. Everything restored byte-identically (`ea71f48c…` / `2eb451e2…`); working tree matches the pre-run snapshot exactly; operator's `:8000` (pid 70791, unchanged) and `:3000` untouched and healthy.

## Q/A verdict — phase-80.5, cycle 2

**Harness compliance (5-item audit, all PASS):** researcher first (`research_brief_80.5.md` 03:25, `gate_passed: true`, 10 read-in-full, envelope present; I re-derived **38 unique URLs — matches**) → contract 03:27 → results/live_check 03:47. `masterplan.json` is **unmodified vs HEAD**, so the criteria are definitionally unamended; all 5 are byte-identical substrings of the contract (programmatic, 5/5). 80.5 still `pending`, `retry_count: 0`. Zero `phase=80.5` lines in `harness_log.md` (log-last, correct) — with the one prior CONDITIONAL this is the **2nd**, so the auto-FAIL rule does not fire. **Not verdict-shopping:** evidence changed (`d41b053d`→`ea71f48c`, `f41d183d`→`2eb451e2`, 20→23 tests).

**Deterministic, verbatim:** `npm test -- PortfolioAllocationDonut` → `Tests 23 passed (23)`, bare exit **0**. `npx tsc --noEmit -p tsconfig.json` exit **0**. `npx eslint` on the git-derived changed set (non-empty asserted) exit **0** — 1 warning, `react-hooks/immutability` on `acc`, which I confirmed is present at HEAD:147, pre-existing. No `.py` in the diff. `git add -An` scope is clean: 13 files, all 80.5-related plus hook-appended audit JSONLs.

**Test counts re-derived independently:** HEAD `it(` = **14**, now **23** → 9 new. Claim reproduces.

**All 10 mutations re-run — all KILLED.** M1 kills 1; M2/M3/M4/M5/M6/M7/M8/M9 kill 1 each. **M1b kills 3, not the 2 claimed** (verbatim: `expected 4 to be 3` — matching Main's quote, so it is the same mutation — plus `expected 'static' to be 'absolute'` and `expected 'h-full flex flex-col rounded-xl borde…' to contain 'relative'`).

**Criteria: all 5 independently verified MET.** I parsed the raw hovered MCP snapshot myself: 1 tooltip node (`ref=e646`), 0 `title` nodes, centre `5.6%` == legend `5.6%`, tooltip DOM-nested inside `region "Allocation"`. §B/§C arithmetic re-derives exactly (130px uniform scroll, all four Δ-from-tabpanel and all heights identical; tooltip 1062..1391 × 422..476 contained in 1045..1408 × 278..493).

### Findings

1. **The regression class was NOT swept — a second, NEW instance ships.** Main fixed the exact line cycle 1 named (`onFocus`) and guarded that line. But the diff also newly added `onMouseLeave={scheduleClose}` to the **tooltip itself** (`:386`). Probe: focus a slice (tooltip opens), pointer enters then leaves the tooltip, advance past the grace window → **tooltip gone while the focus trigger is still held**. It **passes on HEAD** (HEAD's tooltip has no mouse handlers) and **fails on shipped** — new in this diff. Mechanism confirmed by mutation: deleting `:386` makes the probe pass *while the shipped suite stays 23/23 green*. Same SC 1.4.13 *Persistent* violation, same diff, unguarded and undisclosed — while the artifacts present the WCAG work as a strict improvement. Root cause is the single `hoverIdx` conflating hover and focus; separate `focusIdx`/`hoverIdx` fixes this **and** the two pre-existing holes below.
2. **Two probes fail on shipped AND on HEAD** (focus held while an unrelated hover ends; blur wiping a mouse-held tooltip) → **pre-existing, not introduced.** Recording so finding 1 is not overstated.
3. **M1b "KILLED — 2 failed" is stale** — measured **3**. It is a cycle-1 number (20-test suite, before the containing-block guard existed) carried into a cycle-2 table that presents all 10 as measured together; it recurs 3× (`experiment_results:141`, `live_check:124`, `live_check:149` "M1b kills 2") and in the spawn prompt. Direction is conservative, but it proves the matrix was not fully re-run after the new tests — in the row called load-bearing.
4. **Two unguarded paths of the identical `openAt`/`cancelClose` shape.** Mutating the **arc** or the **legend** `onMouseEnter` from `openAt(i)` to `setHoverIdx(i)` leaves **23/23 green** while breaking slice→slice and row→row traversal (my PROBE-1/PROBE-2 go red). M6 guards only the focus path.
5. **The new containing-block guard admits false negatives.** It asserts `parentElement.className` *contains* `"relative"`. `relative`→`lg:relative` survives 23/23 (verified `"lg:relative …".includes("relative") === true`) — the tooltip would escape to the viewport below 1024px. Reparenting the tooltip into the donut wrapper (also `relative`) also survives. Acceptable given jsdom has no layout, but weaker than its name ("offset parent is the positioned chart row") claims.
6. **A comment in the test file claims a pin it does not provide.** `:5-8`: *"Pinned by the test below so a drift in the component makes a test fail."* Measured: `TOOLTIP_CLOSE_DELAY_MS` 200 → **0** leaves **23/23 green** (my drift probe goes red). `TOOLTIP_GRACE_MS` is a local duplicate, not a pin — the Hoverable guard cannot distinguish a real grace period from a zero one.
7. **Box measurements are not archived.** **No** `.playwright-mcp/*.yml` on disk carries any `box=`; the criterion-1 table exists only as Main's prose. Everything structural I verified myself; the numbers I could not.
8. **The masterplan's own "ADDITIONAL SUCCESS CRITERION"** (*a test asserting mouseEnter on the `<svg>` root does NOT mount the tooltip*) has **no test and no disclosure** anywhere in the contract or results. Not in the immutable array, so no criterion fails — but it is uncovered.

**Corrections check:** finding 5's correction is **accurate, measured** — under M3 the unscoped positive still passes (vacuous) while the unscoped negative fails (catches). Finding 6's correction is **accurate** (215/656 match; 1038→1045, 359→363 do not). Neither introduced a new error. One mutation (Escape-without-cancel) I judged an **equivalent mutant** — no behavioural differential — so it is not a finding.

**Required §1c disclosure:** I could not take my own capture — `:3100`/`:8001` are down (`000`) and server lifecycle is Main's. UI judgment rests on Main-produced boxes plus the raw snapshot I parsed independently. **Degraded path, second consecutive cycle.**

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 5 immutable criteria independently verified MET and reproduced (masterplan unmodified vs HEAD so criteria are definitionally unamended; 23/23 tests exit 0, tsc 0, eslint 0; all 10 mutations re-run and all KILLED; 14->23 test count re-derived from HEAD; 38-URL gate envelope, box arithmetic, and the on-disk hovered snapshot's 1-tooltip/0-title/5.6%-agreement/DOM-containment all reproduce). Withheld from PASS on three measured items: (1) the regression CLASS was not swept -- Main fixed the exact onFocus line cycle 1 named, but the same diff newly added onMouseLeave={scheduleClose} to the TOOLTIP at :386, so a pointer leaving the tooltip wipes a tooltip keyboard focus is still holding; my probe PASSES on HEAD and FAILS on shipped, and deleting :386 makes it pass while the suite stays 23/23 green -- a new, unguarded, undisclosed SC 1.4.13 Persistent violation of the identical class; (2) the M1b row reports 'KILLED -- 2 failed' but measures 3 -- a cycle-1 number carried into the cycle-2 table (repeated 3x across two artifacts), proving the matrix was not fully re-run after the 3 new tests; (3) the test file's TOOLTIP_GRACE_MS comment claims the delay is 'pinned by the test below', but setting the component constant 200->0 leaves all 23 green. Two further probe failures were verified PRE-EXISTING (they fail on HEAD too) and are explicitly NOT counted against this diff. Counter: this is the 2nd CONDITIONAL on 80.5; a 3rd auto-FAILs.",
  "violated_criteria": [
    "Regression_Class_Not_Swept: tooltip onMouseLeave (:386) wipes a focus-held tooltip -- new in this diff, unguarded",
    "Unreproducible_Claim: M1b reported as killing 2 tests, measures 3 (stale cycle-1 count in 3 places)",
    "Vacuous_Pin: TOOLTIP_GRACE_MS comment claims a drift pin the suite does not provide (200->0 stays green)",
    "Unguarded_Path: arc and legend onMouseEnter openAt->setHoverIdx both survive 23/23",
    "Weak_Guard: containing-block test is a class-substring assert (lg:relative and reparenting both survive)",
    "Missing_Assumption: masterplan's ADDITIONAL SUCCESS CRITERION (svg-root hover mounts no tooltip) untested and undisclosed"
  ],
  "violation_details": [
    {
      "violation_type": "Overgeneralization",
      "action": "Remediation of cycle-1 finding 1 changed onFocus to openAt(i) and added one guard for that instance, but left onMouseLeave={scheduleClose} on the tooltip element itself (PortfolioAllocationDonut.tsx:386), which this same diff introduced",
      "state": "Evaluator probe PROBE-4 on the UNMUTATED shipped tree: focus(arc[0]) -> mouseEnter(tooltip) -> mouseLeave(tooltip) -> advanceTimersByTime(600) => tooltip is null. 'AssertionError: PROBE-4: focus still on arc[0]; leaving the tooltip closed it (SC 1.4.13 Persistent): expected null not to be null'. The SAME probe file run against git show HEAD:<path> PASSES (HEAD's tooltip carries no mouse handlers), so it is NEW in this diff. Mutation Q-TIPLEAVE (delete :386) makes PROBE-4 pass while the shipped suite stays 'Tests 23 passed (23)' -- so the path is also unguarded. Undisclosed in all four artifacts, whose WCAG section is presented as a strict accessibility improvement.",
      "constraint": "WCAG 2.2 SC 1.4.13 Persistent: 'The additional content remains visible until the hover or focus trigger is removed...'; contract_80.5.md Do-no-harm HARD STOP: 'any change making the component less accessible than today'"
    },
    {
      "violation_type": "Contradiction",
      "action": "experiment_results_80.5.md:141 and live_check_80.5.md:124/:149 report mutation M1b as 'KILLED -- 2 failed' inside a table headed 'Mutation matrix -- 10 mutations, all killed'",
      "state": "Re-run of the identical mutation (faithful in-flow restore of the tooltip to the card root; verified identical by the quoted assertion 'expected 4 to be 3') yields 'Tests 3 failed | 20 passed (23)': the structural guard, the computed-position test, AND the cycle-2 containing-block test. 2 is arithmetically impossible once the containing-block test exists, since a card-root tooltip's parent className cannot contain 'relative'. Every other row's count reproduces exactly, so this single row was not re-measured after the 3 new tests were added.",
      "constraint": "qa.md 4b -- every numeric claim must reproduce from the command that produces it; a cycle-2 artifact must not present a cycle-1 measurement as current"
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "PortfolioAllocationDonut.test.tsx:5-8 declares 'const TOOLTIP_GRACE_MS = 200' with the comment 'Mirrors TOOLTIP_CLOSE_DELAY_MS in the component. Pinned by the test below so a drift in the component makes a test fail rather than silently changing how long a tooltip lingers.'",
      "state": "Mutation Q-DELAY0 (component constant 200 -> 0) => 'Tests 23 passed (23)'. The Hoverable test fires mouseEnter on the tooltip synchronously before advancing timers, so it cannot distinguish a usable grace window from a zero one; my drift probe (assert tooltip alive 100ms after mouseleave) goes red under Q-DELAY0 and green on shipped. The test constant is a local duplicate that no assertion compares against the component's value.",
      "constraint": "qa.md 4c -- a guard that cannot fail when its subject is broken does not count; shape 4 (tautology / assertion true by construction)"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Mutations Q-ARC and Q-LEGEND: change the arc's and the legend row's onMouseEnter from openAt(i) to setHoverIdx(i), dropping cancelClose on the two mouse-open paths",
      "state": "BOTH SURVIVED -- 'Tests 23 passed (23)' each. Behavioural differential confirmed by execution, not reasoning: Q-ARC turns PROBE-1 red (arc[0] -> mouseleave -> arc[1], tooltip vanishes 200ms after the pointer lands on the next slice) and Q-LEGEND turns PROBE-2 red (same on legend rows). The M6 guard added this cycle covers only the focus path, so the primary mouse interaction -- traversing between adjacent slices -- has no regression guard.",
      "constraint": "qa.md 4c -- name the concrete mutation that makes each guard fail; a surviving mutant with a measured behavioural differential is a coverage gap"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "The new test 'the tooltip's offset parent is the positioned chart row, not the viewport' asserts tip.parentElement?.className toContain 'relative'",
      "state": "Mutation Q-RELSUB (relative -> lg:relative) SURVIVED 23/23 -- verified in node that 'lg:relative flex items-center gap-4 flex-1'.includes('relative') === true, so the tooltip would resolve against the viewport below the 1024px breakpoint, i.e. the operator-rejected escaping behaviour, with a green suite. Mutation Q-REPARENT (move the tooltip inside the donut wrapper, which also carries 'relative') likewise SURVIVED 23/23 while positioning the tooltip against the 128px donut box. The test verifies neither the offset parent nor which element establishes the block. Accepted as the best available in jsdom (getBoundingClientRect is all zeros, offsetParent unavailable) and criterion 2 is separately proven live -- recorded as a guard-strength WARN, not a criterion failure.",
      "constraint": "Criterion 2 -- 'the rejected portaled/escaping-tooltip behaviour is NOT reintroduced'"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "masterplan.json 80.5 description states 'ADDITIONAL SUCCESS CRITERION: a test asserting that mouseEnter on the <svg> root (or on a centre <text>) does NOT mount the tooltip'",
      "state": "No such test exists among the 23 (the nearest, 'slice circles use fill=none...', asserts fill and pointer-events ATTRIBUTES, not mount behaviour). Neither contract_80.5.md, experiment_results_80.5.md nor live_check_80.5.md mentions the requirement or scopes it out. It is not in the immutable success_criteria array, so no criterion fails; it is an uncovered, undisclosed step requirement. NOTE: in jsdom such a test would be near-vacuous (no hit-testing, and mouseenter does not bubble), which is a legitimate reason to omit it -- but that reasoning was never written down.",
      "constraint": "qa.md 4 Contract completeness -- every stated requirement mapped to covering evidence, or explicitly scoped out"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "live_check_80.5.md B/C present browser_snapshot(boxes:true) tables as the verbatim evidence for criterion 1",
      "state": "grep of every .playwright-mcp/*.yml returns ZERO files containing 'box=' -- the archived at-rest snapshot (page-2026-07-26T01-31-06-307Z.yml) does not even contain the Allocation region. The numeric table is therefore Main's prose only and cannot be independently reproduced; I verified everything STRUCTURAL in the hovered snapshot myself (1 tooltip node ref=e646, 0 title nodes, centre 5.6% == legend 5.6%, DOM containment) and the table's internal arithmetic is exact (uniform 130px scroll, all heights and all four Delta-from-tabpanel identical, at-rest height 215 and heading y=656 reproducing the masterplan baseline). Weighed as a disclosed-degraded evidence path consistent with cycle 1, not a criterion failure; the fix is to archive the boxes:true output alongside the PNGs.",
      "constraint": "qa.md 1c -- a verdict resting on a Main-produced capture must say so; qa.md 4b -- a numeric claim must carry the command that reproduces it"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "criteria_byte_diff_vs_masterplan",
    "masterplan_unmodified_vs_HEAD",
    "verification_command_exit_code",
    "tsc_noEmit",
    "eslint_git_derived_changed_files",
    "md5_baseline_reproduction",
    "test_count_rederivation_from_HEAD",
    "mutation_matrix_rerun_M1_M1b_M2_M3_M4_M5_M6_M7_M8_M9",
    "evaluator_mutations_QARC_QLEGEND_QBLUR_QRELSUB_QREPARENT_QESCNOCANCEL_QTIPLEAVE_QDELAY0",
    "probe_suite_shipped_vs_HEAD_differential",
    "rounding_scope_probe_under_M3",
    "raw_playwright_snapshot_independent_parse",
    "box_arithmetic_rederivation",
    "research_gate_envelope_and_url_rederivation",
    "artifact_mtime_ordering",
    "harness_log_conditional_count",
    "verdict_shop_evidence_change_check",
    "git_add_dry_run_commit_scope",
    "service_health_8000_3000"
  ]
}
```

**For Main — the three blocking items are all small and named:** (1) `PortfolioAllocationDonut.tsx:386` — split `focusIdx` from `hoverIdx` (that one change also closes the two pre-existing holes), and extend the persistence guard to the tooltip-leave path plus the arc/legend traversal paths; (2) re-run M1b and correct **2 → 3** in all three places; (3) either make `TOOLTIP_GRACE_MS` a real pin (import the constant, or assert the tooltip is alive mid-window) or delete the false comment. Items 5–8 are WARN-level and can be dispositioned in prose. Full working notes: `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/df87839b-b9ab-4177-abf5-a397a5e2dc58/scratchpad/qa805_mut.py` and `qa805_run.sh` (probe files were deleted; the tree is clean).

---

# Follow-up — Main's remediation of the CYCLE-2 CONDITIONAL (2026-07-26)

Suite 23 → **28**. Matrix 10 → **14**, all re-run in one pass.

## Finding 1 — I fixed the INSTANCE, not the CLASS. Root cause now removed.

This is the finding that matters, and the Q/A is right twice over: my cycle-1 remediation
fixed the exact line named (`onFocus`) and shipped a **second, new** instance of the same
defect in the same diff — `onMouseLeave={scheduleClose}` on the tooltip itself, so focusing
a slice and running the pointer through the tooltip killed it while focus was still held.
It passes on HEAD and fails on my shipped code, so I introduced it.

Root cause, as the Q/A diagnosed: **one `hoverIdx` conflating two independent inputs.**
Every SC 1.4.13 *Persistent* hole in this component reduced to that. Fixed at the root:

```
const [hoverIdx, setHoverIdx]  // pointer
const [focusIdx, setFocusIdx]  // keyboard
const activeIdx = focusIdx ?? hoverIdx
```

`scheduleClose` now clears **hover only**; blur clears **focus only**. Four probes added
covering the whole class. **Measured against the pre-split component: 3 of the 4 were RED**
(the fourth was already green there — see below).

## Finding 2 — two of those holes were PRE-EXISTING, not mine

The Q/A separated them explicitly so finding 1 would not be overstated, and it is right:
"focus held while an unrelated hover ends" and "blur wiping a mouse-held tooltip" fail on
**HEAD** as well. The same split fixes them. Recorded so the blame is accurate in both
directions — I introduced two of the four, not four.

## Finding 3 — M1b was stale at "2"; measured **3**. FIXED, and the lesson recorded.

A cycle-1 number (20-test suite, before the containing-block guard existed) carried into a
cycle-2 table that presented all rows as measured together — in the row I had called
load-bearing. **The entire matrix has now been re-run in a single pass against the current
28-test suite.** The lesson is not the digit: a matrix must be re-run in full whenever
tests are added, or its rows silently become claims.

## Finding 4 — two unguarded paths of the identical shape — FIXED

Mutating the **arc** or **legend** `onMouseEnter` from `openAt(i)` to `setHoverIdx(i)` left
23/23 green while breaking slice→slice and row→row traversal. New traversal test; **M10**
and **M11** both KILLED.

**Honest correction to my own test comment:** I wrote that all four new PERSISTENT probes
were "RED before the split". Measured, **three** were — the traversal probe was already
green there, because `openAt` cancelled. It guards finding 4's mutation instead. The
comment now states the measurement rather than the assumption.

## Finding 5 — the containing-block guard admitted false negatives — FIXED

`className.includes("relative")` accepts `lg:relative`, which establishes a containing block
only ≥1024px — below that the tooltip escapes to the viewport, the operator-rejected
behaviour, with a green suite. Now `classList.contains("relative")`. **M13** KILLED.

## Finding 6 — a comment claiming a pin that did not exist — FIXED, after a wrong first attempt

The test file declared `const TOOLTIP_GRACE_MS = 200` and claimed it was "pinned by the test
below". Nothing pinned it.

**My first fix was wrong and I caught it by mutating it.** I exported the constant and
imported it — which made the coupling real but the drift *invisible*: `200 → 350` left all
27 tests green, because the suite now compared the component against itself. That is
strictly worse than the comment it replaced.

Corrected: keep the import (timer assertions stay correct) **and** add an explicit
`expect(TOOLTIP_CLOSE_DELAY_MS).toBe(200)`. **M14** now KILLED.

## Not addressed

The Q/A's finding-5 note that reparenting the tooltip into the donut wrapper (also
`relative`) would survive is **still true** — the guard pins that *a* positioned parent
exists, not *which*. Given jsdom has no layout, pinning the specific ancestor would mean
asserting on DOM shape that legitimately changes. Left as a known limit rather than papered
over; the live Playwright geometry in §B is what actually establishes containment.
