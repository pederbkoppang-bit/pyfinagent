STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 91.13
WRITTEN: 2026-08-20T20:58:39Z

# Q/A cycle-2 write-first record -- step 91.13

Spawn context: FRESH Q/A, cycle 2, prior_verdict=CONDITIONAL (per spawn prompt, ADVISORY).
Immutable criteria (as supplied):
 1. `grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx` returns 0 after the fix
    (glow prop removed from the Total Cost card), OR all 4 cards made visually consistent by
    deliberate design with a documented reason.
 2. A Playwright screenshot of the Cost & Token Usage panel shows a uniform border/background
    across all 4 stat boxes.

## Progress log

- [t0] Read `.claude/agents/qa.md` in full from disk (runtime read; Workflow path).
- [t0] Created this WIP record.

## Prior-attempt / prior-verdict EVIDENCE (gathered, not applied)

`python scripts/qa/qa_wip.py 91.13 --spawned-at 2026-08-20T20:58:39Z`:
  source_present=true, attempt_number=2 (status "ok", is_lower_bound=false),
  prior_attempts=1, records_retained=2 (GAUGE, not counter),
  records_pruned_known=null, identity_checked=true,
  prior_records=[verdict_wip_91.13__20260820T204117Z.md]

`python scripts/qa/verdict_history_86_21.py --step 91.13 --evidence-only`:
  status = "no_rows_for_step"; verdicts = (none).

CROSS-CHECK: prior_attempts (1, auto) > ledger verdict count (0) => **LEDGER IS STALE**
for this step. Sequence from the ledger is therefore UNRELIABLE. Main's spawn prompt
states prior_verdict=CONDITIONAL, cycle=2 -- ADVISORY ONLY (Main is the constrained
party). Independent corroboration that a cycle-1 spawn existed: the prior WIP record
file above, mtime 2026-08-20 22:41 local. I did NOT infer its verdict from record
body word-frequency (qa.md forbids it).

## A. HARNESS-COMPLIANCE AUDIT (5 items)

mtimes (LOCAL time; `stat -f %Sm` prints local, the trailing Z in my format string is
a formatting artifact, NOT UTC):
  research_brief_91.13.md     2026-08-20 22:32:30
  contract_91.13.md           2026-08-20 22:39:32
  CostDashboard.tsx           2026-08-20 22:39:39
  captures_91.13/*.png        2026-08-20 22:40:46
  live_check_91.13.md         2026-08-20 22:58:21
  experiment_results_91.13.md 2026-08-20 22:58:28
  (my WRITTEN stamp 20:58:39Z == 22:58:39 local)

A1 research-gate-before-contract: research (22:32:30) < contract (22:39:32). ORDER OK.
   Envelope/gate_passed: PENDING (checked below).
A2 contract-before-generate: contract (22:39:32) < code artifact (22:39:39, +7s) <
   capture (22:40:46). ORDER OK.
A3 experiment_results present: YES (2434 bytes).
A4 log-last: PENDING (check harness_log + masterplan status).
A5 no-verdict-shopping: evidence CHANGED after the cycle-1 spawn (22:41:17):
   live_check_91.13.md CREATED 22:58:21 and experiment_results_91.13.md UPDATED
   22:58:28 -- both AFTER cycle-1's Q/A. NOT verdict-shopping. NOTE: the capture PNG
   itself is UNCHANGED (22:40:46) -- the delta is the live_check artifact, which is
   precisely the gap cycle 1 flagged.

## B. DETERMINISTIC CHECKS

B1 IMMUTABLE COMMAND:
   $ grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx
   stdout: 0     shell exit: 1 (grep exits 1 on zero matches -- expected; the
   criterion is worded on the RETURNED COUNT, which is 0)
   Independent corroboration: `grep -n -i 'glow' frontend/src/components/CostDashboard.tsx`
   returns NOTHING (exit 1) -- no glow token of any casing survives in the file.

B2 SCOPE OF THE 91.13 DIFF:
   `git diff HEAD -- frontend/src/components/CostDashboard.tsx` is EXACTLY 1 line:
     -        <BentoCard glow>
     +        <BentoCard>
   at line 85. Nothing else in that file changed. Matches the claim verbatim.

B3 WORKING-TREE SCOPE (concurrency disclosure): 13 frontend files are modified vs
   HEAD, NOT 1. The other 12 belong to SIBLING steps running in the same session
   (91.9 observability phase-tag, 91.22 chart-tooltip CHART_TOOLTIP_ITEM_STYLE,
   phase-tag copy edits). I inspected the full `git diff HEAD -- frontend/` and found
   NO edit to CostDashboard.tsx, BentoCard.tsx or globals.css beyond the 1 line above.
   So no unintended production change WITHIN 91.13's scope; the co-resident diffs are
   other steps' and are not 91.13's to grade.

B4 TSC: `cd frontend && npx tsc --noEmit` -> TSC_EXIT=0.

B5 ESLINT: `cd frontend && npx eslint .` -> exit 1, "85 problems (26 errors, 59
   warnings)". Grouped by file via `-f json`: ALL 26 errors are in build output dirs
   (.next-audit-36-12/, .next-functional/ webpack runtime chunks). ZERO errors in
   frontend/src/. Pre-existing repo condition, not introduced by this diff. Gate PASSES
   for the diff scope.

B6 PYTHON LINT GATE: `git diff --name-only HEAD -- '*.py'` is EMPTY -> gate N/A
   (no Python touched). Backend runtime smoke (S1d) N/A: no backend/** change.

## C. PIXEL RE-DERIVATION OF THE SHIPPED CAPTURE (my own, not Main's numbers)

handoff/current/captures_91.13/cost_kpi_row_91.13.png -- 1120x140 RGB.
Card column runs detected independently (threshold on the navy-800/70 body colour):
  card1 x[3,267] card2 x[286,550] card3 x[569,833] card4 x[852,1116]  (4 cards,
  each 265px, gaps exactly 18px)
Interior background median RGB, y=4..12:
  card1 [12,18,36] std [0,0,0]
  card2 [12,18,36] std [0,0,0]
  card3 [12,18,36] std [0,0,0]
  card4 [12,18,36] std [0,0,0]
Top border pixel (mid-card column): [30,41,59] for ALL FOUR (border-navy-700).
Gap regions, full height:
  gap1 x[268,285] mean [6.67,11.75,29.03] max [30,41,59] maxBlueExcess 23.5
  gap2 x[551,568] mean [6.67,11.75,29.03] max [30,41,59] maxBlueExcess 23.5
  gap3 x[834,851] mean [6.67,11.75,29.03] max [30,41,59] maxBlueExcess 23.5
  -> all three gaps identical to 2dp; the 23.5 blue-excess is the BORDER colour
     (59-(30+41)/2 = 23.5), not a halo.
Left outer margin mean [16.00,23.24,41.10] == right outer margin mean
  [16.00,23.24,41.10] -- identical, so the leftmost card is not haloed relative to
  the rightmost.

CRITICAL CAVEAT I ESTABLISHED MYSELF (this is the vacuity risk in criterion 2's
literal wording): `alpha-score-glow` is `animation: pulse-glow 3s infinite`
(globals.css:61-68) whose ONLY effect is an OUTER `box-shadow`
(0 0 20px rgba(56,189,248,.3) at 0%/100%, 0 0 40px + 0 0 60px at 50%). It changes
NEITHER `border-navy-700` NOR `bg-navy-800/70`. So an oracle that measures ONLY
"border and background RGB" is VACUOUS against a glow -- it would read identical
with the glow present. The oracle that can actually discriminate is the GAP /
OUTER-MARGIN region, which is what I measured above. Positive control PENDING.

## D. LIVE POSITIVE CONTROL -- EXECUTED BY ME, NOT REASONED

The vacuity risk above had to be settled by EXECUTION (an infinite CSS animation can
be cancelled to its un-animated state by Playwright's screenshot path, which would
make the whole screenshot oracle blind to a glow). I therefore captured a LIVE
positive control of the IDENTICAL CSS class in the SAME app:

`frontend/src/app/performance/page.tsx:178` renders `<BentoCard glow>` (Win Rate) next
to two plain `<BentoCard>` siblings -- the same 1-of-N-uniform-row shape as the bug.
I navigated there MYSELF (`browser_navigate http://localhost:3000/performance`),
CONFIRMED the URL was `/performance` and NOT `/login`, confirmed the page had settled
(Win Rate 0.0%, 0W/3L, cost table fully populated, console: 0 errors / 0 warnings),
and captured at 1440x900 -> `.playwright-mcp/qa9113_poscontrol_perf_row.png`.

RESULT: the glow IS captured -- visibly and measurably. Playwright does NOT cancel it.
13px band immediately ABOVE each card:
  Win Rate  (GLOW)  mean RGB [ 6.31, 20.68, 40.76]  maxBlueExcess 33.5
  Avg Return(plain) mean RGB [ 2.00,  6.00, 23.00]  maxBlueExcess 19.0
  Beat Bench(plain) mean RGB [ 2.00,  6.00, 23.00]  maxBlueExcess 19.0
18px GAP INTERIOR (apples-to-apples with the 91.13 capture's gaps):
  gap(glow|plain)  mean [5.808, 19.020, 38.834] max [11,35,59] meanBE 26.42
  gap(plain|plain) mean [2.000,  6.000, 23.000] max [ 2, 6,23] meanBE 19.00

=> CALIBRATED ORACLE: "no glow" signature is EXACTLY [2,6,23] / BE 19.00 with zero
   variance; "glow" lifts G by ~13 and B by ~16 on the mean and raises max to [11,35,59].

## E. APPLYING THE CALIBRATED ORACLE TO THE SHIPPED 91.13 CAPTURE

  gap1 INTERIOR x[272,281] mean = max = [2, 6, 23]  meanBE 19.000
  gap2 INTERIOR x[555,564] mean = max = [2, 6, 23]  meanBE 19.000
  gap3 INTERIOR x[838,847] mean = max = [2, 6, 23]  meanBE 19.000
  BYTE-IDENTITY of the three full 140x18x3 gap blocks (numpy array_equal):
    gap1==gap2 True   gap1==gap3 True   gap2==gap3 True
  Outer left margin block == mirrored outer right margin block: True

If card1 (Total Cost) still glowed, gap1 would carry the halo and gap3 would not --
byte-identity across all three gaps is impossible under that hypothesis. All three
gap interiors read the calibrated NO-GLOW signature with ZERO variance.
CRITERION 2: **MET**, on an executed, calibrated, discriminating oracle.

## F. CAPTURE PROVENANCE -- CORROBORATED FROM ARTIFACTS MAIN DID NOT AUTHOR

I could NOT re-drive the Cost tab myself. Two independent reasons, the second of which
I verified in source rather than taking from Main:
  (i)  `browser_click` is ABSENT from my tool surface (navigate/snapshot/screenshot/
       console only), and I have no ToolSearch to load it.
  (ii) `frontend/src/components/ReportTabs.tsx:20` holds tab state in
       `useState(tabs[0]?.id ?? "overview")` -- there is NO searchParams read and NO
       deep-link, so the Cost tab is reachable ONLY by a click. VERIFIED BY ME.
This is §1c's explicitly-degraded fallback (tools absent from my surface) and I
disclose it. I compensated with third-party corroboration:
  - `.playwright-mcp/page-2026-08-20T20-40-11-561Z.yml` (22:40:11 local, 35s BEFORE the
    shipped PNG and 32s AFTER the code edit) contains
    `button "Cost $8.21" [active]` plus `Total Cost / $8.21 / Total Tokens / 154.8K /
    ... Deep Think Calls / "4"` and `MRVL` -- the exact panel, tab-active, on a real
    report, matching the PNG's values.
  - `.playwright-mcp/console-2026-08-20T20-39-45-550Z.log` logs TWO completed
    `[Fast Refresh] rebuilding -> done` cycles (1428ms, 408ms) -- the edit WAS hot-
    reloaded into the running page before the capture. So the PNG is post-fix, not stale.
  - The PNG's machine-regularity (three byte-identical gap blocks, zero-variance card
    interiors, mirrored margins) is characteristic of a genuine browser raster.

## G. MUTATION MATRIX vs CRITERION 1's GREP GUARD (executed, 6 cells)

  M1 `<BentoCard glow>` restored                 -> grep 1  KILLED
  M2 `<BentoCard glow={true}>`                   -> grep 1  KILLED
  M3 `<BentoCard className="alpha-score-glow">`  -> grep 0  SURVIVES grep
  M4 multi-line `<BentoCard\n  glow\n>`          -> grep 0  SURVIVES grep
  M5 glow moved to a SIBLING card (line 91)      -> grep 1  KILLED
  M6 `glow` default flipped true in BentoCard.tsx-> grep 0  SURVIVES grep

  Coverage by criterion 2's now-CALIBRATED pixel oracle:
    M3 -> card1 glows -> gap1 diverges from gap3 -> KILLED by criterion 2.
    M4 -> identical render to M1 -> KILLED by criterion 2.
    M6 -> ALL FOUR cards glow. SURVIVES BOTH criteria -- but note criterion 1's own
          OR-branch explicitly BLESSES "all 4 cards made visually consistent by
          deliberate design", and criterion 2 asks only for UNIFORMITY. So M6 is a
          property of the immutable criteria as written, NOT a defect in the shipped
          work, and the criteria are immutable. NOTE-level, no verdict effect.
  => NEITHER guard is vacuous, and they are COMPLEMENTARY: the grep kills the literal
     regression (M1/M2/M5), the screenshot kills the disguised ones (M3/M4).
     Per Goodenough-Gerhart this licenses only "these 6 mutations were classified",
     never a global no-vacuity claim.

## H. CODE-REVIEW HEURISTICS (5 dimensions) -- evaluated, no findings

Dim1 security: 1-line JSX prop removal; no secret, no LLM path, no subprocess, no sink.
Dim2 trading-domain: no execution/risk/perf-metrics path touched.
Dim3 quality: `glow` remains an optional prop on BentoCard (BentoCard.tsx:9,13,19);
  the two other consumers (`performance/page.tsx:178`, `GlassBoxCards.tsx:31`) still
  compile -- tsc exit 0. NOT a consumer-contract-break (nothing removed from the API).
Dim4 anti-rubber-stamp: not financial logic; 1 line of presentation. No tautological or
  over-mocked test added. Mutation matrix executed above.
Dim5 evaluator anti-patterns: prior CONDITIONAL -> PASS is NOT sycophancy because the
  EVIDENCE CHANGED (live_check_91.13.md created 22:58:21, absent at the 22:41:17 cycle-1
  spawn; experiment_results updated 22:58:28). I also re-derived every number myself
  and added a live positive control cycle 1 did not have.

## I. CLAIM AUDIT (§4b) -- claims reproduced

  "1 line"                     -> git diff = 1 insertion / 1 deletion. REPRODUCES.
  "grep returns 0"             -> 0 (stdout), re-run at end of eval, still 0. REPRODUCES.
  "3 siblings already plain"   -> lines 85/91/97/103 all bare `<BentoCard>`. REPRODUCES.
  "7 external sources in full" -> 7 distinct URLs in the read-in-full table, Tier-1/2
                                  heavy (W3C SC 2.2.2, MDN, ActiveCampaign design
                                  system) + an explicitly ADVERSARIAL source and a
                                  PARTIAL-NEGATIVE finding. REPRODUCES, not padded.
  "AlphaScoreCard has 0 render sites" -> `grep -rn AlphaScoreCard frontend/src/`
                                  returns exactly 1 hit: its own definition at
                                  GlassBoxCards.tsx:25. REPRODUCES.
  Criteria copied verbatim     -> exact string match of BOTH masterplan criteria in
                                  contract_91.13.md. REPRODUCES.
  MINOR IMPRECISION (NOTE, no verdict effect): experiment_results.md:22 and
  live_check_91.13.md:16 both say the boxes share "identical `bg-navy-800/70`
  borders/backgrounds". The BACKGROUND is bg-navy-800/70 (composites to [12,18,36]);
  the BORDER is `border-navy-700` ([30,41,59]), a different token. The substantive
  claim (all four identical) reproduces exactly; only the token name is conflated.
  NOTE 2: the shipped capture is cropped to the 4-stat-box ROW and excludes the
  "Cost & Token Usage" heading card above it (CostDashboard.tsx:73-81). Criterion 2's
  operative clause is about the 4 STAT BOXES, all of which are in frame, so this does
  not undercut it.

## J. HARNESS COMPLIANCE -- CLOSED

A1 research-gate: brief_status COMPLETE, gate_passed true,
   external_sources_read_in_full 7 (>=5), urls_collected 32 (>=10),
   recency_scan_performed true ("## Recency scan (2024-2026)", 3 findings),
   audit_class false so coverage.dry not required. Contract cites the brief
   (contract_91.13.md:4-7, :41). PASS.
A2 contract-before-generate: contract 22:39:32 < code 22:39:39 < capture 22:40:46. PASS.
A3 experiment_results present + non-empty. PASS.
A4 log-last: `grep -F 'phase=91.13' handoff/harness_log.md` -> NO MATCH (exit 1);
   masterplan status = "pending". LOG has not run, step not flipped. PASS.
A5 no-verdict-shopping: evidence CHANGED after the cycle-1 spawn. PASS.
live_check gate: handoff/current/live_check_91.13.md exists, 1942 bytes, references
   the capture, carries a fenced block. PASS.
HEAD re-checked at end of eval: b81c2b38 (unchanged during evaluation).

## K. CRITERION ROLL-UP

C1 "grep returns 0 after the fix (glow prop removed from the Total Cost card)"
   -> **MET**. stdout 0; no `glow` token of any casing survives in the file; the
      diff is exactly the prop removal at line 85. (Satisfied on the FIRST branch of
      the OR, so the "documented reason" branch is not engaged.)
C2 "a Playwright screenshot of the Cost & Token Usage panel shows a uniform
    border/background across all 4 stat boxes"
   -> **MET**. All 4 card interiors [12,18,36] std 0.0; all 4 top borders [30,41,59];
      three gap blocks BYTE-IDENTICAL; gap interiors exactly the calibrated no-glow
      signature [2,6,23]; margins mirrored. Oracle proven discriminating by an
      EXECUTED live positive control on the identical CSS class.

VERDICT: PASS. No unintended production change within 91.13's scope; harness
compliance clean; both immutable criteria met on independently re-derived evidence.

COMPLETED: 2026-08-20T21:08:52Z
