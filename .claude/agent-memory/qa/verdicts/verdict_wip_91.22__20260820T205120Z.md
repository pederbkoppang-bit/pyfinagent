STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 91.22
WRITTEN: 2026-08-20T20:51:20Z

# Q/A write-first record -- step 91.22 (Recharts Tooltip itemStyle contrast)

## A. HARNESS COMPLIANCE (5 items)
1. research-gate-before-contract: PASS. research_brief_91.22.md envelope
   brief_status=COMPLETE, gate_passed=true, tier=simple,
   external_sources_read_in_full=7 (>=5), urls_collected=26 (>=10),
   recency_scan_performed=true. mtime 22:32:42 < contract 22:42:24.
   contract_91.22.md:4-6 cites the brief by path.
2. contract-before-generate: PASS WITH A DISCLOSED 32s DEVIATION.
   Full mtime chain:
     22:32:42 research_brief_91.22.md
     22:41:52 frontend/src/lib/chart-tooltip-style.ts   <-- 32s BEFORE the contract
     22:42:24 contract_91.22.md
     22:42:43 .. 22:45:30  the other 13 edits (all AFTER)
     22:48:19 experiment_results_91.22.md
   contract_91.22.md:32 discloses it verbatim: "(done during research write-up)".
   Sequence is RESEARCH -> research artifact -> CONTRACT -> GENERATE, not a
   contract fitted to finished code. NOTE, not a blocker.
3. experiment_results present: PASS (22:48:19, after every edit).
4. log-last: PASS. `grep -c -F 'phase=91.22' handoff/harness_log.md` = 0;
   masterplan status = pending; `git log --oneline -20 | grep -F 91.22` = none.
5. no-verdict-shopping: PASS -- this is attempt 1.
   qa_wip.py 91.22 --spawned-at 2026-08-20T20:51:20Z ->
     source_present=true, attempt_number=1, attempt_number_status="ok",
     attempt_number_is_lower_bound=false, prior_attempts=0, prior_records=[],
     records_retained=1 (this spawn's own record; a GAUGE, not a counter).
   verdict_history_86_21.py --step 91.22 --evidence-only ->
     status="no_rows_for_step", verdicts=(none).
   Cross-check prior_attempts(0) vs ledger rows(0): equal -> no staleness signal.
   Sequence: no prior verdicts recorded for this step.

## B. DETERMINISTIC

### Immutable verification command -- exit=0 (run twice, start and end of eval)
```
frontend/src/components/SectorDashboard.tsx:1
frontend/src/components/TransformerForecastPanel.tsx:1
frontend/src/components/PaperReconciliationChart.tsx:1
frontend/src/components/OptimizerInsights.tsx:2
frontend/src/components/RedLineMonitor.tsx:1
frontend/src/components/StockChart.tsx:2
frontend/src/components/StrategyDetail.tsx:1
exit=0
```
Non-zero for all 7 named files. Sum = 9 = exactly the 9 Tooltip instances the
masterplan audit_basis enumerated. CRITERION 2 MET.
Comment-token trap checked: all 9 matching lines enumerated verbatim, every one
is a real JSX prop `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` sitting on the line
directly after its `<Tooltip` (SectorDashboard:123, TransformerForecastPanel:120,
PaperReconciliationChart:146, RedLineMonitor:192, OptimizerInsights:159/236,
StockChart:229/317, StrategyDetail:85). Zero comment matches. The import line
does NOT match the case-sensitive grep (constant is UPPER_SNAKE).

### Independent scope derivation (git/grep, NOT the author's list)
`grep -rn '<Tooltip' frontend/src/` => 16 files, **21** instances (excluding the
doc-comment occurrence in the new chart-tooltip-style.ts).
Per file  tooltips / itemStyle / `content={`:
  backtest/page.tsx            3 / 3 / 0
  paper-trading/nav/page.tsx   1 / 1 / 0
  reports/page.tsx             2 / 2 / 0
  OptimizerInsights            2 / 2 / 0
  PaperReconciliationChart     1 / 1 / 0
  RedLineMonitor               1 / 1 / 0
  SectorDashboard              1 / 1 / 0
  StockChart                   2 / 2 / 0
  StrategyDetail               1 / 1 / 0
  TransformerForecastPanel     1 / 1 / 0   => 10 files / 15 default-content, ALL fixed
  BudgetDashboard              1 / 0 / 1
  ComputeCostBreakdown         1 / 0 / 1
  MfeMaeScatter                1 / 0 / 1
  OptimizerProgressChart       1 / 0 / 1
  PerfProgressChart            1 / 0 / 1
  SharpeHistoryChart           1 / 0 / 1   => 6 files / 6 custom-content, excluded
15 + 6 = 21. Zero default-content instances missed. KNOWN-MEMBER RECALL: the
9 masterplan-named members are all present and all fixed.

### Exclusion of the 6 custom-content files -- mechanism VERIFIED, not accepted
`grep -rn 'itemStyle'` across all 6 => NONE reads the prop. Recharts applies
itemStyle only inside DefaultTooltipContent, which never runs when `content` is
supplied. Setting itemStyle there would be a guaranteed no-op. Their render
functions use explicit readable Tailwind tokens (text-slate-100/200/300/400/500,
text-amber-300) -- none renders black text, so the exclusion leaves no live
instance of the reported defect. Disclosed in contract + experiment_results and
queued as a follow-up. Correct call.

### Root cause independently reproduced
SectorDashboard.tsx:132 `<Bar dataKey="return" radius=...>` sets NO `fill`;
colour comes from per-`<Cell>` fills. Recharts derives tooltip `entry.color` from
the SERIES fill, not the Cell -> undefined -> DefaultTooltipContent falls back to
`'#000'`. Exactly the operator's screenshot.

### BEHAVIOURAL MUTATION MATRIX -- executed against installed recharts 2.15.4
`node -e` + react-dom/server rendering the real DefaultTooltipContent (no files
written). Emitted inline styles on `.recharts-tooltip-item`:
| cell | mutant | result |
|---|---|---|
| M1 | itemStyle REMOVED, entry.color undefined (pre-fix state) | `color:#000` -> 1.18:1  DEFECT REPRODUCED |
| M2 | itemStyle = {color:'#e2e8f0'} (shipped fix) | `color:#e2e8f0` -> 14.48:1  KILLED |
| M3 | entry.color='#10b981', no itemStyle | `color:#10b981` -> background-coupled, as the research claimed |
| M4 | entry.color='#10b981' + itemStyle | `color:#e2e8f0` -> itemStyle WINS (spread order proven behaviourally) |
| M5 | contentStyle.color='#e2e8f0' ONLY (the removed code) | item row still `color:#000` -> "never reached item rows" is TRUE |
| M6 | label style with/without contentStyle.color | both `margin:0` -> label carries no own colour; inherits |
=> itemStyle is LOAD-BEARING at runtime. The fix is not a source-level gesture.

### In-memory mutation cells on the grep guard (no file written; `sed | grep -c`)
- strip `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` from SectorDashboard -> count 0
  => the immutable command CAN fail. Not vacuous.
- replace it with `itemStyle={{ color: "#000" }}` -> count still 1
  => SURVIVOR. The immutable command cannot distinguish a readable colour from
  black (vacuity shape #3, literal-kept-behaviour-stripped). This is a property
  of criterion 2 as written ("non-zero count"), and criterion 1's "readable
  colour" half is covered by three INDEPENDENT behavioural guards: the measured
  constant (14.48:1), the three measured live captures, and M1/M2 above. Per
  qa.md 4c wiring this is a named limit, not a blocking finding, because the
  criterion does not rest on the grep alone.

### Every quantified claim in the new constant's doc comment REPRODUCES
- installed recharts 2.15.4 (node -e require(recharts/package.json).version) OK
- es6/component/DefaultTooltipContent.js:58 == `color: entry.color || '#000'`
  exact line, exact literal OK; `_objectSpread({...defaults}, itemStyle)` OK
- contrast recomputed by me (WCAG relative luminance):
    #e2e8f0 on #0f172a = 14.48:1 (claim 14.48) OK
    #e2e8f0 on #1e293b = 11.87:1 (claim 11.87) OK
    #000000 on #0f172a =  1.18:1 (claim 1.18)  OK
  Both >= 7:1 -> AAA; matches rules/frontend.md text-slate-100 tier.

### Regression probe on the 3 `contentStyle.color` REMOVALS
RedLineMonitor / StrategyDetail / TransformerForecastPanel set NO labelStyle, so
contentStyle.color was NOT fully dead -- it reached the LABEL by inheritance from
the wrapper div. After removal the label inherits from globals.css:12-14
`body { color: #e2e8f0 }` -- the SAME value. CONFIRMED LIVE: the RedLineMonitor
capture's label "2026-08-04" is legible with no labelStyle anywhere.
Residual: "dead no-ops" is imprecise (redundant with the body token, not
unreachable) and leaves a latent coupling to that token. NOTE.

### Lint / typecheck
- `npx tsc --noEmit` -> exit=0
- `npx eslint <the 11 changed frontend files>` -> exit=0, 0 errors, 6 warnings.
  All 6 are pre-existing react-hooks warnings at lines this diff never touches
  (backtest 412/483/495/505, OptimizerInsights 428, StockChart 86 vs the diff's
  962/1377/1417, 158/235, 228/316 + imports).
- No *.py in the diff -> ruff gate N/A. No backend/** -> runtime smoke N/A.

## C. LIVE UI (qa.md 1c)

### Capture authorship -- EXPLICITLY-DEGRADED FALLBACK, disclosed
The three hover-state captures were produced by MAIN, not by me. My granted tool
surface is navigate / snapshot / screenshot / console ONLY -- there is no hover,
click or type tool, and /signals holds the ticker in local `useState` with no URL
param (signals/page.tsx:15), so I can neither load the Sector deep-dive nor
trigger a Recharts hover state. That is the documented "tools absent from your
surface" fallback. Compensating independent evidence below.

### My own live captures (1440x900, :3000, real NextAuth session)
- http://localhost:3000/signals -> URL confirmed /signals (NOT /login), sidebar
  shows pytest@localhost. 0 console errors. Page requires interaction to load a
  ticker, so no chart.
- http://localhost:3000/ -> URL confirmed /. 0 console errors. Red Line Monitor
  renders correctly under the CURRENT post-fix source. Page had SETTLED:
  GATE/KILL/CYCLE/LAST segments are populated (NOT em-dashes), NAV/P&L/Sharpe
  all resolved. I started and killed no server.

### Quantitative measurement of Main's captures (PIL, tight tooltip crops)
Freshness: captures 22:46:50 / 22:47:20 / 22:47:44 -- all AFTER every source
edit (22:41:52 .. 22:45:30). Post-fix, not stale.
| capture | brightest px | contrast vs #0f172a | px darker than bg |
|---|---|---|---|
| sector_rotation_tooltip_fixed.png (146,154,272,230) | (226,232,240) = #e2e8f0 | 14.48:1 | 2 (noise) |
| stockchart_tooltip_spotcheck.png (862,505,976,622)  | (226,232,240) = #e2e8f0 | 14.48:1 | 105 (crop edge over the chart, not glyphs) |
| redlinemonitor_tooltip_spotcheck.png (855,603,982,673) | (226,232,240) = #e2e8f0 | 14.48:1 | 1 (noise) |
Visual read: Sector Rotation shows "Healthcare" AND "Return : 16.9%" both legible
-- the exact chart and the exact value row the operator reported. StockChart
shows "Mar 5 / Volume : 42.6M / Close : $75.62 / SMA 50 : $81.46". RedLineMonitor
shows "2026-08-04 / nav : 23803.94". All authenticated, all post-fix.

## CRITERIA
1. MET -- 9/9 named instances + 6/6 additional default-content instances (15/15)
   set itemStyle from ONE shared constant; #e2e8f0 = 14.48:1 / 11.87:1 (AAA);
   the 6 custom-content exclusions are mechanically proven inert and disclosed.
2. MET -- immutable command exit=0, non-zero for every listed file.
3. MET -- live post-fix capture of the Sector Rotation hover; label AND value
   measured at exactly #e2e8f0 / 14.48:1. (capture authorship disclosed above)
4. MET -- 2 spot-checks (StockChart, RedLineMonitor) measured identically;
   RedLineMonitor doubles as the regression probe for the removal.

## NON-BLOCKING FINDINGS
N1 contract-before-generate 32s deviation, disclosed in the contract itself.
N2 research_brief_91.22.md:199/:227 summary arithmetic is one low ("20 instances",
   "10 files / 14 instances"; true 21 and 15). The brief's file:line ANCHOR TABLE
   is complete -- all 21 anchors are listed -- and GENERATE applied 15, so there
   is no coverage gap; only the two summary numbers are wrong.
   experiment_results' "~15" matches reality.
N3 "3 dead contentStyle.color no-ops" is imprecise (see regression probe).
N4 TREE NOT FROZEN DURING EVALUATE. `git diff --name-only HEAD` carries 4 files
   with non-91.22 edits: app/observability/page.tsx (phase-91.9),
   components/CostDashboard.tsx (phase-91.13 BentoCard glow), and two comment-only
   `phase-X.Y` -> `phase X.Y` rewordings at app/page.tsx:465 and
   app/backtest/page.tsx:1519 that I could not attribute to 91.9/91.13/91.18/91.22.
   All are comment / UI-copy only with zero tooltip effect, so no criterion is
   touched -- but the backtest one sits INSIDE a 91.22 file, so a per-step
   auto-commit for 91.22 sweeps in a foreign edit unless staged selectively.
N5 immutable-command survivor (black-literal mutant) -- see mutation cells.
N6 capture authorship -- see above.

## WORST-OF-N LENSES
correctness PASS / does-it-reproduce PASS / scope-honesty PASS -> worst = PASS.

COMPLETED: 2026-08-20T21:02:40Z
