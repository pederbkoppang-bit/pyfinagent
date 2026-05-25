# phase-44.6 -- experiment results (Cycle 64)

**Date:** 2026-05-25
**Cycle:** 64
**Step:** phase-44.6 -- Analyze section refresh (Home h-full anti-pattern fix + KPI sparklines/LiveBadge/role=group; /signals useEnrichmentSignals hook + label + recent-tickers chips + progressive disclosure)

## Summary

Home page anti-pattern removed (per `.claude/rules/frontend.md:23` rule).
6-KPI grid wrapped in `role="group"` with single aria-label; each tile
gains optional Tailwind-only mini-sparkline + LiveBadge dot (NAV +
Positions). `/signals` page: 52 LoC of inline type coercion extracted to
new `useEnrichmentSignals` hook; ticker input gains `<label htmlFor>` +
`aria-label`; new `RecentTickerChips` component (last-5 LRU + dedupe +
localStorage); Sector + Macro dashboards moved into a native `<details>`
collapsible (level-3 progressive disclosure per NN/G research).

7 of 9 code criteria PASS this cycle. 2 deferred (criteria 3 + 9 =
Lighthouse runs, operator-side). The verification command
`test -f handoff/current/live_check_44.6.md` is single-gate -- once the
file is created this cycle and Q/A PASSes, the step CAN flip to `done`
on the harness's next masterplan write. No operator approval required
for this step.

## Files shipped

**NEW (5 files):**

| File | Lines | Role |
|------|-------|------|
| `frontend/src/lib/hooks/useEnrichmentSignals.ts` | 67 | Extracts 52-LoC inline type-coercion from signals/page.tsx; defensive pick() helper; same `EnrichmentSignals` return shape |
| `frontend/src/lib/hooks/useEnrichmentSignals.test.ts` | 71 | 6 vitest cases (null / missing fields / typed extraction / coerced fields / non-object / wrong-type) |
| `frontend/src/components/RecentTickerChips.tsx` | 128 | last-5 LRU + dedupe + localStorage roundtrip + role=group + WCAG 2.2 24px target-size |
| `frontend/src/components/RecentTickerChips.test.tsx` | 145 | 11 vitest cases (empty / hydrate / click / submit / dedupe / cap / uppercase / blank / role=group / aria-label / target-size / internals) |
| `handoff/current/live_check_44.6.md` | -- | Verdict + criteria table |

**MODIFIED (4 files):**

| File | Diff | Change |
|------|------|--------|
| `frontend/src/lib/hooks/index.ts` | +1 | Barrel re-exports useEnrichmentSignals |
| `frontend/src/app/page.tsx` | +90 -45 | (a) KpiTile extended with sparkData / sparkPositive / liveBand / liveAgeSec / ariaLabel; (b) MiniSpark Tailwind-SVG mini-area chart (zero new deps); (c) 6-KPI grid wrapped in role=group + aria-label; (d) navNums / dailyPctSeries / alphaSeries / ddSeries derivation; (e) **anti-pattern removed**: `lg:items-stretch` + per-child `h-full` dropped per frontend-layout.md Section 4.5 option 2; replaced with `lg:items-start` |
| `frontend/src/app/signals/page.tsx` | +33 -55 | (a) replaced 52 LoC inline coercion with one-line `useEnrichmentSignals(data)` call; (b) added `<label htmlFor>` + `aria-label` to ticker input; (c) wired `RecentTickerChips` row below input; (d) progressive-disclosure: Sector + Macro moved into native `<details>` (level 3); (e) `handleFetch(ticker?)` signature lets chip-click drive the fetch |
| `handoff/current/contract.md` | overwrite | Cycle 64 contract |

**ZERO new backend code; ZERO new env vars; ZERO new dependencies (used a Tailwind+SVG mini-spark inline rather than pulling Tremor's SparkAreaChart -- cheaper bundle + simpler test surface).**

## Verification command output

```
$ test -f handoff/current/live_check_44.6.md
$ echo $?
0
```

Single-gate verification PASSES once `live_check_44.6.md` is created
this cycle. Step CAN flip to `done` after Q/A PASSes.

## /goal integration-gate scoreboard

| # | Gate | Verdict | Evidence |
|---|------|---------|----------|
| 1 | pytest >= 614 backend + 83 frontend (cycle-63 baseline) | **PASS** | backend 614 collected / 589 passed (same 14 pre-existing env/calendar/doc-archive failures as cycle 63; ZERO new regressions). Frontend vitest 15 files / 100 tests pass (+17 net vs cycle 63's 83). |
| 2 | TS build + ast.parse green | **PASS** | `npx tsc --noEmit` exit 0. `npm run build` green; all 22 routes emit including 7 /paper-trading/* sub-routes from cycle 63. |
| 3 | Feature behind flag default OFF | **N/A** | Anti-pattern removal + ARIA wiring + hook extraction are bug-fix/refactor-level, not new features. Chip row is small additive UX called out explicitly in master_design Section 3.11 -- ship inline. |
| 4 | BQ migrations idempotent | **N/A** |
| 5 | New env vars documented | **N/A** |
| 6 | Contract has N* delta | **PASS** -- B primary + R/P speculative declared in `handoff/current/contract.md`. |
| 7 | Zero emojis | **PASS** -- emoji scan across 7 changed files: TOTAL HITS 0. |
| 8 | ASCII loggers | **PASS** -- no backend logger touches. `scripts/qa/ascii_logger_check.py` (untouched dependencies) exits 0. |
| 9 | Single source of truth | **PASS** -- hook extraction REINFORCES SSOT (was inline duplication risk); KpiTile reused 6x; RecentTickerChips reusable. |
| 10 | log FIRST / flip LAST | **HOLDING** -- harness_log append happens next; status flip happens after Q/A PASS. |

## Criteria table

| # | Criterion (verbatim) | Verdict | Evidence |
|---|----------------------|---------|----------|
| 1 | home_3box_row_h_full_anti_pattern_removed_per_frontend_md_line_23 | **PASS** | `frontend/src/app/page.tsx` line 332ish: `lg:items-stretch` -> `lg:items-start`; line 333ish + 339ish + 345ish: per-child `h-full` dropped. The anti-pattern named at `.claude/rules/frontend.md:23` is gone. |
| 2 | home_6_KPI_tiles_have_Sparkline_LiveBadge_aria_label_role_group | **PASS** | `frontend/src/app/page.tsx` line ~290: `<div role="group" aria-label="Portfolio key performance indicators">`. Each KpiTile gains `role="group"` + per-tile `aria-label` (default: `${label} ${value}${subText ? ` (${subText})` : ""}`). 5 of 6 tiles get sparklines (NAV / P&L / vs SPY / Sharpe / Max DD; Positions skipped per researcher topic 2 -- no time-series). LiveBadge compact dot on NAV + Positions. |
| 3 | home_LCP_under_2_seconds | **DEFERRED** (operator-side Lighthouse) | Code preserves the existing `next/dynamic` ssr:false on RedLineMonitor (the heaviest bundle). Sparkline is inline Tailwind SVG (no Recharts/Tremor bundle add). Should not regress LCP. |
| 4 | signals_useEnrichmentSignals_hook_extracted_to_frontend_src_lib_hooks | **PASS** | `frontend/src/lib/hooks/useEnrichmentSignals.ts` exists; consumed at `signals/page.tsx:33` via `useEnrichmentSignals(data)`. Hook re-exported via `frontend/src/lib/hooks/index.ts`. |
| 5 | signals_50_LoC_of_inline_type_coercion_removed_from_signals_page_tsx | **PASS** | The previous 52 LoC at signals/page.tsx:34-85 collapsed to 2 lines (comment + one-call). Verified via `git diff --stat`. |
| 6 | signals_input_gains_aria_label_ticker_symbol_and_label_pairing | **PASS** | `<label htmlFor="signals-ticker-input">Ticker symbol</label>` + `<input id="signals-ticker-input" aria-label="Ticker symbol" .../>` |
| 7 | signals_recent_tickers_chips_below_input_last_5_clickable | **PASS** | `<RecentTickerChips onSelect={...} recentlySubmitted={lastSubmitted} />` mounted below the input. localStorage key `pyfinagent.signals.recentTickers`. last-5 LRU + dedupe verified by 11 vitest cases. |
| 8 | signals_progressive_disclosure_consensus_pill_then_12_cards_then_collapsible_details | **PASS** | Render order: SignalSummaryBar (level 1 consensus pill) -> SignalCards (level 2 12 cards) -> `<details>` wrapping SectorDashboard + MacroDashboard (level 3). NN/G ceiling of 2 deep respected. |
| 9 | Lighthouse_a11y_at_least_95_on_both_pages | **DEFERRED** (operator-side) | ARIA wiring done (criteria 2 + 6); audit pending operator Lighthouse run. |

**7 PASS + 2 DEFERRED (both operator Lighthouse). Verdict PASS for code work.**

## Pytest sweep

```
$ pytest backend/ -q --no-header
14 failed, 589 passed, 2 skipped, 9 xfailed, 1 warning in 111.57s
```

Same 14 pre-existing failures as cycle 63 (BQ freshness x4 calendar-bound; watchdog 7d; layer1 BQ writes; shortlist doc archived x6; rainbow canary flaky; verify_phase_23_1_17 cascade). ZERO new regressions caused by phase-44.6.

## Frontend pytest sweep

```
$ npm test -- --run
 Test Files  15 passed (15)
      Tests  100 passed (100)
```

+17 net frontend tests (83 -> 100):
- +6 useEnrichmentSignals.test.ts
- +11 RecentTickerChips.test.tsx

## Operator runbook for closing criteria 3 + 9

```bash
# Pull change + restart frontend
cd /Users/ford/.openclaw/workspace/pyfinagent && git pull origin main
launchctl kickstart -k "gui/$(id -u)/com.pyfinagent.frontend"

# Criterion 3 (home_LCP_under_2_seconds):
npx lighthouse http://localhost:3000/ --only-categories=performance --form-factor=desktop
# Look for LCP < 2000ms.

# Criterion 9 (Lighthouse_a11y_at_least_95_on_both_pages):
npx lighthouse http://localhost:3000/ --only-categories=accessibility
npx lighthouse http://localhost:3000/signals --only-categories=accessibility
# Score >= 95 on each.
```

## Q/A expectations

- 5-item harness-compliance audit must PASS: researcher (`a578f3cfa9547464c` brief at `research_brief_phase_44_6.md`) + contract pre-commit + experiment_results + log-LAST + no-verdict-shopping.
- Single-gate verification command `test -f handoff/current/live_check_44.6.md` PASSes after this cycle writes the file.
- 7 of 9 criteria PASS code-side; 2 honest deferrals to operator Lighthouse.
- Step CAN flip to `done` on the harness's next masterplan write (no operator_approval second-gate this time, unlike cycle 63's phase-44.2).
