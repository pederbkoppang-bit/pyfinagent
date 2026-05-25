# Contract -- phase-44.6 Analyze section refresh

**Step id:** 44.6
**Cycle:** 64 (2026-05-25)
**Hypothesis:** Fixing the documented Home page equal-height anti-pattern + extracting the `useEnrichmentSignals` hook + adding KPI sparklines/LiveBadge/ARIA + signals page recent-tickers chips + progressive-disclosure shape will (a) remove the canonical anti-pattern called out in `.claude/rules/frontend.md:23`, (b) reduce signals page noise by ~52 LoC of inline type coercion, and (c) advance UX-DoD criterion 6/8/9 on the Analyze surface.

## Research gate

- Researcher subagent `a578f3cfa9547464c`, tier=simple, executed 2026-05-25.
- External sources read in full: **9** (>= 5 floor). NN/G progressive disclosure + Every Layout sidebar + Tremor SparkAreaChart x2 + MDN role=group + MDN role=region + MDN align-items + W3C WAI-ARIA APG region + W3C WAI-ARIA APG toolbar.
- Snippet-only sources: 14.
- Recency scan (2024-2026): performed; "no findings overturn canonical sources; ARIA 1.3 + subgrid Baseline 2024 are additive".
- Search queries: 3-variant discipline across 5 topics.
- Internal codebase audit: 20 file:line entries.
- **gate_passed: true.**
- Brief: `handoff/current/research_brief_phase_44_6.md`.

## North-star (N*) delta

- **B (Burn) primary:** -1 documented anti-pattern (Home page equal-height +per-child h-full); -52 LoC inline type coercion on signals page; lower operator-cognitive-load on /signals (recent-tickers chips + progressive disclosure).
- **R (Risk) speculative:** Better ARIA semantics + KPI sparkline trend visibility helps the operator spot trend regressions earlier; magnitude unknown.
- **P (Profit) speculative:** Marginal -- this is housekeeping + presentation, not signal-generation.

Defended because the anti-pattern is explicitly named in the project's frontend rules (`frontend.md:23`), the coercion is a 52-LoC type-safety hazard (any backend schema drift silently goes through `as unknown as Record<string, unknown>` cast), and the chip-row + progressive disclosure are NN/G-documented operator-efficiency patterns (researcher source #1 + #2).

## Scope (code work)

Executes 8 of 9 success criteria from `.claude/masterplan.json::phase-44.6.verification.success_criteria`:

| # | Criterion (verbatim) | This cycle? | Approach |
|---|----------------------|-------------|----------|
| 1 | home_3box_row_h_full_anti_pattern_removed_per_frontend_md_line_23 | YES | Remove `lg:items-stretch` from `frontend/src/app/page.tsx:283` AND `h-full` from the three children at lines 284 + 291 + 298. Per `frontend-layout.md` Section 4.5 option 2 ("items-start; accept visible asymmetry"). |
| 2 | home_6_KPI_tiles_have_Sparkline_LiveBadge_aria_label_role_group | YES | Wrap the 6-KpiTile grid in `<div role="group" aria-label="Portfolio key performance indicators">`. Extend `KpiTile` to optionally accept `sparkData` + `liveBand`/`liveAgeSec` props. 5 of 6 tiles get sparklines (NAV / P&L / vs SPY / Sharpe / Max DD); Positions tile gets a count change indicator but no time-series sparkline (no daily series available). LiveBadge compact dot shown on NAV + Positions (the live-fetched ones). |
| 3 | home_LCP_under_2_seconds | DEFERRED (Lighthouse operator-side) | Code keeps `next/dynamic` ssr:false on Recharts already. No regression introduced. |
| 4 | signals_useEnrichmentSignals_hook_extracted_to_frontend_src_lib_hooks | YES | Move 52 LoC at `frontend/src/app/signals/page.tsx:34-85` into `frontend/src/lib/hooks/useEnrichmentSignals.ts`. Hook signature: `useEnrichmentSignals(data: AllSignals | null): EnrichmentSignals | null`. Returns same shape so consumer code is a one-line replacement. |
| 5 | signals_50_LoC_of_inline_type_coercion_removed_from_signals_page_tsx | YES (CONFIRMED by criterion 4) | The hook extraction IS the coercion removal. |
| 6 | signals_input_gains_aria_label_ticker_symbol_and_label_pairing | YES | Add `<label htmlFor="signals-ticker-input">Ticker symbol</label>` + `aria-label="Ticker symbol"` on the `<input>` at signals/page.tsx ~line 104. (Currently no label or aria-label.) |
| 7 | signals_recent_tickers_chips_below_input_last_5_clickable | YES | New `<RecentTickerChips>` row below the input: localStorage key `pyfinagent.signals.recentTickers`. On submit, prepend ticker (deduped, max 5). Each chip click calls `setTicker + go`. `role="group"` per researcher source #4. Tailwind `py-1.5` for WCAG 2.2 24px target-size. |
| 8 | signals_progressive_disclosure_consensus_pill_then_12_cards_then_collapsible_details | YES | Render order: consensus pill -> 12 cards grid -> Sector + Macro inside native `<details>` (level-3 disclosure). NN/G ceiling of 2 disclosure levels respected. Need to identify the existing render structure + restructure. |
| 9 | Lighthouse_a11y_at_least_95_on_both_pages | DEFERRED (operator-side) | ARIA + label wiring done; audit pending operator Lighthouse run. |

## Out of scope this cycle (operator-side)

- Criterion 3 (home_LCP_under_2_seconds) -- Lighthouse runs are operator-side per /goal.
- Criterion 9 (Lighthouse_a11y_at_least_95) -- same.

The verification command is `test -f handoff/current/live_check_44.6.md` (single-gate, no operator_approval requirement). Once `live_check_44.6.md` is created this cycle, the verification command PASSES. Step CAN flip to `done` after Q/A PASS -- a meaningful difference from phase-44.2's two-gate AND.

## Plan steps

1. **Extend `KpiTile` component.** Add optional `sparkData?: { date: string; value: number }[]`, `liveBand?: FreshnessBand`, `liveAgeSec?: number | null`, `ariaLabel?: string` props. Add Tremor `SparkAreaChart` (or Recharts mini if Tremor doesn't fit) when sparkData present. Conditionally render compact LiveBadge.
2. **Wrap 6-KPI grid in role="group".** `<div role="group" aria-label="Portfolio key performance indicators" ...>` at `frontend/src/app/page.tsx:227`.
3. **Remove the equal-height anti-pattern.** Drop `lg:items-stretch` at line 283; drop `h-full` at 284 + 291 + 298. Confirm with `git diff` + production build.
4. **Extract `useEnrichmentSignals` hook.** New `frontend/src/lib/hooks/useEnrichmentSignals.ts` (~70 LoC); add to barrel at `frontend/src/lib/hooks/index.ts`. Replace `const enrichmentSignals = data ? {...}` block in signals/page.tsx with `const enrichmentSignals = useEnrichmentSignals(data);`.
5. **Add label + aria-label to signals page input.** Add `<label htmlFor="signals-ticker-input">Ticker symbol</label>` (visually-hidden if needed) + `aria-label` on input.
6. **Build `RecentTickerChips`** at `frontend/src/components/RecentTickerChips.tsx` (~80 LoC). localStorage with the canonical key. Hooks into the existing setTicker + submit handlers.
7. **Progressive disclosure of /signals.** Identify the current render order. Move Sector + Macro into a native `<details>` block at the bottom (level 3). Keep consensus pill + 12 cards at level 1 + 2.
8. **Vitest coverage.** New test files: `KpiTile.test.tsx` (sparkline/LiveBadge/aria), `useEnrichmentSignals.test.ts`, `RecentTickerChips.test.tsx` (localStorage round-trip + chip dedupe).
9. **Verification gates.** pytest backend >= 614; vitest >= 83; tsc --noEmit exit 0; npm run build green; ascii_logger_check exit 0; emoji scan 0 hits.
10. **Honest deferrals** documented (criteria 3 + 9 = operator Lighthouse).

## Files planned

NEW:
- `frontend/src/lib/hooks/useEnrichmentSignals.ts`
- `frontend/src/lib/hooks/useEnrichmentSignals.test.ts`
- `frontend/src/components/RecentTickerChips.tsx`
- `frontend/src/components/RecentTickerChips.test.tsx`
- `frontend/src/components/KpiTile.tsx` (or kept inline + extracted; decide during implementation)
- `frontend/src/components/KpiTile.test.tsx`
- `handoff/current/live_check_44.6.md`

MODIFIED:
- `frontend/src/app/page.tsx` (anti-pattern removal + KpiTile extension call sites + role=group wrapper)
- `frontend/src/app/signals/page.tsx` (hook call + label + chips + progressive disclosure restructure)
- `frontend/src/lib/hooks/index.ts` (barrel export the new hook)

ZERO backend changes.

## Verification command (immutable per masterplan)

```
test -f handoff/current/live_check_44.6.md
```

Single-gate verification (no operator_approval required). After this cycle creates the file + Q/A PASSes, the masterplan step CAN flip to `done`.

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 backend + 83 frontend (current baseline) | Run both. No backend changes; frontend net +X tests. |
| 2 | TS build + ast.parse green | `npx tsc --noEmit` + `npm run build`. |
| 3 | New feature behind flag default OFF | Anti-pattern removal + ARIA + label additions are bug-fix-level; not new features. Hook extraction is a refactor. Chips component is a small additive feature -- could feature-flag but the master_design explicitly calls for it; ship inline. |
| 4 | BQ migrations idempotent | N/A. |
| 5 | New env vars documented | N/A. |
| 6 | Contract has N* delta | DONE. |
| 7 | Zero emojis | Grep on all changed files. |
| 8 | ASCII loggers | N/A (frontend). `scripts/qa/ascii_logger_check.py` exit 0. |
| 9 | Single source of truth | Hook extraction REINFORCES SSOT (was inline duplication). KpiTile reused 6x. RecentTickerChips reusable for future input pages. |
| 10 | log FIRST / flip LAST | harness_log append BEFORE masterplan flip. Step CAN flip this cycle (single-gate verification command satisfied after live_check is written). |

## Circuit-breaker plan

- If pytest count drops -> revert + investigate.
- If tsc errors -> revert offending file.
- If scope risks > 3 cycles -> stop + file blocker.

## Contract sign-off

Authored AFTER researcher returned `gate_passed: true`. N* delta declared. 7 success criteria targeted; 2 deferred (operator Lighthouse). Status CAN flip to `done` this cycle if Q/A PASSes (no operator_approval second-gate this time).
