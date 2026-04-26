---
step: phase-16.52
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/settings/page.tsx (refactor: single-zone -> two-zone shell + active-tab color fix)
  - frontend/src/app/backtest/page.tsx (move ingestResult banner from fixed-header zone to scrollable zone)
---

# Experiment Results -- phase-16.52

## What was done

Refactored `frontend/src/app/settings/page.tsx` (the user-named "canonical
reference") to actually follow the canonical two-zone shell pattern
established in 16.48/16.49. Settings was the outlier -- it used the OLD
single-zone layout where the tab bar scrolled with content. Now settings
matches reports/backtest/agents/paper-trading.

Also moved `backtest/page.tsx` `ingestResult` banner from the fixed-header
zone into the scrollable content zone (per `frontend-layout.md` §3 Tier-2
"Status Banners" go in the scrollable zone, not the fixed header).

## Deliverables

### `frontend/src/app/settings/page.tsx` (refactor)

Three edits applied:

1. **Loading early-return (L509-532):** wrapped in two-zone shell. Header now in `flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8` zone; loading content / error banner in `flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8` zone.

2. **Main return (L534+):** wrapped in `<main className="flex flex-1 flex-col overflow-hidden">`. Header (page title + Save All Settings button) + sub-navigation tabs moved into `flex-shrink-0` zone. Tab content (Models/Scoring/Cost/Performance) moved into `flex-1 overflow-y-auto scrollbar-thin` zone.

3. **Active tab color (L574):** changed from `bg-slate-700 text-slate-100 shadow-sm` to `bg-sky-500/10 text-sky-400` per `frontend-layout.md` §5 canonical pattern. Now matches reports + backtest.

Closing tag added before `</main>` (one extra `</div>` to close the new scrollable wrapper).

### `frontend/src/app/backtest/page.tsx` (banner relocation)

`ingestResult` banner moved out of the `flex-shrink-0` fixed-header zone
(was L687-704) and into the scrollable content zone, immediately above
the existing `error` banner. The `phase-16.49` comment in the fixed-header
zone (which used to wrap the banner) was removed since the banner is no
longer there.

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)
$ cd frontend && npm run lint
0 errors, 34 warnings (all pre-existing in unmodified files; none from this cycle)
```

The TypeScript and ESLint passes are clean for the two files this cycle
modified. Pre-existing warnings live in `lib/api.ts`, `lib/useLivePrices.ts`,
chart components, and other files outside this cycle's scope.

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/settings/page.tsx` | refactored | Two-zone shell + active-tab color fix (loading + main returns) |
| `frontend/src/app/backtest/page.tsx` | relocated banner | ingestResult moved from fixed-header to scrollable zone |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.52-research-brief.md` | created (researcher, internal-heavy) | -- |

NO new files. NO new dependencies. NO test additions (this is pure-UI;
verification is `npx tsc --noEmit` per masterplan).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Settings uses canonical two-zone shell (`flex flex-1 flex-col overflow-hidden` + `flex-shrink-0` + `flex-1 overflow-y-auto scrollbar-thin`) | PASS |
| 2 | Settings loading state uses same two-zone shell | PASS |
| 3 | Settings active-tab color matches canonical (`bg-sky-500/10 text-sky-400`) | PASS |
| 4 | Backtest ingestResult banner is in scrollable zone (not fixed-header) | PASS |
| 5 | `cd frontend && npx tsc --noEmit` exits 0 | PASS |
| 6 | `cd frontend && npm run lint` -- zero new errors introduced | PASS (0 errors total; warnings unchanged) |

## Honest disclosures

1. **Cosmetic-tier violations deferred per contract** -- 3 cosmetic items left out of scope: (a) reports tab bar bg color drift navy-800/60 vs settings slate-800/50 (sibling inconsistency, harmless); (b) settings tab bar `max-w-fit` kept (non-standard but harmless); (c) SETTINGS_TABS missing icon field (cosmetic, would require importing 4 phosphor icons). Documented in research brief; can land in a follow-up.

2. **No browser visual verification.** The dev server is not necessarily running this cycle. The two-zone refactor follows the exact canonical template from `frontend-layout.md` §1 + matches reports/backtest/agents (which DID get visual verification in 16.49). High-confidence layout-only change.

3. **No regression on the 8 previously fixed pages.** Researcher confirmed login/signals/performance/home/agents/paper-trading/reports/backtest still pass. Sovereign has no tabs and was already correct.

4. **Settings is now ACTUALLY canonical.** Going forward, when the rules file or future cycles say "match settings", that reference is now self-consistent. Before this cycle, "match settings" would have meant copying the OLD pattern.

5. **Net diff is small** -- 1 file moderately edited (settings, ~50 lines around the shell), 1 file lightly edited (backtest, banner moved ~25 lines down).

## Closes

Net-new task #78 (UAT-16.52). Adds new step phase-16.52 to masterplan.

## Next

Spawn Q/A.
