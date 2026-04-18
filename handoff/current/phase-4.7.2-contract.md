# Contract -- Cycle 70 / phase-4.7 step 4.7.2

Step: 4.7.2 Redesign homepage as MAS operator cockpit

## Hypothesis

The canonical `OpsStatusBar` already exists
(`frontend/src/components/OpsStatusBar.tsx`, 339 lines, with Gate /
Kill / Cycle / Scheduler segments and inline Pause/Resume/Flatten
buttons) but is only embedded on `/paper-trading` today. The homepage
still renders a portfolio-snapshot layout without operator status.

Redesigning `frontend/src/app/page.tsx` to (a) mount OpsStatusBar at
the top of the content area and (b) register a `KillSwitchShortcut`
component (Ctrl+Shift+H -> confirm + flatten-all + pause) satisfies
criteria 1 and 2. A production build + Lighthouse run against
localhost:3000 satisfies criteria 3 and 4.

## Scope

Files created / modified:

1. **MODIFY** `frontend/src/app/page.tsx`
   - Keep the existing KPI hero (NAV, P&L, vs SPY, Positions, +
     Sharpe, + Open orders -> 6-tile row per frontend-layout §4).
   - Insert `<OpsStatusBar />` above the KPI hero.
   - Insert `<KillSwitchShortcut />` listener (no-visual component).
   - Follow the two-zone page shell per frontend-layout §1 (fixed
     header zone + scrollable content zone).
2. **NEW** `frontend/src/components/KillSwitchShortcut.tsx`
   - Client component; registers a window keydown listener.
   - Ctrl+Shift+H (Windows/Linux) / Cmd+Shift+H (macOS) -> show a
     native `window.confirm("Emergency halt: flatten all positions
     and pause paper trading?")`.
   - On confirm: call `postPaperFlattenAll()` then
     `postPaperKillSwitch(true, "keyboard-shortcut")`.
   - Surface an accessible toast / aria-live region with the result.
   - No-op outcome if the call fails; never crashes the page.

## Immutable success criteria

1. `ops_status_bar_present`: homepage mounts `<OpsStatusBar />`
2. `kill_switch_shortcut_present`: homepage mounts
   `<KillSwitchShortcut />` and the component registers a keydown
   handler for Ctrl+Shift+H / Cmd+Shift+H.
3. `lighthouse_perf_ge_90`: Lighthouse 13 performance category
   score >= 0.90 on `http://localhost:3000` when served via
   `next start` (production build).
4. `fmp_le_1_5s`: FMP was removed in Lighthouse 13 (Chrome Developers
   notice). Interpreted per research consensus as **LCP <= 1.5s**
   (the documented semantic successor). Evaluator may additionally
   sanity-check FCP <= 1.5s if LCP is flaky.

## Verification (immutable, from masterplan.json)

    cd frontend && npm run lighthouse -- --url http://localhost:3000 \
      --output json --output-path handoff/lighthouse_home.json \
      --only-categories=performance --chrome-flags='--headless=new' && \
    python -c "import json; \
      d=json.load(open('frontend/handoff/lighthouse_home.json')); \
      assert d['categories']['performance']['score'] >= 0.9"

Environmental precondition for verification: the frontend must be
running under `next start` (production build), not `next dev`. Dev
mode cannot hit 0.9 because of source maps + HMR overlay + React
double-render (per Next.js deployment docs).

## References

- Anthropic Harness Design (dual evaluator, immutable verification)
- https://nextjs.org/docs/app/getting-started/deploying
- https://developer.chrome.com/docs/lighthouse/performance/first-meaningful-paint
  (FMP removed; LCP is successor)
- https://web.dev/articles/lcp
- https://unlighthouse.dev/tools/lighthouse-score-calculator
  (LH13 weights: TBT 30%, LCP 25%, CLS 25%, FCP 10%, SI 10%)
- https://www.nngroup.com/articles/dashboards-preattentive/
- frontend/src/components/OpsStatusBar.tsx (canonical ops bar)
- .claude/rules/frontend-layout.md section 4.5 (operator status
  pattern: ONE dense bar, not stacked cards)
