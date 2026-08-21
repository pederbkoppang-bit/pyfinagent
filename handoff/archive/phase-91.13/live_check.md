# Live Check -- phase-91.13
Step: report-detail Cost & Token Usage panel's Total Cost stat box has a stray glow class

## Required evidence (per masterplan step 91.13's `verification.live_check`)
"Playwright screenshot of the panel showing consistent styling across all 4 boxes"

## Evidence

Playwright MCP navigation to a real MRVL report (`http://localhost:3000/reports/MRVL?date=...`)
behind the real, authenticated NextAuth session. Navigated `/reports` -> clicked the MRVL row ->
clicked the Cost tab -> captured the KPI row element directly.

**Screenshot:** `handoff/current/captures_91.13/cost_kpi_row_91.13.png`

All 4 stat boxes (Total Cost $8.21, Total Tokens 154.8K, LLM Calls 31, Deep Think Calls 4) render
with identical `border-navy-700` borders and `bg-navy-800/70` backgrounds (two distinct tokens,
both identical across all 4 boxes) -- no glow, no visual outlier, on any box. This
matches the fix at `frontend/src/components/CostDashboard.tsx:85`, which removed the `glow` prop
from the Total Cost `<BentoCard>` so it renders identically to its 3 siblings.

## Verification command re-run
```
$ grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx
0
```

## Cycle-1 Q/A note
Cycle-1 Q/A (CONDITIONAL) independently re-derived the code-level fix as correct via a 6-cell
mutation matrix (glow-restore, `glow={true}`, `className="alpha-score-glow"`, multi-line prop,
glow-moved-to-sibling -- all killed or explicitly covered by criterion 2) and a pixel-level
measurement of the shipped capture (all 4 card borders/backgrounds RGB-identical, inter-card gaps
zero-variance) plus an executed positive control (a synthetic glow render measurably diverges from
the shipped capture, proving the pixel oracle discriminates). The sole gap was this file's absence,
not the fix -- Q/A's own tool surface lacks `browser_click`, so it could not re-drive the click-only
Cost tab itself and instead corroborated Main's capture via the Playwright server's own artifact
timestamps. This file closes that gap.
