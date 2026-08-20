# Experiment Results -- phase-91.13
Step: report-detail Cost & Token Usage panel's Total Cost box has a stray glow class

## What was built/changed
`frontend/src/components/CostDashboard.tsx:85` -- removed the `glow` prop from the Total Cost
`<BentoCard>`, matching its 3 siblings (Total Tokens, LLM Calls, Deep Think Calls), which already
render as plain `<BentoCard>`.

## File list
- `frontend/src/components/CostDashboard.tsx` (1 line)

## Verbatim verification command output
```
$ grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx
0
```

## Live capture
`handoff/current/captures_91.13/cost_kpi_row_91.13.png` -- Playwright navigation to a real MRVL
report (`http://localhost:3000/reports/MRVL?date=...`), Cost tab, behind the real authenticated
NextAuth session. All 4 stat boxes (Total Cost $8.21, Total Tokens 154.8K, LLM Calls 31, Deep Think
Calls 4) now render with identical `border-navy-700` borders and `bg-navy-800/70` backgrounds
(corrected: these are two distinct tokens, not one -- both are identical across all 4 boxes) --
no glow on any box.
Verified via Next.js Fast Refresh, no restart needed for this frontend-only change.

## Follow-up items queued (out of scope for this step, per the research brief's explicit recommendation not to widen it)
1. `app/performance/page.tsx:178` -- the same `alpha-score-glow` misapplication on the Win Rate card, same 1-of-N-uniform-row pattern.
2. WCAG 2.2 SC 2.2.2 gap: 4 of 5 infinite CSS animations (`shimmer`, `pulse-glow`, `spin-slow`, `gemini-bounce`) have no `prefers-reduced-motion` guard; only the NumberFlow tint is guarded (`globals.css:178-183`).
Both to be filed as new masterplan steps after this round of fixes completes.

## Artifact shape
- Code diff: 1 line, prop removal.
- Live evidence: Playwright screenshot, real authenticated session, real report data.

---

## Follow-up (Q/A cycle 1 returned CONDITIONAL -- fixed, not just disclosed)

Q/A cycle 1 verdict: CONDITIONAL. The evaluator independently re-derived the code-level fix as
correct (6-cell mutation matrix, all killed or covered; pixel measurement of the shipped capture
showing all 4 boxes RGB-identical; an executed positive control proving the pixel oracle
discriminates a real glow render). The sole gap: `handoff/current/live_check_91.13.md` was missing
while masterplan step 91.13 sets `verification.live_check`. Fixed: `live_check_91.13.md` now
exists, referencing the already-captured (and now Q/A-corroborated) screenshot. A fresh Q/A is
being spawned against this updated evidence.
