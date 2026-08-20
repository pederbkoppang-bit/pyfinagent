# Sprint Contract -- phase-91.13
Step: report-detail Cost & Token Usage panel's Total Cost box has a stray glow class its siblings don't get

## Research Gate
researcher (tier=simple) gate_passed=true.
Brief: `handoff/current/research_brief_91.13.md`.
- 7 external sources read in full (floor is 5).
- Recency scan performed: 3 new 2025-2026 findings, all sharpening (not contradicting) the canonical guidance.
- Key findings:
  - External consensus (5 of 7 sources): a KPI strip is a uniform component set; emphasis belongs to position/size/contrast/scarce-semantic-color, not per-card decoration. A mature design system (ActiveCampaign, read in full) offers no per-card decorative variant mechanism at all.
  - Glow IS a legitimate pattern -- but only as a **200-400ms transient tied to a data change** (Smashing Magazine 2025). pyfinagent's is `3s infinite` -- an order of magnitude longer and unbounded, i.e. a different, unsupported mechanism.
  - Total Cost is already first in the row -- position already gives it salience (pre-attentive literature: attention decays left-to-right across a repeated row). The glow adds motion, not hierarchy.
  - **WCAG 2.2 SC 2.2.2 applies and has no exception here**: an auto-starting, indefinite (3s infinite), non-essential animation requires a pause/stop/hide mechanism or a `prefers-reduced-motion` guard. pyfinagent has exactly one enforced reduced-motion rule in the whole frontend (NumberFlow tint); `pulse-glow` is unguarded.
  - Project's own `.claude/rules/frontend-layout.md` already decides this: emphasis belongs in typography/color tokens, and §9 requires "consistent micro-interactions... same transitions everywhere."
  - Two adjacent findings are OUT OF SCOPE for this step's immutable criteria (which name `CostDashboard.tsx` only): a second, identical misapplication at `app/performance/page.tsx:178` (Win Rate card, same 1-of-N-uniform pattern), and the broader `prefers-reduced-motion` gap (4 of 5 infinite animations project-wide lack a guard: shimmer, pulse-glow, spin-slow, gemini-bounce). Both queued as new discovered-defect steps after this cycle, not folded in here.
  - Dead-code finding (not fixed here, just recorded so a future cleanup doesn't treat the class as load-bearing): `AlphaScoreCard` (`GlassBoxCards.tsx:25`, the component the `alpha-score-glow` class was named for, with genuine `text-7xl` size hierarchy) has 0 render sites repo-wide.

## Hypothesis
`frontend/src/components/CostDashboard.tsx:85` applies `<BentoCard glow>` to the Total Cost card only, among 4 otherwise-identical sibling KPI cards (`:91`, `:97`, `:103`, all plain `<BentoCard>`). Per the research consensus and the project's own layout rules, removing the `glow` prop restores visual consistency across the row with no loss of real information (position already carries the row's implicit hierarchy).

## Success Criteria (immutable)
```
grep -c 'BentoCard glow' frontend/src/components/CostDashboard.tsx
```
Plus sub-criteria (copied verbatim from `.claude/masterplan.json` phase-91 step 91.13):
- the command above returns 0 after the fix (glow prop removed from the Total Cost card), OR all 4 cards are made visually consistent by deliberate design with a documented reason
- a Playwright screenshot of the Cost & Token Usage panel shows a uniform border/background across all 4 stat boxes

## Plan (PRE-commit; will NOT diverge in Generate)
1. Edit `frontend/src/components/CostDashboard.tsx:85`: remove the `glow` prop from the Total Cost `<BentoCard>`, matching its 3 siblings.
2. Run the immutable grep command, confirm it returns 0.
3. Capture a live Playwright screenshot of a report's Cost & Token Usage panel showing all 4 boxes visually uniform.
4. Do NOT touch `app/performance/page.tsx:178`, `globals.css`'s animation guards, or `GlassBoxCards.tsx` -- all queued separately per the research's explicit recommendation not to widen this step.

## Scope honesty / out-of-scope
- `app/performance/page.tsx:178` (the second glow misapplication) -- queued as a new step.
- The `prefers-reduced-motion` gap on `shimmer`/`pulse-glow`/`spin-slow`/`gemini-bounce` -- queued as a new step (broader than this one card; affects animations independent of this fix).
- `AlphaScoreCard` dead-code removal -- noted, not actioned; folded into the follow-up queue item's notes so a future reader knows the class survives this fix only via dead code.

## References
- Research brief: `handoff/current/research_brief_91.13.md`
- Filed from: `.claude/masterplan.json` phase-91 step 91.13 (originally 86.139, renumbered during the same-day phase-91 split)
