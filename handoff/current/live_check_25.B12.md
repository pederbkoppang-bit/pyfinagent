# Live-check placeholder — phase-25.B12

**Step:** 25.B12 — Missing states + tab icons sweep
**Date:** 2026-05-12

## Live-check field
> "Visual check: performance loading state, sovereign error state, paper-trading tab icons all present"

## Pre-deployment evidence
- 9/9 verifier PASS
- TS clean (`npx tsc --noEmit`)
- ESLint 0 errors; canonical icon barrel preserved
- All 3 pages have `phase-25.B12` attribution

## Post-deployment operator workflow
1. Visit `/performance` while data loads → PageSkeleton (animated pulse blocks), not bare text
2. Disable backend or block `/api/sovereign/red-line` → visit `/sovereign` → rose-bordered banner "RedLine data unavailable: ..." with Retry
3. Visit `/paper-trading` → each tab pill shows a Phosphor icon (Wallet/Receipt/Chart/etc.) before the label

**Audit anchor for next bucket:** 25.A11 (wire /paper-trading/learnings backend).
