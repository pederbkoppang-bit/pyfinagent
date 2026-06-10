# Live-check placeholder -- phase-25.B

**Step:** 25.B -- Remove cosmetic aliasing patch after 25.A decouples calls (P2 cleanup)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Code review: no is_lite_dup references in main branch post-25.B"

## Pre-deployment evidence
- 6/6 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_B.py`).
- Backend AST clean; frontend TS clean (excluding pre-existing 25.A12 Playwright-not-installed noise).
- grep on `is_lite_dup` across backend/ + frontend/ returns 0 matches.
- grep on `lite_path` returns 0 matches (the field name was also dropped from the frontend Signal interface).
- 1 behavioral round-trip verifies the RiskJudge entry shape post-cleanup is exactly `{agent, role, rationale, weight}` with no `lite_path` key.

## Post-deployment operator workflow
1. Pull main, rebuild frontend:
   ```
   git pull origin main
   cd frontend && npm run build
   ```
2. Verify no compilation warnings about `lite_path` or `is_lite_dup`.
3. Open a recent paper-trade rationale drawer in the UI; the RiskJudge row should render cleanly with weight + rationale, no amber "lite-path" badge.

## Closes audit basis
phase-24.4 F-2 RESOLVED. The cosmetic-patch detection block was dead code after 25.A made the Risk Judge an independent LLM call. Code is now smaller + clearer.

**Audit anchor for next bucket:** 25.C (P2; surface Layer-1 28-skill outputs in drawer) OR 25.B7 (P2; yfinance fallback counter) OR 25.D9.1 (caller-side Files API adoption).
