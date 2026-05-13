# Live-check placeholder -- phase-25.P

**Step:** 25.P -- Weekly autoresearch summary Slack notification
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "After next Sunday cron, Slack post with champion vs challenger, promotions, regressions"

## Pre-deployment evidence
- 4/4 verifier PASS including behavioral mock round-trip on
  `run_meta_evolution_cycle` confirming `severity="P3"` +
  `error_type="meta_evolution_weekly_summary"` is dispatched.
- AST clean on both touched modules.
- formatter behavioral round-trip confirms 4-block Block Kit shape.

## Post-deployment operator workflow
1. Pull main + restart backend (or wait for next Sunday tick at 02:00 ET):
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Manually trigger one weekly cycle to test:
   ```
   python -c "from backend.meta_evolution.cron import run_meta_evolution_cycle; print(run_meta_evolution_cycle())"
   ```
3. Expected Slack alert via webhook:
   ```
   [P3] Weekly autoresearch cycle completed
   started_at: 2026-05-13T... finished_at: ...
   duration_seconds: ... error_count: 0
   cron_allocations: {...} provider_allocations: {...}
   archetype_count: 7
   ```

## Note on champion/challenger fields
The live-check text in the masterplan mentions "champion vs challenger,
promotions, regressions" -- these fields are NOT in
`run_meta_evolution_cycle`'s results dict; they live in
`backend/autoresearch/friday_promotion.py`. Surfacing them would
require either widening the dict OR a separate friday_promotion-specific
digest. Documented as 25.P.1 follow-up.

## Closes audit basis
bucket 24.5 F-5(g) RESOLVED.

**Audit anchor for next bucket:** 25.B10.1 (lesser-secret SecretStr),
25.A10.1 (.mcp.json env var fix), 25.P.1 (champion/challenger surfacing),
follow-ups.
