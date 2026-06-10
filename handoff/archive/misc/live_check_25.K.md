# Live-check placeholder — phase-25.K

**Step:** 25.K — Wire kill-switch state changes to Slack
**Date:** 2026-05-12

## Live-check field
> "Manual kill-switch press → Slack alert delivered within 30s"

## Pre-deployment evidence
- 7/7 verifier PASS
- Backward compat: `pause_signals()` (no args) still works
- 2 new async helpers exposed for other callers (paper_trader, autonomous_loop, API endpoints) to wire into

## Post-deployment confirmation (operator test)
1. From Slack bot process: call `pause_signals(app)` (e.g., via `/admin pause` slash command — to be added)
2. Confirm P0 Slack post `[P0] Kill Switch Activated` in alerts channel
3. Confirm iMessage delivery to +47...
4. Verify `handoff/kill_switch_audit.jsonl` shows the pause event

## Deferred (phase-25.K.1)
Backend-side breach detection at `autonomous_loop.py:316` (after `ks_check.get("triggered") is True`) ALSO firing Slack. Requires cross-process pattern: backend writes alert event to BQ, Slack bot polls + dispatches. Not blocking for 25.K which handles the rollback path.

**Audit anchor for next bucket:** 25.A8 (cost-budget HARD-BLOCK, depends on 25.A9 done).
