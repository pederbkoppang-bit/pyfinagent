# Live-check placeholder — phase-25.6

**Step:** 25.6 — No-stop-on-entry hard block
**Date:** 2026-05-12

## Live-check field
> "BQ paper_positions for any new position post-25.6 has stop_loss_price NOT NULL"

## Pre-deployment evidence
- 8/8 verifier PASS
- None-check ordered BEFORE portfolio fetch (synthesized stop flows into BQ write)
- `if price > 0:` guard prevents zero-stop degenerate
- getattr fallback 8.0 if setting missing

## Post-deployment verification
After next autonomous cycle that triggers a BUY (with the None-check chain):
```
SELECT ticker, stop_loss_price, avg_entry_price, created_at
FROM pyfinagent_pms.paper_positions
WHERE status='OPEN' AND created_at > '<deploy_timestamp>'
ORDER BY created_at DESC LIMIT 10
```
Expect all rows to have stop_loss_price NOT NULL.

**Audit anchor for next bucket:** 25.J (trade confirmation Slack — last P0 in sprint).
