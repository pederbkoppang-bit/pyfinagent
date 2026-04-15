# Contract — Cycle 13: Phase 4.4.3.4 All monitoring crons operational

## Target
`docs/GO_LIVE_CHECKLIST.md` item 4.4.3.4: watchdog, morning, and evening crons are scheduled and have fired in the last 24 hours.

## Current State
- `scheduler.py` has only morning digest cron (APScheduler, `"cron"` trigger at `morning_digest_hour`)
- No watchdog or evening cron exists
- Settings only has `morning_digest_hour`

## Plan
1. Add `evening_digest_hour` (default 17) and `watchdog_interval_minutes` (default 15) settings
2. Add `_watchdog_health_check` job (interval trigger, 15 min) to scheduler.py
3. Add `_send_evening_digest` job (cron trigger, evening hour) to scheduler.py
4. Add `format_evening_digest` formatter to formatters.py
5. Write drill `scripts/go_live_drills/monitoring_crons_test.py`

## Success Criteria
- SC1: `start_scheduler` registers exactly 3 jobs: morning_digest, evening_digest, watchdog_health_check
- SC2: watchdog uses interval trigger (~15 min), morning/evening use cron triggers
- SC3: settings.py has all three config fields
- SC4: formatters.py has format_evening_digest
- SC5: drill exits 0 verifying SC1-SC4 via code inspection
