# Research — Cycle 13: Phase 4.4.3.4 Monitoring Crons

## Sources
1. **APScheduler 3.x User Guide** (https://apscheduler.readthedocs.io/en/3.x/userguide.html) — AsyncIOScheduler supports both `"cron"` and `"interval"` triggers via `add_job()`. Interval trigger takes `minutes=N` kwarg.
2. **Better Stack APScheduler Guide** (https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/) — Pattern: interval jobs for health checks, cron jobs for daily summaries. `replace_existing=True` prevents duplicate registration.
3. **Slack scheduling best practices** (https://api.slack.com/messaging/scheduling) — Post health alerts only on failure to avoid noise. Evening digests should summarize day's activity.

## Takeaways
- Watchdog: use `"interval"` trigger with `minutes=15`, hit `/api/health`, post to Slack only on failure
- Evening digest: use `"cron"` trigger mirroring morning pattern, summarize day's trading activity
- All three jobs get `replace_existing=True` and explicit `id=` for idempotent registration
