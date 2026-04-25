---
step: phase-16.18
cycle_date: 2026-04-25
forward_cycle: true
---

# Experiment Results -- phase-16.18

## What was done

Read-only live API smoke. Verbatim execution of the chained curl probes from the immutable verification command. No code changes.

### Files touched
- `handoff/current/contract.md` (rolling)
- `handoff/current/experiment_results.md` (this)
- `handoff/current/phase-16.18-research-brief.md` (researcher)

## Verification results (verbatim stdout per probe)

### Sovereign endpoints (public)
- `GET /api/sovereign/red-line?window=30d` → series rows: **31**, events: 0
- `GET /api/sovereign/leaderboard` → entries: **2**, source: `strategy_deployments_view`
- `GET /api/sovereign/compute-cost?window=30d` → daily rows: **31** (5 providers per row)

### Paper-trading endpoints
- `GET /api/paper-trading/status` → **HTTP 200**, scheduler_active: **True**
- `GET /api/paper-trading/portfolio` → **HTTP 200**, nav: None, positions: 0 *(disclosure below)*
- `GET /api/paper-trading/kill-switch` → **HTTP 200**, payload:
  ```json
  {"paused":false,"pause_reason":null,
   "sod_nav":9499.5,"peak_nav":9499.5,"current_nav":9499.5,
   "breach":{"daily_loss_breached":false,"daily_loss_pct":0.0,"daily_loss_limit_pct":4.0,
             "trailing_dd_breached":false,"trailing_dd_pct":0.0,"trailing_dd_limit_pct":10.0,
             "any_breached":false},
   "thresholds":{"daily_loss_limit_pct":4.0,"trailing_dd_limit_pct":10.0}}
  ```

### OWASP headers (probed against public `/api/health`)
```
x-content-type-options: nosniff
x-frame-options: DENY
x-xss-protection: 0
referrer-policy: strict-origin-when-cross-origin
cache-control: no-store
```
**5/5 headers present.** Note: `x-xss-protection: 0` is the OWASP-2025-correct value (the deprecated `1; mode=block` is exploitable). The stale documentation in `.claude/rules/security.md:16` should be updated to match the code.

### Frontend route reachability (NextAuth redirect)
```
302 /sovereign
302 /paper-trading
302 /performance
302 /backtest
302 /signals
302 /reports
302 /agents
302 /settings
```
**8/8 return 302** (NextAuth login redirect, expected for unauthenticated curl).

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | sovereign_endpoints_return_data | PASS | 31 + 2 + 31 rows across the three |
| 2 | paper_trading_status_200 | PASS | HTTP 200, scheduler_active=True |
| 3 | kill_switch_paused_false | PASS | `paused: false`, no breaches, sod_nav=peak_nav=current_nav=$9499.50 |
| 4 | owasp_headers_present_5_of_5 | PASS | 5/5 headers, x-xss-protection=0 (OWASP-current) |
| 5 | all_authed_routes_200_or_302 | PASS | 8/8 routes return 302 |

## Honest disclosures

1. **`/api/paper-trading/portfolio` returned `nav: None`**. The kill-switch endpoint shows current_nav=$9499.50, so the system DOES know NAV. The portfolio endpoint's response shape may have NAV under a different key (e.g., `portfolio.nav` vs top-level `nav`), or it returns None when zero positions are open. Not flagging as a regression because: (a) HTTP 200 was returned, (b) the criterion is `paper_trading_status_200`, not "nav populated", (c) kill-switch path proves NAV calc works. Worth a follow-up to confirm portfolio endpoint's expected payload shape.

2. **Stale doc in `.claude/rules/security.md:16`** says `X-XSS-Protection: 1; mode=block` but actual code emits `0`. Code is correct (per OWASP 2025); doc is wrong. Trivial fix recommended but not in this cycle's scope.

3. **OWASP header probe targets public `/api/health`** per researcher finding #2 (auth-failure path returns early before headers are added). This is the correct test target, not a 401-protected endpoint.

4. **All 8 frontend routes 302 (not 200)** because the test browser is unauthenticated. NextAuth middleware redirects to `/login`. The criterion explicitly accepts both. To verify a 200, we'd need persistent auth cookies (out of scope; same pattern as 10.5.7 lighthouse).

5. **scheduler_active: True** confirms APScheduler picked up the daily 14:00 ET cron job. Important for Monday.

6. **kill_switch.paused: false** at start-of-day NAV $9499.50 confirms the kill switch is armed but not engaged. Last operator action was a `resume` per the Explore report. Ready for Monday.

7. **No code changes this cycle.**

## No-regressions

`git diff --stat` shows the same stale tree from prior cycles (page.tsx hero, hook fix, frontend-layout, masterplan additions). Nothing new this cycle.

## Cycle-2 fix (post-Q/A-CONDITIONAL)

Q/A's first verdict on this cycle was **CONDITIONAL** with one blocker:

> `backend/main.py:134` instantiates `AsyncIOScheduler()` with no `timezone=` kwarg. `paper_trading.py:651-658` calls `_scheduler.add_job(..., "cron", hour=settings.paper_trading_hour, ...)` with no `timezone=` either. Live `/api/paper-trading/status` returns `next_run: 2026-04-27T14:00:00+02:00` -- that's CEST (host's Europe/Oslo TZ), not ET. At 14:00 CEST the US market is at 08:00 ET, **90 minutes before market open at 09:30 ET**.

Per the approved plan ("very few touch code (only fixes that surface from verification, and only if blocking Monday)") this qualifies. **Code change applied:**

| Path | Diff |
|------|------|
| `backend/api/paper_trading.py` | +2 / -0 |

Lines added:
- `from zoneinfo import ZoneInfo` (top of imports)
- `timezone=ZoneInfo("America/New_York"),` argument added to `_scheduler.add_job(...)` call inside `_add_scheduler_job(settings)`

**Backend bounced** to pick up the change (kill PID 8301 → launchd respawned as PID 43839; `/api/health` HTTP 200).

**Re-probe:**
```
$ curl -sS http://127.0.0.1:8000/api/paper-trading/status | python3 -c "import json,sys; d=json.load(sys.stdin); print('next_run:', d.get('next_run'))"
next_run: 2026-04-27T14:00:00-04:00
```
**`-04:00` = EDT (US Eastern Daylight Time)**. PASS.

This means Monday 2026-04-27 the cycle fires at 14:00 ET = 19:00 UTC = 21:00 CEST, which is **after US market close (16:00 ET)**. That's actually a different concern: the daily cycle runs AFTER the trading day, not DURING it. Worth confirming with the operator that 14:00 ET is the intended trigger time (between market open 09:30 and close 16:00 — yes, 14:00 is mid-session). Mistake re-read: 14:00 ET is 4 hours before close, mid-session — correct. Fix is good.

## Updated success criteria assessment (post-fix)

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | sovereign_endpoints_return_data | PASS (unchanged) | 31 + 2 + 31 rows |
| 2 | paper_trading_status_200 | PASS (re-probed post-restart) | HTTP 200 |
| 3 | kill_switch_paused_false | PASS (unchanged) | paused=false |
| 4 | owasp_headers_present_5_of_5 | PASS (unchanged) | 5/5 |
| 5 | all_authed_routes_200_or_302 | PASS (unchanged) | 8/8 → 302 |
| **bonus** | scheduler_timezone_explicit_ET | NEW PASS | `next_run: 2026-04-27T14:00:00-04:00` |

## Next

Spawn FRESH Q/A on the changed evidence (per CLAUDE.md cycle-2 flow: "spawning a fresh Q/A AFTER fixing blockers and updating the files IS the documented pattern"). Q/A reads the updated experiment_results.md (this file) + the diff + the live re-probe.
