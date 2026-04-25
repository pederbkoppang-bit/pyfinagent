---
step: phase-16.18
title: Live API smoke (sovereign + paper-trading + auth + OWASP)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.18

## Research-gate summary

`handoff/current/phase-16.18-research-brief.md` (gate_passed=true, 6 in-full, 16 URLs, recency scan present).

Critical findings load-bearing for the plan:
1. **security.md is stale** on `X-XSS-Protection`: docs say `1; mode=block` but code emits `0` (correct per OWASP 2025). I'll assert against the running code, not stale docs.
2. **OWASP headers are NOT on auth-failure responses** -- the 401 path returns early before the middleware adds headers. So the OWASP header check must hit a PUBLIC path, e.g., `/api/health` (which the verification command does correctly).
3. **Sovereign endpoints are public** per `_PUBLIC_PATHS` in `backend/main.py:228`. No auth needed.
4. **`/api/paper-trading/kill-switch`** requires auth — returns `{paused: bool, ...}` shape per `backend/api/paper_trading.py:304-314`. The verification command in masterplan does NOT pass auth. Will need to check actual behavior — either it's in _PUBLIC_PATHS or it returns 401.
5. **Frontend 302 redirect** behavior on unauthenticated routes per `middleware.ts:29-31`. Acceptable per criterion `all_authed_routes_200_or_302`.

## Hypothesis

All sovereign endpoints return ≥1 row (verified live earlier today). Paper-trading status returns 200 with kill_switch.paused=false (or operator must resume — kill_switch_audit last entry was a resume per the Explore report). Frontend routes 302 on unauth. OWASP headers all 5 present on /api/health.

## Success Criteria (verbatim from masterplan)

```
curl -sS http://127.0.0.1:8000/api/sovereign/red-line?window=30d && curl -sS http://127.0.0.1:8000/api/sovereign/leaderboard && curl -sS http://127.0.0.1:8000/api/sovereign/compute-cost?window=30d && curl -sS http://127.0.0.1:8000/api/paper-trading/status && curl -sS http://127.0.0.1:8000/api/paper-trading/portfolio && curl -sS http://127.0.0.1:8000/api/paper-trading/kill-switch && curl -sI http://127.0.0.1:8000/api/health | grep -iE 'x-content-type|x-frame|referrer-policy|cache-control'
```

- sovereign_endpoints_return_data
- paper_trading_status_200
- kill_switch_paused_false
- owasp_headers_present_5_of_5
- all_authed_routes_200_or_302

## Plan steps

1. Hit each sovereign endpoint, count rows
2. Hit each paper-trading endpoint; honest disclosure if 401 on kill-switch
3. OWASP header probe against /api/health
4. Frontend route loop: 8 authed routes, expect 200 or 302
5. Spawn Q/A

## What Q/A must audit

1. Each endpoint independently re-verified
2. OWASP header assertion targets public path (per researcher's #2 finding)
3. X-XSS-Protection value matches actual code (`0`), not stale security.md
4. kill_switch.paused value -- if `true`, document why and how to resume before Monday
5. Stale tree git status (carry-forward from 16.16/16.17)
