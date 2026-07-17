# live_check — step 64.2 (Functional specs for all 22 routes)

## Timed full-run transcript (immutable command)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
    LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line
...
  [28/28] [functional] › tests/e2e-functional/system.spec.ts:35:5 › system: navigation from /agents to /agent-map
  28 passed (1.2m)
# exit 0 | wall-clock 73s  (criterion 3: << 15-min ceiling)
```

**28 tests / 22 routes / 6 family spec files**, all green on the Mac against the :3100 auth-bypass server.

## Coverage (criterion 1: one spec per family, ≥22 routes)

| Spec file | Routes | Interaction |
|---|---|---|
| `smoke.spec.ts` (home) | `/` | sidebar nav → /signals |
| `system.spec.ts` | /agents, /agent-map, /cron, /observability | nav /agents→/agent-map |
| `analysis.spec.ts` | /signals, /backtest, /learnings, /reports, /performance | fill ticker input |
| `settings.spec.ts` | /settings, /login | sidebar nav → cockpit |
| `sovereign.spec.ts` | /sovereign, /sovereign/strategy/[id] | sidebar nav → /reports |
| `paper-trading.spec.ts` | /paper-trading(→positions), positions, trades, nav, reality-gap, exit-quality, manage, learnings(→/learnings) | positions → Trades tab |

= **22 routes**. Each route asserts (criterion 2): primary data region renders + zero console.error + zero 5xx (+ zero
pageerror). Paper-trading subpages assert their route-distinctive `#panel-<subpage>` (not just the shared layout
heading).

## Do-no-harm (operator :3000 untouched)

```
pre  :3000 /login -> 200
[functional suite: 28 passed, 73s]
post :3000 /login -> 200          # unchanged -- distDir isolation (64.1) holds
git status frontend/next-env.d.ts frontend/tsconfig.json -> (clean; globalTeardown restored)
```

The suite runs ONLY on the isolated :3100 (LIGHTHOUSE_SKIP_AUTH, distDir=.next-functional). `tsc --noEmit` exit 0;
`eslint` exit 0. Pure test-infra — no frontend/backend code changed (all 22 assertion targets are existing selectors;
no new data-testid needed).

## Method disclosure

Playwright 1.60.0 functional project (gated on LIGHTHOUSE_SKIP_AUTH; isolated :3100 skip-auth server;
distDir=.next-functional; globalTeardown restores TS files). Dev-vs-prod trade-off disclosed (runs against `next dev`;
acceptable given 73s runtime + type-filtered warnings). One GENERATE fix: /agents testid was behind a non-default tab
→ switched to the always-rendered `<h2>Multi-Agent System</h2>`.
