# Experiment Results -- Cycle 68 / phase-4.7 step 4.7.0

Step: 4.7.0 Route inventory + 30-day usage telemetry

## What was generated

1. **scripts/harness/frontend_route_inventory.py** (NEW):
   - Walks `frontend/src/app/**/page.tsx` (excludes api/).
   - Runs `git log --since=30.days --name-only -- frontend/src/app`
     to build per-file commit counts over the trailing 30-day window.
   - Maps each file to its route path (root page.tsx -> "/").
   - Honest labeling: `usage_source = "git_activity_30d"` (proxy,
     NOT real page-views). A follow-up step will wire a first-party
     `/api/telemetry/pageview` endpoint so future windows carry
     actual opens.
   - `DOCS_GROUND_TRUTH` appends ground-truth tags to the
     `usage_source` string for known-production surfaces
     (/backtest, /paper-trading, /agents, /); it NEVER modifies the
     numeric `opens_30d`.

2. **handoff/frontend_usage.json** (generated; 12 routes).

## Inventory (12 routes, 30-day commit counts)

    /backtest         47  (git_activity + primary_user_surface)
    /paper-trading    12  (git_activity + primary_user_surface)
    /settings          5
    /                  4  (git_activity + dashboard_landing)
    /agents            4  (git_activity + mas_cockpit)
    /performance       4
    /reports           4
    /signals           4
    /analyze           3
    /compare           3
    /portfolio         2
    /login             1

No route has opens_30d=0 in this window; /login is the lowest at 1.

## Verification run (immutable, verbatim)

    $ python scripts/harness/frontend_route_inventory.py
    {"wrote": ".../handoff/frontend_usage.json",
     "routes": 12, "usage_source": "git_activity_30d"}

    $ test -f handoff/frontend_usage.json && \
      python -c "import json; d=json.load(open('handoff/frontend_usage.json')); \
        assert all('opens_30d' in r for r in d['routes'])"
    exit=0

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| every_route_has_usage_count | PASS (12/12 integer opens_30d) |
| usage_source_named | PASS (top-level + per-route non-empty) |

## Known limitations / follow-ups (non-blocking)

- git_activity is a PROXY, not true page-views. Future 30-day
  windows will carry real data once the pageview beacon lands;
  for today's window the proxy discriminates (/backtest 47 vs
  /login 1) enough to unblock step 4.7.1 "remove or merge zero-
  open pages" decision-making.
- Masterplan verification command uses bare `python`; requires
  `source .venv/bin/activate` first (standard project convention).
  Harness-verifier flagged this as an environment note, not a
  criteria violation.
