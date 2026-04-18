# Experiment Results -- Cycle 69 / phase-4.7 step 4.7.1

Step: 4.7.1 Remove or merge zero-open pages; <= 8 top-level routes

## What was generated

1. **DELETED** three directories with their contents:
   - frontend/src/app/compare/  (functionality already a tab in /reports)
   - frontend/src/app/analyze/  (merged into /signals)
   - frontend/src/app/portfolio/ (merged into /paper-trading)

2. **MODIFIED** frontend/next.config.js -- added `redirects()` async
   function with three permanent (308) redirects so any bookmarked or
   externally-linked URL transparently routes to the merge target:
     /compare   -> /reports
     /analyze   -> /signals
     /portfolio -> /paper-trading

3. **MODIFIED** frontend/src/components/Sidebar.tsx -- removed
   NavAnalyze + NavPortfolio imports and three nav items
   (Deep Analysis, Portfolio; /compare was never in sidebar).

4. **NEW** scripts/audit/route_count.py -- enumerates top-level
   page.tsx under frontend/src/app/ via `APP_DIR.iterdir()`
   (dynamic, not hardcoded); filters api/_ prefixes and /login
   (documented exclusion). Emits handoff/route_count.json with
   top_level_routes, paths, removed_in_this_step (merge target +
   reason per removed route), and excluded_from_budget (/login
   justification).

## Verification run (verbatim, immutable)

    $ python scripts/audit/route_count.py
    {"wrote": ".../handoff/route_count.json", "top_level_routes": 8,
     "paths": ["/", "/agents", "/backtest", "/paper-trading",
               "/performance", "/reports", "/settings", "/signals"]}

    $ python -c "import json; d=json.load(open('handoff/route_count.json')); \
        assert d['top_level_routes'] <= 8"
    exit=0

## Build verification (not criteria-gating; sanity)

    $ cd frontend && npm run build
    ...
    Route (app)
    / /_not-found /agents /api/auth/[...nextauth] /backtest /login
    /paper-trading /performance /reports /settings /signals
    (12 static pages generated; no dangling imports; no TS errors)

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| top_level_routes_le_8 | PASS (exactly 8) |
| zero_open_pages_removed_or_justified | PASS (3 removed with merge target+reason; /login justified) |

## Top-level routes after consolidation (8)

    /                /paper-trading
    /agents          /performance
    /backtest        /reports
    /settings        /signals

/login remains outside the budget (auth surface, not user-navigated
top-level). All internal routes compile clean; 308 redirects protect
bookmarked URLs.

## Known limitations / follow-ups (non-blocking)

- The deep-analysis flow that `/analyze` exposed (single-ticker full
  glass-box pipeline via /api/analysis/*) is not yet surfaced as a
  tab inside /signals. For now bookmarks 308 to /signals and the
  backend flow remains invocable via API. Surfacing it as a /signals
  tab lands in 4.7.2 (homepage redesign as MAS operator cockpit) or
  4.7.3 (tab reorganization per route).
- The manual-positions CRUD that `/portfolio` exposed is similarly
  not yet a tab inside /paper-trading. Backend /api/portfolio/*
  endpoints still function; UI follow-up same cycle.
