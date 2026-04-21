# Contract -- Cycle 69 / phase-4.7 step 4.7.1

Step: 4.7.1 Remove or merge zero-open pages; <= 8 top-level routes

## Hypothesis

Three routes are consolidation candidates based on the 4.7.0 inventory
and the Explore verdict:

- `/compare` -- REDUNDANT. Functionality already exists as a tab in
  `/reports` (reports/page.tsx lines 328-620). Standalone page
  duplicates the logic.
- `/analyze` -- OVERLAPS /signals (both single-ticker flows). Deep
  Analysis flow is a heavier glass-box pipeline; consolidating keeps
  one ticker-analysis entry point.
- `/portfolio` -- OVERLAPS /paper-trading (both track positions).
  Paper-trading is the primary surface (12 opens vs portfolio 2).

Delete the three directories and add 308 permanent redirects in
`next.config.js` so any bookmark / stale link routes to the target.
Removal and redirect land in the same commit (no 30-day analytics
window needed for internal tool; per research_4.7.1.md Q3).

After consolidation:
  Top-level user routes = /, /agents, /backtest, /paper-trading,
  /performance, /reports, /settings, /signals = 8 (meets criterion).
  /login excluded from top-level count (auth surface).

## Scope

Files created / modified / deleted:

1. **DELETE** `frontend/src/app/compare/` (entire directory)
2. **DELETE** `frontend/src/app/analyze/` (entire directory)
3. **DELETE** `frontend/src/app/portfolio/` (entire directory)
4. **MODIFY** `frontend/next.config.js` -- add `redirects()` with
   three 308 permanent redirects
5. **MODIFY** `frontend/src/components/Sidebar.tsx` -- remove the
   three corresponding nav items
6. **NEW** `scripts/audit/route_count.py` -- enumerate page.tsx under
   `frontend/src/app/`, exclude `/login` and `/api/*`, emit
   `handoff/route_count.json`

## Immutable success criteria

1. `top_level_routes_le_8`: `handoff/route_count.json` has
   `top_level_routes <= 8` (strict <= 8 per masterplan verification).
2. `zero_open_pages_removed_or_justified`: every removed route listed
   with merge target + rationale in the JSON; /login stays,
   justification recorded ("auth surface; opens_30d=1 in 4.7.0 window
   is a floor for access events").

## Verification (immutable, from masterplan.json)

    python scripts/audit/route_count.py && \
    python -c "import json; d=json.load(open('handoff/route_count.json')); \
      assert d['top_level_routes'] <= 8"

## Additional self-imposed check (not criteria-gating but recorded)

- `cd frontend && npm run build` succeeds after edits (type-check +
  compile). This catches import references to removed routes. If
  build fails, we revert and iterate.

## References

- https://nextjs.org/docs/app/guides/redirecting (308 permanent,
  next.config redirects is idiomatic for static path removal)
- https://nextjs.org/docs/app/api-reference/config/next-config-js/redirects
- handoff/frontend_usage.json (4.7.0 inventory)
- frontend/src/components/Sidebar.tsx:21-55 (NAV_SECTIONS)
- frontend/src/app/reports/page.tsx:328-620 (existing compare tab)
