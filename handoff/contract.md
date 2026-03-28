# Phase 2.6.2 Contract — Budget Dashboard (Phase A: Visibility)

## Hypothesis
A simple budget overview page showing known costs and projections gives Peder immediate visibility into burn rate without waiting for BQ billing export.

## Success Criteria
1. New Settings > Budget page (or Budget tab on settings) showing:
   - Known fixed costs (Claude Max $200/mo, Mac Mini amortized)
   - Estimated variable costs (BQ, LLM API usage)
   - Monthly burn rate estimate
   - Runway projection (cash runway at current burn)
2. Backend API: GET /api/budget/summary returning structured cost data
3. Clean UI following all frontend conventions

## Scope
Phase A only (Budget Visibility). Phase B (Cost Autoresearch) deferred — needs more data.
Lightweight implementation using known/estimated data, not full BQ billing integration.

## Started
2026-03-28 23:24 Oslo
