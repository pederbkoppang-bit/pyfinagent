# Phase 2.6.1 Contract — Harness Dashboard

## Hypothesis
Exposing harness cycle data, evaluator critiques, and validation results on the backtest page gives Peder full visibility into the optimization process without reading markdown files.

## Success Criteria
1. New "Harness" tab on backtest page showing cycle history, evaluator scores, validation results
2. Backend API endpoints serving parsed handoff files as JSON
3. Frontend renders all data with proper loading/error/empty states
4. No existing functionality broken

## Fail Conditions
- Backtest page crashes or existing tabs break
- API endpoints return errors for missing files (should return empty gracefully)
- Build fails

## Started
2026-03-28 23:11 Oslo
