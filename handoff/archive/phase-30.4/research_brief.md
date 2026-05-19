# Research Brief — phase-31.0.1 Stage 1 Smoketest (RE-SPAWN)

**Tier:** deep | **Effort:** max | **Date:** 2026-05-20
**Scope:** Verify `screen_universe(tickers=["AAPL","MSFT","NVDA","JPM"])`
returns 4 enriched candidate dicts with `sector` + score populated.

## Objective

Confirm/refute the prior researcher's finding that `screen_universe`
does NOT populate `sector` by default in production, recommend
appropriate test design for Stage 1 smoketest, and back it with 20+
external sources on multi-factor screening, sector-enrichment, and
smoke-test best-practice.

## Search-query composition (three-variant discipline)

| Variant | Topic | Sample query |
|---------|-------|--------------|
| 2026 frontier | multi-factor screening | "multi-factor stock screening momentum 2026" |
| 2025 last-2-yr | composite scoring | "composite factor score backtest 2025" |
| year-less canonical | GICS, smoke-test | "GICS sector classification methodology", "smoke test patterns trading pipeline", "fixture seam testing implicit dependency" |

## Code-audit findings (file:line anchors)

(populated below as I read internal code)

## Pass 1 — Broad coverage (20+ sources read in full)

### Quantitative screening canonical criteria

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

### GICS sector classification

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

### Composite-score factor weighting

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

### Survivorship bias mitigation

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

### End-to-end smoke-test patterns

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

### Function-under-test isolation

| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|

## Pass 2 — Adversarial cross-validation

(populated below as adversarial findings emerge)

## Snippet-only sources (context; not counted toward gate)

| # | URL | Kind | Why not fetched in full |
|---|-----|------|------------------------|

## Recency scan (last 2 years)

(populated below)

## Application to pyfinagent — Stage 1 test design

(populated after research completes)

## JSON envelope

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "adversarial_tags_present": false,
  "gate_passed": false
}
```
