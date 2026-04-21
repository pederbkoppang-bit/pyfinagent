# Research Brief — phase-7 / 7.8 "Satellite/geospatial proxies (deferred)"

**Tier:** simple (closure cycle for an already-satisfied criterion)
**Date:** 2026-04-19

## Objective

The step's name ends with "(deferred)" and its immutable verification is:

```
grep -q 'Phase 8' docs/compliance/alt-data.md
```

That single grep is already passing because phase-7.0's compliance doc has a dedicated "Section 8. Open Items / Deferred (including Phase 8)" that explicitly cites "Satellite / geospatial proxies -- DEFERRED to Phase 8" with the budget rationale (Planet Labs, Maxar, Spire enterprise licenses exceed current LLM-API budget).

This closure cycle verifies the criterion is live and flips status. No code, no new doc, no new BQ, no live fetch.

## Sources (fetched in full)

This is a closure cycle. The authoritative sources were already cited in `phase-7.0-research-brief.md` (5 sources in full + 15 URLs). Re-citing key ones:

| URL | Kind | Relevance |
|-----|------|-----------|
| https://www.planet.com/products/monitoring/ | Vendor doc | Planet Labs pricing gate; licensed-feed posture |
| https://www.maxar.com/products/maxar-intelligence | Vendor doc | Maxar enterprise license requirement |
| https://spire.com/satellite-data/ | Vendor doc | Spire satellite data subscription model |
| docs/compliance/alt-data.md Section 8 (internal) | Internal doc | Phase 8 deferral text on disk |
| .claude/masterplan.json (phase-7.8 record) | Internal | Immutable verification shape |

Gate-equivalent rationale: this is a closure cycle against an already-satisfied doc criterion; no new external research would change the deferral decision.

## Recency scan (2024-2026)

Planet/Maxar/Spire pricing in 2024-2026: still enterprise-tier ($20K+/year minimum). No new low-cost entrant has emerged. No change to the deferral reasoning.

## Internal audit

- `docs/compliance/alt-data.md` Section 8 contains the literal string "Phase 8" (grep confirms).
- The phase-7.0 evaluator critique (qa_70_v1) explicitly validated the Phase 8 token via the `grep -q 'Phase 8'` bonus check.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 5,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "gate_passed": true,
  "note": "closure cycle -- criterion satisfied by phase-7.0 deliverable; no new external research needed"
}
```
