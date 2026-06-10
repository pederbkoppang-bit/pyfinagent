# Research Brief — phase-53.3: Data-stack elevation (BigQuery cost/perf + partition/cluster discipline + feature freshness/lineage)

Tier: **complex** | Researcher session | Date: 2026-06-10 | Gate: **IN PROGRESS**

THE TASK: Audit the HOT BQ query paths (signals, prices, fundamentals, macro) in `backend/db/bigquery_client.py`, measure current bytes-scanned via $0 dry-run, identify partition/cluster/SELECT-* gaps, and land a BOUNDED set of correctness-preserving QUERY-LEVEL optimizations (add date/partition WHERE filters, column pruning, LIMIT). Operator-gated: NO DROP / unqualified DELETE / schema repartition. Preserve the 30s fallback-query timeout rule. Results must be byte-identical.

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
| TBD | | | | |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| TBD | | |

## Recency scan (2024-2026)
TBD

## Search-query variants (3 per topic)
TBD

## Key findings (external)
TBD

## Internal code inventory — HOT query paths
| File:line | Role | SELECT *? | Partition/date filter? | Status |
| TBD | | | | |

## Dry-run bytes measurement (current / $0)
TBD

## Table partition/cluster state
TBD

## Freshness / lineage
TBD

## Recommended BOUNDED correctness-preserving query optimizations (land this cycle)
TBD

## Operator-gated schema recommendations (separate)
TBD

## Do-no-harm risks (results must be identical)
TBD

## Research Gate Checklist
- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total
- [ ] Recency scan performed + reported
- [ ] Full pages read (not abstracts)
- [ ] file:line anchors for every internal claim

## JSON envelope
```json
{
  "tier": "complex",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "gate_passed": false
}
```
