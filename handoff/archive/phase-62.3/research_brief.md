# Research Brief -- phase-62.4 (goal-away-ops): guardrail/budget sentinel scripts/away_ops/sentinel.sh

Tier: moderate (caller-set). Date: 2026-06-12. Researcher: Layer-3 (merged Explore). STATUS: IN PROGRESS -- filling incrementally.

Step scope: scripts/away_ops/sentinel.sh printing {metered_llm_usd_today, baseline_usd,
kill_switch_paused, flags_match_tokens, ok} JSON; exit 0 healthy; metered-figure source PINNED
in script header; tamper tests (synthetic cost row above baseline -> non-zero exit + named gate;
behavior flag w/o matching token -> non-zero exit + named gate); verify 62.3 wrapper pre-flight
wiring assumption (missing-or-failing sentinel -> digest-only) + document wrapper-test leg.

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| (pending) | | | | |

## Snippet-only (context, not gate)

(pending)

## Search queries (three-variant discipline)

(pending)

## Recency scan (2024-2026)

(pending)

## Key findings

(pending)

## Internal code inventory

### 1. Metered-cost source (THE critical question)

(pending)

### 2. Flag-vs-token reconciliation

(pending)

### 3. Kill-switch state read

(pending)

### 4. Wrapper sentinel contract (run_away_session.sh)

(pending)

## Consensus vs debate

(pending)

## Pitfalls

(pending)

## Application to pyfinagent

(pending)

## Risks & gotchas + GO/NO-GO

(pending)

## Research Gate Checklist

- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total
- [ ] Recency scan performed + reported
- [ ] Full pages read for the read-in-full set
- [ ] file:line anchors for every internal claim

```json
{"tier": "moderate", "external_sources_read_in_full": 0, "snippet_only_sources": 0, "urls_collected": 0, "recency_scan_performed": false, "internal_files_inspected": 0, "gate_passed": false}
```
