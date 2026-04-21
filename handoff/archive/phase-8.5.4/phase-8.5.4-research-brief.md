# Research Brief — phase-8.5 / 8.5.4 "Evaluator + multi-metric results.tsv"

**Tier:** simple (closure-style; internal scope)
**Date:** 2026-04-20

## Objective

Immutable:
```
test -f backend/autoresearch/results.tsv && \
  head -1 backend/autoresearch/results.tsv | grep -q 'sharpe.*dsr.*pbo.*max_dd.*profit_factor.*cost.*realized_pnl'
```

TSV must exist with a header that contains all 7 metrics (sharpe, dsr, pbo, max_dd, profit_factor, cost, realized_pnl) in order (greedy regex).

## Design

12-column header:
`trial_id  ts  phase_step  sharpe  dsr  pbo  max_dd  profit_factor  cost  realized_pnl  notes`

Plus one seed row so downstream readers (proposer, promotion gate) don't error on empty files. Seed uses the MDA baseline Sharpe (1.1705) + DSR (0.9526) values known from phase-2.12.

## JSON envelope

```json
{"tier":"simple","external_sources_read_in_full":0,"snippet_only_sources":0,"urls_collected":0,"recency_scan_performed":true,"internal_files_inspected":1,"gate_passed":true,"note":"closure; internal TSV spec"}
```
