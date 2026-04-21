# Experiment Results — phase-5 restructure (meta-action)

**Step:** meta-decision — rewrite phase-5 in `.claude/masterplan.json`.
**Date:** 2026-04-19. **Cycle:** 1.

## What changed

Exactly one file edited: `.claude/masterplan.json`.

### Phase ordering (last 5 entries)

```
29  phase-11   done
30  phase-12   done
31  phase-13   blocked
32  phase-14   done
33  phase-5    pending   <- moved from index 11 to 33 (last slot)
```

### Phase-5 record shape

| Field | Old | New |
|---|---|---|
| name | "Multi-Market Expansion" | "Multi-Market Expansion (15-step)" |
| status | pending | pending (unchanged) |
| depends_on | [] | [] (unchanged) |
| gate | null | null (unchanged) |
| steps count | 3 (all placeholders) | **15 concrete sub-steps** |
| archived_legacy_steps | — | 3 old steps preserved for audit |
| path_decision | — | `"cross-cutting-first-then-markets"` with contract + brief pointers |
| open_issues | — | 4 entries (EODHD budget, IBKR infra, market priority, CFTC compliance) |

### 15 new steps

```
5.1  Broker Abstraction Layer                       (depends_on [])
5.2  Data Provider Abstraction (yfinance + EODHD)   (depends_on [5.1])
5.3  Multi-Asset BQ Schema Extension                 (depends_on [5.2])
5.4  Multi-Asset Risk Engine Extension               (depends_on [5.1])
5.5  Crypto Market Integration (Alpaca Crypto)       (depends_on [5.1,5.2,5.3,5.4])
5.6  Options Integration (Alpaca Options Level 3)    (depends_on [5.4,5.5])
5.7  FX Integration (OANDA Practice)                 (depends_on [5.1,5.2,5.3,5.4])
5.8  Futures Integration (IBKR via ib_insync)        (depends_on [5.1,5.2,5.3,5.4])
5.9  International Equities (EODHD + IBKR)           (depends_on [5.2,5.7])
5.10 Expanded ETF Universe                           (depends_on [5.5])
5.11 Cross-Market Regime Detection                   (depends_on [5.5,5.7])
5.12 Cross-Market Signal Generation                  (depends_on [5.11])
5.13 Multi-Asset Backtest Engine Extension           (depends_on [5.4,5.5,5.11])
5.14 Multi-Market Autonomous Loop Integration        (depends_on [5.5,5.7,5.8,5.12,5.13])
5.15 Phase-5 Integration Test + Go/No-Go Gate        (depends_on [5.14])
```

Every step has:
- Non-null `verification.command`
- Non-empty `verification.success_criteria` (3-6 criteria each)
- `harness_required: true`
- `status: pending`
- `contract: null`
- `depends_on` list

## Verbatim diff (from the Python applier)

```
=== final phase ordering (last 5) ===
29  phase-11   status=done
30  phase-12   status=done
31  phase-13   status=blocked
32  phase-14   status=done
33  phase-5    status=pending

=== phase-5 summary ===
name: Multi-Market Expansion (15-step)
status: pending
num steps: 15
archived_legacy_steps count: 3
open_issues: 4
path_decision: cross-cutting-first-then-markets

step ids: ['5.1', '5.2', '5.3', '5.4', '5.5', '5.6', '5.7', '5.8', '5.9', '5.10', '5.11', '5.12', '5.13', '5.14', '5.15']
```

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `phase_5_has_15_steps` | PASS | `len(phase-5.steps) == 15`, ids 5.1–5.15. |
| 2 | `phase_5_moved_to_end` | PASS | Phase-5 is index 33 (last) of `mp["phases"]`. |
| 3 | `each_step_has_verification` | PASS | All 15 have non-null `verification.command` + non-empty `success_criteria`. |
| 4 | `open_issues_recorded` | PASS | 4 open issues: EODHD budget, IBKR infra, market priority, CFTC compliance. |
| 5 | `json_valid` | PASS | `json.load(...)` round-trip without error. |

## Caveats (transparency)

1. **No code changes.** Only masterplan metadata. Any of the 15 new steps will still require its own full harness cycle (research → contract → generate → Q/A → log → flip) when executed.
2. **Open issues flagged, not resolved.** EODHD budget, IBKR infra choice, and market-priority ordering all remain owner decisions. Steps that require these inputs (5.2 / 5.8) cannot start until resolved.
3. **Archived legacy preserved.** The 3 old placeholder steps are retained under `phase-5.archived_legacy_steps` so any audit can reconstruct the prior state.
4. **Phase-5.5 (External Data-Source Audit, `done`) is a distinct phase** from `phase-5.steps[5]` (`5.5` Crypto Market Integration). The naming collision is historical — clear in the masterplan because they live on different `phase` records. Flagged for awareness.
5. **Dependencies form a DAG.** Verified by construction: 5.1 / 5.4 have no cross-step deps; 5.5–5.10 fan out from 5.1–5.4; 5.11–5.15 are the top of the DAG. No cycles.

## Pre-Q/A self-check

- `json.load(open('.claude/masterplan.json'))` round-trip OK.
- Phase-5 is at `mp["phases"][-1]`.
- 15 steps, all with verification.
- 4 open issues recorded.
- Archived legacy preserved.
- `harness_log.md` NOT yet appended (log-last discipline).
