---
step: phase-10.7.1
title: Alpha Velocity metric + BQ table
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-10.7
---

# Sprint Contract -- phase-10.7.1

## Research-gate summary

`handoff/current/phase-10.7.1-research-brief.md`. tier=moderate, 7 in-full, 18 URLs, recency scan, gate_passed=true.

## Key findings load-bearing for the plan

1. **"Alpha velocity" is NOT a standard term** in 2024-2026 quant/agentic literature. The closest analogs are QuantaAlpha's IC-slope-per-iteration (arXiv 2602.07085) and AgentEvolver's "convergence velocity" (arXiv 2511.10395). We have latitude to define it.

2. **Recommended formula (Candidate B, Sharpe-slope):** `alpha_velocity_score = (sharpe_end - sharpe_start) / window_days` with `n_obs >= 20` guard. Picks Sharpe-slope over IC-slope because (a) feeds from existing `paper_snapshots` NAV without new signal-return alignment, (b) fits in one cycle, (c) DSR gate (Bailey & Lopez de Prado) applies as downstream filter.

3. **BQ table:** `pyfinagent_pms.alpha_velocity_samples`, partitioned by `window_start`, clustered on `[strategy_id, macro_regime]`. 11 columns: strategy_id, window_start, window_end, n_obs, sharpe_start, sharpe_end, alpha_velocity_score, window_days, macro_regime, components_json, computed_at.

4. **Test path is root-level `tests/meta_evolution/`** (matches masterplan 10.7.2-10.7.7 verification commands). Module package at `backend/meta_evolution/` (NEW). Existing `backend/agents/meta_coordinator.py` is DEPRECATED — do NOT extend.

5. **Scope fits one cycle:** 5 new files, ~230 LOC. No BQ auth in tests (FakeBQ stub pattern from `backend/tests/test_paper_trading_v2.py:33-80`).

6. **Closes the open feedback loop** noted in `docs/architecture/trading-mas-evaluation.md` §2 — alpha velocity IS the metric that surfaces "is the system actually learning?"

## Hypothesis

A new `backend/meta_evolution/alpha_velocity.py` module (`AlphaVelocityComputer` class with `compute(strategy_id, window_start, window_end, n_obs, sharpe_start, sharpe_end, macro_regime) -> dict` + `persist(bq_client, sample_dict)`) plus `scripts/migrations/create_alpha_velocity_table.py` and `tests/meta_evolution/test_alpha_velocity.py` with 6 test cases covers the deliverable. Verification: `pytest tests/meta_evolution/test_alpha_velocity.py -v` passes 6/6.

## Success Criteria (verbatim, immutable)

```
python -m pytest tests/meta_evolution/test_alpha_velocity.py -v
```

Per masterplan 10.7.1's `success_criteria` field: not separately enumerated. Implicit: pytest exit 0 with all tests passing.

## Plan steps

1. Create `backend/meta_evolution/__init__.py` (empty package marker)
2. Create `backend/meta_evolution/alpha_velocity.py` — `AlphaVelocityComputer` class + helpers (~80 lines)
3. Create `scripts/migrations/create_alpha_velocity_table.py` — mirrors `create_strategy_deployments_view.py:1-196` pattern (~70 lines, supports `--apply`, `--verify`, `--dry-run`)
4. Create `tests/meta_evolution/__init__.py` (empty package marker)
5. Create `tests/meta_evolution/test_alpha_velocity.py` — 6 test cases per research brief (~120 lines)
6. Run `pytest tests/meta_evolution/test_alpha_velocity.py -v` from repo root
7. Spawn Q/A

## What Q/A must audit

1. All 5 files created; no existing code modified
2. Pytest exit 0 with 6 PASSes (no skips, no fails)
3. Formula matches research brief's Candidate B (Sharpe-slope with n_obs >= 20 guard)
4. BQ schema matches research brief (11 columns, partitioned, clustered)
5. Migration script supports `--dry-run` (does NOT actually create the table this cycle — Q/A or user runs `--apply` separately)
6. No regression: existing pytest suite still 177/178 (1 env skip) on `backend/tests/`
7. Code follows codebase conventions (no emojis, ASCII logging, type hints, docstrings)
