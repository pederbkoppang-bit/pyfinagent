# Sprint Contract — phase-24.6 — Backtest Engine + Walk-Forward + Live-vs-Backtest

**Cycle:** phase-24 cycle 12
**Date:** 2026-05-12
**Step ID:** 24.6
**Priority:** P2

## Research-gate
`gate_passed: true` (tier=moderate). 6 sources: arxiv 2512.12924 walk-forward 2025, reasonabledeviations AFML CPCV notes, FICO champion-challenger, Anthropic built-multi-agent, quantinsti walk-forward, Wikipedia walk-forward.

```json
{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":9,"urls_collected":15,"recency_scan_performed":true,"internal_files_inspected":8,"gate_passed":true}
```

## Hypothesis
Backtest engine correct but outputs don't flow back into live trading. Live-vs-backtest drift not measured. Seed stability undocumented.

**Researcher verdict: PARTIALLY CONFIRMED:**
- Seed stability: `random_state=42` hardcoded (`backtest_engine.py:725,749,886,914`). Endpoint exists at `GET /api/backtest/harness/seed-stability`. `handoff/data/seed_stability_results.json` may be stale/absent — not in git status.
- Live-vs-backtest gap: NO explicit `live_realized_sharpe vs backtest_predicted_sharpe` computation. `paper_go_live_gate.py:91-94` uses NAV divergence as a PROXY (conservative but imprecise). `perf_metrics.py:84-106` computes paper Sharpe independently; never compared to backtest champion.
- MDA flow: backtest → live works (`backtest_engine.py:59-75`, `mda_cache.json`). One-directional channel; no live → backtest feedback for warmstart.
- Optimizer plateau: last 62 experiments all discarded since 2026-04-21; planner should have triggered strategy-switch at exp 11 (Rule 1) but ran to 62.
- Reconciliation gates at 5% NAV divergence + 30% SR gap (`paper_go_live_gate.py:38`); consistent with 20-30% industry decay benchmark.

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. findings_audits_walk_forward_correctness
12. findings_audits_seed_stability
13. findings_audits_live_vs_backtest_reconciliation_drift

**Verifier:** `python3 tests/verify_phase_24_6.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 53 log
5. live_check_24.6.md
6. Flip
