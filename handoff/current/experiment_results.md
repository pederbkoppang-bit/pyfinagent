# Experiment results — step 70.0 (Research gate + design pack)

**Phase/step:** phase-70 → 70.0 | **Date:** 2026-07-16 | **Type:** research + design only (offline, $0, NO production code)

## What was produced

1. **`handoff/current/research_brief_70.0.md`** (229 lines) — research-gate output. Sections: 3-variant
   query disclosure; source table (7 read-in-full + 14 snippet-only); mandatory recency scan (last 2 yrs);
   internal code audit (6 files); design recommendations for (a)/(b)/(c); JSON gate envelope.
   Gate envelope: `gate_passed=true`, `external_sources_read_in_full=7`, `snippet_only_sources=14`,
   `urls_collected=48`, `recency_scan_performed=true`, `internal_files_inspected=6`, tier=complex.
   Launched via Workflow structured-output (Opus 4.8, $0 Max rail, stall-immune) because this session's
   Agent-tool roster is fable-snapshotted — the sanctioned path per `feedback_workflow_qa_when_subagents_stall`.
2. **`handoff/current/contract.md`** — step 70.0 contract; verbatim immutable criteria; research summary;
   hypothesis; plan; boundaries. Written BEFORE the design (mtime-proven).
3. **`handoff/current/design_trade_diversity_70.md`** — the design pack (GENERATE deliverable):
   - (a) SOFT profit-aware sector diversification: two-part (soft diversity penalty at rank time
     `(1-w_d)^(j-1)` + min-K-sector round-robin on the analyze slice) + Unknown-bucket exemption; explicit
     rationale for NOT hard-neutralizing (cites the -0.166 2026-06-01 replay + Ehsani-Harvey-Li FAJ 2023);
     files/flags: screener.py rank_candidates, autonomous_loop.py:838, portfolio_manager.py:272/319/360;
     flags `paper_soft_sector_diversity_enabled`/`_w`, `paper_min_k_sectors_analyzed` (default OFF);
     validation via scripts/ablation/sector_neutral_replay.py w_d×K grid before any token.
   - (b) ATOMIC cross-sector swap: pre-flight aggregate validation (Saga/SagaLLM — drop the whole pair,
     never a half-swap), cash-bound + $50 floor, compensating buy-back, cross-sector HHI-reducing rotation;
     files portfolio_manager.py:594/620/675 + autonomous_loop.py:1262-1320; flags
     `paper_atomic_swap_enabled`/`paper_cross_sector_rotation_enabled` (default OFF); depends on
     paper_swap_churn_fix.
   - (c) BUY-gate observability: structured skip-reason ledger (VeritasChain/arXiv 2607.02830) + fix the
     swallowed BudgetBreachError + reconcile the hidden $1 session budget vs visible $2 cap; files
     autonomous_loop.py:90/:925/:966-970; do-no-harm split (logging un-flagged, ceiling change flag-gated).
   - Downstream step map (70.1–70.5) + rejected alternatives (hard neutralization, 2PC, un-gated budget raise).

## Verification command output (verbatim)

```
$ bash -c 'test -f handoff/current/research_brief_70.0.md && test -f handoff/current/design_trade_diversity_70.md && grep -q "gate_passed" handoff/current/research_brief_70.0.md && grep -Eqi "sector.?neutral|diversif" handoff/current/design_trade_diversity_70.md && grep -Eqi "atomic|rollback|two-leg|swap" handoff/current/design_trade_diversity_70.md && grep -Eqi "budget|cost cap|gate visibility|observab" handoff/current/design_trade_diversity_70.md'
VERIFICATION: PASS (exit 0)
```

mtime ordering (research → contract → design; contract BEFORE generate):
```
1784221757  research_brief_70.0.md
1784221919  contract.md
1784221981  design_trade_diversity_70.md
```

## Do-no-harm

70.0 is design + research only. NO production code changed; NO live-loop behavior change; NO risk-limit
threshold moved; `historical_macro` untouched; $0 metered (Workflow on the Opus Max rail). All downstream
behavior changes are specified as flag-gated default-OFF with a validation-before-token gate.

## Scope honesty

This step delivers the DESIGN for 70.1–70.5; it does NOT implement any fix. The soft-diversification
approach is deliberately conservative (soft tilt, not hard neutralization) precisely because the internal
replay showed hard neutralization hurts long-only returns — activation is gated on a backtest that must
beat the incumbent OOS and clear DSR/PBO.
