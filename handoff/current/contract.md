# Contract — step 70.0 (Research gate + design pack)

**Phase:** phase-70 (Trade diversity + changeable fund)
**Step:** 70.0 — Research gate + design pack (offline, $0, NO production code). Opens the phase.
**Priority:** P1 | harness_required: true
**Cycle:** 1 | Date: 2026-07-16

## Research-gate summary (gate PASSED)

Researcher role run via the Workflow structured-output path (Opus 4.8, $0 Max rail, stall-immune —
this session's Agent-tool roster is fable-snapshotted; the Workflow path is the sanctioned launch per
`feedback_workflow_qa_when_subagents_stall` + phase-71.1). Envelope: **gate_passed=true**, tier=complex,
**7 external sources read in full** (floor 5), 14 snippet-only, 48 URLs collected, recency scan performed,
6 internal files audited. Brief: `handoff/current/research_brief_70.0.md` (229 lines, written write-first).

Key external grounding:
- **Ehsani, Harvey & Li — "Is Sector Neutrality in Factor Investing a Mistake?"** (Financial Analysts
  Journal, May 2023): long-only books are more likely to BENEFIT from keeping sector exposure (78% of
  trials); decision rule: neutralize iff Sharpe(across/within) < their correlation — rarely holds long-only.
  → validates our internal 2026-06-01 replay (hard sector-neutral = -0.166 long-only Sharpe, screener.py:71-73).
- **arXiv 2601.08717v1** (Jan 2026): soft diversification penalty `max[(1-w)ROI - w·Risk - w_d·θ₁·HHI]`,
  tunable `w_d∈[0,1]` (0 = byte-identical). The profit-aware, non-neutralizing shape we adopt.
- **microservices.io Saga + SagaLLM (arXiv 2503.11951, Mar 2025)**: multi-leg transactions guarantee
  "either a fully committed state S′ or a coherent rollback to S"; independent GlobalValidationAgent (no
  self-validation) — aligns with our Q/A doctrine.
- **VeritasChain audit-trail (Dec 2025) + arXiv 2607.02830**: log the full decision lifecycle;
  rejections/skips are first-class REJ events carrying skip_reason + decision_factors + trace correlation —
  the exact anti-pattern our silent BUY-gates exhibit.

## Hypothesis

The three operator symptoms (S1 can't-change-setting, S2 no cross-sector buys, S3 too few trades) can be
closed by a design that: (a) diversifies the analyzed candidate set with a SOFT, profit-aware tilt (NOT hard
sector-neutralization, which the replay proved hurts long-only returns) that is flag-gated default-OFF and
backtest-validated before activation; (b) makes the swap/rotation path atomic (pre-flight aggregate
validation, never a half-swap), cash-bounded, and cross-sector-capable; (c) makes every BUY-gate observable
via a structured skip-reason ledger and reconciles the hidden $1 session budget with the visible $2 cap.
70.0 produces the DESIGN + research basis only — no production code, no live-loop behavior change.

## Immutable success criteria (verbatim from masterplan.json 70.0)

1. research_brief_70.0.md exists with an honest JSON gate envelope (gate_passed, >=5 external sources read
   in full, recency scan performed) covering: soft vs hard sector-diversification for long-only momentum
   books, atomic multi-leg paper-order execution / rollback patterns, and gate/limit observability for
   autonomous trading loops
2. design_trade_diversity_70.md specifies the soft-diversification algorithm (and why it does NOT
   hard-neutralize the ranking, citing the 2026-06-01 replay), the atomic cross-sector swap design, and the
   BUY-gate visibility design -- each with the exact files/flags it will touch
3. The design reaffirms the binding boundaries: flag-gated DARK-until-token, $0 metered, paper-only, no
   change to risk-sector-caps as risk limits, and a paper/backtest gate before any diversification
   activation token

Verification command (immutable):
`bash -c 'test -f handoff/current/research_brief_70.0.md && test -f handoff/current/design_trade_diversity_70.md && grep -q "gate_passed" handoff/current/research_brief_70.0.md && grep -Eqi "sector.?neutral|diversif" handoff/current/design_trade_diversity_70.md && grep -Eqi "atomic|rollback|two-leg|swap" handoff/current/design_trade_diversity_70.md && grep -Eqi "budget|cost cap|gate visibility|observab" handoff/current/design_trade_diversity_70.md'`

## Plan

1. (DONE) Research gate → research_brief_70.0.md + gate envelope.
2. (this contract, written BEFORE generate — mtime-ordered.)
3. GENERATE: write `handoff/current/design_trade_diversity_70.md` — the design pack for (a) soft
   diversification, (b) atomic cross-sector swap, (c) BUY-gate observability, each with exact files/flags +
   the do-no-harm/flag-gating posture + the validation-before-token gate. Map each design block to the
   downstream implementation step (70.1/70.2/70.3/70.4/70.5).
4. Write experiment_results.md (artifact list + verification command output).
5. EVALUATE: fresh Q/A via Workflow structured-output (harness-compliance audit first, then deterministic
   verification.command, then LLM judgment of the 3 criteria + do-no-harm).
6. LOG: append harness_log.md (after PASS). 7. DECIDE: flip 70.0 → done.

## Boundaries (binding)

$0 metered, free APIs only, paper-only; do-no-harm (kill-switch limits / stops / risk-sector-caps /
DSR>=0.95 / PBO<=0.5 byte-untouched — 70.0 is design-only so it touches NONE of these); every live-loop
behavior change (in the DOWNSTREAM steps) ships flag-gated DARK-until-token with an ON-vs-OFF $0 diff;
diversification must not lower risk-adjusted OOS P&L (paper/backtest gate before any activation token);
hysteresis banned; historical_macro frozen; harness stays exactly 3 agents.

## References

- `handoff/current/research_brief_70.0.md` (this step's research gate)
- `handoff/current/confirmed_findings.json` (phase-70 audit register, 17 confirmed)
- `handoff/current/goal_phase70_trade_diversity_DRAFT.md`
- Code sites: autonomous_loop.py:838/:90/:1262-1320, portfolio_manager.py:303-369/:594/:620/:675,
  screener.py rank_candidates/build_sector_map/:71-73, paper_trader.py:182/:308
