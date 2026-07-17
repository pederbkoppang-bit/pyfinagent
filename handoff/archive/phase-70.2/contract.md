# Contract — step 70.2 (S2: soft, profit-aware cross-sector diversification)

**Phase:** phase-70 | **Step:** 70.2 | **Priority:** P1 | harness_required: true
**Cycle:** 1 | Date: 2026-07-17 | **Type:** backend + ML (screener/loop/portfolio), flag-gated default-OFF, $0, paper-only, DARK-until-token

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0). Envelope: **gate_passed=true**, tier=moderate,
**6 external sources read in full**, 10 snippet-only, 34 URLs, recency scan performed, 5 internal files audited.
Brief: `handoff/current/research_brief_70.2.md`. Builds on the 70.0 design pack.

Key grounding: arXiv 2601.08717 (soft HHI penalty, `w_d=0` recovers baseline exactly, shades-never-zeroes);
arXiv 2408.09168 (multinomial-blend leader-pick → sector round-robin); Bailey-Lopez de Prado DSR (JPM 2014,
on MONTHLY returns per phase-69.2) + Bailey/Borwein/LdP/Zhu PBO; internal 2026-06-01 replay (hard-neutral
= -0.166 long-only Sharpe → hard neutralization REJECTED). Springer 2026: linear penalties concentrate
pathologically → multiplicative rank-decay preferred.

## Hypothesis

The monosector funnel (S2) is closed by SOFT diversification: (a) a multiplicative rank-decay penalty in
`rank_candidates` — within each sector the j-th candidate by raw composite order `*= (1-w_d)^(j-1)` (leader
untouched, keeps across-sector momentum, `w_d=0` → byte-identical); (b) a min-K-sector round-robin on the
analyze slice so ≥K GICS sectors reach the analyzer; (c) exempt the "Unknown" sector from the count/NAV caps so
enrichment failure can't freeze the funnel. All flag-gated default-OFF; the OOS-P&L check runs on the existing
$0/macro-free ablation harness; activation is operator-token-gated.

## Immutable success criteria (verbatim from masterplan.json 70.2)

1. Under the new flag (default-OFF), the analyzed top-N candidate set for a representative cycle spans >=2
   distinct GICS sectors (proven by a test/fixture or a $0 dry-run log), and the book can consequently hold
   positions in >=2 sectors
2. The change is SOFT (does not hard-neutralize the momentum ranking) and is justified against the 2026-06-01
   hard-neutral-hurts-returns replay; a paper/backtest check shows it does not lower risk-adjusted OOS P&L
   before activation
3. A ticker-meta / sector enrichment failure no longer buckets everything into 'Unknown' and freezes the book
   at the count cap -- unresolved-sector candidates are handled without collapsing distinct real sectors together
4. Flag OFF -> live loop byte-identical to pre-change (ON-vs-OFF $0 diff in the results); DARK-until-token

Verification command (immutable):
`bash -c 'grep -Eqi "sector" backend/services/autonomous_loop.py && ls backend/tests/ | grep -Eqi "70_2|diversif|sector" && python -c "import ast; ast.parse(open(\'backend/services/autonomous_loop.py\').read())"'`
Live check: `experiment_results_70.2` ON-vs-OFF evidence: OFF analyzed-set monosector (reproduces today), ON
analyzed-set spans >=2 sectors; $0, historical_macro untouched.

## Plan

1. (DONE) research gate → research_brief_70.2.md.
2. (this contract, before generate.)
3. GENERATE (all default-OFF ⇒ byte-identical):
   - `settings.py`: 4 flags modelled on momentum_52wh_tilt — `paper_soft_sector_diversity_enabled` (bool F),
     `paper_soft_sector_diversity_w` (float 0.0, ge0 le1), `paper_min_k_sectors_analyzed` (int 0, ge0 le11),
     `paper_unknown_sector_cap_exempt` (bool F).
   - `screener.py`: `_apply_soft_sector_diversity(scored, w)` (sets composite_score_raw + `(1-w)^(j-1)` shade
     per sector by raw order) + params `soft_sector_diversity=False, soft_sector_diversity_w=0.0`, inserted as
     a NEW block after the sector_neutral/multidim/52wh blocks and BEFORE `scored.sort()` (:483).
   - `autonomous_loop.py`: add the diversity flag to the `build_sector_map` gate (:433) so candidates carry
     sectors; thread the two kwargs into the `rank_candidates` call (:703); replace the analyze slice (:838)
     with `_min_k_sector_slice(new_candidates, N, K)` when K>0 (leader-of-each-of-K-top-sectors, then fill by
     score, then re-sort by composite_score; graceful when <K sectors; K=0 → plain slice).
   - `portfolio_manager.py`: guard the count cap (:359-369) and NAV-pct cap (:394-408) with
     `_unk_ok = paper_unknown_sector_cap_exempt AND sector=='Unknown'` → skip enforcement for Unknown; OFF →
     byte-identical.
   - `scripts/ablation/sector_neutral_replay.py`: add soft-diversity configs (baseline + soft_w grid) so the
     $0/macro-free harness reports OOS Sharpe + sector spread + turnover vs baseline.
   - `backend/tests/test_phase_70_2_soft_diversity.py`: deterministic (network-free) tests — OFF byte-identical;
     soft penalty shading; min-K slice spans ≥2 sectors on a monosector-heavy fixture; Unknown exemption.
4. Run: verification command; import-smoke the changed modules; pytest the new test; run the extended ablation
   replay ($0, yfinance, macro-free) for the ON-vs-OFF OOS-P&L + sector-spread evidence.
5. EVALUATE: fresh Q/A via Workflow structured-output (harness-compliance first, then verification, then the 4
   criteria + do-no-harm + the OFF-byte-identical proof). NOT a UI step → live_check is the ON-vs-OFF evidence
   block in experiment_results, not Playwright.
6. LOG (after PASS); 7. FLIP 70.2 → done.

## Boundaries (binding)

$0 metered, free APIs (yfinance) only, paper-only; every behavior change flag-gated default-OFF (DARK-until-
token) with an ON-vs-OFF $0 diff; NO risk-limit threshold / stop / kill-switch / DSR>=0.95 / PBO<=0.5 gate
moved; historical_macro FROZEN (the ablation replay is macro-free — confirmed); hysteresis BANNED; hard
sector-neutralization REJECTED (soft only). Activation (flipping the flags in prod) is operator-token-gated,
gated on OOS Sharpe >= incumbent + DSR>=0.95 + PBO<=0.5. Harness stays exactly 3 agents.

## References

- `handoff/current/research_brief_70.2.md`, `design_trade_diversity_70.md`, `confirmed_findings.json` (#1/#2/#5/#14)
- Code: screener.py:299-303/411/450-484/703, autonomous_loop.py:433/703/837-838,
  portfolio_manager.py:319/359-369/394-408, settings.py:442-443, scripts/ablation/sector_neutral_replay.py
