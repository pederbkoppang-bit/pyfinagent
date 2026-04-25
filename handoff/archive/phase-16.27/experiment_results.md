---
step: phase-16.27
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverable: docs/architecture/trading-mas-evaluation.md
---

# Experiment Results -- phase-16.27

## What was done

Wrote a 278-line design doc evaluating whether/how to add a trading-decision MAS to pyfinagent. NO code written. Doc grounded in 2024-2026 literature + internal-code audit.

### Files touched

| Path | Action | Size |
|------|--------|------|
| `docs/architecture/trading-mas-evaluation.md` | CREATED | 278 lines |
| `handoff/current/contract.md` | rewrite (rolling) | — |
| `handoff/current/experiment_results.md` | rewrite (this) | — |
| `handoff/current/phase-16.27-research-brief.md` | created (researcher) | 60+ lines |

## Verification (verbatim, immutable)

```
$ test -f docs/architecture/trading-mas-evaluation.md && grep -qE 'Recommendation|Estimated benefit|Plug-in point' docs/architecture/trading-mas-evaluation.md && echo "verification PASS"
verification PASS

$ wc -l docs/architecture/trading-mas-evaluation.md
278
```

## Success criteria assessment

| # | Criterion | Result | Section in doc |
|---|-----------|--------|----------------|
| 1 | design_doc_exists | PASS | file exists |
| 2 | current_state_mapped | PASS | §1 (autonomous_loop.run_daily_cycle full flow + 28-agent Layer-1 + rule-based decide_trades + execution_router) |
| 3 | options_compared | PASS | §3 (Alpha 5-agent vs Beta 3-agent vs Gamma + learner) + §4 (Refined Beta) |
| 4 | plug_in_point_identified | PASS | §7 (`autonomous_loop.py:207-217` between mark-to-market and decide_trades; new file `backend/services/fund_manager.py`) |
| 5 | recommendation_present | PASS | §10 ("Ship Refined Beta as follow-up cycle, NOT today; A/B flag default OFF; DSR not raw SR; 6-week comparison; Gamma deferred 6+ months") |
| 6 | no_code_written | PASS | only `docs/architecture/trading-mas-evaluation.md` created; backend/frontend untouched |

## Key recommendations from the doc (summary for Q/A)

1. **Refined Beta** (vs full Beta): pyfinagent already has 2 of 3 typical Beta agents (Analyst = Layer-1 synthesis; Risk Officer = Risk Judge in `risk_debate.py`). Only **Fund Manager** is new. ~80 lines, 1-2 days.

2. **NOT for Monday paper-trading day-1.** Untested reasoning layer between signal and execution. Risk > reward at launch. Build skeleton when paper-trading is steady-state (≥2 weeks).

3. **Memory wiring (Gamma) is the biggest single lever** (HedgeAgents +39.3% SR ablation), but defer 6+ months — needs production data + Q/A guardrails first.

4. **A/B harness must use DSR, NOT raw SR.** Lopez de Prado overfit floor. 6-week minimum holdout.

5. **Cost-benefit is positive but slim.** ~$3/month LLM cost vs ~$20/month estimated alpha (1bps daily on $10k paper). Research-grade, not production-grade alpha.

6. **HONEST disclosure:** the +24.49% Sharpe lift cited from HedgeAgents is ONE point on ONE benchmark, NOT a generalizable rate.

## Honest disclosures

1. **Doc is recommendation, NOT decision.** Peder owns the build/no-build call.

2. **No code shipped.** Trading MAS does not exist; `TRADING_MAS_ENABLED` is not a setting.

3. **Alpha estimates are speculative.** Best-defensible point is HedgeAgents +24.49% on their benchmark. Could be +0% on pyfinagent's universe.

4. **The doc references commit-state IDs that may drift** (line numbers in `autonomous_loop.py:207-217`). The plug-in point description should be re-verified before any code work.

5. **Q/A escalation-clause check:** this is a PURE research-deliverable cycle, not a missing-function-pattern cycle. Should not trigger the 16.21 escalation clause regardless of verdict.

## No-regressions

`git diff --stat` — only `docs/architecture/trading-mas-evaluation.md` added (new file). Zero backend/frontend code changes.

## Next

Spawn Q/A to audit honesty + completeness of the doc. If PASS → log + flip → 16.28 (reconciliation cycle for 16.15 + 16.2 + 16.3 status).
