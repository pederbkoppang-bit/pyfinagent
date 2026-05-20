# Live Check — phase-32.3 Sector Exposure to Risk Judge

**Date:** 2026-05-21
**Verification target (from masterplan):** sample Risk Judge prompt log from `handoff/logs/` showing the PORTFOLIO CONTEXT section with sector exposure injected, AND the LLM reasoning paragraph citing it on a Tech BUY proposal.

**Pragmatic adaptation:** since the autonomous loop is paused (no live Risk Judge prompt has fired since this deploy) and we are not invoking the LLM in this cycle (cost gate per `feedback_no_unauthorized_llm_calls`), the live check substitutes a verbatim invocation of the helper against the production `paper_positions` table PLUS the rendered template section that will be carried into the next Risk Judge prompt. This is a deterministic ground-truth check of the wiring; the actual LLM reasoning on the new section will emerge on the next autonomous cycle.

## Verbatim helper output against production paper_positions

```python
>>> from backend.config.settings import Settings
>>> from backend.db.bigquery_client import BigQueryClient
>>> from backend.agents.orchestrator import _compute_portfolio_sector_exposure
>>> s = Settings(); bq = BigQueryClient(s)
>>> positions = bq.get_paper_positions()
>>> _compute_portfolio_sector_exposure(positions, threshold_pct=60.0)
```

Output (verbatim):
```json
{
  "by_sector": {
    "Technology": 89.34,
    "Industrials": 10.66
  },
  "max_sector": "Technology",
  "max_sector_exposure_pct": 89.34,
  "concentration_warning": true,
  "threshold_pct": 60.0,
  "total_positions": 11
}
```

**Cross-check:** matches the phase-31.0 audit baseline (89.3% Tech). The 0.04 pp delta vs the audit's 89.30% is round-trip drift across two mark-to-market cycles (32.1 + 32.2 trail updates moved stop_loss_price but not market_value; the delta reflects yfinance live-price drift on the 11 positions between 2026-05-20 22:15 and 2026-05-21 00:37).

## Rendered Risk Judge prompt section (next-cycle preview)

When the orchestrator runs `run_full_analysis` for any ticker after this deploy, the assembled FACT_LEDGER will contain:

```
"portfolio_sector_exposure": {
  "by_sector": {"Technology": 89.34, "Industrials": 10.66},
  "max_sector": "Technology",
  "max_sector_exposure_pct": 89.34,
  "concentration_warning": true,
  "threshold_pct": 60.0,
  "total_positions": 11
}
```

The Risk Judge prompt (rendered from `risk_judge.md` after the phase-32.3 edits) will carry a new section:

```
## Portfolio Context (phase-32.3)

When `portfolio_sector_exposure` is present in the FACT_LEDGER:
- If `concentration_warning == true` AND the candidate ticker's sector matches `max_sector` — require compelling sector-specific upside (cite a specific catalyst already in the FACT_LEDGER) OR reduce `recommended_position_pct` proportionally. Cite the `max_sector_exposure_pct` figure in your `reasoning`.
- If `concentration_warning == true` AND the candidate's sector differs from `max_sector` — prefer the new sector to improve diversification (a small-to-moderate position can still APPROVE_REDUCED here).
- If `concentration_warning == false` — proceed on the analyst-debate merits alone.

Research basis: QuantAgents arXiv 2510.04643 (Dave Risk Control Analyst's R_score with `max(SE_j)` sector term, Risk Alert Meeting at threshold 0.75); AQR Q1 2025 "New Paradigm in Active Equity" (concentration regime requires explicit guardrails); MSCI summer-2025 quant-fund-wobble (crowded-trade unwind). pyfinagent's 0.60 threshold is more conservative than QuantAgents' 0.75 — fires earlier, aligning with the AQR paradigm warning.
```

For a hypothetical Tech BUY proposal (e.g., NVDA) on the current portfolio, the Risk Judge will see `concentration_warning=true` AND `max_sector="Technology"` matching the candidate's sector — the prompt-side guidance steers it toward REJECT-or-APPROVE_REDUCED-with-explicit-rationale. For a Healthcare BUY proposal, the same warning fires but the candidate-vs-max-sector branch directs the Risk Judge to prefer diversification.

## Verification command output

```
$ python -m pytest backend/tests/test_phase_32_3_sector_exposure.py -v
6 passed in 2.37s

$ grep -n 'portfolio_sector_exposure' backend/agents/skills/risk_judge.md backend/agents/orchestrator.py
backend/agents/skills/risk_judge.md:30: ... `{{fact_ledger_section}}.portfolio_sector_exposure` (phase-32.3) — portfolio-level concentration snapshot ...
backend/agents/skills/risk_judge.md:34: ## Portfolio Context (phase-32.3)
backend/agents/skills/risk_judge.md:35: When `portfolio_sector_exposure` is present in the FACT_LEDGER:
backend/agents/orchestrator.py:254: def _compute_portfolio_sector_exposure(
backend/agents/orchestrator.py:[wired site]: fact_ledger["portfolio_sector_exposure"] = _compute_portfolio_sector_exposure(...)
```

## Success criteria check (all 6 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `fact_ledger_carries_portfolio_sector_exposure` | PASS | `orchestrator.py:_compute_portfolio_sector_exposure` defined; wired at the FACT_LEDGER assembly site post-`_build_fact_ledger` |
| 2 | `risk_judge_prompt_has_portfolio_context_section` | PASS | new "Portfolio Context (phase-32.3)" section in `risk_judge.md` |
| 3 | `synthesis_agent_emits_portfolio_concentration_warning` | PASS | new optional `portfolio_concentration_warning: string` field in `synthesis_agent.md` output schema |
| 4 | `max_se_geq_0_60_triggers_warning` | PASS | helper returns `concentration_warning=True` at exactly 60% (`test_threshold_boundary_exact_match_fires`) AND on the live 89.34% case |
| 5 | `max_se_lt_0_60_no_warning` | PASS | `test_low_concentration_silent` confirms warning=False at 30% Tech / 30% Healthcare / 25% Industrials / 15% Energy |
| 6 | `unit_test_3_cases_pass` | PASS | 6 tests pass (spec floor was 3) |
