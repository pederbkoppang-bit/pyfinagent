# Sprint Contract — phase-32.3 Surface Sector Exposure to Risk Judge

**Step ID:** `phase-32.3`
**Date:** 2026-05-21
**Cycle type:** Implementation. Prompt-only + orchestrator helper. NO BQ migration.

---

## Research-Gate Summary

- **Tier:** moderate. **gate_passed:** true (per researcher subagent).
- **Brief:** `handoff/current/research_brief.md`.
- **FACT_LEDGER assembly site confirmed at `backend/agents/orchestrator.py:1485-1490`:**
  ```python
  fact_ledger = _build_fact_ledger(report["quant"])
  fact_ledger_json = json.dumps(fact_ledger, indent=2, default=str)
  report["_fact_ledger"] = fact_ledger
  self._fact_ledger_json = fact_ledger_json  # available to all agent methods
  ```
  All ~15 agent prompts (RAG, Market, Competitor, Insider, Options, etc.) receive `fact_ledger=self._fact_ledger_json` as a kwarg. The Risk Judge consumes the FACT_LEDGER via the `{{fact_ledger_section}}` placeholder in `risk_judge.md:76`.
- **QuantAgents R_score formula re-verified:** `R_score = w1·β_p + w2·(1/LR) + w3·max(SE_j) + w4·σ_p`. Sector-exposure term is the MAX over sectors (`max_j SE_j`), not HHI. Risk Alert Meeting trigger threshold = 0.75. The masterplan's 0.60 threshold is MORE conservative (fires earlier), aligning with AQR Q1 2025's concentration paradigm guidance for current Mag-7 era equity portfolios.

---

## Hypothesis

The Risk Judge cannot reason about portfolio concentration risk because the FACT_LEDGER it consumes is PER-TICKER. Add a `portfolio_sector_exposure` block to the FACT_LEDGER carrying:
- `by_sector`: `{sector_name: pct_of_total_value}` dict
- `max_sector`: name of the most-concentrated sector
- `max_sector_exposure_pct`: float (0-100)
- `concentration_warning`: bool (true when max ≥ 60)
- `threshold_pct`: 60.0 (the trigger)
- `total_positions`: int (count of positions)

When the Risk Judge sees `max_sector_exposure_pct = 89.3` and `concentration_warning = true`, it can argue against new Tech BUYs without code-side blocking. The `synthesis_agent.md` output schema gains an optional `portfolio_concentration_warning: string` narrative field so the Risk Judge sees BOTH raw exposure data AND the synth's narrative warning pre-debate.

Live signal: current portfolio is 89.3% Tech (10/11 positions). Once this lands, every new Tech BUY proposal hits the Risk Judge with an explicit concentration warning at prompt time.

---

## Success Criteria (IMMUTABLE — from `.claude/masterplan.json::phase-32.3.verification.success_criteria`)

1. `fact_ledger_carries_portfolio_sector_exposure`
2. `risk_judge_prompt_has_portfolio_context_section`
3. `synthesis_agent_emits_portfolio_concentration_warning`
4. `max_se_geq_0_60_triggers_warning`
5. `max_se_lt_0_60_no_warning`
6. `unit_test_3_cases_pass` (we will ship ≥5)

Verification command (must pass):
```bash
python -m pytest backend/tests/test_phase_32_3_sector_exposure.py -v && \
grep -n 'portfolio_sector_exposure' backend/agents/skills/risk_judge.md backend/agents/orchestrator.py
```

Live check requirement: `handoff/current/live_check_32.3.md` shows (a) the `portfolio_sector_exposure` dict computed against the current 11-position portfolio (Technology 89.3%, Industrials 10.7%, max=Tech, concentration_warning=True), AND (b) a verbatim FACT_LEDGER snippet (rendered as JSON from the orchestrator) showing the new field carried through, AND (c) the Risk Judge prompt template rendered with the new PORTFOLIO CONTEXT section visible.

---

## Immutable Hard Guardrails (verbatim from `implementation_plan.hard_guardrails`)

1. Prompt-only + read-only computation — NO change to `portfolio_manager.py` pre-trade sector caps (those exist separately and are correct).
2. NO change to `decide_trades` flow.
3. Skill-file edits to `risk_judge.md` and `synthesis_agent.md` trigger an InstructionsLoaded reload but DO NOT require Claude Code session restart (Layer-2 skills, not Layer-3 agents).
4. Per CLAUDE.md: separation-of-duties on agent edits applies to `.claude/agents/` ONLY, not `backend/agents/skills/` — no Peder-review-required hold.

Plus global overnight goal guardrails (NO `AskUserQuestion`, NO mutating Alpaca, scope honesty).

---

## Plan Steps

1. **RESEARCH** ✅ done.
2. **PLAN** ✅ this file.
3. **GENERATE** — add `_compute_portfolio_sector_exposure(positions: list[dict], threshold_pct: float = 60.0) -> dict` PURE FUNCTION near `_build_fact_ledger` in `orchestrator.py`. Wire it in at the FACT_LEDGER assembly site (line 1487): fetch positions on demand via a one-shot `BigQueryClient(self.settings)` (matching the existing pattern at line 588), call the helper, merge result into the `fact_ledger` dict under key `portfolio_sector_exposure`. Fail-open (log warning + set to None on any exception so analysis never breaks).
4. **SKILLS** — update `risk_judge.md` with a new "PORTFOLIO CONTEXT (phase-32.3)" section consuming `fact_ledger.portfolio_sector_exposure`. Update `synthesis_agent.md` output schema with optional `portfolio_concentration_warning: string` field.
5. **TESTS** — `backend/tests/test_phase_32_3_sector_exposure.py`. 5 cases per `test_specs`. Pure-function tests + 1 integration test that the helper output reaches the FACT_LEDGER dict.
6. **VERBATIM RESULTS** — write `handoff/current/experiment_results.md`.
7. **EVALUATE** — spawn `qa` ONCE.
8. **LIVE CHECK** — invoke the helper against the LIVE `paper_positions` (11 rows, 89.3% Tech) and quote the resulting dict. Render a simulated Risk Judge prompt section using the new template. Write to `handoff/current/live_check_32.3.md`.
9. **LOG** — append cycle block to `handoff/harness_log.md`.
10. **FLIP** — `phase-32.3.status: pending → done`. Manual commit `phase-32.3:`.

---

## Implementation crib

**Helper signature (pure function, easy to test):**
```python
def _compute_portfolio_sector_exposure(
    positions: list[dict],
    threshold_pct: float = 60.0,
) -> dict:
    """Compute sector concentration from paper_positions rows.

    Returns:
        {
            "by_sector": {sector: pct_of_total_market_value, ...},
            "max_sector": <sector with highest exposure>,
            "max_sector_exposure_pct": <float 0-100>,
            "concentration_warning": <bool, True iff max >= threshold_pct>,
            "threshold_pct": <threshold_pct>,
            "total_positions": <int>,
        }
    """
```

**Caller wiring at `orchestrator.py:1485-1490`:**
```python
fact_ledger = _build_fact_ledger(report["quant"])
# phase-32.3: portfolio-level sector exposure into the per-ticker fact ledger
try:
    from backend.db.bigquery_client import BigQueryClient
    _bq = BigQueryClient(self.settings)
    _positions = _bq.get_paper_positions()
    fact_ledger["portfolio_sector_exposure"] = _compute_portfolio_sector_exposure(
        _positions, threshold_pct=60.0,
    )
except Exception as _pse_exc:
    logger.warning(
        "phase-32.3: portfolio_sector_exposure compute failed (non-fatal): %s",
        _pse_exc,
    )
    fact_ledger["portfolio_sector_exposure"] = None
fact_ledger_json = json.dumps(fact_ledger, indent=2, default=str)
```

**risk_judge.md addition** (after `## Data Inputs` block at lines 22-29):
```
## Portfolio Context (phase-32.3)
- `portfolio_sector_exposure` (read from FACT_LEDGER) — dict with by_sector / max_sector / max_sector_exposure_pct / concentration_warning / threshold_pct.
- When `concentration_warning == true` AND the candidate ticker's sector matches `max_sector`, require compelling sector-specific upside (cite a specific catalyst in the FACT_LEDGER) or reduce `recommended_position_pct` proportionally. Cite the figure in your `reasoning`.
- When `concentration_warning == true` but the candidate's sector differs from `max_sector`, prefer the new sector to improve diversification.
```

**synthesis_agent.md output schema addition** — add to the output JSON:
```json
"portfolio_concentration_warning": "<optional 1-2 sentence narrative if portfolio_sector_exposure.concentration_warning is true; null/omit otherwise>"
```

---

## References

- Researcher brief at `handoff/current/research_brief.md` (this cycle).
- Phase-32.1 + 32.2 commits `24d03224`, `2d973b13` (the helper this cycle complements).
- FACT_LEDGER assembly at `backend/agents/orchestrator.py:254` (`_build_fact_ledger`) + line 1487 (assignment to `self._fact_ledger_json`).
- QuantAgents arXiv 2510.04643 R_score (re-verified this cycle).
- AQR Q1 2025 paradigm + MSCI 2025 wobble (phase-31.0 audit references).
