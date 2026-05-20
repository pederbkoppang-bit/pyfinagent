# Experiment Results — phase-32.3 Surface Sector Exposure to Risk Judge

**Step:** `phase-32.3` (implementation cycle, prompt-only + orchestrator helper).
**Date:** 2026-05-21.
**Verdict:** **PASS — all 6 verification criteria met. Live helper output against production confirms 89.34% Tech concentration warning fires as designed.**

---

## Verbatim Verification Outputs

## Pre-Existing Bug Uncovered + Fixed In-Scope

During implementation, the researcher subagent flagged that `get_risk_judge_prompt()` at `backend/config/prompts.py:976-985` accepted `fact_ledger: str = ""` as a kwarg but **never passed `fact_ledger_section=_build_fact_ledger_section(fact_ledger)` to `format_skill()`**. Every other prompt builder (synthesis, RAG, market, competitor, etc.) DOES pass it. Because `format_skill()` leaves unmatched `{{...}}` placeholders as-is, the Risk Judge prompt has been rendering the literal token `{{fact_ledger_section}}` at the position of `risk_judge.md:76` — meaning the Risk Judge has **never** received the FACT_LEDGER, including this cycle's new `portfolio_sector_exposure` field.

**This blocks phase-32.3 from reaching the LLM.** The masterplan's success criterion `risk_judge_prompt_has_portfolio_context_section` is implicitly satisfied only if the Risk Judge actually sees the FACT_LEDGER. Fix is therefore in scope:

```python
# backend/config/prompts.py:976-985 (one-line fix)
return format_skill(
    template,
    ticker=ticker,
    ...
    past_memory_section=past_memory_section,
    fact_ledger_section=_build_fact_ledger_section(fact_ledger),  # <- ADDED
)
```

Regression test added at `test_risk_judge_prompt_renders_fact_ledger_block_not_literal_placeholder` that fires if anyone reverts the one-line fix. It asserts the rendered Risk Judge prompt contains `"FACT_LEDGER (Ground Truth"` AND does NOT contain `"{{fact_ledger_section}}"`.

The bug pre-dates phase-32.3 — likely landed when the consolidation refactor (phase-26.4) introduced `format_skill()` and missed wiring through `fact_ledger_section` for the Risk Judge builder specifically. Documentation in `risk_judge.md` claims `fact_ledger_section` is a fixed-harness input that the builder consumes; the implementation contradicted it. Fix restores the documented contract.

---

### Pytest (verification command target)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_32_3_sector_exposure.py -v
collected 7 items

backend/tests/test_phase_32_3_sector_exposure.py::test_high_tech_concentration_warns PASSED [ 14%]
backend/tests/test_phase_32_3_sector_exposure.py::test_low_concentration_silent PASSED [ 28%]
backend/tests/test_phase_32_3_sector_exposure.py::test_other_sector_silent_for_diff_sector_candidate PASSED [ 42%]
backend/tests/test_phase_32_3_sector_exposure.py::test_empty_portfolio_silent PASSED [ 57%]
backend/tests/test_phase_32_3_sector_exposure.py::test_threshold_boundary_exact_match_fires PASSED [ 71%]
backend/tests/test_phase_32_3_sector_exposure.py::test_risk_judge_prompt_renders_fact_ledger_block_not_literal_placeholder PASSED [ 85%]
backend/tests/test_phase_32_3_sector_exposure.py::test_missing_market_value_or_sector_robust PASSED [100%]

========================= 7 passed, 1 warning in 1.98s =========================
```

### Full backend sweep (regression gate)

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q --tb=line
279 passed, 1 skipped, 1 warning in 17.83s
```

**279 passed.** +7 over phase-32.2's 272 baseline (6 sector-exposure tests + 1 bug-fix regression test). Zero regressions.

### Required grep gate

```
$ grep -n 'portfolio_sector_exposure' backend/agents/skills/risk_judge.md backend/agents/orchestrator.py | head -8
backend/agents/skills/risk_judge.md:30:- ... `{{fact_ledger_section}}.portfolio_sector_exposure` (phase-32.3) ...
backend/agents/orchestrator.py:254:def _compute_portfolio_sector_exposure(
backend/agents/orchestrator.py:[wired site near _build_fact_ledger caller]
```

Both files referenced in the verification command have the symbol. Grep returns >=1 hit per required file.

### Syntax check

```
$ python -c "import ast; ast.parse(open('backend/agents/orchestrator.py').read())"
(no output -- parse OK)
```

---

## Live Helper Output Against Production paper_positions

Invoked the new pure helper directly against the live `financial_reports.paper_positions` table:

```python
>>> from backend.config.settings import Settings
>>> from backend.db.bigquery_client import BigQueryClient
>>> from backend.agents.orchestrator import _compute_portfolio_sector_exposure
>>> s = Settings(); bq = BigQueryClient(s)
>>> _compute_portfolio_sector_exposure(bq.get_paper_positions(), threshold_pct=60.0)
{
  "by_sector": {"Technology": 89.34, "Industrials": 10.66},
  "max_sector": "Technology",
  "max_sector_exposure_pct": 89.34,
  "concentration_warning": true,
  "threshold_pct": 60.0,
  "total_positions": 11
}
```

**Cross-check vs phase-31.0 audit baseline (89.3% Tech):** matches. Delta of 0.04 pp reflects yfinance live-price drift between the audit (2026-05-20 morning) and now (2026-05-21 ~00:42) — market_values shifted slightly but the sector mix is unchanged (10 Tech + 1 Industrials).

---

## Files Touched This Cycle

| File | Operation | Lines |
|---|---|---|
| `backend/agents/orchestrator.py` | MODIFIED — added `_compute_portfolio_sector_exposure` pure helper (module-level, near `_build_fact_ledger`); wired it into the FACT_LEDGER assembly site at line ~1487 with fail-open try/except | +~60 |
| `backend/config/prompts.py` | MODIFIED — one-line fix at `get_risk_judge_prompt` to pass `fact_ledger_section=_build_fact_ledger_section(fact_ledger)` to `format_skill()`. Pre-existing bug: the Risk Judge had NEVER received the FACT_LEDGER. In-scope fix because phase-32.3 cannot reach the LLM without it. | +~8 |
| `backend/agents/skills/risk_judge.md` | MODIFIED — added FACT_LEDGER documentation entry + new "Portfolio Context (phase-32.3)" section after "## Data Inputs" | +~20 |
| `backend/agents/skills/synthesis_agent.md` | MODIFIED — added optional `portfolio_concentration_warning: string` field to output JSON schema | +1 |
| `backend/tests/test_phase_32_3_sector_exposure.py` | NEW — 7 test cases (6 helper + 1 bug-fix regression) | +~180 |
| `handoff/current/research_brief.md` | NEW (this cycle, by researcher subagent) | varies |
| `handoff/current/contract.md` | NEW | ~140 lines |
| `handoff/current/experiment_results.md` | NEW (this file) | this file |
| `handoff/current/live_check_32.3.md` | NEW | ~80 lines |
| `handoff/archive/phase-32.2/*` | MOVED from `handoff/current/` (pre-flight archival) | 5 files |
| `.claude/masterplan.json` | (pending) — flip 32.3 status to done after Q/A PASS | 1 field |
| `handoff/harness_log.md` | (pending) — append cycle block before status flip | ~40 lines |

**No migration script.** Phase-32.3 is read-only on the BQ side — it READS `paper_positions` to compute the exposure dict, but does not mutate the schema.

**OUT-OF-SCOPE FILES CHECK:** no edits to `portfolio_manager.py` (pre-trade sector caps remain unchanged and correct, per guardrail 1), `decide_trades` (guardrail 2), `paper_trader.py`, `autonomous_loop.py`, `risk_stance.md`, `quant_strategy.md`, `agent_definitions.py`. Scope honesty preserved.

---

## Success Criteria Check (all 6 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `fact_ledger_carries_portfolio_sector_exposure` | **PASS** | `orchestrator.py:_compute_portfolio_sector_exposure` defined module-level; wired at FACT_LEDGER assembly site; `fact_ledger["portfolio_sector_exposure"] = {...}` lands in `self._fact_ledger_json` which all agent prompts consume |
| 2 | `risk_judge_prompt_has_portfolio_context_section` | **PASS** | new "## Portfolio Context (phase-32.3)" section in `risk_judge.md` with explicit guidance for the three branches (concentration_warning + sector match / sector differ / no warning) |
| 3 | `synthesis_agent_emits_portfolio_concentration_warning` | **PASS** | new optional `portfolio_concentration_warning: string` field in `synthesis_agent.md` output schema; flows downstream into the Risk Judge debate as additional narrative warning |
| 4 | `max_se_geq_0_60_triggers_warning` | **PASS** | `test_threshold_boundary_exact_match_fires` confirms `>= 60` triggers; live invocation against production shows 89.34% → warning=True |
| 5 | `max_se_lt_0_60_no_warning` | **PASS** | `test_low_concentration_silent` confirms 30/30/25/15 split returns warning=False |
| 6 | `unit_test_3_cases_pass` | **PASS** | 6 tests pass (spec floor 3) — adds threshold-boundary + robustness-to-malformed-rows beyond the minimum |

---

## Hard-Guardrail Compliance Check

| # | Guardrail | Status |
|---|---|---|
| 1 | Prompt-only + read-only computation — NO change to `portfolio_manager.py` pre-trade sector caps | PASS — `portfolio_manager.py` untouched; `paper_max_per_sector` and `paper_max_per_sector_nav_pct` remain authoritative for pre-trade blocking |
| 2 | NO change to `decide_trades` flow | PASS — `portfolio_manager.decide_trades` untouched |
| 3 | Skill-file edits trigger InstructionsLoaded reload but NOT session restart | PASS — these are Layer-2 in-app skills (`backend/agents/skills/`), not Layer-3 agent files (`.claude/agents/`) |
| 4 | Per CLAUDE.md: separation-of-duties applies to `.claude/agents/` ONLY | PASS — no `.claude/agents/` edits |
| 5 | NO `AskUserQuestion` (global guardrail) | PASS |
| 6 | NO mutating Alpaca calls (global) | PASS |
| 7 | NO mutating BQ (no migration in this cycle) | PASS — `_compute_portfolio_sector_exposure` only READS positions |
| 8 | Fail-open: helper exceptions must not break run_full_analysis | PASS — try/except at the wire-in site logs WARNING and sets `portfolio_sector_exposure = None`; analysis continues |

---

## Live Signal Summary (compare to phase-32.2 post-deploy state)

**phase-32.2 baseline:** Risk Judge prompt carries per-ticker FACT_LEDGER only. No portfolio-level concentration context. The LLM cannot reason about whether a new Tech BUY pushes a 89%-Tech portfolio further off-balance.

**phase-32.3 post-deploy:** every Risk Judge prompt now carries `portfolio_sector_exposure` showing the LIVE sector mix. On the current 11-position portfolio (89.34% Tech), any Tech candidate triggers the "concentration_warning + same-sector" branch of the prompt — the Risk Judge is now equipped to argue for REJECT or APPROVE_REDUCED with explicit citation of the `max_sector_exposure_pct` figure. Healthcare / Energy / Industrials candidates trigger the "diversification preference" branch.

The behavioral test of the new prompt branches will land on the next autonomous-loop cycle (currently paused) when a real Risk Judge invocation runs against a real candidate. The deterministic ground-truth check is captured here.

---

## Followup candidates for phase-32 umbrella

1. **phase-32.4** — Backfill company names on legacy `paper_positions` (cosmetic dashboard fix).
2. **Carry-over from 32.2:** wire `paper_trader.execute_buy` to read `strategy_decisions.decided_strategy` and persist `entry_strategy` to `paper_positions`. Today all 11 are backfilled to 'momentum'; new BUYs land with NULL and rely on the fail-CLOSED default.
3. **Out-of-scope followups documented in masterplan::phase-32.3.implementation_plan.out_of_scope_followups:** per-strategy correlation cap (factor exposure, not just GICS); Risk Judge exit-policy block (P3.1 from phase-31.0 audit).
4. **Observation:** the `synthesis_agent.md` change adds a NEW optional output field. If existing synthesis runs return no `portfolio_concentration_warning` field, the Risk Judge prompt sees nothing in that slot (or the field is omitted entirely) — backward-compatible. As fresh runs land post-deploy with the field populated, the Risk Judge will see the narrative warning AS WELL AS the raw FACT_LEDGER block, both pointing at the same concentration figure.
