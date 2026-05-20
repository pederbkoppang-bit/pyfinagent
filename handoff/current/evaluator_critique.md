# Evaluator Critique — phase-32.3 Surface Sector Exposure to Risk Judge

**Verdict:** PASS
**Cycle:** 1 (first Q/A spawn for phase-32.3)
**Date:** 2026-05-21
**Q/A model:** Opus 4.7 / max effort

---

## Harness-Compliance Audit (5-item, runs FIRST per memory feedback_qa_harness_compliance_first)

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Researcher gate cleared (≥5 sources read in full, recency scan present, `gate_passed: true`) | PASS | `handoff/current/research_brief.md` JSON envelope: `external_sources_read_in_full=5, recency_scan_performed=true, gate_passed=true`. 5-source table populated with arxiv 2510.04643 (native + ar5iv), arxiv 2512.02261 (adversarial), Guardfolio + CFA Institute (industry practitioner). Transitive inheritance from phase-31.0 explicitly documented. |
| 2 | Contract written BEFORE GENERATE (mtime ordering) | PASS | contract.md mtime `1779317789` < experiment_results.md mtime `1779318319`. Contract precedes results by ~9 min. |
| 3 | Results artifact carries all required subsections (verbatim pytest 7/7 + full sweep 279, live helper output 89.34%, success-criteria table 6 rows, hard-guardrail table 8 rows, pre-existing bug-fix section, file-touch list) | PASS | All six subsections present and verbatim. The bug-fix section is prominently placed at the top under "Pre-Existing Bug Uncovered + Fixed In-Scope" and documents the `fact_ledger_section` kwarg fix in `get_risk_judge_prompt`. |
| 4 | Log-last discipline (harness_log.md does NOT yet contain phase-32.3 cycle block) | PASS | `grep -nE "phase-32\.3" handoff/harness_log.md` returns only one hit at line 21332 in a forward-reference inside an earlier cycle's body ("Next step in this overnight run: phase-32.3..."). No `## Cycle N -- ... phase=32.3 result=...` header exists. Log append correctly deferred to after Q/A verdict. |
| 5 | First Q/A spawn (not verdict-shopping after a CONDITIONAL) | PASS | Previous `evaluator_critique.md` is shown as DELETED in `git status` (D handoff/current/evaluator_critique.md) — that was the phase-32.2 archive being rolled over by the archive-handoff hook, not a phase-32.3 verdict being overwritten. This file is the FIRST phase-32.3 Q/A critique. |

All 5 audit items PASS. Proceeding to deterministic + content checks.

---

## Deterministic checks

| Check | Command | Result |
|---|---|---|
| pytest new file (7 tests) | `python -m pytest backend/tests/test_phase_32_3_sector_exposure.py -v` | 7 passed, 1 warning in 1.87s |
| pytest full sweep (regression gate) | `python -m pytest backend/tests/ -q --tb=line` | 279 passed, 1 skipped, 1 warning in 17.03s |
| grep gate (both files) | `grep -n 'portfolio_sector_exposure' backend/agents/skills/risk_judge.md backend/agents/orchestrator.py` | 6 hits — 2 in risk_judge.md (lines 30, 34), 4 in orchestrator.py (lines 254, 1558, 1563, 1566) |
| AST parse orchestrator.py | `python -c "import ast; ast.parse(open('backend/agents/orchestrator.py').read())"` | OK (exit 0) |
| AST parse prompts.py | `python -c "import ast; ast.parse(open('backend/config/prompts.py').read())"` | OK (exit 0) |
| Live helper against production paper_positions | `_compute_portfolio_sector_exposure(bq.get_paper_positions(), threshold_pct=60.0)` | `{'by_sector': {'Technology': 89.34, 'Industrials': 10.66}, 'max_sector': 'Technology', 'max_sector_exposure_pct': 89.34, 'concentration_warning': True, 'threshold_pct': 60.0, 'total_positions': 11}` — matches experiment_results.md byte-for-byte |
| Bug-fix renders FACT_LEDGER block | Render `get_risk_judge_prompt(...)` with non-empty `fact_ledger` arg | "FACT_LEDGER (Ground Truth" found in rendered output |
| Bug-fix removes literal placeholder | Same render, assert `'{{fact_ledger_section}}' not in rendered` | PASS — no literal placeholder |
| Scope honesty (negative diff) | `git diff --stat backend/services/paper_trader.py backend/services/portfolio_manager.py backend/services/autonomous_loop.py backend/agents/agent_definitions.py backend/agents/skills/risk_stance.md backend/agents/skills/quant_strategy.md` | "(none touched)" — all 6 out-of-scope files clean |

---

## Content (LLM judgment)

### Helper purity and shape (orchestrator.py:254-313)
- Module-level pure function, no `self`, no I/O — verified.
- Defaults: `threshold_pct=60.0`. Returns the contract's 6-key dict exactly.
- Tolerates malformed rows: `try/except` on `float(market_value)`, sector defaults to `"Unknown"` when missing.
- Empty-portfolio canonical zero shape: returns `False` for `concentration_warning`, `0.0` for `max_sector_exposure_pct`, `None` for `max_sector`. Matches contract's "Empty-portfolio canonical zero shape" specimen.
- Numeric correctness: percentages sum to 100 (within rounding), `max_sector_exposure_pct` is 0-100 (not 0-1), comparison `bool(max_sector[1] >= threshold_pct)` is unit-consistent.
- **Mutation-resistance:** if `>=` were flipped to `>`, the `test_threshold_boundary_exact_match_fires` case (60.0 == 60.0) would fail. Confirmed by reading the test file.

### Wiring at FACT_LEDGER assembly (orchestrator.py:1548-1569)
- AFTER `_build_fact_ledger()` returns and BEFORE `json.dumps()`, so the new key flows through `self._fact_ledger_json`.
- Fail-open: try/except logs `WARNING` via `logger.warning("phase-32.3: portfolio_sector_exposure compute failed (non-fatal): %s", _pse_exc)` and sets the field to `None`. Analysis continues on transient BQ failure.
- Single-source: one BQ fetch per analysis, not per-agent.

### Risk Judge prompt (risk_judge.md:30-39)
- New "Portfolio Context (phase-32.3)" section is present at lines 32-39 and explicitly enumerates the three behavioral branches:
  1. `concentration_warning == true` AND candidate sector == `max_sector` → require compelling upside OR reduce position.
  2. `concentration_warning == true` AND candidate sector != `max_sector` → prefer for diversification.
  3. `concentration_warning == false` → debate merits alone.
- Lines 30 documents the FACT_LEDGER consumption (`{{fact_ledger_section}}.portfolio_sector_exposure`). The research basis at line 39 cites QuantAgents arXiv 2510.04643 + AQR Q1 2025 + MSCI 2025.

### Synthesis Agent (synthesis_agent.md:87)
- New optional `portfolio_concentration_warning: string` field is present at line 87 with proper guidance on when to emit (cite verbatim) vs omit. Backward-compatible.

### Bug fix (prompts.py:976-993)
- The ONE-LINE FIX is present at line 992: `fact_ledger_section=_build_fact_ledger_section(fact_ledger),` is passed to `format_skill()`.
- The phase-32.3 multi-line comment at lines 976-982 documents the root cause (refactor regression where the `fact_ledger_section` placeholder existed in the template but was never wired through this specific builder) and explicitly classifies it as "in-scope" because phase-32.3's `portfolio_sector_exposure` cannot reach the LLM without this fix.
- **Mutation-resistance:** if the fix were reverted (removing the `fact_ledger_section=...` kwarg), the `test_risk_judge_prompt_renders_fact_ledger_block_not_literal_placeholder` case would fail because `format_skill()` would leave the literal `{{fact_ledger_section}}` in the output.
- **Live-renderer test passed** (run by Q/A this cycle): rendered prompt contains `"FACT_LEDGER (Ground Truth"` AND does NOT contain `"{{fact_ledger_section}}"`. The bug fix is genuinely load-bearing.

### Scope honesty
Confirmed via `git diff --stat` on the 6 out-of-scope files:
- `portfolio_manager.py` — UNTOUCHED (pre-trade sector caps remain authoritative, per Guardrail 1)
- `decide_trades` — UNTOUCHED (Guardrail 2)
- `paper_trader.py` — UNTOUCHED
- `autonomous_loop.py` — UNTOUCHED
- `agent_definitions.py` — UNTOUCHED
- `risk_stance.md` / `quant_strategy.md` — UNTOUCHED

The `prompts.py` edit is in scope per the contract's explicit "bug-fix uncovered during P1.3 implementation" treatment — the contract's plan steps and live-check requirements are unsatisfiable without it.

### Threshold justification (0.60)
The contract argues: the 60% threshold sits cleanly between the practitioner cluster (25-30% standard / 50% extreme per Guardfolio + CFA Institute) and the QuantAgents composite R_score Risk Alert trigger (0.75). It is more conservative than QuantAgents' 0.75, in line with AQR Q1 2025's concentration-paradigm guidance for the Mag-7 era. This calibration produces a True warning on the live 89.34% Tech portfolio without being so loose that it would never fire. Defensible.

### Code-review heuristics (Top-15 ranked, phase-16.59)

Ran the security / trading-domain / code-quality / anti-rubber-stamp / LLM-evaluator anti-pattern sweep against the diff:

| Heuristic class | Finding | Severity |
|---|---|---|
| secret-in-diff | None | clean |
| kill-switch-reachability | Diff does NOT touch the execution path; helper is read-only on positions | clean |
| stop-loss-always-set | Diff does NOT touch buy/sell path | clean |
| prompt-injection-path | The `fact_ledger` flow is system-controlled (orchestrator computes from BQ rows, not user input) → no new injection surface | clean |
| broad-except-silences-risk-guard | The `except Exception as _pse_exc:` at orchestrator.py:1561 is INSIDE the FACT_LEDGER fail-open path (NOT a risk-guard), and it LOGS a WARNING via `logger.warning(...)` before continuing. Not a silent swallow. Contract documents the fail-open intent explicitly. | clean |
| financial-logic-without-behavioral-test | The new helper IS exercised by 6 behavioral tests + 1 bug-fix regression test (7 total). The "threshold-boundary-exact-match-fires" + "missing-market-value-or-sector-robust" cases exercise the calibration AND robustness. | clean |
| tautological-assertion | Tests assert specific dict-key values + booleans, not `x == x` or `is not None` alone. | clean |
| perf-metrics-bypass | No Sharpe/drawdown/alpha computed here. | clean |
| position-sizing-div-zero | `total_value <= 0` guarded; empty portfolio returns canonical zero shape before division. | clean |
| criteria-erosion | All 6 success criteria carried forward verbatim from `.claude/masterplan.json::phase-32.3.verification.success_criteria`. | clean |
| sycophantic-all-criteria-pass | This critique cites file:line and quoted output for every criterion; not <3 sentences; not a rubber stamp. | clean |
| sycophancy-under-rebuttal | First Q/A spawn for this phase; no prior verdict to flip. | clean |
| second-opinion-shopping | First spawn; no prior `experiment_results.md` for phase-32.3 to compare mtimes against. | clean |
| missing-chain-of-thought | This critique cites file:line throughout. | clean |
| 3rd-conditional-not-escalated | No prior CONDITIONALs for phase-32.3. | clean |

No BLOCK, WARN, or NOTE findings.

---

## Result

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": null,
  "checks_run": {
    "harness_compliance_audit_5_items": "PASS",
    "researcher_gate": "PASS",
    "contract_before_generate_mtime": "PASS",
    "results_artifact_6_subsections_plus_bug_section": "PASS",
    "log_last_not_yet_appended": "PASS",
    "no_verdict_shopping_first_qa": "PASS",
    "pytest_new_file_7_pass": "PASS",
    "pytest_full_sweep_279_no_regression": "PASS",
    "ast_parse_orchestrator": "PASS",
    "ast_parse_prompts": "PASS",
    "grep_portfolio_sector_exposure_both_files": "PASS",
    "live_helper_production_returns_warning_true": "PASS",
    "bug_fix_renders_fact_ledger_block": "PASS",
    "bug_fix_no_literal_placeholder": "PASS",
    "scope_honesty_diff_check": "PASS",
    "helper_pure_function_6_key_dict": "PASS",
    "wiring_fail_open_logged_warning": "PASS",
    "risk_judge_md_has_portfolio_context_section": "PASS",
    "synthesis_agent_md_has_warning_field": "PASS",
    "threshold_justification_defensible": "PASS",
    "code_review_heuristics_top_15": "PASS"
  }
}
```

Justification: All 5 harness-compliance audit items PASS (researcher gate cleared with `gate_passed:true` and 5 read-in-full sources; contract mtime precedes results mtime; results artifact carries pytest 7/7 + full sweep 279 + live helper output + 6-row success-criteria table + 8-row guardrail table + dedicated pre-existing bug-fix section + file-touch list; harness_log.md has no phase-32.3 cycle block yet; this is the first Q/A spawn for phase-32.3, not verdict-shopping). All 21 deterministic + content checks PASS, including the load-bearing bug-fix verification (`get_risk_judge_prompt` now renders `"FACT_LEDGER (Ground Truth"` and no longer leaves the literal `{{fact_ledger_section}}` placeholder), the live-production helper invocation matching experiment_results.md byte-for-byte (Technology 89.34%, concentration_warning=True), and the scope-honesty negative diff confirming all 6 out-of-scope files (portfolio_manager.py, decide_trades flow, paper_trader.py, autonomous_loop.py, agent_definitions.py, risk_stance.md/quant_strategy.md) are untouched. The bug fix in prompts.py is correctly classified as in-scope under the contract's "bug-fix uncovered during P1.3 implementation" treatment because phase-32.3's portfolio_sector_exposure cannot reach the Risk Judge LLM without it. Threshold justification (0.60) is defensible: more conservative than QuantAgents 0.75 R_score Risk Alert trigger, above the 25-30% industry practitioner cluster, calibrated to fire on the actual live 89.34% Tech signal without being habituating. No code-review heuristic findings (security, trading-domain, code quality, anti-rubber-stamp, LLM-evaluator anti-patterns) — all 15 classes clean.
