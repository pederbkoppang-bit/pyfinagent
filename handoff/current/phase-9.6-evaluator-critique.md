# Q/A Critique — phase-9.6 (nightly outcome rebuild) — REMEDIATION v1

**Verdict: PASS**  **qa_id:** qa_96_remediation_v1  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_96_remediation_v1",
  "violated_criteria": [],
  "checks_run": ["syntax_ast_parse", "pytest_3_of_3", "three_handoff_files_exist", "contract_mtime_le_results_mtime", "verbatim_verification_output_quoted", "no_log_append_yet", "research_gate_envelope_7in_full_17urls", "spot_read_line_27_compute_outcomes", "spot_read_line_30_fail_open_except", "mutation_observation_line_44", "llm_judgment_carry_forward_disclosure", "llm_judgment_schema_deferral", "llm_judgment_url_reality"],
  "reason": "All 5 deterministic gates green. Research gate cleared (7 in full, 17 URLs, three-variant, recency, gate_passed=true). Cross-phase carry-forward to job_runtime.py:112-113 honestly disclosed. Schema deferrals (mae/mfe/return_pct/holding_period/strategy_id) explicitly scoped. Mutation observation on line 44 (pnl>0 vs >=0) recorded as Q/A note, not a blocker."
}
```

## Protocol audit (5/5 PASS)

1. Researcher envelope: 7 in full, 17 URLs, three-variant, recency, gate_passed=true.
2. Contract mtime (18:36:14) < results mtime (18:36:27).
3. Verbatim verification output quoted.
4. No 9.6 log append yet.
5. Cycle v1; not verdict-shopping.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse | exit 0 |
| pytest -q | 3 passed, exit 0 |
| Handoff files | all 3 present |
| Line 27 `_compute_outcomes(trades)` + line 30 fail-open except | matches brief |
| Mutation line 44 `>` → `>=` | test stays green (no pnl==0 fixture) — flagged as non-blocking Q/A observation |

## LLM judgment

Cross-phase carry-forward: `job_runtime.py:112-113` mark-on-heartbeat-exit bug honestly disclosed with file:line anchor + phase-9.1 hardening route. Not swept.

Schema deferrals (mae, mfe, return_pct, holding_period, strategy_id) named with industry-standard justification (TraderSync/Tradewink/TradesViz), explicitly scoped out.

Fail-open alert-path gap correctly routed to phase-9.8 cost-budget channel at hardening.

Gross-vs-net PnL semantics flagged for confirmation before production wiring; today's stub returns `[]` so no misclassification risk.

17 URLs across three search variants, tier 2-4 sources (QuantConnect Lean, OneUptime BQ, TraderSync, Tradewink, TradesViz). No tier-5 padding.

**Non-blocking Q/A observations for follow-up (phase-9.1 hardening):**
1. Add `pnl == 0` fixture to `test_outcomes_win_loss_classification` to mutation-proof the binary boundary
2. Fix `job_runtime.py:112-113` mark-on-success semantics

Cleared for log append + masterplan status confirmation.
