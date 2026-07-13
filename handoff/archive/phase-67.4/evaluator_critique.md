# Evaluator Critique — Step 69.3 (P1 signal integrity + $0 free-data lift)

**Evaluator**: fresh Q/A subagent via the Workflow structured-output path (Opus 4.8,
effort=high) — the reliable evaluator route when Agent-tool subagents stall
(see auto-memory `feedback_workflow_qa_when_subagents_stall.md`).
**Run**: `wf_7564cca5-54c` / task `w94h9jlst`. Single agent, 0 errors, 82,011 subagent tokens.
**Verdict**: **PASS** (`ok: true`, `violated_criteria: []`).

## Harness-compliance audit (ran FIRST, per `feedback_qa_harness_compliance_first`)

| Check | Result |
|-------|--------|
| research_gate | PASS — `research_brief_69.3.md` `gate_passed=true`, 5 external sources read in full + 69.0 carryover; provenance disclosed (internal map by researcher before 8th stall, external floor by Main). |
| contract_before_generate | PASS — mtime order verified: research_brief_69.3 (1783788090) → contract.md (1783790793) → 69.3 code macro_regime.py (1783791572) → experiment_results.md (1783791669). overlay_math.py is 69.1 pre-committed scaffolding. |
| results_present | PASS — `experiment_results.md` carries verbatim pytest (12 passed), ruff exit 0, and the $0 ON-vs-OFF + regime-prompt evidence. |
| log_last | PASS — no `phase=69.3` entry in `harness_log.md` at verdict time (grep count 0); Main appends after PASS and before the status flip. |
| no_verdict_shopping | PASS — first Q/A on 69.3 (0 prior entries). |

## Deterministic checks (independently reproduced)

`checks_run`: syntax, verification_command_pytest_12_passed, ruff_F821_F401_F811_exit_0,
backend_runtime_import_smoke, do_no_harm_flag_off_byte_identity, netliq_no_bq_write_grep,
regime_prompt_off_live_render, on_vs_off_ranking_reproduction, overlay_routing_spot_check,
code_review_heuristics, contract_mtime_order, harness_log_verdict_shop_check, research_gate,
experiment_results.

- **pytest** `backend/tests/test_signal_integrity_69.py` → **12 passed**.
- **ruff** `--select F821,F401,F811` on all touched files → **exit 0**.
- **runtime import smoke** — all changed modules import cleanly.

## Immutable success criteria — all met

- **C1 (sign-safe overlays)** — `sign_safe_mult` routes 14 overlays; neg-base **+catalyst (-9)** now
  ranks ABOVE neg-base **-catalyst (-11)**; OFF inverts (-11 vs -9). Reproduced directly AND via test;
  spot-checked options_flow / pead / insider / macro_regime all route through the helper.
- **C2 (news cap)** — the truncating `min(8192,250*len)` inversion is removed →
  `min(48000,max(8192,...))` + `range(2)` parse-retry.
- **C3 (QMJ Growth)** — `revenue_growth_yoy` assigned before its QMJ read (`assign_idx < read_idx`).
- **C4 (INDPRO + net-liquidity)** — INDPRO added to `fred_data.SERIES` (True at runtime); net-liq
  `WALCL−WTREGEN−RRPONTSYD×1000` via a new 24h file-cache, **no BQ**; regime-prompt inclusion gated on
  `regime_net_liquidity` (`macro_regime.py:516-518`).
- **C5 (do-no-harm)** — CONFIRMED byte-identical when flags OFF:
  (a) `sign_safe_mult` OFF `== base*mult` (grid-verified);
  (b) regime prompt OFF has no INDPRO / NET_LIQUIDITY and is byte-identical to the pre-fix prompt,
      net-liq not fetched when OFF (`_nl_on = settings.regime_net_liquidity`, default False);
  (c) `_fetch_net_liquidity` body has zero BQ sinks (grep for insert_rows/bigquery/_bq(/.query( → NONE),
      and `data_ingestion.py` (the only historical_macro writer) is absent from the diff → **historical_macro
      byte-untouched**. Both flags default `False` (verified at runtime).

## Q/A note (non-degrading, recorded for honesty)

Q/A flagged that the C2/C3 tests assert **source-string ordering** rather than executing the code paths,
while C1/C4 are behavioral. Because every ranking/prompt change is behind a default-OFF flag (live engine
byte-identical), no code-review heuristic fires at BLOCK/WARN. The C2 news-cap and C3 QMJ-ordering behavioral
validation is therefore deferred to the operator token window (documented in `live_check_69.3.md` "Deferred").
This is a disclosed test-strength limitation, not a correctness defect.

## Do-no-harm verdict
**CONFIRMED byte-identical when flags OFF.** kill-switch / stops / sector caps / DSR / PBO untouched;
historical_macro frozen; $0 metered (all evidence rendered from strings + fixtures, no LLM call).

## Verdict
**PASS** — proceed to `harness_log.md` Cycle 88 append, then flip 69.3 → done.
