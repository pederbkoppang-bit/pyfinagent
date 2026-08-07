# Contract — Step 72.0.2: standard-tier fail-forward on rail-dead (FLAG-GATED DARK)

- **Step id:** 72.0.2 (P0, Restoration R2)
- **Tier:** moderate — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 177

## Research-gate summary

`handoff/current/research_brief_72.0.2.md` — gate_passed: **true** (7 sources read in full, 33 URLs, recency scan §2.4). Decisive findings the plan is built on:

1. **TWO step premises are wrong** (brief §1.1, §3.1): the cited seam `llm_client.py:1983-2042` is `BatchClient`; `make_client` starts at `:2044`, the CC-rail branch is `:2114-2133` (re-verified live this cycle). And the lite path that actually emits the HOLD **bypasses `make_client` entirely** (`_run_claude_analysis` builds `anthropic.Anthropic` directly at `autonomous_loop.py:2478`) — a provider-order-only fix cannot satisfy criterion 1. **Two seams are required.**
2. **Rail-dead never raises** (§1.2): the chain is silent-empty (`rail_guard_skipped` → regex-fail → fabricated `score=5/confidence=0` HOLD at `:2549` / un-marked twin in the Gemini path `:2795-2800`) → `$0` book.
3. **The Vertex None-trap** (§1.4): every non-orchestrator caller passes `vertex_model=None`; a naive reroute yields `GeminiClient(model=None)` — same $0 with extra steps. The bundle must be built **in-seam from ADC** (`get_genai_client()`, mirroring `orchestrator.py:571+625`).
4. **Fallback-to-alternative-provider is textbook Azure Open-state doctrine, but a fallback without a quality floor is the named anti-pattern**, and schema-validity is NOT quality (arXiv:2604.25359: "nearly perfect JSON, yet a sizeable fraction of the leaf values … are wrong"). Two-stage deterministic $0 floor; **explicitly rejected**: an LLM-judge floor (doubles metered spend on the budget-constrained path).
5. **Per-provider breaker separation is doctrine** (Azure): the fail-forward is a strict READER of `rail_guard_status()`; a Gemini failure must never feed `_rail_guard_record_failure`.
6. **402 `billing_error` is real and NOT auto-retried** (Anthropic docs); on the CC rail it surfaces as non-zero exit/empty envelope, so criterion 2 is satisfied by *not adding* a retry arm (`rail_guard_skipped` empties are already never retried, `orchestrator.py:924-940`).
7. Determinism, not parameter count, is the finance-relevant substitution axis (arXiv:2511.07585) — pin `temperature=0.0, top_k=1` (already the idiom at `llm_client.py:2099`); do NOT reuse the local-LLM rejection memory as evidence against this step (different metric, different models).

## Immutable success criteria — verbatim from `.claude/masterplan.json` 72.0.2 `verification`

- Command: `bash -c 'python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" backend/agents/llm_client.py'`
- C1: "With the flag ON and the cc_rail probe dead, standard-tier analyses run on Vertex-Gemini and produce real (non-degraded) scores; with the flag OFF behavior is byte-identical legacy"
- C2: "The circuit-breaker (claude_rail_breaker_threshold) and probe alerting are unchanged; no retry loop on 402 billing errors (non-retryable per Anthropic docs)"
- C3: "Cost note recorded: Vertex calls are metered -- the flag activation line in the decision sheet must state expected per-cycle cost"
- live_check: `handoff/current/live_check_72.0.2.md`: a live cycle log line showing a standard-tier analysis served by the fail-forward provider while the rail probe was dead, scoring non-degraded.

NOTE (deferral shape, recorded up front): the live_check requires an INDUCED flag-ON cycle → real METERED Vertex calls → operator-gated under the standing `$0 metered` constraint. This step therefore ends this cycle as **built-dark + Q/A-verdicted + flip HELD by the live_check gate**, with a new operator ask (#13) carrying the induced-capture recipe and the C3 cost line — the 61.2 pattern.

## Explicit decisions

- **D1 — Seam A** (`llm_client.py` CC-rail branch, before the `ClaudeCodeClient` return): flag ON + (`rail_skipped` OR `breaker_tripped` from `rail_guard_status()`) + `paper_failforward_model` is `gemini-*` + `get_genai_client()` non-None → return `GeminiClient` over an in-seam ADC `GeminiModelBundle` (`base_config={"temperature": 0.0, "top_k": 1}`). ANY miss → fall through to today's rail path (fail-open). Reads `rail_guard_status()` ONLY — never the mutators.
- **D2 — Seam B** (`_select_lite_analyzer`): gains an optional `settings=None` param (call sites `:1882`/`:1983` pass it); flag ON + standard model `claude-*` + rail dead → returns new `_run_failforward_analysis`; all other inputs → identical returns to today.
- **D3 — `_run_gemini_analysis` gains `model_override: str | None = None`**; the model-resolution + Gemini-only guard is EXTRACTED to a pure helper `_resolve_lite_gemini_model(settings, model_override)` (the hard-raise at `:2756-2762` preserved verbatim in message), called at the function top so misconfig fails before any I/O. Default `None` = today's behaviour.
- **D4 — quality floor** = pure module-level predicate `_failforward_floor_ok(inner_analysis)`: stage-1 structural (dict; `action` in {BUY,SELL,HOLD}; `confidence` numeric 0-100 non-None; `score` numeric 1-10; non-empty `reason`) + stage-2 degenerate-signature rejection (`_parse_failed` marker OR `confidence == 0` — the fabrication tell mirrored from `_degraded_scoring_check:2240-2247`). Floor-fail → the result is marked `_degraded: True` (the honest 61.2 path — `_fold_degraded_for_trading` drops it when the integrity flag is ON; never fabricated).
- **D5 — provenance stamps** on every fail-forward-served result: `_failforward: True`, `_failforward_provider`, `_failforward_reason` (from `rail_guard_status`) — the repo-local `gen_ai.fallback.*` analogue; what makes C1's "real (non-degraded)" auditable.
- **D6 — two settings fields** beside the rail knobs: `paper_rail_failforward_enabled` (False, DARK; description states both states + metered-cost consequence + operator promotion per the `paper_synthesis_integrity_enabled` template) and `paper_failforward_model` (default `GEMINI_WORKHORSE` imported from `model_tiers` — never the literal; 2.5 family EOL 2026-10-16).
- **D7 — breaker isolation:** no call to `_rail_guard_record_failure/_success`, no probe/P1 change; proven by a status-snapshot-unchanged test AND a diff-scope assertion that `claude_code_client.py` + the probe block are untouched.
- **D8 — no new retry arm** (C2): the fail-forward is a single substitution per call.
- **D9 — 2×2 with 61.2** (brief §3.9): tests cover both diagonal cells (failforward × integrity); 61.2's flag is NOT flipped; record that promoting 72.0.2 shrinks the degraded-row population 61.2's 142/170 baseline was measured on (re-derive, don't reuse).
- **D10 — 4000.x interlock:** flag stays OFF during any 4000.x rail-measurement window (free — OFF is byte-identical); stated in the ask.
- **D11 — C3 cost line is DERIVED, not asserted:** standard-tier calls/cycle measured from `llm_call_log` (bounded BQ query) × the published Vertex `gemini-2.5-flash` rate (cited with access date), written into the decision sheet + ask #13.

## Plan

1. Settings fields (D6). 2. Seam A + `_build_vertex_bundle` helper (D1). 3. Seam B + override param + helper extraction + floor + provenance (D2-D5). 4. `backend/tests/test_phase_72_0_2_rail_failforward.py` — behavioural, $0, using the public `rail_guard_disable/reset` seam (66.1 precedent) + stubbed `get_genai_client`. 5. Mutation matrix M1-M7 (brief §3.6) on a scratchpad mirror. 6. Lint gate BEFORE Q/A. 7. C3 derivation + ask #13 + experiment_results → qa-verdict → transcribe → harness_log. Status flip only when the live_check exists (operator) — this cycle records the deferral.

## References

`research_brief_72.0.2.md` (Azure circuit-breaker doctrine; arXiv:2511.07585, 2604.25359, 2512.16959; Anthropic error docs; futureagi quality-floor; Portkey negative evidence). Test seams: `test_phase_78_1_c_block_rail.py` (`_S` idiom), `test_phase_66_1_rail_guard.py` (guard-state precedent).
