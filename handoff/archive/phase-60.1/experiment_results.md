# Experiment Results -- Step 60.1 (GENERATE)

**Step:** 60.1 -- Deep-pipeline restoration + honest-degradation alarm (AW-4). **Date:** 2026-06-11.

## What was built

### 1. Live model smoke + repin (criterion 1)

- NEW `scripts/debug/smoke_vertex_model.py` -- one-shot Vertex availability smoke, two legs per model (plain text + JSON-schema structured output), exit 0 iff all pass. Uses the project's own `get_genai_client()` shim (the production rail).
- LIVE SMOKE RESULTS (us-central1, the project's pinned region, 2026-06-11):
  - `gemini-3.1-flash-lite` -- FAIL 404 NOT_FOUND (Google's named replacement is NOT served in this region/project)
  - `gemini-3-flash-preview` -- FAIL 404 NOT_FOUND
  - `gemini-2.5-flash` -- PASS both legs (text `OK`, structured `{'status': 'ok', 'score': 1}`)
  - `gemini-2.5-flash-lite` -- PASS both legs (rejected per contract gotcha: conflicting 2026-07-22 retirement capture; a 6-week runway repeats AW-4)
  - **Chosen pin: `gemini-2.5-flash`** ($0.30/$2.50 per Mtok). RETIREMENT TRIGGER 2026-10-16 documented at the constant.
- NEW single source of truth `backend/config/model_tiers.py::GEMINI_WORKHORSE` -- the next retirement is a one-line fix.
- Repin sweep (every LIVE occurrence; verified by repo-wide grep -- remaining hits are comments/historical rows only): model_tiers.py (gemini_enrichment, layer1_swappable), orchestrator.py `_GEMINI_FALLBACK`, multi_agent_orchestrator.py MAS fallback x3, backend/autonomous_loop.py:75 + evaluator_agent.py:88 + harness_memory.py:322,503 defaults, meta_evolution/directive_review.py:159 + directive_rewriter.py:202, slack_bot/mcp_tools.py:204, api/agent_map.py:131, news/sentiment.py:81, scripts/harness/run_autonomous_loop.py:74, agents/_inventory.json (33 model fields), settings.py:35 description, api/settings_api.py (`_VALID_MODELS` retired-id REMOVED so the operator cannot select a dead model; AVAILABLE_MODELS row), frontend settings/page.tsx (picker set, fallback default, retired label), backend/.env.example, cost_tracker.py (2.5-flash pricing corrected 0.15/0.60 -> 0.30/2.50 per the 2026-06-11 pricing page; retired row KEPT for historical cost computation), ARCHITECTURE.md + .claude/cron_budget.yaml comments. 6 pre-existing test files updated off the literal retired pin.

### 2. TWO additional live-discovered AW-4 root-cause legs (found by running the immutable live verification, not by code reading)

- **Gemini 2.5 thinks by default.** First live MU run failed: `Market timed out after 90s` x3. gemini-2.5-flash has dynamic thinking ON by default (2.0-flash had none). FIX in `llm_client.py::GeminiClient.generate_content`: when the caller did NOT opt into thinking, explicitly send `ThinkingConfig(thinking_budget=0)` for gemini-2.5 non-pro models (2.5-pro rejects budget=0, min 128, left untouched; the `enable_thinking` opt-in path unaffected).
- **The 90s step budget races the Claude Code CLI rail.** Second live MU run: market passed ~30s, competitor failed 3x90s with `claude_code_invoke: success duration_ms=88862` logged at the moment of the step timeout -- the away week's "90s agent timeouts" leg is the CLI rail's 60-90s round-trip racing the 90s step budget (the rail's own subprocess timeout is 120s, so the step gave up while the call was still in flight). FIX: `_resolve_step_timeout()` in orchestrator.py -- grounded calls at the 90s default get 180s (2.5-flash grounded tail latency), and rails declare `recommended_step_timeout` (ClaudeCodeClient = 150s > its 120s subprocess timeout). Explicit caller budgets only ever raised, never lowered. Inner GeminiClient hang guard raised 120 -> 240s so it sits above every step budget.

### 3. KR-aware tagged skip (criterion 2; design choice: honest skip now, DART deferred -- recorded in contract)

- `orchestrator.py::AnalysisOrchestrator._is_sec_covered()` -- suffix-derived via the canonical `backend.backtest.markets.market_for_symbol`.
- `run_full_analysis` gates the three SEC-bound stages for non-US tickers with persisted `report["skipped_stages"]` tags (land in `full_report_json` -> BQ-auditable): `ingestion_sec`, `quant_cf_sec` (quant dict built by NEW `_quant_from_yfinance()` -- CF-shape-compatible, SEC-only fields explicit Nones with honest source strings), `rag_sec_filings` (skips the LLM call entirely). The 26+ market-agnostic agents run.
- No Cloud Function redeploy (the CF's CIK abort is gated orchestrator-side, before the call).

### 4. Fallback-rate alarm (criterion 3)

- Capture: `_run_single_analysis`'s fallback except now stamps `_fallback_reason` (exception class + message) + `_intended_path: "full"` onto the lite result. Deliberate `lite_mode=True` analyses carry NO tag -- the alarm can never fire on an operator's lite choice.
- NEW pure predicate `_fallback_rate_check(analyses, threshold)` -> (fire, n_fallback, n_total, per-ticker reasons). Strict `>` semantics: 2/4 quiet, 3/5 fires, away-week 100% always fires.
- Wired in the SAME block as the 56.2 degraded-scoring guard (autonomous_loop.py, same `raise_cron_alert` path, distinct `error_type="fallback_rate"`, severity P1, per-ticker reasons in details; `summary["fallback_rate"]` + `summary["fallback_reasons"]` stamped). NOT a parallel bespoke path.
- NEW `settings.fallback_alarm_threshold` (default 0.5, configurable).

### 5. Lite/full provenance operator-visible (criterion 4; digest leg -- UI untouched, no Playwright required)

- `_persist_analysis` stamps `_path` (+ `_fallback_reason` when present) INTO the persisted `full_report_json`; manual-path writers (api/analysis.py + tasks/analysis.py) stamp `_path: "full"` likewise.
- `bigquery_client.get_recent_reports` SELECT adds `JSON_VALUE(full_report_json, '$._path') AS analysis_path` (NULL for pre-tag rows).
- `api/models.py::ReportSummary` carries optional `analysis_path` (the route validates through it -- without this the field is silently dropped).
- Digest `formatters.py` Recent Analyses lines render a backticked `[lite]`/`[full]` marker; pre-tag rows render unchanged. No emojis.

## Files changed (21 code/config + 1 new test + 1 new script)

backend/config/{model_tiers,settings}.py, backend/agents/{orchestrator,llm_client,multi_agent_orchestrator,evaluator_agent,harness_memory,cost_tracker,claude_code_client}.py, backend/agents/_inventory.json, backend/{autonomous_loop}.py, backend/meta_evolution/{directive_review,directive_rewriter}.py, backend/slack_bot/{mcp_tools,formatters}.py, backend/news/sentiment.py, backend/api/{agent_map,settings_api,models,analysis}.py, backend/tasks/analysis.py, backend/db/bigquery_client.py, backend/services/autonomous_loop.py, frontend/src/app/settings/page.tsx, backend/.env.example, ARCHITECTURE.md, .claude/cron_budget.yaml, scripts/harness/run_autonomous_loop.py, scripts/debug/smoke_vertex_model.py (NEW), backend/tests/test_phase_60_1_deep_pipeline.py (NEW, 22 tests) + 6 pre-existing test files repinned.

## Verification command output (verbatim)

```
$ python -m pytest backend/tests -k 'fallback_alarm or model_pin or 60_1' -q
22 passed, 780 deselected, 1 warning in 5.78s        (exit 0; live_check existence satisfied at EVALUATE time)
```

FULL suite: `python -m pytest backend/tests -q` -> `784 passed, 12 skipped, 6 xfailed, 1 warning in 80.21s` (exit 0; was 762 pre-60.1).
Touched files in the separate tests/ tree: `26 passed`. Frontend: `npx tsc --noEmit` exit 0; `npx eslint src/app/settings/page.tsx` 0 errors (2 pre-existing warnings). tests/ tree collection errors (db.tickets_db import path) are PRE-EXISTING and unrelated (canonical suite is backend/tests, per Cycle 48-51 precedent).

## Live verification (see live_check_60.1.md for verbatim evidence)

- Smoke: gemini-2.5-flash exit 0 (text + structured legs).
- Full-orchestrator US analysis (MU, d1fbcc82) **COMPLETED end-to-end on the live stack** -- all 15 steps; BQ row `_path="full"` with all enrichment keys populated (synthesis 46,661 chars; away-week lite rows have NONE of those keys). THREE diagnostic failures en route (market thinking-timeout; competitor CLI-rail race x2), each root-caused, fixed, deployed (launchctl kickstart), re-run -- the live failures ARE the AW-4 evidence: they reproduce the away week's two US failure legs exactly.
- KR analysis (005930.KS, 21e5868b) **COMPLETED end-to-end** -- BQ row `_path="full"` with the three `skipped_stages` tags verbatim, quant from yfinance (Samsung Electronics Co., Ltd.), synthesis 41,508 chars. First KR full-path row in the table.
- DISCLOSED: both rows carry final_score 0.0 from the pre-existing CC-rail "Synthesis-Final returned invalid JSON" fail-open flake (same class visible pre-60.1; full cycles 05-22..05-27 scored normally on this rail). Criterion-1/2 demand end-to-end completion + populated fields + tags -- met; the flake is routed to the cycle_block_summary candidate list, and in autonomous cycles such rows now trip the 56.2 degraded-scoring guard instead of passing silently.
- Live provenance check: `GET /api/reports/?limit=4` -> `005930.KS -> 'full' | MU -> 'full' | SNDK -> None | HPE -> None`.
- **Cycle-2 addition (first Q/A CONDITIONAL fixed):** the settings-page repin IS a UI change -> live Playwright capture taken per the 59.2 rule: `handoff/current/captures_60.1/settings-model-picker-60-1.png` + `settings-snapshot.yml` (skip-auth :3100 workflow; pickers show "Gemini 2.5 Flash $0.3/2.5", no retired entry, saved labels render; :3000 untouched). live_check_60.1.md §F wording corrected.
- Burn disclosure: appended to live_check_58.1.md spend ledger (~$0.5-1.0 metered est., Gemini legs only; CC rail $0 flat-fee).

## Artifact shape

- Alarm artifact: P1 `fallback_rate` cron alert with `per_ticker_reasons` dict (ticker -> exception class + message).
- BQ artifact: `full_report_json.$._path` ("lite"/"full"), `$._fallback_reason` (fallback rows), `$.skipped_stages[]` ({stage, reason}) on KR full-path rows.
- Digest artifact: Recent Analyses line suffix `` `[lite]` `` / `` `[full]` ``.
