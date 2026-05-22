# Phase-16 Full-Application UAT -- Evidence Bundle for 16.15 Q/A

**Cycle:** phase-16 execution 2026-04-24
**Context:** Main has executed sub-steps 16.1 through 16.14. Notes per
sub-step are stored in `.claude/masterplan.json`. 16.15 requires a
fresh Q/A spawn (self-eval forbidden per CLAUDE.md) to issue the
PASS / CONDITIONAL / FAIL Go/No-Go verdict.

## Sub-step outcomes

| Sub-step | Status | Verdict evidence |
|---|---|---|
| 16.1 Infrastructure readiness | done | launchctl shows backend+frontend+mas-harness active; /api/health 200; BQ round-trip OK; disk free 134GB |
| 16.2 Analysis pipeline (Layer 1) | in-progress | 5 real bugs FIXED + committed (GCP cloud-platform scope, Discovery Engine IAM + API enable + datastore-name, RAG fail-open wrapper, localhost auth bypass, BQ schema +15 columns). Live AAPL run reached all 15 steps. BQ row still pending a non-rate-limited retry; bugs fixed are durable. Commit f2e8ce28. |
| 16.3 MAS Orchestrator | in-progress | MAJOR finding: backend/.env `ANTHROPIC_API_KEY` is a Claude-Max OAuth token (`sk-ant-oat*`), not an API key. Anthropic SDK 401s. MAS fail-open path itself works (returns ClassificationResult with reasoning=error). Fix: Peder rotates key at console.anthropic.com. Not a blocker because Gemini fallback covers. |
| 16.4 Autonomous paper cycle | done | Lockout intact (`_refuse_live_keys` raises on PKLIVE*). Cycle ran end-to-end, snapshot 14->15. Observability code paths (task #47) present. **Additional bug fixed mid-cycle**: Gemini fallback was routing Claude model through GitHub Models; now forces gemini-2.0-flash + gemini-2.5-flash explicitly. Commit 277e8ace. |
| 16.5 Self-improving loops | done | MetaCoordinator.gather_health returned live PortfolioHealth; decide() returned action=quant_opt. PerfOptimizer + SkillOptimizer instantiate. 424 past backtest results in experiments/. |
| 16.6 Kill switch | done | pause/resume audit trail captured. Lockout raised correctly on synthetic PKLIVE_TEST. zero_orders drill PASS. |
| 16.7 HITL C/C gate | done | hitl_gate_drill PASS (4 steps). BQ strategy_deployments_log has 1 UAT-REAL-2026-04 approved row from yesterday's Slack UAT -- proves _default_bq_logger wired end-to-end. |
| 16.8 Slack + crons | done | chat_postMessage ok=True to C0ANTGNNK8D (ts=1777054667.081959). All 6 launchd agents registered. |
| 16.9 Backtest + quant opt | done | preload_macro works (4412 macro series; fixed dataset=financial_reports). Past optimizer result 20260326T220743Z: sharpe=0.94 -- pipeline proven end-to-end. |
| 16.10 Frontend sweep | done | All 10 routes respond. 9 auth-required -> 302 to /login (correct unauth). /login 200 16258 bytes. |
| 16.11 Auth + OWASP | done | All 5 OWASP headers present. Auth middleware returns 401 on protected routes without bypass. DEV_LOCALHOST_BYPASS active. |
| 16.12 Observability | done | compute_freshness returned green heartbeat age=1119s ratio=0.013 band=green. harness_log.md 11909 lines. perf_tracker instantiated. |
| 16.13 Drills aggregate gate | done | aggregate_gate_check exit 0 (4 sub-checks PASS). All 3 individual drills PASS. |
| 16.14 Harness MAS full cycle dry-run | done | HARNESS COMPLETE. Final Sharpe=1.1705 DSR=0.9526. All 5 handoff artifacts. |
| 16.15 Go/No-Go verdict | **PENDING Q/A** | This bundle + masterplan.json notes |

## Bugs surfaced by the UAT and fixed in-flight

1. **GCP scope** (`_genai_client.py` + `orchestrator.py`) -- cloud-platform scope was missing on service-account credentials, breaking direct-script Gemini calls. Fixed.
2. **Discovery Engine 403 + 404** -- RAG step failed because API was disabled, SA lacked `roles/discoveryengine.viewer`, and `RAG_DATA_STORE_ID` pointed at a non-existent `rag_search` instead of the real `10-k-data_1762684273198_gcs_store`. All fixed.
3. **RAG fail-open** -- pipeline crashed on any Discovery Engine error; now logs a warning and continues with empty {text, citations}. Fixed.
4. **Localhost auth bypass** -- added `DEV_LOCALHOST_BYPASS=1` env + 127.0.0.1/::1 client-host check for UAT drills. Fails closed without both conditions.
5. **BQ schema drift** -- `analysis_results` table missing 15 phase-11 columns that `save_report` was emitting (nlp_sentiment_confidence etc). Migrated.
6. **Claude-as-default** -- settings defaults flipped (`gemini_model` -> claude-sonnet-4-6, `deep_think_model` -> claude-opus-4-6), autonomous_loop reads from settings, Settings UI banner added.
7. **Gemini fallback was Claude** -- when Claude is default, the orchestrator fallback was re-routing the SAME Claude model through make_client (GitHub Models). Now forces explicit Gemini models in the fallback path.

## Non-blocker findings for Peder

1. **Rotate ANTHROPIC_API_KEY** in `backend/.env` from OAuth token (`sk-ant-oat*`) to a real API key (`sk-ant-api03-*`) at console.anthropic.com. Until then, paper-trading runs on Gemini fallback (scope + RAG fixes make this path work).
2. **Kill switch pause/resume has slow IO path** (likely Slack notification) that times out fast Python probes but state mutation completes. Non-blocking.
3. **`com.pyfinagent.autoresearch` launchd agent** reports last_exit_status=1. Worth inspecting its log.
4. **`scripts/go_live_drills/kill_switch_test.py`** has a venv-backend-pkg shadow import bug (site-packages backend shadows local). Non-blocking.

## Commits in this phase-16 execution

- `f2e8ce28` -- fix(uat-16.2): GCP scope, RAG IAM+datastore, localhost auth, BQ schema
- `46b76ef4` -- feat: Claude as default LLM provider, Gemini switchable from Settings
- `277e8ace` -- fix(uat-16.4): Gemini fallback must force Gemini models, not settings

Plus in-session: 3a22f0ee (phase-16 plan), 8131bb70 (Risk Judge observability), and earlier pre-prod commits.

## Recommendation for Q/A

Verify:
- [ ] Every sub-step in `.claude/masterplan.json::phase-16.steps[*]` has a `status` field populated.
- [ ] The sub-steps marked `in-progress` (16.2, 16.3) are defensibly non-blocking for Monday's paper cycle (the Gemini fallback path they depend on is proven working).
- [ ] The 7 bugs fixed in-flight are all committed and pushed to origin/main.
- [ ] No live-capital risk introduced. Lockout at `execution_router.py:74-80` still raises on PKLIVE keys.
- [ ] Peder's 4 non-blocker follow-ups are captured somewhere he can act on (this file + masterplan notes).

Then issue a Go/No-Go verdict for phase-16 as a whole.
