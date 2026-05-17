# Pre-Prod Smoke Test — 2026-05-16

**Auditor:** Claude (Opus 4.7, max effort, harness skipped per single-shot audit scope)
**Target go-live:** end of May 2026
**Scope:** candidate-discovery → analysis → paper-trade light path
**Explicitly out of scope:** AutonomousLoop full-path, AlpacaBroker live execution
**Modality:** read-only audit. No code modified.

## Executive summary

Backend (v6.8.9) and Slack bot are alive; the daily paper-trade scheduler is firing and the most-recent run (`cycle_id=74c322b2`, 2026-05-16 18:21:55 → 18:25:43 UTC) completed all six published steps. Portfolio is at $22,748.62 NAV / +13.74% since inception, vs benchmark +13.97%.

But the light path has three load-bearing defects that prevent the system from learning or persisting most of what it does:

1. **B-1 (Full path dead):** Every full-pipeline call routes through `claude-sonnet-4-6`, which the `llm_client.create_llm_client()` factory tries to dispatch via GitHub Models *first* — and GitHub Models requires `GITHUB_TOKEN`, which is unset. Every full attempt errors and falls back to the lite Claude analyzer. The 28-skill Layer-1 pipeline has not run in production for at least ~24 h of the audit window.
2. **B-2 (Lite-path persistence broken):** Every lite analysis tries to write 5 columns (`consumer_sentiment`, `revenue_growth_yoy`, `quality_score`, `momentum_6m`, `rsi_14`) that the `financial_reports.analysis_results` BigQuery table does not have. **14 of 15** analyses in the live cycle failed to persist. Reports dashboard is therefore lagging reality.
3. **B-3 (CoALA semantic-learning loop unfed):** `outcome_tracking` and `agent_memories` are both at **0 rows** lifetime. The Step-9 "Learn from closed trades" branch is gated on `closed_tickers` being non-empty; in 30 days of paper trading the cycle has had ~1 SELL and no successful outcome write. The debate stack reads `agent_memories` for past lessons but the table is empty — so no learning is occurring.

Each is fixable inside a single PR (see Go-live blockers, §A).

The scaffolding underneath is largely correct: the 6-step paper-trade loop is wired, Risk Judge governs sizing on the lite path, the kill-switch is armed (4 % daily / 10 % trailing-DD), gates and freshness probes exist, the Slack bot dispatches three slash commands and a deploy flow, signal endpoints (earnings tone, anomalies, sector) return real data. Nothing in the audit suggests the architecture is wrong; what's broken is configuration drift (B-1), an unfinished BQ schema migration (B-2), and a path that is reachable but conditional in a way it has never reached (B-3).

---

## Test environment

| Item | Value |
|---|---|
| Date | 2026-05-16 |
| Backend | `uvicorn backend.main:app --host 0.0.0.0 --port 8000`, pid 52623, version 6.8.9 |
| Frontend | next dev on :3000 (302 redirect on root, confirms middleware live) |
| Slack bot | `python -m backend.slack_bot.app`, pid 52648 (Socket Mode) |
| MCP servers attached | data, backtest, signals (all `status: ok` from `/api/health`) |
| Project | `sunny-might-477607-p8` |
| BQ dataset (paper + reports) | `financial_reports` (us-central1) |
| Backend log | `backend.log` (capture window starts ~2026-05-15 19:00 UTC) |
| Live cycle observed | run-now triggered at 2026-05-16 18:21:55 UTC, completed 18:25:43 UTC (~3:48 wall) |

---

## §1. MultiAgentOrchestrator + PlannerAgent — **PARTIAL**

- `MultiAgentOrchestrator` lives at `backend/agents/multi_agent_orchestrator.py:1` (Slack-tier conversational MAS, three roles: Main, QA, Research). It is **not** in the paper-trade path — that path is driven by `backend/services/autonomous_loop.py::run_cycle()`. The orchestrator runs on Slack message dispatch only.
- `PlannerAgent` at `backend/agents/planner_agent.py:55` is research-harness-tier: it proposes parameter/feature changes when invoked by `scripts/harness/run_harness.py`. Not in the live cycle.
- Task decomposition for the actual cycle is hardcoded as 9 numbered steps inside `autonomous_loop.run_cycle()` (`autonomous_loop.py:197–620`); not LLM-planned.
- Verdict: orchestration is fine for the Slack surface but the *paper-trade* cycle is procedural, not planner-driven. This is an expected design choice — the user's scope is the cycle, so the orchestrator is incidental here.

**Pass.** No blockers. Document the split (Slack=MAS, cycle=imperative) in onboarding so newcomers don't expect Planner to drive trades.

---

## §2. Signal sourcing (Earnings Tone, NLP Sentiment) — **PASS**

Both surfaces return real data when probed independently:

- `GET /api/signals/AMD/earnings-tone` → `signal=CONFIDENT`, mgmt confidence 8/10, evidence phrases, 55k-char transcript, Q1 2026
- `GET /api/signals/AMD/anomalies` → 5 anomalies, 3 high severity (z=2.13 vol, z=7.68 P/E gap)
- `GET /api/signals/AMD/sector` → Technology / XLK, 26.53 % 3-mo sector return, 100 % 3-mo relative

Function-level: `backend/tools/earnings_tone.py` (Yahoo Finance transcript scrape + Gemini tone classification), `backend/tools/nlp_sentiment.py` (Vertex text-embedding-005 against bullish/bearish corpora). Invoked from `orchestrator.run_earnings_tone_agent()` (orchestrator.py:1019) and `run_nlp_sentiment_agent()` (orchestrator.py:1049). But the orchestrator is only entered through the **full path**, which is dead (see B-1) — so in production, signal sourcing is currently a screening-only contribution: the screener picks 502 tickers, narrows to 10 candidates, the lite analyzer sees raw OHLC + fundamentals but does NOT call the 28-skill enrichment.

**Pass** as standalone endpoints. **Functionally idle in the live cycle until B-1 is fixed.**

---

## §3. Layer-1 Analysis Pipeline (28 skills) — **FAIL (light path) / N/A (full path)**

The 28 skill prompts live under `backend/agents/skills/*.md`. Mapping each to its invoker in `orchestrator.py`:

| Skill | Invoker | In pipeline? | Skip conditions |
|---|---|---|---|
| alpha_decay_agent | `run_alpha_decay_agent` (1089) | NO — external only | — |
| alt_data_agent | `run_alt_data_agent` (1035) | YES (Step 6) | sector skip: Utilities, Real Estate |
| anomaly_agent | `run_anomaly_agent` (1056) | YES (7) | — |
| bias_detector | `detect_biases()` in `bias_detector.py` | YES (12b) | — |
| competitor_agent | `run_competitor_agent` (891) | YES (5) | — |
| critic_agent | `get_critic_prompt` (prompts.py:461) | YES (11 reflection loop, max 2 iter) | — |
| debate_stance (Bull/Bear) | `debate.run_debate` (124) | YES (8) | skip Devil's Advocate if `lite_mode` |
| deep_dive_agent | `run_deep_dive_agent` (907) | YES (10) | skip if `lite_mode` |
| earnings_tone_agent | `run_earnings_tone_agent` (1019) | YES (6→7) | — |
| enhanced_macro_agent | `run_enhanced_macro_agent` (1026) | YES (9) | — |
| info_gap_agent | `detect_info_gaps()` in `info_gap.py` | YES (6b) | — |
| insider_agent | `run_insider_agent` (991) | YES (7) | — |
| market_agent | `run_market_agent` (882) | YES (4) | — |
| moderator_agent | `get_moderator_prompt` (debate.py:640) | YES (8) | — |
| nlp_sentiment_agent | `run_nlp_sentiment_agent` (1049) | YES (7) | — |
| options_agent | `run_options_agent` (998) | YES (7) | — |
| patent_agent | `run_patent_agent` (1012) | YES (7) | sector skip: Financial Services, Utilities, Real Estate |
| quant_model_agent | `run_quant_model_agent` (1076) | YES (7) | — |
| quant_strategy | (none) | **NO** — optimizer-only, used by `quant_optimizer.py` | — |
| rag_agent | `run_rag_agent` (859) | YES (3) | `_rag_available` fail-open returns `{"text":"","citations":[]}` |
| risk_judge | `risk_debate.run_risk_debate` (956) | YES (12c) | skip if `lite_mode` or `data_quality < min` |
| risk_stance (Aggressive/Conservative/Neutral) | risk_debate.py | YES (12c) | skip if `lite_mode` or `data_quality < min` |
| scenario_agent | `run_scenario_agent` (1063) | YES (7) | — |
| sector_analysis_agent | `run_sector_analysis_agent` (1042) | YES (7) | — |
| sector_catalyst_agent | (none) | **NO** — referenced as template placeholder in synthesis only | — |
| social_sentiment_agent | `run_social_sentiment_agent` (1005) | YES (7) | — |
| supply_chain_agent | (none) | **NO** — template placeholder | — |
| synthesis_agent | `get_synthesis_prompt` (prompts.py:350) | YES (11) | reflection loop max 2 iter |

**Wired into a 15-step pipeline:** 24 / 28.

**Dead skill files (have `.md`, never invoked anywhere):** 4
- `alpha_decay_agent` — external (strategy router), not pipeline
- `quant_strategy` — optimizer-only
- `sector_catalyst_agent` — referenced as variable in synthesis prompt template (`{sector_catalyst}`), never actually invoked → injects nothing
- `supply_chain_agent` — same situation; `{supply_chain_signal}` template placeholder, never invoked

**Lite-path skips (active TODAY because the full path is dead):**
- Devil's Advocate (`debate.py:249`)
- `deep_dive_agent` (Step 10)
- `risk_stance` 3-way debate + `risk_judge` (Steps 12c) — replaced by a single lite-Risk-Judge LLM call with hardcoded system prompt `_LITE_RISK_JUDGE_SYSTEM` (`autonomous_loop.py:818–830`)
- Debate rounds reduced from 2 to 1 (`orchestrator.py:1395`)

**Sector skips:** financial-services tickers skip `patent_agent`; utilities and real-estate skip both `patent_agent` and `alt_data_agent`.

**Verdict:** the full Layer-1 pipeline has been silently inactive in production. Until B-1 is fixed, the 24 in-pipeline skills do not run; only the lite Claude analyzer's two LLM calls (trader + risk-judge) execute.

**Blocks go-live light-path?** No — the lite path is what the user said is in scope. But the **4 dead skill files** are a documentation hazard (the team will keep onboarding new contributors who read them and assume they are live). Mark them dead in `_inventory.json` or delete.

---

## §4. Debate stack (Bull / Bear / DA / Moderator / Synthesis) — **N/A on light path**

- `debate.run_debate()` at `backend/agents/debate.py:124`. `BullAgent`, `BearAgent`, `ModeratorAgent`, `SynthesisAgent`, `DevilsAdvocateAgent` are **prompt templates, not classes** (`get_bull_agent_prompt`, etc.). The "agent" identity is the prompt + structured-output schema (`DevilsAdvocateResult`, `ModeratorConsensus` in `agents/schemas.py:117`).
- Rounds: parameter `max_debate_rounds` defaults to 2; reduced to 1 in lite mode.
- Synthesis has a reflection loop bounded by `synthesis_iterations <= 2` with `critic_agent` review.
- **In the light path: the debate stack does not run at all.** The lite Claude analyzer is two single-turn LLM calls (trader + risk-judge); no bull/bear, no DA, no moderator, no synthesis.
- Test coverage: indirect only (via `test_autonomous_loop_integration.py`). No `test_debate.py`. Schema parsing / JSON-fallback paths untested.

**Verdict:** light path go-live = acceptable. Full path is currently unreachable (B-1).

---

## §5. Quality gates (BiasDetector, R&D, Analyst, RiskJudge) — **PARTIAL on light path**

- **BiasDetector** (`backend/agents/bias_detector.py:62 detect_biases()`). 5 bias classes checked: tech-sector, confirmation, recency, anchoring, source-diversity. Called from full-path Step 14. **Not called in the lite path.** The lite path has no bias gate.
- **"R&D Agent"** — does not exist as a separate component. `meta_coordinator.py:63` maps `"patent_signal": ["patent_innovation_agent"]`, but `patent_innovation_agent` is just an alias for the Layer-1 `patent_agent` skill. The original architecture's "R&D Agent (locked)" was a planning artifact; in code it is the `patent_agent` skill which is alive but only runs on the full path.
- **AnalystAgent** — at `backend/api/agent_map.py:75` maps `"analyst_agent": "mas_communication"`. This is a Slack-tier conversational role (Sonnet 4.6), invoked when a user types `@assistant`. **Not in the trade cycle.**
- **RiskJudge** — `risk_debate.py::run_risk_debate` runs the full 3-stance debate + judge verdict on the full path. On the lite path, `_run_lite_risk_judge_call` (`autonomous_loop.py` ~ line 818–950) issues a single LLM call with a hardcoded system prompt that demands volatility/concentration/valuation reasoning. The lite judge's verdict feeds `decide_trades()` for position sizing.

**Light-path verdict:** only Risk Judge (single-call lite version) is active. BiasDetector, BullBear debate, and the 3-stance risk debate are bypassed. This is acceptable for go-live IF the user accepts the lite path's reduced rigor; but **document explicitly** that bias detection is OFF on the live path.

**Note on persistence:** the lite Risk Judge's `recommended_position_pct` is computed and passed into `PaperTrader.execute_buy(risk_judge_position_pct=…)` — but `paper_trader.py` never writes that field into the `paper_positions` row. Confirmed: every position in `/api/paper-trading/portfolio` returns `risk_judge_position_pct: null`. See B-5.

---

## §6. EvaluatorAgent + CommunicationAgent — **NOT IN LIVE CYCLE**

- `backend/agents/evaluator_agent.py::EvaluatorAgent.evaluate_proposal()` is invoked only by `backend/api/backtest.py` (post-backtest harness step). It is **never called inside the live paper-trade cycle**. No call site in `autonomous_loop.py`. So live trades have no evaluator gate; they have only the (lite) risk-judge.
- `CommunicationAgent` is an AgentConfig at `agent_definitions.py:128` (model `mas_communication` → Sonnet 4.6). It is the Slack-tier responder, used by `MultiAgentOrchestrator` when the user posts a question in Slack. **Not part of the trade cycle.**

**Light-path verdict:** acceptable for go-live IF the user accepts that live trades skip evaluator review. Long-term, wiring the evaluator into the live cycle (post-decide, pre-execute) is a recommended hardening.

---

## §7. Paper trade execution (PortfolioManager → PaperTrader → ExecutionRouter) — **PARTIAL**

- `decide_trades()` at `backend/services/portfolio_manager.py:41` implements sell-first-then-buy. Sell triggers: stop-loss, explicit SELL recommendation, downgrade. Buy phase consumes Risk-Judge `recommended_position_pct`.
- `PaperTrader.execute_buy()` at `backend/services/paper_trader.py:85`. Writes to BQ `paper_trades` via `_safe_save_trade()` and creates/updates the `paper_positions` row. **Does NOT persist `risk_judge_position_pct`** — see B-5.
- `ExecutionRouter._current_mode()` at `backend/services/execution_router.py:65` reads `EXECUTION_BACKEND` env (default `bq_sim`). Live values supported: `bq_sim` (mock fill at last close), `alpaca_paper` (real paper-trading at Alpaca), `shadow` (write both, BQ owns state). **Live Alpaca keys (`PKLIVE…`) are hard-rejected.** Today, the deployment runs in `bq_sim` mode (Alpaca keys are not set).
- The cycle's Step 7 (`Executing`) iterates orders and calls `trader.execute_buy()` / `execute_sell()`. ExecutionRouter is invoked **inside** the PaperTrader's save path, not as a top-level router.
- Live cycle observation: `cycle_id=74c322b2` → `trades_executed: 0`. The cycle decided no buys/sells because no candidate cleared the lite-Risk-Judge gate; all 15 analyses also failed to persist (B-2), so the decide step had no fresh signals to act on.

**Verdict:** the wiring is correct, but B-2 (analyses don't persist) and B-5 (risk_judge_position_pct doesn't persist) both leave audit-trail gaps. Live trading on the light path is technically functional (we have 15 historic trades in BQ; portfolio is up +13.74 %), but you cannot reconstruct sizing intent from the BQ paper_positions table.

---

## §8. OutcomeTracker — **DEAD**

- `backend/services/outcome_tracker.py::OutcomeTracker.evaluate_recommendation()` is called from `_learn_from_closed_trades()` (`autonomous_loop.py:1087`).
- The call site (Step 9 in the cycle, `autonomous_loop.py:572–578`) is **gated on `closed_tickers` being non-empty.** Most cycles have no closes, so Step 9 silently skips.
- BigQuery confirmation:

| Table | Row count | Latest write |
|---|---|---|
| `financial_reports.outcome_tracking` | **0** | never |
| `financial_reports.agent_memories` | **0** | never |
| `financial_reports.paper_trades` | 15 | 2026-05-01 (TER SELL) |
| `financial_reports.signals_log` | 32 | 2026-05-16 |
| `financial_reports.analysis_results` | 54 | (older — most recent persists are failing per B-2) |

The single SELL in 30 days (`TER` on 2026-05-01) did not trigger an outcome write either — `_learn_from_closed_trades` looks up the matching analysis by `analysis_id`, but `analysis_id` is missing on most paper_trades rows. The chain is gated three deep and never reaches the writer.

**Verdict:** **CoALA semantic-learning loop is silently dead.** This is B-3. The debate prompts read `agent_memories` for past lessons (`orchestrator.py:1798–1810`) but the table is empty, so the debate-stack memory injection contributes nothing in production today.

---

## §9. CoALA memory (episodic, semantic, procedural) — **PARTIAL**

CoALA (Sumers et al. 2024) prescribes WORKING / EPISODIC / SEMANTIC / PROCEDURAL.

| Layer | Implementation | Storage | Live use |
|---|---|---|---|
| Working | not labeled — context window only | — | implicit |
| Episodic | `HarnessMemory.append_episodic / load_episodic` at `agents/harness_memory.py:119, 151` | `memory/YYYY-MM-DD.md` files | called only from `multi_agent_orchestrator._save_to_memory()` (Slack tier). **Not called in the trade cycle.** |
| Semantic | `HarnessMemory.load_semantic` (read-only); programmatic write API missing; uses `MEMORY.md` | file | session init only — **not in cycle** |
| Procedural | `backend/agents/skills/*.md` loaded at orchestrator init | filesystem | static; no update loop |

The actual learning hub in production is **not** a CoALA layer per se but `FinancialSituationMemory` in `agents/memory.py:57` — BM25-indexed lessons persisted to BigQuery `agent_memories`. It is loaded into bull / bear / moderator / risk-judge instances at orchestrator init (`orchestrator.py:567`) and used in the debate phase (`orchestrator.py:1798–1810`) — but agent_memories is empty (B-3), so it contributes nothing.

**Verdict:** the architecture is CoALA-adjacent but the loop is broken at two places:
- procedural layer is static (no self-prompt-update); acceptable for go-live, but flag as post-launch.
- semantic-learning is wired but unfed (B-3).

**No end-to-end test** exercises write → retrieve across episodic / semantic / procedural.

---

## §10. Slack interface — **PASS (with gaps)**

- Entry: `backend/slack_bot/app.py` — Socket Mode handler (AsyncSocketModeHandler) — pid 52648 running.
- Slash commands registered in `backend/slack_bot/commands.py`:
  - `/analyze TICKER` (line 91)
  - `/portfolio` (line 131)
  - `/report TICKER` (line 147)
- Keyword-routed (non-slash):
  - `"status"` (line 258) — memory, plan, git, backtest
  - `"clear queue"` (line 234) — destructive: kills processes, drops queue
  - `deploy …` (line 236) — routes through `self_update.handle_deploy_command()`
- Thread response: `thread_ts = message.get("thread_ts") or message.get("ts")`; streaming via `StreamingHandler.stream_response()` (`streaming_integration.py:71–87`).
- MCP tools (`backend/slack_bot/mcp_tools.py`): `search_messages`, `search_channels`, `search_files`, `search_users`, `post_message`, `create_canvas`, `read_channel_history`, `read_thread`. Executor scaffolded; some implementations marked TODO.
- Governance (`backend/slack_bot/governance.py`): `AuditLogger` (**in-memory only — TODO: persist to BQ**), `HumanInTheLoopManager`, `RateLimiter` (60 req/hr), `ContentDisclaimer`.
- Error/fallback: every command wraps the upstream API call in `try/except` and surfaces a truncated 200-char error to the user.

**Gaps:**
- **G-1 (medium):** `AuditLogger` is in-memory — restart loses the trail. Persist to BigQuery before go-live or compliance has a gap.
- **G-2 (medium):** `handle_deploy_command` has **no permission check** — any message in any channel the bot reads can trigger `deploy update`. Add an admin gate before go-live.
- **G-3 (low):** `SLACK_SIGNING_SECRET` is not set in the env; Socket Mode tolerates this, but the documentation recommends setting it.

Slack bot log shows healthy heartbeats, no claude-API errors, hourly signal-warmup jobs completing in <1 ms.

**Verdict:** functional. G-1 and G-2 block compliance-grade launch but not the trading mechanic.

---

## §11. Self-update deploy — **PASS (no real dry-run, no auth)**

- Entry: `backend/slack_bot/self_update.py::handle_deploy_command()` at line 436.
- Real-run: `deploy update` → record old commit → `git stash` → `git pull origin main` (60 s timeout) → diff-stat → syntax-validate 7 key files → kill old processes (`active_slack_monitor`, `slack_response_agent`, `slack_mention_checker`, `imsg_responder_tickets`) → `_schedule_restart()` async (3 s, kill, exec, verify-alive, retry once).
- "Dry-run": there is **no explicit dry-run flag**. `deploy status` shows `git status --short` and commits-behind; `deploy diff` shows the pull diff without pulling. These are previews, not a real dry-run.
- Rollback: `git revert --no-commit HEAD && git commit -m "Rollback: revert …"` → restart.
- Safety: syntax check is a hard gate (deploy aborts if any of the 7 key files fails `ast.parse`). Stash + restore on pull failure. **No in-flight-cycle check** — deploying mid-trade is theoretically possible.
- Audit: `DEPLOY_LOG = logs/deploy.log` exists on disk only after first successful run; file is **absent** in current workspace, meaning no deploy has happened on this machine via this surface yet.
- Auth: none. See G-2 above.

**Smoke test attempted:**
- Read-only dry-run inspection only — did not invoke `deploy update` in this audit (would restart the bot and confound the rest of the audit).
- `deploy status` would have been benign — but per instructions, "do not modify code unless approved" — restarting the bot is arguably a state mutation, so I did not run it. Recommend running `deploy status` from Slack as a real test (it only reads).

**Verdict:** the mechanic is sound. Two gaps:
- D-1 (low): no real dry-run path; `deploy diff` is the workaround.
- D-2 (high): no caller-auth check — same as G-2.

---

# A. Go-live blockers (must fix before end of May)

| ID | Severity | Component | Symptom | Root cause | Fix |
|---|---|---|---|---|---|
| **B-1** | **CRITICAL** | LLM client routing | Every full-path analysis errors with `Model 'claude-sonnet-4-6' requires a GitHub Token (GITHUB_TOKEN) but none is set` and falls back to lite. Observed on every ticker in the 18:21:55 live cycle. | `llm_client.create_llm_client()` (`backend/agents/llm_client.py:1740–1758`) checks the `GITHUB_MODELS_CATALOG` **first**; `claude-sonnet-4-6` is in the catalog (line 416). If `GITHUB_TOKEN` is unset, it raises before reaching the Anthropic-direct branch at line 1761. ANTHROPIC_API_KEY is set and works for `/api/paper-trading/*`, so the model is reachable — just not via this router. | Two options. (a) Set `GITHUB_TOKEN` in `backend/.env` (Peder's Copilot Pro PAT). (b) Re-order the factory so Anthropic-direct wins when `anthropic_api_key` is set and `github_token` is not (i.e., turn the catalog check into a fallback rather than a gate). I recommend (b) — it's defensive and survives future PAT rotations. |
| **B-2** | **CRITICAL** | BQ schema | 14 of 15 analyses in the live cycle failed to persist to `financial_reports.analysis_results` with `no such field: consumer_sentiment` / `revenue_growth_yoy` / `quality_score` / `momentum_6m` / `rsi_14`. These are the Phase-11 Autoresearch FEATURE_TO_AGENT bridge columns. | `BigQueryClient.save_report()` writes these 5 fields (`bigquery_client.py:113–117`); the BQ table is on 85 columns; the 5 are missing. The 15 sibling Phase-11 columns ARE present, so the schema migration was applied partially. | Run a one-off schema migration adding the 5 columns to `sunny-might-477607-p8.financial_reports.analysis_results`. All are `FLOAT64 NULLABLE`. Mirror the idempotent pattern in `scripts/migrations/migrate_bq_schema.py`. |
| **B-3** | **HIGH** | Outcome / memory loop | `outcome_tracking` and `agent_memories` both at 0 rows lifetime. The debate stack's BM25 lesson injection contributes nothing. | Step 9 (`_learn_from_closed_trades`) is gated on `closed_tickers != []`. Even when there IS a SELL (TER, 2026-05-01), the inner lookup requires `analysis_id` on the paper_trades row, which is mostly null. | (a) Decouple Step 9: also run an unconditional `outcome_tracker.evaluate_all_pending()` once per cycle (reads open positions, not just closes). (b) Backfill `analysis_id` on existing paper_trades rows. Track as a single masterplan step. |
| **B-4** | **HIGH** | Cycle-history API | `/api/paper-trading/cycles/history` returns `steps`, `candidates`, `trades_executed`, `signals_logged`, `analysis_cost` as `null` for every historical row except possibly the very latest. | The history endpoint likely reads a thin snapshot table while the in-memory `summary` carries the rich fields only for the live run. Endpoint shape needs to be reconciled with `paper_portfolio_snapshots` schema (which I didn't deep-dive). | Add the missing fields to the snapshot row writer in `Step 8 (Final snapshot)` of `autonomous_loop.py:561–571`, and to the read query in `backend/api/paper_trading.py::cycles_history`. |
| **B-5** | **HIGH** | Position sizing audit | `risk_judge_position_pct` is computed and passed into `PaperTrader.execute_buy(risk_judge_position_pct=…)` but **not** persisted to `paper_positions`. Every position in `/api/paper-trading/portfolio` returns `null`. | `paper_trader.py:181–200` builds `pos_row` without including `risk_judge_position_pct`. The column either exists in BQ but isn't written, or doesn't exist at all (verify with `mcp__bigquery__describe-table` once dataset location is unified). | Add the field to the pos_row dict; if the BQ column is missing, run an `ALTER TABLE` migration. Same migration script as B-2. |
| **B-6** | **MEDIUM** | Trades schema gaps | `/api/paper-trading/trades` returns `null` for `executed_at`, `signal_score`, `sector` on every trade. Compliance can't reconstruct trade timing. | Either the writer doesn't populate these fields, or the API serializer drops them. Likely the former — `_safe_save_trade()` in paper_trader.py only writes a minimal set. | Add the three fields to `_safe_save_trade()` and to the BQ `paper_trades` schema if missing. |
| **B-7** | **MEDIUM** | Slack governance | `AuditLogger` is in-memory only (lost on restart); `handle_deploy_command()` has no permission check. | Both flagged TODO in `governance.py` and `self_update.py`. | Two PRs: (a) persist `AuditLog` to `pyfinagent_data.unified_sar_log` or a new table; (b) gate deploy commands on `user_id in admin_set`. Set `SLACK_SIGNING_SECRET` while you're at it. |

---

# B. Acceptable for launch on the light path (document; do not fix)

| Item | Note |
|---|---|
| 4 dead skill files (`alpha_decay_agent`, `quant_strategy`, `sector_catalyst_agent`, `supply_chain_agent`) | Mark dead in `_inventory.json` and add a one-line README warning. Two of these (`sector_catalyst`, `supply_chain`) are referenced as variables in synthesis prompt templates — those variables resolve to empty strings, which is fine, but it's misleading to future contributors. |
| Lite path skips BiasDetector, full 3-stance risk debate, deep_dive, devil's advocate, debate-stance | Explicit user choice. Document in CLAUDE.md that the light path is the production path. |
| EvaluatorAgent not called in the live cycle | Acceptable for go-live; recommend wiring it in post-launch as a sanity gate. |
| `coordinator.action: quant_opt` on Sharpe -6.70 | This is the MetaCoordinator advising the system to run quant-optimization. It is not actually triggering one — just flagging — so it doesn't affect live trading. Investigate post-launch why Sharpe is computed as -6.7 when realized portfolio is +13.74 %; likely a metric-definition divergence in `MetaCoordinator.gather_health()`. |
| `agent_memories` empty | will fix on B-3 once outcome writer is unblocked. |
| Paper_trades freshness band RED (last_tick_age_sec=173920s) | Caused by the 2-day-with-no-fills weekend pattern; band thresholds (1.5x cycle interval) are too tight for a daily cycle. Tune `cycle_interval_sec` or `warn/critical_ratio`. |
| Sector concentration (Technology 54.82 %) | Risk-policy concern, not a system bug. Surface in dashboard but defer policy fix to post-launch. |

---

# C. Post-launch masterplan (deferred items, with severity)

These should be added to `.claude/masterplan.json` as new steps after the May 31 cutover.

| Step (proposed) | Severity | Title | Why |
|---|---|---|---|
| `phase-24.0` | HIGH | Wire AutonomousLoop full path end-to-end | Today the lite path is the live path. After B-1 fix, run the 28-skill full path on at least one shadow ticker per cycle and compare lite vs full decisions; this gives empirical justification for the lite-mode choice. |
| `phase-24.1` | HIGH | Wire AlpacaBroker live execution behind a feature flag | `ExecutionRouter` already supports `alpaca_paper` and `shadow` modes. Add the live-Alpaca path with a hard-block on `PKLIVE…` until the user signs off; first run it in `shadow` mode for ≥30 days. |
| `phase-24.2` | HIGH | Wire EvaluatorAgent into the live cycle | Add a post-decide, pre-execute call to `evaluator.evaluate_proposal()` that can veto a trade. Currently evaluator is backtest-only. |
| `phase-24.3` | MEDIUM | Backfill `analysis_id` on paper_trades; re-run outcome tracking | One-time SQL backfill plus running outcome_tracker.evaluate_all_pending() once over the historical window will seed `agent_memories` with real lessons. |
| `phase-24.4` | MEDIUM | Persist Slack `AuditLogger` to BQ + add deploy-command admin gate | B-7 hardening for compliance. |
| `phase-24.5` | MEDIUM | Reconcile `MetaCoordinator.gather_health().sharpe_ratio` with `perf_metrics.py` | Sharpe -6.7 in coordinator vs realized +13.74 % NAV is suspicious. Likely a sign convention or windowing bug. |
| `phase-24.6` | MEDIUM | Wire CoALA episodic + semantic memory into the trade cycle | Today `HarnessMemory.append_episodic/load_episodic` are only called from the Slack-tier orchestrator. Have each cycle write a one-line episodic summary and have the debate stack include the last 7 days of episodic context. |
| `phase-24.7` | MEDIUM | Procedural-memory update loop | Wire `skill_optimizer.py` (already exists) to actually rewrite skill prompts based on outcome-tracker feedback. Closes the CoALA procedural loop. |
| `phase-24.8` | LOW | Add real `--dry-run` flag to `deploy update` | Currently only `deploy diff` previews. A real dry-run would run the syntax check + restart-readiness probes without pulling. |
| `phase-24.9` | LOW | Delete or annotate 4 dead skill files | `_inventory.json` truth-up + README warning. |
| `phase-24.10` | LOW | End-to-end test for CoALA write → retrieve | One pytest that drives outcome_tracker → agent_memories write → debate prompt retrieval. Catches B-3 regressions. |
| `phase-24.11` | LOW | Tune paper_trades freshness band for daily-cycle gap | Adjust warn/critical ratios so weekend gaps don't flash RED. |
| `phase-24.12` | LOW | Add in-flight-cycle check to `deploy update` | Block deploy when a cycle is mid-run. |
| `phase-24.13` | LOW | Persist `risk_judge_position_pct`, `executed_at`, `signal_score`, `sector` on paper_trades and paper_positions (already covered by B-5/B-6; this step is the audit follow-up after fix lands) | Re-verify via `/api/paper-trading/portfolio` and `/api/paper-trading/trades`. |

---

# D. Live-system evidence captured

Captured at 2026-05-16 18:21–18:25 UTC. Quoted verbatim from `curl` against the running backend and `backend.log`.

```
$ curl -s http://localhost:8000/api/health
{"status":"ok","service":"pyfinagent-backend","version":"6.8.9",
 "mcp_servers":{"data":{"status":"ok"},"backtest":{"status":"ok"},"signals":{"status":"ok"}},
 "limits_digest":"edf822591bb17c9d8f62f4f50a8fca72f11690b21884b7cd2f0988e0e2c9bad4"}

$ curl -sX POST http://localhost:8000/api/paper-trading/run-now
{"status":"started","started":true,"message":"Daily cycle triggered"}

# 3:48 later:
$ curl -s http://localhost:8000/api/paper-trading/status | jq '.loop.last_result'
{
  "status":"completed",
  "steps":["screening","analyzing","mark_to_market","deciding","executing","snapshot"],
  "cycle_id":"74c322b2",
  "started_at":"2026-05-16T18:21:55.893471+00:00",
  "screened":501,"candidates":10,"new_to_analyze":2,"reeval_tickers":13,
  "signals_logged":1,"trades_executed":0,"closed_tickers":[],
  "analysis_cost":0.15,
  "coordinator":{"action":"perf_opt","reason":"p95 latency 6570ms > 500ms threshold",
                 "health":{"sharpe":-6.44,"accuracy":0.0,"p95_latency_ms":6570.3}}
}

# from backend.log during the cycle (B-1):
20:22:19 W [autonomous_loop] Full orchestrator failed for STX: Model 'claude-sonnet-4-6'
  requires a GitHub Token (GITHUB_TOKEN) but none is set. Add GITHUB_TOKEN=ghp_... to
  backend/.env -- falling back to lite Claude analyzer
... [same line for AMD, FIX, MU, KEYS, GEV, COHR, ON, INTC, DELL, GLW, CIEN, LITE, SNDK, WDC]

# from backend.log during the cycle (B-2):
20:22:22 W [autonomous_loop] Failed to persist lite analysis for STX:
  Failed to save report: [{'index':0,'errors':[{'reason':'invalid',
  'location':'momentum_6m','message':'no such field: momentum_6m.'}]}]
... [14 similar with rotating field names: consumer_sentiment, revenue_growth_yoy,
     quality_score, momentum_6m, rsi_14]

# BQ row counts (us-central1) for B-3:
outcome_tracking:   0 rows
agent_memories:     0 rows
paper_trades:      15 rows
signals_log:       32 rows
analysis_results:  54 rows
```

---

# E. Components I did NOT cover (transparency)

- **Frontend smoke**: did not load the dashboard in a browser. 302 redirect on `/` confirms middleware is alive, but I did not click through `/backtest`, `/portfolio`, `/reports` to confirm the UI renders the live data. Recommend a quick visual smoke before the go-live cutover.
- **Backtest engine**: out of audit scope. Healthy per `/api/backtest/status` (MCP server `backtest` reports `ok`).
- **Active deploy of `deploy update`**: chose not to actually run it during the audit (would restart the bot mid-audit). Recommend `deploy status` (read-only) from Slack as the next live test.
- **Full Layer-1 pipeline once B-1 is fixed**: cannot exercise the 28 skills until the LLM router is unblocked. Re-run this audit after B-1 lands.
- **Slack command live test**: did not post `/analyze AMD` from Slack myself; relied on log evidence that the bot is up and responding. Recommend Peder post `/analyze AAPL` after B-1 lands to capture an end-to-end Slack→full-pipeline trace.

---

*End of audit. Approve B-1 through B-7 fixes individually; each is a single PR. Add the §C masterplan steps after May 31.*
