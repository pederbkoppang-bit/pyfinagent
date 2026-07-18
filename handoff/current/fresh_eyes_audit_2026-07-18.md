# Fresh-Eyes Audit Register -- 2026-07-18 (Fable 5 session, cloud)

Basis for the long-term cloud goal (`goal_long_term_fable5_cloud.md`). Three
parallel read-only auditors swept surfaces NOT covered by the 67/71/72/73
audits, each carrying an explicit dedup ledger. Every finding below was
verified against source with file:line evidence and confirmed absent from
`.claude/masterplan.json`, `design_pack_73/`, `frontier_map_73.md`,
`money_diagnosis_72.md`, and `harness_proposals.json` at audit time.
Convert findings into masterplan steps (phase-74+) via the full protocol;
mark the register row with the step id when adopted.

**ADVERSARIAL VERIFICATION (2026-07-18, ultracode dynamic workflow
`wf_1b92b344-335`, 26 independent refute-oriented verifiers, 26/26
returned):** 21 CONFIRMED, 3 PARTIAL (B4, D4, E1 -- corrections applied
inline below), 1 REFUTED (E5 -- streaming-buffer race does not exist; DML
MERGE path; residual non-atomicity note kept), 1 DUPLICATE (B3 -- covered
by masterplan step 61.5). Findings below carry a [VERDICT] tag; unmarked
claims inside a finding were verified as stated.

## A. Guard integrity (SEV-1 -- terminal-risk-guard defects)

- **A1. flatten_all lacks per-position error isolation + post-flatten
  verification.** `paper_trader.py:987-1013` loop has no try/except; on a
  kill-switch breach `check_and_enforce_kill_switch` (`:1099-1100`) calls
  `flatten_all()` THEN `state.pause()` -- one raising `execute_sell` unwinds
  the call, leaving the book neither flat nor paused. `execute_sell`
  returning None also lets a position silently stay open with no
  `get_positions()==[]` post-check. Enforcement call at
  `autonomous_loop.py:1263` is itself unguarded.
- **A2. Kill-switch pause not enforced at the execution primitives.**
  `execute_buy` (`paper_trader.py:123`) / `execute_sell` (`:391`) never check
  `is_paused()`; `signals_server.publish_signal`
  (`backend/agents/mcp_servers/signals_server.py:409-425`) executes a BUY
  after only `risk_check` -- trades can fire while paused. `risk_server.py:181`
  proves the gate exists but is unwired on this path.
- **A3. Safety-chain SPOF.** `autonomous_loop.py:1258-1279`: Step 5.5
  kill-switch and 5.6 stop-loss enforcement share one unguarded path; a
  single exception downs multiple independent safety primitives + snapshot.

## B. Execution realism (gates phase-68 go-live)

- **B1. Default bq_sim fills are frictionless.** `execution_router.py:83-125`
  `_bq_sim_fill` returns exact close; ADV/partial-fill logic (:103) reachable
  only from `shadow_submit` (:292-304), never from `submit_order` (:271-276);
  `paper_trader.py:273-276`/`:435-438` never pass `adv`. The only slippage
  model (0.3% fixed, `:139-141`) is unreachable in default mode.
- **B2. Fills/stops price off stale daily bars.** `paper_trader.py:1258-1277`
  `_get_live_price` uses `history(period="1d")` (prior session's close
  intraday) for sells/stops/kill-switch/MTM, while the dashboard uses 1m bars
  (`live_prices.py:112`) -- booked fills diverge from what the operator sees.
- **B3. Side-blind fills. [DUPLICATE -> step 61.5]** `execution_router.py:99`
  fills BUY and SELL at the same price (defect real, line exact), but
  masterplan step 61.5's "optional per-market half-spread bps"
  (masterplan.json:14563; audit_basis :14559 cites execution_router.py:85-126)
  already carries the remedy. Caveat: 61.5's spread model is config-gated
  default-OFF -- if the DEFAULT path must be side-aware unconditionally,
  extend 61.5's scope; do not open a new step.
- **B4. No price-staleness gate on the execution path. [PARTIAL --
  corrected]** Integrity check is default-OFF and pre-LLM only
  (`backend/services/autonomous_loop.py:2346`, enforcement :2350-2354 inside
  `_run_claude_analysis`); the phase-50.4 calendar gate never gates US
  tickers (comment :529, impl :538-539; whole block skipped when
  paper_markets==["US"], :522). CORRECTION: the automated loop is scheduled
  mon-fri (`paper_trading.py:1307`), so the unguarded automated-path
  exposure is WEEKDAY US MARKET HOLIDAYS (cron fires, gate returns True,
  yfinance serves prior close); weekend stale-close fills occur only via the
  manual/API triggers (`paper_trading.py:1031/1279/1355`). The only
  pre-trade price gate (phase-30.6, `paper_trader.py:173-202`) is
  default-OFF, BUY-only, divergence-based -- cannot catch uniform staleness.

## C. Corporate actions + data resilience

- **C1. Splits never re-base open positions.** Stored qty/avg-entry/stop
  (`paper_trader.py:363-374`) are raw; split-adjusted live prices produce a
  ~50% phantom loss on a 2:1 split -> spurious stop fire or full-book
  kill-switch flatten (`:653`, `:1092-1101`). No adjustment logic repo-wide.
- **C2. Delisted/halted tickers mark at a frozen price forever.**
  `paper_trader.py:564-565` falls back to last-known price; no age gate, no
  forced exit -- dead lines silently inflate NAV.
- **C3. Single-vendor yfinance SPOF.** All live pricing, benchmark, screener,
  ingestion paths are yfinance (unofficial, rate-limited); an outage routes
  every mark into C2's stale path with no second source and no circuit alarm.
- **C4. mark_to_market cost.** O(N) serial fetches + full SPY history refetch
  every cycle, uncached, growing daily (`paper_trader.py:561-563`, `:1304-1310`).
- **C5. Universe silently collapses to a hardcoded fallback** on any Wikipedia
  scrape change (`screener.py:29-61`) -- warning-only, operator never paged.

## D. Pipeline truth + efficiency (Layer-1/2)

- **D1. supply_chain agent NEVER runs.** `orchestrator.py:1406` reads
  `report.get("supply_chain", "No supply chain data.")` -- never assigned
  anywhere; `get_supply_chain_prompt` never called; yet `_inventory.json:56`
  + `agent_map.py:93` show it as a live Gemini agent. Synthesis permanently
  receives the placeholder string for every ticker.
- **D2. sector_catalyst synthesis slot fed PATENT text.**
  `orchestrator.py:1390` maps `report["patent"]["text"]` into
  `sector_catalyst_report` (:1426); real sector-catalyst prompt never
  invoked. Mislabeled evidence inside the decision-critical synthesis prompt.
- **D3. No per-agent signal->outcome attribution.**
  `signal_attribution.py:95-109` is a fixed BUY->1.0/HOLD->0.5 display
  heuristic; the only ablation harness targets ML quant features, not LLM
  agents. Nobody can say which of the 28 agents earn their tokens.
- **D4. Roster drift / mislabeled agents. [PARTIAL -- corrected]**
  bias_detector_skill (`_inventory.json:46`, `agent_map.py:85`) and
  info_gap_agent (`_inventory.json:63`, `agent_map.py:99`) are deterministic
  (zero LLM calls; live path runs `detect_biases()`/`detect_info_gaps()`,
  `orchestrator.py:38,44`) yet badged gemini-2.5-flash / layer1_swappable;
  `quant_strategy` is optimizer-only but counted as a pipeline agent
  (`_inventory.json:42,68`; `agent_map.py:102`). CORRECTION: conflict_detector
  is also deterministic but has NO inventory/agent_map node, so it is not
  mislabeled -- dropped from the list.
- **D5. Dead orchestrator methods.** `run_macro_agent` (:1121) and
  `run_alpha_decay_agent` (:1310) have zero callers; skills still enumerate.
- **D6. No Gemini context caching.** Fact-ledger + skill bodies re-billed
  fresh ~15-28x per ticker; `cache_control` exists only on the Claude path
  (`llm_client.py:1387`); gemini-2.5-flash explicit caching unused on the
  dominant-spend path.
- **D7. Deep-dive N+1 loop.** `orchestrator.py:1143-1153` -- serial
  per-question RAG calls with `time.sleep(2)`, unbounded by question count.
- (Cleared: enrichment/debate/synthesis output limits ARE honored.)

## E. Surface hardening (security / config / storage)

- **E1. POST /api/harness/monthly-approval is UNAUTHENTICATED. [PARTIAL --
  corrected]** `main.py:406-423` `_PUBLIC_PATHS` prefix-skips **16** paths
  (not 14) incl. the champion/challenger deployment approval
  (`monthly_approval_api.py:184-196` -- no route/router auth dependency,
  mutates approval state via `record_approval`); CORS admits the whole
  100.x tailnet with credentials (`main.py:399-400`). security.md documents
  only 5 skip-auth prefixes. No masterplan step tracks this.
- **E2. Core deps unpinned** (fastapi/pydantic/pandas/sklearn/cryptography
  all `>=`); Docker rebuild can silently pull breaking majors.
- **E3. Prod/CI Python drift.** Dockerfile `python:3.11-slim` vs CI + rules
  pinned 3.14.
- **E4. BQ retention absent.** `llm_call_log` / `harness_learning_log`
  partitioned but no partition_expiration; AI telemetry 10-50x volume.
- **E5. [REFUTED -- streaming-buffer race does not exist.]** paper_positions
  is written exclusively via DML (`save_paper_position` is a MERGE,
  `bigquery_client.py:593-632`; delete is DML `:634-640`; no
  insert_rows_json touches it), and DML rows never enter the streaming
  buffer -- the codebase adopted DML precisely to avoid this
  (`bigquery_client.py:558,661`; `_run_dml_with_retry:536-548`;
  phase-23.1.15 MERGE-upsert). RESIDUAL (minor, real): the redundant
  delete-then-MERGE pair (`paper_trader.py:593,611`) is non-atomic -- a
  crash between the two DML calls loses the position row; a bare MERGE
  alone would be atomic and sufficient.
- **E6. In-memory jobstores** on all three schedulers -- no crash-recovery
  state (`main.py:267,315`; `slack_bot/scheduler.py:224`).
- **E7. CORS regex vs 401-path emission drift** (`main.py:399` vs `:451`);
  http-only.
- (Cleared: no hardcoded secrets; tracked archive .env empty; misfire
  handling correct; loop-level kill-switch wiring present -- gaps are A1/A2.)
