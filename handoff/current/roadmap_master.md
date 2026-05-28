# Master Roadmap — production-ready + max money (2026-05-28)

Source: `diagnose-prod-and-money` workflow (6 agents) + phase-44.1 research gate.
Governs the multi-cycle run defined in `active_goal.md`. Companion: `ux_roadmap.md`.

## No-trades root cause (adjudicated)
- App emits BUY/SELL recs from the **Layer-1 Gemini pipeline** (on-demand ticker analysis).
- The **autonomous paper loop** (`autonomous_loop.decide_trades`) is a SEPARATE path. It
  builds orders from `candidate_analyses`, which is EMPTY because
  `new_candidates = screener_candidates - held_tickers` empties out (`autonomous_loop.py:658`)
  -> `analyze_tickers = []` -> 0 orders -> `n_trades:0` every cycle.
- **CORRECTION from the 44.1 research gate:** the live screener reads **yfinance live**
  (`autonomous_loop.py:324,372`), NOT `historical_prices` BQ. So stale BQ prices do NOT
  directly block the live loop; they starve the backtest/optimizer + freshness alarms.
- "GATE 0/5" = go-live PROMOTION gate (`paper_go_live_gate.py`), correctly red, NOT the trade blocker.

## Workstreams (money-impact-per-effort)
1. [MONEY] phase-44.1 Restore historical_prices freshness (3 RCs: wrong dest table; 5-ticker
   stub vs full universe; in-memory jobstore loses fires). Prereq for backtest re-baseline. CYCLE 1.
2. [MONEY] phase-44.0 First autonomous trade: fix empty new_candidates so analyze_tickers
   reaches decide_trades; roll sod_date; >=1 real trade. THE no-trades unblocker. CYCLE 2.
3. [OPUS-4.8] phase-44.2 cost_tracker.py missing claude-opus-4-8 pricing (~50x mis-cost
   regression from 8ecc9efe). Add ("claude-opus-4-8",(5.00,25.00)); run /claude-api sweep.
4. [MONEY] Close paper-vs-backtest Sharpe gap (589% all-time vs 30% thr): re-baseline on
   fresh prices + cost-model parity (slippage in paper_trader vs backtest_trader). DoD-2.
5. [MONEY/NORTH-STAR] Dynamic strategy rotation: per-strategy DSR over STRATEGY_REGISTRY,
   weekly promotion of top-DSR strategy to a promoted_strategy BQ table; today STATIC
   (triple_barrier, stamped 2026-04-06). Gated behind #1/#2.
6. [PROD] Learn-loop runtime evidence (outcome_tracking + agent_memories) — unblocks the
   moment #2 produces one sell-close. DoD-6 / phase-35.1.
7. [PROD] 5 consecutive clean cron cycles (DoD-9 / phase-35.3).
8. [PROD/UX] Operator full-control surface + consistent design (see ux_roadmap.md). Priority 7.
9. [HYGIENE] sod_date daily roll; orphaned "started" cycle rows; autoresearch
   langchain_huggingface (owner-gated pip, phase-39.1).
10. [HYGIENE] Doc CONDITIONAL->PASS (DoD-4 coverage wording, DoD-11 deferrals, DoD-14 OWASP).

## Opus 4.8 adoption shortlist
App-side (anthropic==0.87.0, available today):
- cost_tracker.py claude-opus-4-8 pricing — NEW, REGRESSION, fix now (phase-44.2).
- Per-agent max_tokens audit at xhigh (Enrichment 1024 / Synthesis 4096 may truncate reasoning).
- Already adopted: adaptive thinking (llm_client.py:1375), effort pass-through (model_tiers +
  llm_client:1409), structured outputs / batch / prompt caching / 1M context (llm_client:1469/1733/1192).
- NEW low-pri: thinking.display='summarized'; stop_reason=='refusal' branch.
- DEFER to API-key migration: mid-conversation role:'system' messages; context-editing + memory tool.
- DO NOT adopt: fast mode (speed:'fast') — hurts Net System Alpha on batch cron; agent-teams (breaks 3-agent L3).
Coding-env-side (.claude/):
- Run /claude-api migration skill against llm_client.py + model_tiers.py + cost_tracker.py.
- Already adopted: effortLevel=xhigh (Main); model:opus+effort:max (Q/A, Researcher);
  .mcp.json alwaysLoad / continueOnBlock / $CLAUDE_EFFORT.

## Test -> API-key migration checklist
- cost_tracker.py 4.8 pricing correct (else day-1 telemetry 50x wrong).
- model_tiers.py::_LIVE_TIER populated (today TODO_DECIDE_AT_LAUNCH sentinels that raise).
- Re-justify effort:max / xhigh / Opus-4.8 per-token (flat-fee rationale gone off Max) — owner approval.
- Per-agent max_tokens audited; 1M-context surcharge confirmed; TPM/RPM rate limits sized.
- Betas plumbing if context-editing/memory adopted.
- System actually trading + passing DoD gates FIRST (don't migrate cost model at n_trades=0).

## Open questions (resolved/standing)
1. Empty candidate set: held-book saturation vs screener short-circuit — resolve in phase-44.0 (check open positions vs paper_max_positions).
2. daily_price_refresh destination: Option A (redirect write to historical_prices) recommended; price_snapshots has ZERO other consumers (researcher confirmed) -> safe.
3. Dynamic rotation: defer until 44.0 produces sustained trades.
4. autoresearch pip: owner-gated; ask before pip install langchain_huggingface.
