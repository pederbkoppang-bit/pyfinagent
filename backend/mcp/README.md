# pyfinAgent MCP Servers (Phase 3)

Three separate MCP servers expose pyfinAgent's capabilities to Claude via the MCP connector.
Each runs on a separate port for isolation and granular permission control.

## Architecture

```
┌──────────────────────────────────────────┐
│   Claude (with MCP Connector Beta)       │
│                                          │
│   mcp_servers: [                         │
│     pyfinagent-data (port 8101),         │
│     pyfinagent-backtest (port 8102),     │
│     pyfinagent-signals (port 8103)       │
│   ]                                      │
└──────────────────┬───────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    [Port 8101] [Port 8102] [Port 8103]
    data server  backtest    signals
                 server      server
```

## Servers

### 1. pyfinagent-data (Port 8101)
**Purpose:** Read-only data queries (experiments, parameters, metrics).

**Tools:**
- `get_experiments(limit, status, order_by)` → List backtests from experiments DB
- `get_best_params()` → Current best parameters (from optimizer_best.json)
- `get_sharpe_history(window_days)` → Sharpe progression timeline
- `get_portfolio_state(market)` → Current paper trading portfolio

**Security:** Read-only. No state mutation.

### 2. pyfinagent-backtest (Port 8102)
**Purpose:** Run backtests on-demand (controlled access).

**Tools:**
- `run_backtest(params, start_date, end_date)` → Execute full walk-forward backtest
- `run_sub_period_test(params, period)` → Test single sub-period
- `estimate_backtest_time(params)` → Query cost (min, Gemini tokens if applicable)

**Security:** Requires explicit approval. Tracks cost before running. Rate-limited: 1 backtest/min.

### 3. pyfinagent-signals (Port 8103)
**Purpose:** Generate and validate trading signals (read-mostly, gated writes).

**Tools:**
- `generate_signals(date, top_n)` → ML model + screener → signals
- `validate_signal(ticker, direction, confidence)` → LLM gate (this server calls Claude API)
- `get_signal_log(days)` → Historical signal accuracy
- `approve_signal(signal_id)` → Mark signal for paper trading

**Security:** Generate requires market hours (no overnight). Validate calls Claude API (cost gated).

## Deployment (Phase 3)

Each server runs as:
```bash
python -m backend.mcp.data_server      # Port 8101
python -m backend.mcp.backtest_server  # Port 8102
python -m backend.mcp.signals_server   # Port 8103
```

Exposed to Claude via:
```json
{
  "mcp_servers": [
    {
      "name": "pyfinagent-data",
      "url": "http://localhost:8101",
      "auth": { "type": "bearer", "token": "${MCP_SHARED_SECRET}" }
    },
    // ... others
  ]
}
```

## Phase 3 Harness

1. **RESEARCH:** Anthropic MCP spec, LLM decision-making patterns, tool design best practices
2. **PLAN:** Contract with success criteria (LLM Planner must improve exploration over heuristic planner)
3. **GENERATE:** Build 3 servers, integrate with Claude API
4. **EVALUATE:** Heuristic planner vs LLM planner: which explores better? Which converges faster?
5. **DECIDE:** Scale to full LLM governance (PASS) or keep heuristic (FAIL) or hybrid (CONDITIONAL)

## Cost Model

- **Data queries:** Free (BQ fallback if down)
- **Backtest runs:** ~$0.10-0.30 per run (BQ cost + ML)
- **Signal validation:** ~$0.01-0.05 per signal (Claude API)
- **Gate:** Daily budget of $1.00 for signals + backtests

## Future Extensions

- **Phase 4:** Add `signals_server` to publish validated signals to Slack
- **Phase 5:** Multi-market support (routing by market code)
- **Phase 6:** Distribute MCP servers across machines (currently localhost)
