# Phase 3.0 — MCP Server Architecture (backtest + signals plumbing)

## Step
masterplan.json `phase-3` step `3.0` — "MCP Server Architecture"

## Hypothesis
The Phase 3.0 MCP servers (`data_server.py`, `backtest_server.py`, `signals_server.py`) need their TODO stubs replaced with deterministic, side-effect-free plumbing so downstream MAS / Phase 3.1+ work has a stable tool surface to call. Prior session finished `data_server.py`. This contract finishes the deterministic subset of `backtest_server.py` and `signals_server.py` and intentionally defers the model-bound / network-bound subset to Phase 3.1+.

## Scope (in)
1. **`backtest_server.py`**
   - L32 `logger.warning("... — backtest server in stub mode")` em-dash -> `--` (security ASCII rule, defense-in-depth).
   - `get_experiment_list(last_n=None)` — port the stdlib-csv block from `data_server.py:280-356`. Mirror signature, mirror `_to_float` helper, mirror `params_json` JSON parse, mirror `last_n` tail slice.
   - `get_recent_experiments(limit=10)` — implement as a thin delegate: `return self.get_experiment_list(last_n=limit)`. Do not duplicate parsing.

2. **`signals_server.py`**
   - L34 `logger.warning("... — signals server in stub mode")` em-dash -> `--`.
   - `validate_signal(signal: Dict[str, Any])` — signal-intrinsic schema validation. Tolerates partial/missing fields via `.get(...)` — never raises.
   - `risk_check(portfolio: Dict[str, Any], proposed_trade: Dict[str, Any])` — portfolio-extrinsic constraint check. Reads its limits from `get_risk_constraints()` so the threshold table stays single-source-of-truth.

## risk_check evaluation order (canonical FINRA 15c3-5 / FIA whitepaper ordering)
Fail-fast on hard violations, collect soft violations:
1. Schema sanity — `proposed_trade` has `ticker`, `action` in `{"BUY","SELL"}`, `shares > 0`. Hard.
2. Action/state consistency — `SELL` requires existing position with `>= shares`. Hard.
3. Daily trade count — `trades_today < max_daily_trades (5)`. Hard.
4. Per-ticker concentration — projected position notional / total_value <= `max_exposure_per_ticker_pct (10%)`. Hard.
5. Total exposure — sum(positions notional) + proposed notional <= `max_total_exposure_pct (100%)`. Hard.
6. Cash availability — proposed BUY notional <= cash (paper trader has no margin). Hard.
7. Drawdown circuit breaker — if `current_drawdown_pct <= max_drawdown_pct (-15%)`, block all BUYs but allow SELLs (de-risking). Hard for BUY only.

Returns the same shape the stub already advertises so callers don't break. Does NOT call any other MCP server (cross-server coupling is anti-pattern per Anthropic MCP best practices).

## Scope (out, intentional)
- `run_single_feature_test`, `run_ablation_study` — need model-retraining infra not present in dev env; tracked for Phase 3.4.
- `generate_signal` — needs model inference + feature pipeline (Phase 3.2).
- `publish_signal` — needs Slack creds + paper trader commit path (Phase 4.1).
- `get_signal_history` — requires BQ `signals_log` query; Phase 4.2.
- `harness_memory.py`, `multi_agent_orchestrator.py` — already done in prior sessions.
- Cross-server calls — forbidden by MCP best practices.
- Market-hours check inside `risk_check` — would break batch backtests.
- Liquidity-floor check — needs volume data signals_server doesn't own.

## Success criteria (research-backed)
1. **Static syntax**: `ast.parse(...)` clean for both touched files.
2. **Compile**: `py_compile` clean for both.
3. **AST logger ASCII scan**: 0 non-ASCII Constant subtrees inside any `logger.{info,warning,error,debug,critical,exception}` call in either file.
4. **TODO clearance**: 0 occurrences of `TODO:` in `get_experiment_list`, `get_recent_experiments`, `validate_signal`, `risk_check` method bodies.
5. **Behavioral self-check** (stdlib only):
   - `validate_signal({"ticker":"AAPL","signal":"BUY","confidence":0.7,"date":"2026-04-14","factors":["m"]})["valid"]` -> True
   - Missing ticker -> False; signal=`MAYBE` -> False; confidence=1.5 -> False
6. **risk_check determinism**:
   - Empty portfolio + 100-share BUY at $10 against $10000 portfolio: `allowed=True`
   - Same trade with `trades_today` length 5: `allowed=False`, conflicts contain `"max_daily_trades"`
   - SELL of 100 shares with no position: `allowed=False`, conflicts contain `"insufficient_position"`
7. **No new heavyweight imports** (no pandas/numpy on MCP path).
8. **Diff bound**: < 350 added lines across both files. No deletions outside the TODO method bodies and the L32/L34 em-dash sites.

## Fail conditions
- Any non-ASCII string surfacing in a logger call after the change.
- `risk_check` mutates input args.
- `risk_check` raises on missing keys instead of returning a violation.
- New cross-server import of `data_server` or `backtest_server` from `signals_server`.
- Signature breakage of any method already called by `create_*_server()` factories.
- Adding pandas/numpy import to the MCP path.

## Verification commands
```bash
python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/backtest_server.py').read())"
python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())"
python3 -m py_compile backend/agents/mcp_servers/backtest_server.py
python3 -m py_compile backend/agents/mcp_servers/signals_server.py
```

AST logger ASCII scan + behavioral smoke (stub mode is fine; we are not exercising BQ): see experiment_results.md for the full scripts and recorded output.

## Anti-leniency notes for QA evaluator
- Reject if any `risk_check` predicate is implemented in a different order than 1..7 above without explicit justification — order is taken from FINRA 15c3-5 + FIA whitepaper.
- Reject if any logger string anywhere in the touched files contains characters with `ord > 127`.
- Reject if `get_recent_experiments` re-implements the TSV parse instead of delegating.
- Reject if `validate_signal` raises on missing keys.
- Reject if pandas / numpy is imported on the MCP code path.
- Accept if behavioral smoke checks above all return the expected booleans.

## Research provenance
Researcher subagent run completed with 14 URLs across 5 categories (Anthropic MCP docs, FastMCP framework docs, FINRA / SEC 15c3-5 regulatory, FIA / QuestDB / Sterling practitioner whitepapers, QuantConnect platform docs). Key citations driving design:
- Anthropic MCP best practices: deterministic predicates belong in server code, not LLM loop.
- FINRA 15c3-5 + SEC FAQ: regulatory-fatal -> financial-fatal -> soft order, fail-fast on hard violations.
- QuantConnect Algorithm Framework: layered Alpha -> PortfolioConstruction -> RiskManagement -> Execution; `validate_signal` and `risk_check` are complementary, not redundant.
- FIA Best Practices Automated Trading Risk Controls: TOCTOU between validate and execute is a known acceptable risk for paper trading.
- FastMCP Tools docs + Advanced Patterns: keep tools side-effect-free for retry/cache safety.

## Timestamp
Authored: 2026-04-14 by Ford (Opus 4.6, remote agent), session bootstrapped from prior session log `2026-04-14-0500.md`.
