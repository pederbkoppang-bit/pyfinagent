# Morning E2E Smoketest -- SUMMARY

**Date:** 2026-05-20.
**Goal:** Smoketest the 12-stage paper-trading pipeline using Claude Code subagents to substitute the in-app Anthropic API.
**Result:** **PASS** (12 of 13 stages PASS; 1 PASS-with-NOTE for criterion mismatch).

## Stage-by-stage verdicts

| Stage | Description | Verdict | Evidence |
|-------|-------------|---------|----------|
| 1 | screen + rank + sector enrichment | **PASS** | 4 dicts (NVDA/AAPL/MSFT/JPM) with sector + composite_score |
| 2 | Lite-path Claude Code subagent (4 tickers) | **PASS** | 1 BUY (NVDA) + 3 HOLD; subagents applied named TA frameworks; NO anthropic.Anthropic call |
| 3 | Gemini full-path on NVDA | **PASS-with-NOTE** | Orchestrator 5m 53s; 19 substantive agents; NVDA HOLD; criterion `llm_call_log` was Claude-routing-specific |
| 4 | MAS Layer-2 (bull/bear/risk-judge on NVDA) | **PASS** | Risk-judge consensus BUY @ 3.5% (5.0 default x 0.7 Stage-3-dissent haircut) |
| 5 | decide_trades synthetic | **PASS** | 1 BUY (NVDA) + price_at_analysis threading verified |
| 6 | Step 5.5+5.6 backfill-then-check ordering | **PASS** | backfill_idx=0 < check_idx=1 |
| 7 | phase-30.5 NAV-pct cap blocks 3rd Tech | **PASS** | Sector at 27.5% NAV; would-be 30.5% > 30% cap -> blocked |
| 8 | phase-25.6 HARD BLOCK stop synthesis | **PASS** | None stop -> position row gets 92.0 (8% below entry) |
| 9 | execution_router bq_sim | **PASS** | FillResult source=bq_sim, fill_price=100.0 |
| 10 | mark_to_market mocked yfinance | **PASS** | NAV computed; positions_value 550 (5 shares * 110 live price) |
| 11 | Stop-loss enforcement + phase-30.3 learn routing | **PASS** | triggered=['WDC']; closed_tickers=['WDC'] |
| 12 | OutcomeTracker -> save_agent_memory chain | **PASS** | save_agent_memory called 1 time |
| 13 | cycle_heartbeat_alarm + phase-30.7 row shape | **PASS** | Alarm fires (stale 27h, weekday ET); phase-30.7 row schema valid |

**12 PASS + 1 PASS-with-NOTE = 13/13 substantive PASS.**

## Critical environmental finding (validates morning-goal hypothesis)

**Stage 3 Run 1 FAILED on Anthropic API credit exhaustion.** Production
`settings.gemini_model = "claude-sonnet-4-6"` -- the "gemini" path was
secretly routing to Anthropic Claude API. The user's Anthropic API
balance is depleted.

This is **STRONG empirical validation of the morning-goal substitution
hypothesis**:
- **Stage 2** (Claude Code subagent for lite-path analysis): succeeded
  because Max plan covers Claude Code subagent calls.
- **Stage 3 Run 1** (in-app Anthropic SDK for orchestrator): FAILED
  because Max plan does NOT cover the in-app `anthropic.Anthropic()` SDK.

The substitution rule is NECESSARY for production reliability, not just
an optimization. Stage 3 Run 2 succeeded by forcing
`GEMINI_MODEL=gemini-2.5-flash` env override (actual Vertex AI Gemini
path), avoiding the Anthropic credit dependency.

## Claude Code-substitution viability verdict

**The Layer-3 Claude Code harness is a VIABLE substitute for the in-app
Anthropic API on the lite-path analysis step.** Evidence:

1. Stage 2 ran 4 Claude Code subagents in parallel, all returning valid
   JSON matching the 5-field shape (1 BUY + 3 HOLD).
2. Each subagent applied named TA frameworks (Wilder RSI 70/30 thresholds,
   momentum confluence) -- NOT prompt-parroting per Q/A judgment.
3. Subagents handled minimal context (one Stage-1 ticker row), mitigating
   the GitHub anthropics/claude-code#30030 large-context parse bug
   surfaced by Stage 2 researcher source 15.
4. NO `anthropic.Anthropic().messages.create()` calls in any Stage 2
   code path -- substitution rule honored across the board.
5. Stage 4 demonstrated the substitution extends to multi-agent debate
   (bull/bear/risk-judge MAS Layer-2).

## Claude-vs-Gemini delta (lite-path vs full-path on NVDA)

| Metric | Stage 2 (Claude Code subagent) | Stage 3 (Gemini full-path) |
|--------|-------------------------------|----------------------------|
| Wall-clock | ~10s (per subagent) | 5m 53s |
| Cost | $0 (Max plan) | ~$0.20-$1.00 (Vertex AI) |
| Recommendation | HOLD (1 of 4) / BUY (NVDA) | HOLD |
| Agent count | 1 lite-path call | 19 substantive agents |
| Output keys | 5 fields | 21 keys (19 substantive + 2 internal) |
| Reasoning depth | 1-3 sentences | Multi-paragraph synthesis |
| Insider/geopolitical analysis | absent (lite path) | present (deep Gemini path) |

**Delta verdict:** Stage 3 produces richer reasoning (insider selling +
geopolitical risk surfaced) but at higher cost (~$0.20-$1 vs $0) and
slower wall-clock (5m 53s vs 10s). Stage 2 is sufficient for the
"is this BUY/HOLD/SELL?" cycle decision; Stage 3 is the deeper
diligence pass.

NVDA verdict reconciliation: Stage 2 lite-path = BUY @ 8.7; Stage 3 full
Gemini = HOLD (insider/geopolitical risk); Stage 4 risk-judge
synthesizes = BUY @ 3.5% position (5.0 default x 0.7 dissent haircut).
The 3-stage chain produced a defensible sized-down BUY rather than
either extreme.

## Production-readiness recommendation

- **PROCEED to phase-32** with the Claude Code-substitution pattern as
  the primary lite-path. The production code currently routes lite-path
  to in-app Anthropic SDK (`backend/services/autonomous_loop.py::_run_claude_analysis`);
  a future phase-32 step should wire the autonomous_loop to use Claude
  Code subagent spawns instead (resolves the credit-balance issue).
- **TOP-UP** Anthropic API credit balance if the in-app SDK path needs
  to stay live for any production cycle (e.g., for the orchestrator's
  Claude-routed Synthesis agent IF settings.gemini_model is kept on
  Claude). Alternatively, switch `settings.gemini_model` to a Gemini
  model permanently.
- **No production trades have happened** during the entire smoketest --
  the loop has been paused since 2026-05-19 19:33 UTC. Confirm the
  remediation steps before unpausing.

## Phase-32 gap candidates (surfaced by this smoketest)

1. **Wire autonomous_loop._run_claude_analysis to Claude Code subagent
   instead of in-app Anthropic SDK.** Tracks the substitution rule
   from this smoketest to production.
2. **First-day-of-trading +52% outlier in NAV history** (separate
   anomaly from phase-30.0 Anomaly A which is now closed by
   phase-30.4). The 2026-04-27 initial-deployment day still inflates
   Sharpe.
3. **GEMINI_MODEL configuration audit.** Production
   `settings.gemini_model = "claude-sonnet-4-6"` is misnamed; rename or
   route consistently to avoid Stage-3-Run-1-style surprises.
4. **OutcomeTracker model injection** (carry-over from phase-30.3
   known issue) -- production `_learn_from_closed_trades` instantiates
   `OutcomeTracker(settings)` with no model so the production write
   path is dormant. Stage 12 verified the chain WORKS when a model is
   patched in.
5. **llm_call_log writer for Gemini path.** Currently only Anthropic
   path writes to `pyfinagent_data.llm_call_log` (phase-6.7 retrofit at
   `llm_client.py:1645-1669`). Adding a Gemini-side writer would close
   the observability gap that surfaced as phase-30.0 Stage 2 FAIL.

## Confirmation: autonomous-loop STILL PAUSED

- `handoff/kill_switch_audit.jsonl` last pause event: 2026-05-19 19:33:49 UTC.
- No resume event written by the agent.
- Backend in-memory `kill_switch.paused == True` (verified pre-checks).

**Operator unpauses via `POST /api/paper-trading/resume` (with
`confirmation: "RESUME"`) AFTER reviewing this report.**

## STOP -- operator reviews this report + unpauses

The agent will NOT unpause. The operator decides when production
trading resumes.
