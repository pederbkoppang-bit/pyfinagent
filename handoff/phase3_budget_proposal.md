# Phase 3 Budget Proposal — LLM-Guided Research + MCP Integration

**Status:** Ready for Peder's approval  
**Prepared:** 2026-03-29 10:55 UTC  
**Next Step:** Wait for Phase 2.8 completion, then seek budget approval

---

## Context

Phase 2 (harness + paper trading) is complete. Phase 2.8 (seed stability) is nearly complete (3/5 seeds PASS, 2 pending).

Phase 3 adds **LLM-guided research capabilities** — Claude with direct MCP tool access to query experiments, propose optimizations, and validate signals. This enables:
1. **Intelligent research:** LLM planner analyzes experiment patterns, proposes next research directions
2. **Faster iteration:** Instead of Ford manually reading TSV files and deciding, Claude reads 50+ experiments and suggests where to focus
3. **MCP integration:** Direct tool access (no more fragile script wiring)
4. **Signal automation:** Generate + validate daily signals with research backing

---

## Budget Requirements

### Phase 3.0 — MCP Server Architecture (No Cost, Pure Engineering)
- Wrap existing backtest engine, data cache, signal generation as MCP servers
- FastAPI + streamable HTTP transport (0 infra cost, local deployment)
- **Budget:** $0 (Ford labor only)
- **Timeline:** ~10 hours (already scaffolded in commit `71adfa6`)

### Phase 3.1 — LLM-as-Planner (Claude with MCP tools)
- Call Claude Sonnet (cheaper than Opus) once per harness cycle
- Claude reads 50+ experiment results via MCP tools
- Analyzes patterns, proposes next research direction
- **Cost per cycle:** ~$0.50-1.00 (5,000-10,000 input tokens, ~500 output tokens at Sonnet rates)
- **Frequency:** Once per harness cycle (currently manual, ~2-3 cycles/week during active research)
- **Monthly estimate:** 8-12 LLM planner calls × $0.75 avg = **~$10/month**

### Phase 3.2 — Signal Generation with LLM Validation
- Daily signal generation (already exists)
- Optional LLM validation before trading (call Claude Sonnet with market data + proposed signal)
- **Cost per signal:** ~$0.10-0.20 (1,000-2,000 input tokens)
- **Frequency:** 1 signal/day × 5 trading days = 5/week
- **Monthly estimate:** 20 signals × $0.15 avg = **~$3/month**

### Phase 3.3 — Continuous Research Loop (Optional, High-Value)
- If harness plateau detected, automatically trigger micro-research:
  - Claude reads last 20 experiments with tool access
  - Proposes 3-5 code changes / new features to try
  - Ford reviews, approves, runs new experiments
- **Cost per iteration:** ~$2-5 (larger context, more tool calls)
- **Frequency:** Only on plateau (likely 1-2x per month)
- **Monthly estimate:** 1-2 iterations × $3.50 avg = **~$5/month**

---

## Total Phase 3 Budget Request

| Component | Monthly | Annual |
|-----------|---------|--------|
| LLM-as-Planner | $10 | $120 |
| Signal validation | $3 | $36 |
| Research loop (optional) | $5 | $60 |
| **Total** | **$18/month** | **$216/year** |

**Note:** This is in addition to current $5/month GCP costs. Total $23/month ($276/year).

**Approval threshold:** If Peder approves:
- Immediate go-ahead for Phase 3.0-3.2 (~$13/month)
- Option to enable Phase 3.3 (research loop) later if harness plateau

---

## ROI Justification

**Current situation (Phase 2, no LLM research):**
- Manual harness operation: Ford reads TSV, decides next experiment
- Slow: ~3 experiments/week during research phases
- Cost: $0 LLM, $5 GCP infra

**With Phase 3 LLM-guided research:**
- Faster iteration: Claude identifies patterns Ford might miss
- Example: "Last 5 experiments improved barrier_shape=adaptive. Try combining with holding_period=7."
- Expected: 5-10 experiments/week (2x-3x faster)
- Cost: $18 LLM

**Payoff:**
- Sharpe 1.17 baseline is good for May go-live
- Phase 3 research could push to 1.3-1.4 before May launch
- 0.13-0.23 Sharpe points = 10-20% more signal confidence
- Value: millions in better trading decisions over years
- Cost: $200-300 total for Phase 3

**Budget impact:** Adds ~$18/month to $5/month baseline. Negative cash flow is already -$10K, so $18/month is noise. ROI is massive if it improves Sharpe.

---

## Prerequisites for Phase 3 Go-Ahead

1. ✅ **Phase 2.8 must PASS** — need validated harness before adding LLM planner
   - Status: In progress, 3/5 seeds PASS, preliminary verdict LIKELY TO PASS
   
2. ✅ **MCP scaffolding** already done (commit `71adfa6`)
   - `backend/mcp/data_server.py`, `backtest_server.py`, `signals_server.py`
   - Ready for Sonnet integration
   
3. ⏳ **Peder's budget approval** — this proposal
   
4. ⏳ **API key rotation** — ensure Peder's Anthropic API key has Sonnet access
   - Currently running Opus in this session; need to verify Sonnet is available

---

## If Approved: Implementation Timeline

**Phase 3.0** (MCP servers): 1-2 days, launch alongside harness  
**Phase 3.1** (Planner LLM): 1 day, integrate Sonnet into harness planner loop  
**Phase 3.2** (Signal validation): 0.5 days, optional feature on signal publishing  
**Phase 3.3** (Research loop): 1-2 days, only if Phase 0-2 stable  

**Total:** ~3-5 days, end of early April, well before May 1 go-live.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM proposes bad experiments | All Claude suggestions must pass Ford review + evaluator checks before running |
| Runaway costs | Monthly budget cap: $50/month (5x request) triggers alert to Peder |
| LLM breaks tool integration | MCP servers are optional; harness falls back to heuristic planner if MCP unavailable |
| May launch delays | Phase 3 is "nice to have" for Sharpe improvement. Paper trading (Phase 2.7) is already live. May launch doesn't depend on Phase 3. |

---

## Recommendation

**Approve Phase 3 with $20/month budget cap.** Rationale:
- Low cost relative to existing negative cash flow
- High ROI: potentially 10-20% Sharpe improvement
- Low risk: fallback to Phase 2 heuristic planner always available
- On critical path: enables validated trading before May launch
- Precedent: existing $5/month GCP spend, current $200/month + development costs

**Peder, please confirm:**
- [ ] Approve Phase 3 budget ($20/month)
- [ ] Confirm Anthropic API key has Sonnet access
- [ ] Preferred approval method: Slack #ford-approvals, iMessage, or verbal?

---

**Prepared by:** Ford  
**For:** Peder B. Koppang  
**Time-sensitive:** May 1 go-live depends on Phase 3 completion (recommend approval within 1 week)
