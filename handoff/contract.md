# Phase 3.0: MCP Server Architecture — Contract

## Hypothesis
Building MCP servers for pyfinAgent's data, backtest, and signals capabilities will enable Claude (Planner & Evaluator agents) to directly interact with our system, improving research quality and reducing manual prompt engineering.

## Success Criteria (Research-Backed)

### Primary Criteria
1. **Three functional MCP servers deployed:**
   - `pyfinagent-data`: read-only access to BigQuery cache (prices, fundamentals, macro)
   - `pyfinagent-backtest`: backtest engine + experiment queries
   - `pyfinagent-signals`: signal generation pipeline
   - ✅ Minimum: data server works, Planner can query experiments

2. **Claude Messages API MCP connector integration:**
   - Messages API accepts mcp_servers parameter with our three servers
   - Tools correctly exposed via mcp_toolset type
   - Authentication via OAuth token or shared secret working

3. **Planner agent operational:**
   - Claude can query experiment history via `get_experiments` tool
   - Can identify parameter patterns via `compare_params` tool
   - Proposes testable research direction with reasoning
   - Max 2-3 LLM calls per optimization cycle (cost: $2-5/cycle)

4. **Evaluator agent operational (stretch):**
   - Claude can run `run_subperiod` on suspicious Sharpes
   - Can run `run_ablation` to check parameter contribution
   - Provides skeptical critique matching Phase 2 evaluator rigor
   - Rejects overfit patterns with specific evidence

### Secondary Criteria
- [ ] Authentication patterns documented (for Phase 4 expansion)
- [ ] Streaming HTTP transport working (Streamable protocol)
- [ ] Error handling + retry logic in MCP clients
- [ ] Tool definitions match Claude schema expectations

## Fail Conditions
1. Claude Messages API doesn't accept mcp_servers parameter (API version mismatch)
2. Tool calls not executing properly (schema mismatch, auth failure)
3. LLM reasoning too expensive (>$10/cycle) — not economically viable
4. Reliability <95% (frequent timeouts, dropped connections)

## Timeline
- **Research:** 2-4 hours (MCP spec, examples, best practices)
- **Generation:** 4-6 hours (server implementation, deployment)
- **Evaluation:** 1-2 hours (Claude integration, test calls, cost tracking)
- **Total:** ~12-16 hours (can span 2-3 calendar days with breaks)

## Budget Impact
- **MCP servers:** $0 (run on existing backend infrastructure)
- **Claude API calls:** ~$2-5/optimization cycle (~20 cycles/month = $40-100/month)
- **Hosting:** no additional cost (FastAPI on port 8000, same machine)

## References
- [Anthropic MCP Documentation](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)
- Claude Messages API beta: `mcp-client-2025-11-20` (or later)
- FastAPI + Streamable HTTP transport for MCP server implementation
