# Phase 3.0 Contract: MCP Server Architecture

**Date:** 2026-03-29 09:50 UTC

**Research Gate:** ✅ PASSED (10+ sources, 3 read in full)

---

## Hypothesis

The Model Context Protocol (MCP) enables Claude to autonomously improve pyfinAgent's trading strategy by providing direct access to:
1. **Data** (prices, fundamentals, features via read-only resources)
2. **Computation** (backtesting, feature validation via callable tools)
3. **Actions** (signal generation, risk validation via callable tools)

With MCP servers as "universal adapters," Claude becomes a research agent that can:
- Propose new features based on data analysis
- Immediately test features on historical data
- Validate results independently
- Generate trading signals in real-time

This mirrors the Anthropic harness pattern (Planner → Generator → Evaluator) but with MCP enabling direct data/tool access instead of file-based handoffs.

---

## Success Criteria (Research-Backed)

1. **All 3 MCP servers implemented** ✅ Checklist
   - [ ] `pyfinagent-data` server (FastMCP): 7+ resources working
   - [ ] `pyfinagent-backtest` server (FastMCP): 4+ tools working
   - [ ] `pyfinagent-signals` server (FastMCP): 4+ tools working

2. **Integration with Claude** ✅ Checklist
   - [ ] Each server responds to MCP protocol correctly (JSON-RPC)
   - [ ] Claude SDK can discover and call all tools/resources
   - [ ] Type hints auto-generate proper JSON schemas
   - [ ] Docstrings appear as tool/resource descriptions to Claude

3. **Performance Targets** ✅ Checklist
   - [ ] Data resources: <5 second latency (prices, fundamentals)
   - [ ] Backtest tools: <30 second latency (full 27-window backtest)
   - [ ] Signal tools: <10 second latency
   - [ ] No memory leaks (servers stay responsive after 100+ calls)

4. **Security Audit** ✅ Checklist
   - [ ] No private data exposure (portfolio holdings stay private)
   - [ ] Input validation: ticker exists, dates in range, params reasonable
   - [ ] Error handling: graceful failures, no stack traces to Claude
   - [ ] Audit logging: all tool calls + resource accesses logged with timestamp
   - [ ] User consent flow: Claude must ask before running expensive operations

5. **Integration Test** ✅ Checklist
   - [ ] Mock Claude client can connect to all 3 servers
   - [ ] Mock client can query data resource → get valid OHLCV
   - [ ] Mock client can call backtest tool → get valid results
   - [ ] Mock client can call signal tool → get buy/sell decision
   - [ ] All results are interpretable (Claude can understand and act on them)

---

## Fail Conditions

- Any server crashes on malformed input
- Backtest tool hangs for >45 seconds
- Resource queries return stale/incorrect data
- Security audit fails (private data leaked, no consent check)
- Integration test fails to complete

---

## Deliverables

1. **Three FastMCP server implementations**
   - `backend/agents/mcp_servers/data_server.py` (pyfinagent-data)
   - `backend/agents/mcp_servers/backtest_server.py` (pyfinagent-backtest)
   - `backend/agents/mcp_servers/signals_server.py` (pyfinagent-signals)

2. **Server configuration**
   - `backend/agents/mcp_servers/__init__.py` (startup logic)
   - `backend/config/mcp_config.py` (server URIs, timeouts, auth)

3. **Integration tests**
   - `tests/test_mcp_servers.py` (unit tests for each server)
   - `tests/test_mcp_integration.py` (end-to-end with mock Claude client)

4. **Documentation**
   - `docs/MCP_ARCHITECTURE.md` (overview, design decisions)
   - `docs/MCP_SECURITY.md` (security considerations, audit checklist)

5. **Handoff artifacts**
   - `handoff/phase30_experiment_results.md` (what was built)
   - `handoff/phase30_evaluator_critique.md` (PASS/FAIL + evidence)

---

## Timeline & Effort

**Estimated: 20-30 hours over 2-3 work days**

1. **Day 1: Setup + pyfinagent-data (8-10 hours)**
   - FastMCP skeleton
   - Data resources (prices, fundamentals, macro)
   - Unit tests
   - Performance baseline

2. **Day 2: pyfinagent-backtest + pyfinagent-signals (8-10 hours)**
   - Backtest tools (run_backtest, ablation, feature_test)
   - Signal tools (generate_signal, validate, publish)
   - Unit tests for each
   - Error handling + logging

3. **Day 3: Integration + security audit (4-10 hours)**
   - Integration test (mock Claude client)
   - Security audit (consent flows, data privacy)
   - Performance profiling
   - Documentation

---

## Dependencies

- ✅ FastMCP package (pip install fastmcp)
- ✅ Claude SDK (for integration testing)
- ✅ Existing pyfinAgent backend (data access, backtest engine)
- ⏳ Peder's budget approval (Phase 3.1+ costs $20-50/month)

---

## Success Decision Framework

**PASS** if:
- All 5 success criteria checklists ✅
- No fail conditions triggered
- Performance targets met (latencies <30s)
- Security audit signed off
- Integration test PASSES

**CONDITIONAL** if:
- 4/5 criteria met, 1 minor gap (easy fix)
- Performance slightly over targets (e.g., backtest 35s instead of 30s)
- Security audit flags non-critical issue

**FAIL** if:
- Any server crashes
- Security audit fails (data leak)
- Integration test cannot complete
- >2 fail conditions triggered

---

## After Phase 3.0 PASS

Immediately proceeds to:
- **Phase 3.1: LLM-as-Planner** — Claude uses MCP data/tools to propose research
- **Phase 3.2: LLM-as-Evaluator** — Claude validates generator's work via MCP

Both require Peder's budget approval ($2-5 per cycle for Claude API calls).

---

**Contract prepared by:** Ford (automated)
**Date:** 2026-03-29 09:50 UTC
**Status:** Ready for implementation
**Next:** Begin Phase 3.0 GENERATE upon approval
