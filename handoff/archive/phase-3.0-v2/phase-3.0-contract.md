# Sprint Contract -- phase-3.0 MCP Server Architecture

**Written:** 2026-04-19 PRE-commit.
**Step id:** `3.0` in phase-3 (LLM-Guided Research + MCP Integration).
**Parallel-safety:** phase-specific filename.

## Research-gate summary

Researcher spawned today. Envelope: `{tier: moderate, external_sources_read_in_full: 7, recency_scan_performed: true, internal_files_inspected: 14, gate_passed: true}`. Brief at `handoff/current/phase-3.0-research-brief.md` (155 lines).

Key research finding: **phase-3.0 implementation is already complete** (data/signals/backtest/risk MCP servers exist and work; archive has PASS dated 2026-03-29) BUT two contracted doc-deliverables (`docs/MCP_ARCHITECTURE.md` and `docs/MCP_SECURITY.md`) were never created, and `ARCHITECTURE.md:269-276` is stale (omits `risk_server.py` from phase-3.7.3, capability tokens from phase-3.7.7, mcp_health_cron.py from phase-3.5.7). This step is therefore a **narrow documentation consolidation** bundling the decisions already made across phases 3.0, 3.5.0, 3.7.0, 3.7.3, 3.7.6, 3.7.7.

## Hypothesis

Creating the two contracted doc files + updating ARCHITECTURE.md + pinning the Alpaca MCP server in `.mcp.json` closes phase-3.0 as a documentation step without additional code changes.

## Success criteria

The step has an **immutable** verification command in masterplan:
```
source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
```
With success_criteria = `['evaluator_critique_pass', 'no_regressions']`.

Meeting this immutable criterion requires: (a) no code breakage (dry-run cycle exits 0); (b) Q/A returns PASS. Both are met by doc-only work.

**Additional contract criteria (functional):**
1. `docs/MCP_ARCHITECTURE.md` exists with sections:
   - Server inventory (4 servers: data, signals, backtest, risk)
   - Transport (stdio for local, justification)
   - Capability negotiation (FastMCP handshake)
   - Data flow diagram (text description)
   - Phase lineage (3.0 -> 3.5 adoption waves -> 3.7 hardening)
   - Cross-references to ADRs already made: phase-3.7.0 (MCP vs A2A), phase-3.7.6 (guardrails), phase-3.7.7 (capability tokens)
2. `docs/MCP_SECURITY.md` exists with sections:
   - Threat model (trusted stdio vs untrusted remote)
   - Capability-token issuance (`mcp_capabilities.py:1-80`; HMAC-SHA256, 30-min TTL per NIST SP 800-63B-4, 6 roles mapped to fixed scope sets)
   - PII scrub on inbound args
   - Supply-chain pinning + `mcp_health_cron.py` (phase-3.5.7)
   - Output-size caps + debounce (phase-3.7.6)
   - Rate-limiting approach (documented gap: local/trusted-network acceptable; flagged for remote exposure)
3. `ARCHITECTURE.md:269-276` MCP section updated:
   - Table row added for `risk_server.py` (phase-3.7.3)
   - Reference line added for `mcp_capabilities.py` + `mcp_health_cron.py`
   - Cross-link to new `docs/MCP_ARCHITECTURE.md` and `docs/MCP_SECURITY.md`
4. `.mcp.json` Alpaca entry gets an explicit version pin (from `uvx alpaca-mcp-server` to `uvx --from alpaca-mcp-server==<pinned> alpaca-mcp-server`).
5. Immutable verification command: `python scripts/harness/run_harness.py --dry-run --cycles 1` exits 0.
6. No Python code files modified except the Alpaca pin in `.mcp.json`.
7. Non-goal: adding rate-limiting to MCP servers (deferred; research confirmed trusted-stdio is acceptable for the May 2026 go-live scope).

## Plan

1. Determine the current Alpaca MCP server version pin (check `uvx alpaca-mcp-server --version` if network-accessible; otherwise document "UNPINNED-CHECK-BEFORE-PROD").
2. Write `docs/MCP_ARCHITECTURE.md` (~1 page).
3. Write `docs/MCP_SECURITY.md` (~1 page).
4. Update `ARCHITECTURE.md:269-276`.
5. Update `.mcp.json` Alpaca entry.
6. Run `python scripts/harness/run_harness.py --dry-run --cycles 1`, capture output.

## References

- `handoff/current/phase-3.0-research-brief.md` (155 lines)
- `handoff/archive/phase-3.0/contract.md` (original phase-3.0 contract that specified the two missing doc files)
- `backend/agents/mcp_servers/{data,signals,backtest,risk}_server.py` (inventory)
- `backend/agents/mcp_capabilities.py:1-80` (token pattern)
- `backend/services/mcp_health_cron.py` (supply-chain monitor)
- `handoff/archive/phase-3.7.0/`, `phase-3.7.6/`, `phase-3.7.7/` (ADRs already made)
- MCP spec 2024-11-05 + 2025 updates, FastMCP v3.2.4, Anthropic MCP reference servers.

## Researcher agent id

`acf529b0674bc9078`
