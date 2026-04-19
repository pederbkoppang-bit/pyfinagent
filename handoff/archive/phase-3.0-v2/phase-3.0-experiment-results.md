# Experiment Results -- phase-3.0 MCP Server Architecture

**Step:** 3.0 (documentation consolidation)
**Date:** 2026-04-19
**Parallel-safety:** phase-specific filename.

## What was built

Documentation consolidation step closing the phase-3.0 contract gap. The three code deliverables (data/signals/backtest MCP servers) were already implemented and passed earlier (archive PASS 2026-03-29), but two contracted doc deliverables were never written and `ARCHITECTURE.md` was stale. Those gaps are closed here:

- **`docs/MCP_ARCHITECTURE.md`** (new, ~120 lines): single source of truth for MCP server architecture. Sections: server inventory (4 internal + 2 external + harness-injected BQ MCP), transport choice (stdio-only rationale), agent->server data flow, tool-description quality principle (40% lift per Anthropic research), health + supply-chain, related ADR cross-links, known gaps.
- **`docs/MCP_SECURITY.md`** (new, ~110 lines): threat model, HMAC-SHA256 capability tokens with 30-min TTL (NIST SP 800-63B-4), 6 roles with FIXED scope sets, PII scrub regex classes, supply-chain pinning policy, output caps + debounce, rate-limiting deferral rationale, incident response steps.
- **`ARCHITECTURE.md:269-276`** updated: MCP servers table now includes `risk_server.py` (phase-3.7.3); cross-cutting infra section added for `mcp_capabilities.py` + `mcp_health_cron.py`; cross-links to both new docs.
- **`.mcp.json` Alpaca pin**: `uvx alpaca-mcp-server` -> `uvx --from alpaca-mcp-server==2.0.1 alpaca-mcp-server` (pinned to the current latest `2.0.1`, closing the phase-3.7.6 supply-chain action item).

## File list

Created:
- `docs/MCP_ARCHITECTURE.md`
- `docs/MCP_SECURITY.md`

Modified:
- `ARCHITECTURE.md` (MCP section at line 269, expanded from 6 lines to ~22 lines)
- `.mcp.json` (Alpaca version pin)

Non-Python files only. No test changes. No behavior changes.

## Verification command output

### 1. Immutable verification (from masterplan)

```
$ source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
...
[INFO] harness: Wrote handoff/contract.md
[INFO] harness: DRY RUN -- skipping generator and evaluator
[INFO] harness: Appended cycle 1 to harness_log.md
[INFO] harness:
============================================================
[INFO] harness: HARNESS COMPLETE -- 1 cycles finished
[INFO] harness: Final best: Sharpe=1.1705, DSR=0.9526
[INFO] harness: ============================================================
```

Exit 0. No regressions (final best Sharpe preserved at 1.1705; DSR preserved at 0.9526). Both immutable success_criteria met: `no_regressions` ✓ and `evaluator_critique_pass` will be verified by Q/A below.

### 2. Documentation files exist

```
$ ls -la docs/MCP_ARCHITECTURE.md docs/MCP_SECURITY.md
```
Both present, non-empty.

### 3. ARCHITECTURE.md updated

```
$ grep -c "risk_server.py\|MCP_ARCHITECTURE\|MCP_SECURITY\|mcp_capabilities\|mcp_health_cron" ARCHITECTURE.md
```
Shows >=5 matches across the MCP section.

### 4. Alpaca pinned in `.mcp.json`

```
$ grep -A1 "alpaca-mcp-server" .mcp.json
      "args": ["--from", "alpaca-mcp-server==2.0.1", "alpaca-mcp-server"],
```

Pinned to `2.0.1` (current latest, confirmed via `pip index versions alpaca-mcp-server`).

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `docs/MCP_ARCHITECTURE.md` with required sections | PASS (server inventory, transport, data flow, tool-description principle, health, gaps, ADR cross-links all present) |
| 2 | `docs/MCP_SECURITY.md` with required sections | PASS (threat model, capability tokens, PII scrub, supply-chain pin, output caps + debounce, rate-limit deferral, audit log, incident response) |
| 3 | `ARCHITECTURE.md:269-276` MCP section updated | PASS (risk_server added, capabilities + health cron documented, cross-links to new docs) |
| 4 | `.mcp.json` Alpaca pin added | PASS (`alpaca-mcp-server==2.0.1`) |
| 5 | Immutable `--dry-run --cycles 1` exits 0 | PASS (verified) |
| 6 | No Python code files modified (except Alpaca pin non-Python) | PASS (diff shows only `ARCHITECTURE.md`, `.mcp.json`, 2 new `docs/*.md`) |
| 7 | No rate-limiting code added to MCP servers (non-goal) | PASS (no server code touched) |

## Known caveats (transparency to Q/A)

1. **Doc-vs-code audit performed as cycle-2 fix (2026-04-19T~13:30 UTC).**
   Cycle-1 qa_30_v1 CONDITIONAL verdict flagged two divergences:
   (a) MCP_SECURITY.md's Capability Tokens section cited wrong role names
       (`analyst/trader/risk_reviewer/researcher/harness/admin`) and wrong
       scope tokens (`data.read.market`, `signals.publish`, etc.) vs the
       actual code at `backend/agents/mcp_capabilities.py:57-70` which
       defines `{researcher, strategy, risk, evaluator, orchestrator,
       paper_trader}` with coarser scope strings (`data.read`,
       `signals.read/write`, `risk.read/write`, `backtest.read`,
       `trading.write`). Plus the module exports `ROLE_SCOPES` (public, no
       underscore) not `_ROLE_SCOPES`.
   (b) `mcp_audit` BQ table referenced in the Audit log section does not
       exist (no `add_mcp_audit*` migration; zero `mcp_audit` string hits
       in the tree).
   Both are now fixed:
   - Capability Tokens section rewritten against the verified source:
     correct role names, exact scope sets copied from `ROLE_SCOPES`,
     correct exception classes, correct TTL constant, correct verification
     API names, correct `enforce` decorator semantics.
   - PII scrub section rewritten with the actual `_PII_PATTERNS` list
     and the literal `[REDACTED]` marker.
   - Audit log section demoted to "documented gap" with explicit
     acknowledgment that no BQ table exists today; follow-up scoped and
     filed as a phase-3.7.7 / phase-6.9 item, not a phase-3.0 blocker.
2. **`alpaca-mcp-server` version was `uvx` (unpinned implicit latest) prior to this commit.** Pinning to 2.0.1 is a minor behavior change -- the `uvx` cache may download 2.0.1 specifically on next invocation rather than whatever was cached. Not a functional change; supply-chain hardening only.
3. **Documentation is advisory / descriptive.** No executable validators enforce that the docs stay in sync with the code. Standard doc-drift risk. Mitigation: cycle-2 audit established the precedent of grep-verifying every specific claim against source.
