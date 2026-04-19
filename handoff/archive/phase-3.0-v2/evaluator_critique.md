# Phase-3.0 Q/A Critique -- CYCLE 2

- **qa_id**: qa_30_v2
- **cycle**: 2 (fresh respawn on updated evidence)
- **date**: 2026-04-19
- **verdict**: **PASS**

## Anti-verdict-shopping check

- Cycle-1 critique (`phase-3.0-evaluator-critique.md`) contains a
  `## Follow-up` section (lines 105-118) documenting the two fixes
  applied by Main before this respawn. PASS.
- `phase-3.0-experiment-results.md` Known caveats #1 explicitly
  records the cycle-2 fix with exact role/scope corrections and the
  audit-log demotion. PASS.
- Evidence HAS CHANGED between cycle-1 and cycle-2 -> respawn is
  the documented file-based cycle-2 pattern (CLAUDE.md), not
  verdict-shopping.

## 5-item harness-compliance audit (brief)

1. Researcher brief present + gate_passed=true -- PASS (unchanged from cycle-1).
2. Contract pre-committed -- PASS (unchanged from cycle-1).
3. experiment-results accurate post-fix -- PASS (Known caveats updated).
4. harness_log last entry != phase-3.0 -- PASS (will be appended after Q/A PASS, log-last rule).
5. No verdict-shopping -- PASS (evidence changed; Follow-up + Known-caveats both document the change).

## B1 re-verification: Capability Tokens section vs `mcp_capabilities.py:57-70`

`docs/MCP_SECURITY.md:17-34` verified line-by-line against source:

| Claim | Doc | Code | Match |
|---|---|---|---|
| 6 roles | `researcher/strategy/risk/evaluator/orchestrator/paper_trader` (L23-29) | `ROLE_SCOPES` L57-70 | EXACT |
| `researcher` scopes | `{data.read, signals.read, backtest.read}` | same | EXACT |
| `strategy` scopes | `{data.read, signals.read, signals.write, backtest.read}` | same | EXACT |
| `risk` scopes | `{data.read, signals.read, risk.read, risk.write}` | same | EXACT |
| `evaluator` scopes | `{data.read, signals.read, backtest.read, risk.read}` | same | EXACT |
| `orchestrator` scopes | 6 scopes inc. risk.read/write | same | EXACT |
| `paper_trader` scopes | `{data.read, signals.read, trading.write, risk.read}` | same | EXACT |
| Symbol name | `ROLE_SCOPES` (public, L34) | public at L57 | EXACT |
| TTL constant | `TOKEN_TTL_SECONDS = 1800` (L22) | L47 | EXACT |
| Exceptions | `CapabilityError`, `TokenExpiredError`, `TokenInvalidError`, `ScopeViolationError` (L31) | L73-86 | EXACT |
| APIs | `verify_token`, `enforce`, `scrub_args` (L32) | present in module docstring L22-27 | EXACT |

B1: **PASS**. All three cycle-1 divergences fixed.

## B2 re-verification: Audit log section

`docs/MCP_SECURITY.md:89-107` now explicitly labeled "Audit log
(documented gap)" and states "There is NO BQ-backed `mcp_audit`
table" (L91). No claim of an existing phase-3.7.7 migration. A
specific follow-up plan with 3 actions (L102-105). B2: **PASS**.

`grep -r "mcp_audit" backend/ scripts/ docs/` returns exactly 3
hits, all in `docs/MCP_SECURITY.md` (L91, L103, L115 -- the last
is the incident-response step that references the planned table,
acceptable given the documented-gap framing). Zero code/migration
hits, as required.

## Immutable verification re-run

```
$ source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
...
[INFO] harness: HARNESS COMPLETE -- 1 cycles finished
[INFO] harness: Final best: Sharpe=1.1705, DSR=0.9526
```

Exit 0, "HARNESS COMPLETE" emitted. Final best Sharpe/DSR
preserved. PASS.

## MCP_ARCHITECTURE.md spot-check

Re-read. It remains primarily high-level (server inventory,
transport rationale, data-flow, ADR cross-links, known gaps).
Cycle-1 already verified cross-refs (`risk_server.py`,
`mcp_capabilities.py`, `mcp_health_cron.py`, both new docs). No
new falsifiable code claims were introduced in cycle-2. Nothing
further to grep-verify.

## LLM judgment

Evidence has changed: YES. Specific fixes are verifiable by source
(role names, scope sets, symbol name, TTL constant, exception
classes all match `mcp_capabilities.py:47,57-70,73-86`; audit-log
section now honest about the gap). Contract criteria now all
satisfied -- no outstanding doc-vs-code divergence, no fictional
infrastructure claim. This is the documented cycle-2 pattern
working correctly: Main received a CONDITIONAL with specific
blockers, fixed them, updated evidence, and a fresh Q/A confirms
the fix against source.

## violated_criteria

`[]`

## checks_run

`["anti_verdict_shopping", "5_item_audit_brief", "B1_doc_vs_code_role_scope_symbol_ttl_exceptions_apis", "B2_audit_log_gap_language", "grep_mcp_audit_tree_wide", "immutable_harness_dry_run", "mcp_architecture_spot_check"]`

## certified_fallback

`false` -- cycle 2, PASS. No fallback triggered.
