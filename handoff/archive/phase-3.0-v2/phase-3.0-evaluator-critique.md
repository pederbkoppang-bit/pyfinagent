# Phase-3.0 Q/A Critique

- **qa_id**: qa_30_v1
- **cycle**: 1
- **date**: 2026-04-19
- **verdict**: **CONDITIONAL**

## 5-item harness-compliance audit

| # | Check | Status |
|---|-------|--------|
| 1 | Researcher brief present, gate_passed=true | PASS — `handoff/current/phase-3.0-research-brief.md`; envelope: `{external_sources_read_in_full: 7, recency_scan_performed: true, gate_passed: true}` |
| 2 | Contract PRE-committed (mtime < docs) | PASS — contract mtime 1776590925 < MCP_ARCHITECTURE.md 1776597211 < MCP_SECURITY.md 1776597252 |
| 3 | experiment-results present + accurate | PASS (present); accuracy qualified — see H below |
| 4 | harness_log last entry != phase-3.0 | PASS — last entry is phase-2.12-style Cycle 1 dry-run, not phase-3.0 |
| 5 | No verdict-shopping | PASS — first Q/A spawn this cycle |

## Deterministic checks (A–H)

| # | Check | Result |
|---|-------|--------|
| A | MCP_ARCHITECTURE.md sections | PASS — Server inventory(L8), Transport choice(L32), data flow(L42), Tool-description(L75), supply-chain(L87), Related ADRs(L96), Known gaps(L105) all present |
| B | MCP_SECURITY.md sections | PASS — all 8 required headings present (L7,17,29,43,58,65,75,85) |
| C | ARCHITECTURE.md references code + docs | PASS — `risk_server.py` (L276), `mcp_capabilities.py` (L279), `mcp_health_cron.py` (L280), `MCP_ARCHITECTURE.md` (L284), `MCP_SECURITY.md` (L285) |
| D | .mcp.json pins `alpaca-mcp-server==2.0.1` | PASS — exact match found in args |
| E | Harness dry-run exit=0 + "HARNESS COMPLETE" | PASS — re-run by Q/A, emitted "HARNESS COMPLETE -- 1 cycles finished" at 13:16:12 UTC |
| F | No Python code modified | PASS for this step's scope — scope includes ARCHITECTURE.md + .mcp.json + 2 new docs (broader git diff is pre-existing uncommitted work from prior phases) |
| G | **Doc vs `mcp_capabilities.py` accuracy** | **FAIL — 3 factual errors** (see below) |
| H | **`mcp_audit` migration exists** | **FAIL — no migration file; no `mcp_audit` grep hit in `scripts/migrations/` or `backend/`** |

## G — doc-vs-code divergences (hard-verified)

`docs/MCP_SECURITY.md` contains three claims contradicted by `backend/agents/mcp_capabilities.py:57-70`:

1. **Role names wrong.** SECURITY.md:23 lists roles as `analyst, trader, risk_reviewer, researcher, harness, admin`. Code defines `researcher, strategy, risk, evaluator, orchestrator, paper_trader`. Count matches (6) but 5 of 6 names diverge.
2. **Scope granularity overstated.** SECURITY.md:24 cites atomic scopes like `data.read.market`, `signals.publish`, `risk.kill_switch_arm`. Code uses coarser `data.read`, `signals.read`, `signals.write`, `backtest.read`, `risk.read`, `risk.write`, `trading.write`. The "composite scopes are banned" claim is unsupported.
3. **Symbol name.** SECURITY.md:27 says "Do NOT edit `_ROLE_SCOPES`"; code exports public `ROLE_SCOPES` (no underscore) via `__init__`-style imports at `mcp_capabilities.py:26`.

Severity: HIGH. These are exactly the claims Main flagged as "not verified against code" (Known Caveat 2). Leaving them in published docs institutionalizes wrong information for future agents and for incident response.

## H — audit-log infrastructure claim is fiction

`docs/MCP_SECURITY.md:81` claims BQ destination `{project}.{observability_dataset}.mcp_audit` with "schema in a phase-3.7.7 migration file". Verified:

- `ls scripts/migrations/` — no file matching `*mcp*` or `*audit*` beyond `add_api_call_log.py`, `add_llm_call_log.py`.
- `grep -r "mcp_audit" scripts/migrations/ backend/` — zero hits.

The audit log section describes infrastructure that does not exist. Must be downgraded to "planned (gap)" or migration must be created.

## LLM judgment

- **Contract alignment (7 criteria)**: 5 of 7 PASS (file existence, sections, ARCHITECTURE cross-refs, alpaca pin, harness green). Criterion "docs factually consistent with code" FAIL. Criterion "security doc reflects real posture" FAIL on audit log.
- **Research-gate tracing**: PASS — new-doc claims on FastMCP 3.2.4, HMAC-SHA256, NIST 30-min TTL trace to brief's source set.
- **Scope call (docs not code)**: correct call. No code change was needed; the defect is in the docs themselves.
- **Anti-rubber-stamp**: This is the exact case CLAUDE.md warns about — "Don't PASS if the docs have factual errors you can verify against code." Two such errors are verified.
- **Known Caveat #2 disposition**: Main wrote the docs flagging this as uncertain and punted to Q/A. Q/A's job is to close it. **Verdict: CONDITIONAL, not acceptable caveat** — a security doc that names wrong roles, wrong scopes, and a nonexistent audit table is worse than no doc; it misleads future responders.

## violated_criteria

- `docs_consistent_with_code` (Threshold_Not_Met / Contradiction)
- `security_claims_are_implemented` (Unjustified_Inference — `mcp_audit` BQ table)

## violation_details

```json
[
  {
    "violation_type": "Contradiction",
    "action": "docs/MCP_SECURITY.md:23 lists roles {analyst,trader,risk_reviewer,researcher,harness,admin}",
    "state": "backend/agents/mcp_capabilities.py:57-70 defines roles {researcher,strategy,risk,evaluator,orchestrator,paper_trader}",
    "constraint": "Contract: security doc must accurately describe mcp_capabilities.py role/scope model"
  },
  {
    "violation_type": "Contradiction",
    "action": "docs/MCP_SECURITY.md:24 claims atomic scopes e.g. data.read.market, signals.publish, risk.kill_switch_arm; 'composite scopes banned'",
    "state": "code exposes coarser composite-style scopes: data.read, signals.read/write, backtest.read, risk.read/write, trading.write",
    "constraint": "Doc claims must be traceable to code"
  },
  {
    "violation_type": "Unjustified_Inference",
    "action": "docs/MCP_SECURITY.md:81 asserts BQ mcp_audit table with schema in a phase-3.7.7 migration",
    "state": "no file in scripts/migrations/ references mcp_audit; grep of backend/ returns zero hits",
    "constraint": "Security-posture claims must describe implemented infrastructure or be labeled gap"
  }
]
```

## Required fixes (to reach PASS on respawn)

1. Rewrite `docs/MCP_SECURITY.md` "Capability tokens" section using the real role names and scope strings from `mcp_capabilities.py:57-70`. Quote the symbol name `ROLE_SCOPES` (no underscore).
2. Either (a) create the `mcp_audit` BQ migration in `scripts/migrations/add_mcp_audit_log.py` mirroring `add_api_call_log.py`, or (b) rewrite the "Audit log" section as a documented gap with a follow-up phase id.
3. Update Known Caveat #2 in `phase-3.0-experiment-results.md` from "not verified" to either "verified — corrections applied in commit <sha>" or "deferred with ticket".
4. Respawn Q/A with updated evidence (file-based fresh-respawn pattern; NOT verdict-shopping).

## checks_run

`["5_item_audit", "syntax_na_docs_step", "section_headings_A_B", "architecture_md_crossrefs_C", "mcp_json_pin_D", "harness_dry_run_E", "scope_audit_F", "doc_vs_code_divergence_G", "mcp_audit_migration_H", "research_gate_trace", "contract_alignment"]`

## certified_fallback

`false` — cycle 1, retry_count=0. Fixes are small and localized to 2 docs + optionally 1 migration.

---

## Follow-up (2026-04-19T~13:30 UTC, pre-respawn)

Cycle-1 verdict was CONDITIONAL with two specific blockers:
1. `MCP_SECURITY.md` Capability Tokens section claimed wrong roles + scope tokens + private-prefix `_ROLE_SCOPES`; actual code at `backend/agents/mcp_capabilities.py:57-70` defines different roles, coarser scopes, and public-name `ROLE_SCOPES`.
2. `MCP_SECURITY.md` Audit log section referenced an `mcp_audit` BQ table that does not exist (no migration, zero tree hits).

Fixes applied to `docs/MCP_SECURITY.md`:
- **Capability Tokens section rewritten** against verified source: correct 6 roles (`researcher/strategy/risk/evaluator/orchestrator/paper_trader`), exact scope sets from `ROLE_SCOPES`, correct TTL constant name `TOKEN_TTL_SECONDS`, correct exception classes (`CapabilityError` + 3 subclasses), correct API names (`verify_token`, `enforce`). Public-name `ROLE_SCOPES` (not `_ROLE_SCOPES`).
- **PII scrub section rewritten** with the real `_PII_PATTERNS` tuple list (email / phone / anthropic_key / openai_key / jwt / ssn) and the literal `[REDACTED]` marker constant.
- **Audit log section demoted** to a "documented gap" block that explicitly states no BQ table exists today; follow-up is scoped and filed (create `add_mcp_audit_log.py`, add `log_mcp_verify` to `api_call_log.py`, wire inside `verify_token`).

`handoff/current/phase-3.0-experiment-results.md` Known caveats section updated to record the cycle-2 fix and keep the audit trail clear.

Next step: fresh Q/A respawn per CLAUDE.md cycle-2 flow (file-based communication; evidence has changed; new verdict reflects the fix).
