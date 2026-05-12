# Sprint Contract — phase-24.10 — MCP Security Audit

**Cycle:** phase-24 cycle 9
**Date:** 2026-05-12
**Step ID:** 24.10
**Priority:** P1

## Research-gate
`gate_passed: true` (tier=moderate). 6 sources: MCP security spec, arxiv 2603.22489 threat model, WebAuthn passkey security analysis, Red Hat MCP risks, pydantic-settings secrets, financial MCP servers 2026.

```json
{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":10,"urls_collected":16,"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}
```

## Hypothesis
MCP setup minimal (2 servers, deny-listed). Security good but not comprehensive.

**Researcher verdict: CONFIRMED with gaps:**
- Tool poisoning is dominant client-side risk; 5/7 MCP clients lack static validation
- Version pinning correct (`alpaca-mcp-server==2.0.1`, `mcp-server-bigquery==0.3.2`)
- **Gap 1:** Alpaca tool inventory not exhaustively audited (5 denies, no smoke test enumerates the full surface)
- **Gap 2:** `settings.py:87-92,192` uses plain `str` for API keys/secrets — should be `SecretStr`
- **Gap 3:** No Alpaca MCP smoke test (only BigQuery has one)
- Secrets rotation schedule complete + current; no committed secrets in git history
- WebAuthn correct but experimental flag `enableWebAuthn: true`

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. findings_audits_mcp_pinned_servers_alpaca_bigquery
12. findings_audits_deny_rule_surface_completeness
13. findings_audits_nextauth_webauthn_flow
14. findings_audits_backend_env_secret_handling
15. findings_proposes_new_mcps_signals_news_slack

**Verifier:** `python3 tests/verify_phase_24_10.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 50 log
5. live_check_24.10.md
6. Flip
