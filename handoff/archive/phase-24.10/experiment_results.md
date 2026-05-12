---
step: phase-24.10
cycle: 9
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_10.py'
title: MCP infrastructure + permissions + security audit (P1)
---

# Experiment Results — phase-24.10

**Action:** READ-ONLY. Findings + brief + contract. No code changes.

## Verbatim verifier output

```
=== phase-24.10 (mcp-security) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_10_mcp_security_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_modelcontextprotocol_io
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_10_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.10 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_audits_mcp_pinned_servers_alpaca_bigquery
  [PASS] findings_audits_deny_rule_surface_completeness
  [PASS] findings_audits_nextauth_webauthn_flow
  [PASS] findings_audits_backend_env_secret_handling
  [PASS] findings_proposes_new_mcps_signals_news_slack
FAIL (14/15) EXIT=1
```

14/15 PASS. Log-last only FAIL.

## Hypothesis verdict
CONFIRMED with 3 gaps: (1) Alpaca tool inventory not exhaustively audited; (2) plain `str` API keys in settings.py:87-92 should be SecretStr; (3) No Alpaca MCP smoke test. Version pinning correct, deny rules tight, secrets rotation current, WebAuthn correctly wired, no committed secrets.

## Phase-25 candidates (6)
- 25.A10 (P1) — Alpaca MCP tool-surface smoke test
- 25.B10 (P1) — SecretStr migration for API keys
- 25.C10 (P2) — Finnhub MCP (news + signals)
- 25.D10 (P2) — Alpha Vantage MCP
- 25.E10 (P2) — MCP SHA256 digest pinning
- 25.F10 (P2) — Wire secrets_rotation_check to Slack

## Next: Q/A
