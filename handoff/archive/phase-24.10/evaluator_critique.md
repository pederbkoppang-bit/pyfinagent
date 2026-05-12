---
step: phase-24.10
cycle: 9
cycle_date: 2026-05-12
verdict: PASS
qa_agent: qa (merged qa-evaluator + harness-verifier)
spawn: first (no prior CONDITIONAL/FAIL on this step-id)
---

# Q/A Critique — phase-24.10 — MCP infrastructure + security audit (P1)

## 5-item harness-compliance audit
1. **Researcher gate** — CONFIRM. `handoff/current/research_brief.md` ends with
   `{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":10,"urls_collected":16,"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}`.
   6 sources read in full (MCP spec, arXiv 2603.22489, shellnetsecurity passkey,
   Red Hat MCP, pydantic-settings secrets, marketxls financial MCP). Exceeds
   floor of 5; recency scan section present with 2024-2026 findings; three-variant
   query discipline declared.
2. **Contract pre-commit** — CONFIRM. `contract.md` enumerates 15 verbatim
   success criteria (10 common-pack + 5 step-specific: pinned_servers_alpaca_bigquery,
   deny_rule_surface_completeness, nextauth_webauthn_flow, backend_env_secret_handling,
   proposes_new_mcps_signals_news_slack). Verifier command quoted verbatim.
3. **experiment_results step** — CONFIRM. `experiment_results.md` frontmatter
   `step: phase-24.10`, verbatim 15-criteria verifier output block present
   (14 PASS + 1 FAIL log-last).
4. **harness_log not yet** — CONFIRM. `grep -c phase=24.10 handoff/harness_log.md`
   returns 0. Log-last discipline preserved.
5. **First Q/A spawn** — CONFIRM. No prior `phase=24.10` cycle in harness_log;
   3rd-CONDITIONAL counter is 0; this is cycle 9 first pass.

## Deterministic checks
- `python3 tests/verify_phase_24_10.py` → 14/15 PASS, exit 1. Sole FAIL is
  `harness_log_has_phase_24_24_10_cycle_entry` (the log-last criterion that
  by construction cannot pass until AFTER Q/A — this is the intended
  audit-trail of "the log is the LAST step").
- `docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md` present
  alongside the other 8 phase-24 findings docs.
- Findings doc cites `modelcontextprotocol.io` verbatim (canonical URL check).
- Pinned-version evidence verified: `alpaca-mcp-server==2.0.1` and
  `mcp-server-bigquery==0.3.2` cited from `.mcp.json:5,16`.

## LLM-judgment legs
1. **Contract alignment** — PASS. All 8 substantive content anchors land:
   - F-1 pinned servers (alpaca 2.0.1 + bigquery 0.3.2, finding #4)
   - F-2 deny rules (settings.json:153-158, 5 specific tools, finding #5)
   - F-3 NextAuth/passkey wiring (auth.ts:15, finding #7)
   - F-4 secrets in env (no committed secrets, finding #9; plain-str gap finding #6)
   - F-5 rotation schedule (11 secrets, ALPACA 24d, AUTH_SECRET 41d, finding #8)
   - F-6 no Alpaca smoke test (finding #10)
   - F-7 WebAuthn flow (experimental flag flagged, finding #7)
   - F-8 tool poisoning (arXiv 2603.22489 cited as dominant client-side risk, finding #1)
2. **Mutation resistance** — PASS. Content-specific anchors: exact line refs
   (settings.py:87-92, auth.ts:15, .mcp.json:5,16), exact pinned versions,
   exact rotation dates (2026-04-18, 2026-04-01), exact arXiv ID 2603.22489.
   These would not survive a copy/paste of a generic MCP-security writeup.
3. **Anti-rubber-stamp** — PASS. Critique honestly distinguishes WORKING
   (pinning correct, deny rules tight, no committed secrets, rotation current,
   WebAuthn properly wired) from GAPS (Alpaca tool-surface not exhaustively
   audited, SecretStr migration needed, no Alpaca smoke test). Three distinct
   phase-25 candidates queued for the gaps (25.A10, 25.B10, 25.E10).
4. **Scope honesty** — PASS. Open questions disclosed: experimental WebAuthn
   flag (API may change), confused-deputy surface absent only because no HTTP
   MCP transport yet (future addition needs per-client consent), tool-poisoning
   static validation not implemented. Polygon/Finnhub/Alpha Vantage flagged
   as phase-25 candidates rather than overclaimed as in-scope.
5. **Research-gate compliance** — PASS. Contract's "Research-gate" section
   quotes the envelope verbatim and names all 6 sources by topic.

## Violated criteria
None substantive. The single deterministic FAIL is the log-last criterion,
which is BY DESIGN unsatisfiable at Q/A time (log appends AFTER PASS). This
matches the expected 14/15 envelope declared in the prompt.

## Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5/5 harness-compliance CONFIRM; 14/15 verifier PASS with log-last as the sole expected pending; 6 sources read in full; 8/8 content anchors land with line-level mutation-resistant evidence; honest WORKING-vs-GAPS partition with 3 phase-25 candidates queued.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax_via_verifier", "verification_command", "research_brief_envelope", "contract_criteria_count", "experiment_results_verbatim_block", "harness_log_grep", "findings_md_existence", "llm_judgment_5_legs"]
}
```

## Next actions for Main
1. Append `## Cycle 50 -- 2026-05-12 -- phase=24.10 result=PASS` block to
   `handoff/harness_log.md` (log-last criterion will then go green if rerun).
2. Create `handoff/current/live_check_24.10.md` per masterplan
   `verification.live_check` field.
3. Flip `.claude/masterplan.json` step 24.10 status → `done`.
