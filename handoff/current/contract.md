# Contract — Step 56.2

**Step id:** 56.2 — Ops fixes
**Date:** 2026-06-10
**Phase:** phase-56 (fix work, finding-ID-driven; do-no-harm)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=complex, 7 external sources read in full, 16 URLs, recency scan; 16 internal files; envelope `gate_passed: true`)

## Research-gate summary

Shared infra: `raise_cron_alert(_sync)` (`alerting.py:119/:185`) is the canonical fail-open dedup-aware Slack notifier — F-4/F-5 reuse it. F-4: free `claude auth status` probe at cycle start (`autonomous_loop.py:~744`, gated by `paper_use_claude_code_route`), P1 alert on failure; probe where the rail is used (synthetic-transaction principle). F-5 root cause: `_run_claude_analysis` parse-fail fallback (`autonomous_loop.py:1601-1607`); guard at the cycle level after the gather (`:820/:827`) — ALL-zero or N≥3-zero scores → P1 alert + `summary["degraded"]=True` (Write-Audit-Publish: assert before the digest consumes). F-7 CODE CORRECTION: `_fallback_all` emits `round(composite)` not 10.00 — the damping is silently REMOVED; conviction drives top-K (`:723`) so changing the fallback VALUE is a live-selection behavior change → 56.2 makes it LOUD (`meta_scorer_degraded` flag + alert, byte-identical ordering), value-redesign deferred to phase-57 (Confidence-Gate 2026: structural uncertainty → abstention, but not via a silent residual). F-6: `log_llm_call` (`api_call_log.py:203`, auto cycle_id, fail-open) called at the two `claude_code_invoke` callers (`:1580/:1636`); Gemini lite path just needs `_role`/`_ticker` in generation_config (`:1798/:1835`). Criterion-2 root cause PINNED: `ticket_queue_processor._invoke_agent` (`:156-180`) uses the direct Anthropic SDK and does NOT honor `paper_use_claude_code_route` — fix = route through the CLI rail when the flag is set; e2e transcript needs the operator (one-line action escalation allowed by the criterion). F-14: `send_approval_gate` has ZERO callers → remove the dead buttons. F-8: decision-affecting prompt change → ESCALATE to phase-57 (already in that spec). Watchdog: raise the probe timeout 10s→30s (`scheduler.py:485`) — smallest bounded change. F-9: operator-proposal text only (55.1 ruled CORRECTLY-DID-NOT-TRIP → no unit-test fix required). Pytest inventory (13 failures): 2 STALE assertions (UPDATE: opus 4-7→4-8; lock count 14→15 incl. `alerting.py:64`), 2 live-BQ probes (`requires_live` skipif via NEW pytest.ini + env var), 7 moved-doc (repoint to `handoff/archive/phase-23.2.16/`), 2 test-pollution (fix shared state via `reset_default_deduper()` autouse first; honest-reason quarantine as fallback) — blanket-skipping would hide real regressions (watermelon risk).

## Hypothesis

The P0/P1 ops gaps are closable with observability-only changes (probe, guard, logging, dead-code removal) that leave trading behavior byte-identical, plus an honest root-cause-classified test quarantine that turns the full backend suite green without hiding real regressions.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 56.2)

1. "every finding ranked P0/P1 in the 55.3 table is either FIXED with a regression test or explicitly ESCALATED as operator-gated, with the finding-ID map recorded in live_check_56.2.md"

2. "the Slack approval path is exercised end-to-end: typing 'Approve' in the operator channel no longer yields 'Missing API key for provider anthropic' (captured transcript in the live_check), or the residual is escalated with a one-line operator action"

3. "a degraded-scoring guard exists: a cycle whose analyses all score 0.0 (or whose scoring backend is unavailable) is detected and alerted to Slack instead of passing silently, covered by a unit test; the watchdog ReadTimeout fix or a bounded escalation is applied per the 55.2 root cause"

4. "the kill-switch defect is fixed with a unit test reproducing the 06-05 scenario IFF 55.1 ruled SHOULD-HAVE-TRIPPED; any threshold change is presented as an OPERATOR DECISION, never auto-applied; the 16 env-coupled backend test failures are quarantined (skip-markers + reason strings) and backend pytest is green"

**Verification command (immutable):** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -q && test -f handoff/current/live_check_56.2.md`

## Plan (per-finding; tests alongside each fix)

1. **Test hygiene first** (criterion 4): NEW `pytest.ini` (register `requires_live`); skipif the 2 live-BQ probes with exact-dependency reasons; UPDATE the 2 stale assertions (4-7→4-8; lock count 14→15 + roster note); repoint the 7 shortlist-doc tests to the archive path; fix the 2 pollution tests via state-reset fixtures (fallback: honest-reason quarantine). Full suite must exit 0.
2. **F-4** (fix + regression test): `claude_code_health_probe()` in claude_code_client.py (free, no tokens); cycle-start call gated by the route flag; `raise_cron_alert` P1 on failure; own try/except (never breaks a cycle). Tests: probe False on non-zero exit; alert fired once with severity=P1; True on exit-0.
3. **F-5 + F-7** (fix + regression tests): cycle-level degraded-scoring guard after the gather (ALL-degraded or N≥3 zeros → P1 alert + `summary["degraded"]=True`); meta_scorer sets a `fallback_all` indicator consumed as `summary["meta_scorer_degraded"]` (alerted, ordering byte-identical). Tests: all-zero fires; 3/6 fires; 2/6 doesn't; fallback ordering byte-identity (do-no-harm invariant).
4. **F-6** (fix + test): `log_llm_call(provider="claude-code", ...)` after both `claude_code_invoke` callers (ok=False on ClaudeCodeError); `_role`/`_ticker` added to `_run_gemini_analysis` generation_config. Tests: called with right args (mocked); rail-error logs ok=False.
5. **Criterion 2** (fix + test + escalation): `_invoke_agent` honors `paper_use_claude_code_route` (routes via `claude_code_invoke`); unit test asserts the CLI rail is used when flagged (and the direct SDK when not). E2E transcript requires the operator → escalate with the one-line action: "type Approve in #ford-approvals once to confirm" (the criterion's OR-branch).
6. **F-14** (fix): remove the dead `approval_approve/deny` actions block from `governance.py` (zero callers; dead code).
7. **Watchdog** (bounded): `scheduler.py:485` probe timeout 10s→30s (one line; mirrors digest timeout; per the 55.2 root cause the backend was never down).
8. **F-9**: operator-proposal text (SOD re-anchor; thresholds UNCHANGED; dry-run-first recommendation) in live_check — NO code (55.1 verdict makes the unit-test fix not-required; any threshold change stays an operator decision).
9. **F-8**: ESCALATED to phase-57 (decision-affecting prompt change; already in the 55.3 FEATURE spec) — noted in the map.
10. live_check_56.2.md (finding-ID map incl. FIXED vs ESCALATED, pytest summary line, transcript/escalation, F-9 proposal) + experiment_results.md → ONE fresh Q/A → harness_log → flip.

## Constraints

- Every change cites a finding ID; do-no-harm: NO trading-behavior change (the conviction-fallback VALUE stays byte-identical; only observability added); no un-scrubbing ANTHROPIC_API_KEY; no live flag flips; no LLM trading-cycle spend (probe is token-free); launchctl/pip/BQ-writes stay operator-gated.
- Quarantine honesty: classify by root cause; no blanket skips; reasons name the exact dependency.
- ASCII-only logger messages (security.md).

## References

- handoff/current/research_brief.md (researcher 56.2, gate_passed: true)
- Findings: handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md §1 (F-4..F-9, F-14, F-18); 55.2 audit §1 (F-A1/F-D/F-E root causes); 55.1 §8 (kill-switch verdict)
- Code anchors: alerting.py:64,119,185,222; claude_code_client.py:79,110-114,159-162; autonomous_loop.py:698-723,744,820,827,1573-1607,1636,1798,1835; api_call_log.py:203,237-248,276; ticket_queue_processor.py:156-180; governance.py:136-178; scheduler.py:399,435,469-522,485; kill_switch.py:212-217,244-248; paper_trader.py:1034-1035; meta_scorer.py:138-142,240,249-256
- External: Slack Bolt ack() doc, pytest skipping doc, aipatternbook silent-failure, arXiv:2603.09947 (Confidence Gate, 2026), OneUptime heartbeat 2026, Great Expectations WAP, index.dev silent failures
