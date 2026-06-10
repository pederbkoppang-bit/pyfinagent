# Experiment Results — Step 55.2 (GENERATE)

**Step:** 55.2 — Ops incidents + agent-quality audit (away week 2026-06-01 → 2026-06-10). **Date:** 2026-06-10. **Mode:** review-only, $0, NO fixes.

## What was built

| Artifact | Content |
|---|---|
| `handoff/current/55.2-ops-skill-audit.md` | Incident triage with root causes at file:line + severities + stable IDs (F-A1 approve-flow OAuth rail, F-A2 fail-closed + dead button, F-C watchdog event-loop starvation, F-D silent 0.0/10 degraded scoring, F-E llm_call_log observability gap) plus three NEW incidents surfaced live (F-F RiskJudge REJECT advisory-only — DELL bought 06-03 despite REJECT; F-G RiskJudge 10%-vs-30% prompt/config divergence + concentration-blindness; F-H lite-checkbox UI/runtime desync); per-skill firing audit with skill→evidence-source map (verdict: lite 4-agent chain ran, SignalStack on static fallback all week, rag/earnings_tone/insider/patent/news-social DID NOT FIRE); operator's-question answer (NO); N* burn-vs-P&L reconciliation (~$1 burn vs −$132 churn); 3-analysis reasoning spot-check (2 whipsaws + the REJECT case); signal-stability table (35% daily flip rate, mean |Δscore| 1.15, SNDK flip reproduced with direction corrected); look-ahead paragraph; tool reruns |
| `handoff/current/live_check_55.2.md` | Verbatim llm_call_log fire-count tables (away window: 12 rows, ZERO for 06-02..06-09 cycles), incident evidence excerpts (watchdog log lines, 0.0-block BQ rows, REJECT trade row, dead-button grep, CLI auth state), signals-JSON spot-check excerpts, stability table pointer, burn numbers, tool-rerun outputs |

## Key findings (headline)

1. **F-A1 (HIGH):** the approve-flow error originates in the `claude` CLI binary — `claude_code_client.py:163-170` deliberately scrubs the API key to force `~/.claude/` OAuth; the OAuth session was the failure, the .env key was valid all along (direct-Anthropic probes succeeded minutes before the failed Approve). Fix direction is rail-health detection, NOT un-scrubbing.
2. **F-A2:** approval path fails CLOSED on action (by accident); the Approve button is a dead control (no handler).
3. **F-C (LOW):** watchdog/digest timeouts = event-loop starvation during the 18:00Z cycle; every FAIL "(1/3)"; backend never down.
4. **F-D (HIGH):** the "05-28 all-0.0/10" block is dated 05-27 18:02-18:20Z (digest lag); uppercase-HOLD degraded path published via `formatters.py:37` default-0 with no failed-vs-zero guard.
5. **F-E (HIGH):** llm_call_log is blind to the analysis rail (zero rows for 6 of 7 cycles; `log_llm_call` absent from `claude_code_client.py`/lite analyzer). Three masterplan premises corrected (cycle_id exists-but-NULL; labels aren't debate roles; strategy_decisions is rotation-only).
6. **F-F (HIGH, new):** RiskJudge REJECT is advisory-only (`portfolio_manager.py:185,194-198`) — live-proven by the executed DELL 06-03 BUY.
7. **Skills answer:** NO — lite chain only; enrichment skills silent on all 59 rows; conviction overlay was a hardcoded 10.00 stub all week.
8. **Stability:** 16 action flips / 46 pairs (35%), churn by construction (momentum-following with no damping layer); SNDK chronological flip is 5.0-HOLD→7.0-BUY (digest framing reconciles).

## Verification command output (verbatim)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.2-ops-skill-audit.md && test -f handoff/current/live_check_55.2.md && echo PASS
PASS
```

## Honest limitations

- The away-week OAuth-expiry mechanism is bounded, not directly observed (auth state is only inspectable NOW, post-recovery); the bound rests on four converging live evidences (fallback strings, zero logged calls, the 06-01 error, working direct-API probes).
- Per-sector count-cap full adjudication still open (decision-time cap state not persisted) — carried by F-G to phase-56.
- Whether rag/earnings_tone/etc. are SUPPOSED to fire in lite mode (vs only in full mode) is a code-expectation question the goal doc asserted; the audit reports the observed silence and flags the discrepancy rather than adjudicating intent.
- paper_execution_parity.py fails identically to 55.1 (client_order_id reuse) — reported verbatim both times.
