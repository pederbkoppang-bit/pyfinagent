# Experiment Results — Step 56.2 (GENERATE)

**Step:** 56.2 — Ops fixes. **Date:** 2026-06-10. **Mode:** finding-ID-driven fix work; do-no-harm (observability/ops layer only).

## What was built (11 files)

| File | Change | Finding |
|---|---|---|
| `backend/agents/claude_code_client.py` | NEW `claude_code_health_probe()` — free `claude auth status` in the scrubbed env (tests the OAuth rail, not the key); never raises | F-4 |
| `backend/services/autonomous_loop.py` | Cycle-start rail probe + P1 alert; cycle-level degraded-scoring guard (`_degraded_scoring_check` pure predicate + alert + `summary["degraded"]`); conviction-fallback detection (`_all_conviction_fallback` + P1 alert + `summary["meta_scorer_degraded"]`, VALUE byte-identical); `_log_claude_code_call` metering on both CLI-rail legs; `_role`/`_ticker` tags on the Gemini lite path | F-4, F-5, F-7, F-6 |
| `backend/services/ticket_queue_processor.py` | `_spawn_real_agent` honors `paper_use_claude_code_route` (CLI rail; direct SDK unchanged when off); removed a stray emoji from a comment | criterion-2 (F-4 family) |
| `backend/slack_bot/governance.py` | Dead `approval_approve/deny` actions block removed (zero callers); typed-reply instruction instead | F-14 |
| `backend/slack_bot/scheduler.py` | Watchdog probe timeout 10s→30s (bounded F-C fix) | F-C family |
| `backend/services/observability/api_call_log.py` | `reset_buffer_for_test()` re-arms `_last_flush_ts` (root cause of the rainbow-canary suite-order failure) | quarantine (pollution) |
| `backend/tests/test_phase_56_2_ops_fixes.py` | NEW: 18 tests (probe semantics, guard thresholds incl. falsy-zero regression, fallback detection + ordering byte-identity, metering, rail routing) | F-4/5/6/7, criterion-2 |
| `pytest.ini` | NEW: registers `requires_live` marker | quarantine |
| `backend/tests/test_agent_map_live_model.py`, `test_phase_23_2_14_no_reentrant_locks.py` | STALE assertions UPDATED (4-7→4-8; lock count 14→15 with re-audit note) | quarantine (stale) |
| `backend/tests/test_phase_23_2_16_shortlist_doc_presence.py` | Doc path repointed to the archive | quarantine (moved doc) |
| `backend/tests/test_phase_23_2_12_layer1_pipeline_active.py`, `test_phase_23_2_5_kill_switch_no_false_fires.py`, `test_phase_23_2_11_bq_table_freshness.py`, `test_phase_23_2_9_ticker_meta_latency.py` | `requires_live` skipifs with exact-dependency reasons | quarantine (live probes) |

NO code: F-9 (operator proposal presented in live_check §D), F-8/F-3/F-18 (escalated to phase-57 per the map).

## Verification command output (verbatim)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -q
749 passed, 12 skipped, 6 xfailed, 1 warning in 74.67s (0:01:14)
$ test -f handoff/current/live_check_56.2.md && echo PASS
PASS
```

## Key outcomes

1. The three silent away-week failure modes (rail down, degraded scores published, damping removed) now alert P1 to Slack at the cycle level — fail-loud per the Write-Audit-Publish / output-assertion pattern.
2. The approve path no longer depends on the direct-API key when the system runs the Max CLI rail (root cause deeper than 55.2's bound: the route flag was never honored on this path).
3. llm_call_log meters the CLI rail (both legs, ok=False on failure) and the Gemini lite path now stamps agent+ticker — F-E's blindness closed for future audits.
4. Full backend pytest green with an honest, root-cause-classified quarantine (2 stale UPDATED, 7 repointed, 5 live-probes skipped-with-reasons, 1 pollution ROOT-CAUSE-FIXED).
5. A real bug was found and fixed by the new tests themselves: the falsy-zero trap in the degraded-guard predicate (`confidence=0` was being defaulted away).

## Honest limitations

- The end-to-end Approve transcript needs the operator (bot-message filtering + slack-bot restart) — the criterion's one-line-operator-action branch is exercised; expected post-restart behavior documented in live_check §B.
- The fixes take effect in the running backend/slack-bot processes at their next restart (operator window; same note as 56.1).
- F-7's fallback VALUE deliberately unchanged (live-selection impact); the loud alert is the 56.2 deliverable, the redesign is phase-57-gated.
