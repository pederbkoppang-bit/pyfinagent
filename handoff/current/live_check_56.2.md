# live_check_56.2 — Ops fixes: live evidence

**Step:** 56.2. **Date:** 2026-06-10. **Required shape (masterplan):** the approve-flow transcript + the degraded-scoring guard test output + the pytest summary line showing green-with-quarantine.

## A. Finding-ID map (criterion 1 — every P0/P1 from the 55.3 table dispositioned)

| Finding (55.3 §1) | Severity | Disposition in 56.2 | Evidence |
|---|---|---|---|
| F-1 frontend FX rendering | CRITICAL | FIXED in 56.1 | live_check_56.1 |
| F-2 KR trade-ledger rows | HIGH | FIXED in 56.1 (+ operator-gated backfill pending) | live_check_56.1 |
| F-3 RiskJudge REJECT advisory-only | HIGH | **ESCALATED — phase-57 FEATURE candidate** (binding gate is a live-behavior change; config-gated + measured per the 55.3 spec; awaiting `PHASE-57:` reply) | 55.3 §2.6 |
| F-4 claude-CLI rail down silently | HIGH | **FIXED**: `claude_code_health_probe()` (free `claude auth status`, scrubbed env, never raises) + cycle-start wiring gated by `paper_use_claude_code_route` + `raise_cron_alert` P1 on failure | `claude_code_client.py` (new probe), `autonomous_loop.py` (cycle start); tests §C |
| F-5 silent 0.0/10 degraded scoring | HIGH | **FIXED**: cycle-level guard `_degraded_scoring_check` (ALL-degraded or ≥3 zeros → P1 Slack alert + `summary["degraded"]=True`; counts the 05-27 signature: confidence-0 + UPPERCASE rec) | `autonomous_loop.py`; tests §C |
| F-6 llm_call_log blind to the analysis rail | HIGH | **FIXED**: `_log_claude_code_call` meters both CLI-rail legs (lite_trader + lite_risk_judge; ok=False on rail error; envelope token/latency mapping; cycle_id auto-stamps) + `_role`/`_ticker` tags on the Gemini lite path (its client was already instrumented) | `autonomous_loop.py`; tests §C |
| F-7 conviction fallback silently removes damping | HIGH | **FIXED (observability) + ESCALATED (value)**: `_all_conviction_fallback` detection → P1 alert + `summary["meta_scorer_degraded"]`; the fallback VALUE stays byte-identical (it drives top-K selection — changing it is a live-behavior change deferred to the gated phase-57 redesign) | `autonomous_loop.py`; byte-identity test §C |
| F-8 RiskJudge prompt/context divergence | MEDIUM-HIGH | **ESCALATED — phase-57** (decision-affecting prompt change; already inside the FEATURE spec) | 55.3 §2.6 |
| F-9 kill-switch SOD anchor | MEDIUM | **OPERATOR DECISION presented** (§D below); NO code change; thresholds untouched. The 06-05 unit-test fix is NOT required (55.1 ruled CORRECTLY-DID-NOT-TRIP — the criterion's IFF condition is false) | §D |
| F-14 dead approve buttons | LOW | **FIXED**: dead `actions` block removed from `governance.py` (zero callers; replaced with an explicit typed-reply instruction — fail-safe default) | `governance.py` |
| F-18 churn | HIGH (strategy) | **ESCALATED — phase-57 LEVER candidate** (awaiting `PHASE-57:` reply) | 55.3 §2.6 |
| watchdog (F-C family) | LOW | **FIXED (bounded)**: watchdog probe timeout 10s→30s (`scheduler.py`) per the 55.2 root cause (backend never down; event-loop starvation during the 18:00Z cycle) | `scheduler.py` |
| Criterion-2 approve flow | HIGH | **FIXED + one-line operator action** (§B) | `ticket_queue_processor.py`; tests §C |

Nothing was changed without a finding ID. Test-hygiene changes (criterion 4) are root-cause-classified in §E.

## B. Approve-flow: root cause, fix, transcript status (criterion 2)

**Root cause (deeper than 55.2's bound):** the Slack approve path (`handle_any_message` → ticket → `ticket_queue_processor._spawn_real_agent`) **always used the direct Anthropic SDK** and never honored `paper_use_claude_code_route` — so while the trading loop ran on the Max-subscription CLI rail, the operator's only control surface depended on the direct-API account. **Fix:** `_spawn_real_agent` now routes through `claude_code_invoke` (system prompt preserved, 60s timeout) when the route flag is set; the direct-SDK branch is unchanged when the flag is off. Unit tests assert the right rail fires per flag (§C).

**Transcript status — the criterion's OR-branch is exercised:** a true end-to-end transcript requires the OPERATOR to type "Approve" in #ford-approvals (bot messages are filtered by `handle_any_message`, and the slack_bot process must be restarted to load this fix). **One-line operator action: restart the slack bot (`python -m backend.slack_bot.app`) and type `Approve` in #ford-approvals — expect an agent reply via the claude-code rail instead of the missing-key error.** (The CLI rail is verified healthy live: `claude auth status` → loggedIn: true, §55.2 evidence; the probe now guards it every cycle.)

## C. Test output (criterion 3 — degraded guard covered by unit test; criterion 1 — regression tests)

```
$ python -m pytest backend/tests/test_phase_56_2_ops_fixes.py -q
18 passed, 1 warning in 1.97s
```
Covers: rail-probe semantics (exit-code, loggedIn parse, timeout/missing-binary never-raise, env scrub), degraded-guard thresholds (all-zero fires; 3/6 fires; 2/6 quiet; confidence-0+UPPERCASE tell counted — including the falsy-zero-trap regression found WHILE writing the test; empty-cycle quiet), conviction-fallback detection + **ordering byte-identity (do-no-harm)**, llm-metering envelope mapping + ok=False + never-raises, and ticket-rail routing both flag states.

**Full backend suite (criterion 4 — green with quarantine):**
```
$ python -m pytest backend/tests -q
749 passed, 12 skipped, 6 xfailed, 1 warning in 74.67s (0:01:14)
```
Exit code 0. (Verification command's pytest leg verbatim.)

## D. F-9 kill-switch SOD anchor — OPERATOR DECISION (presented, not applied)

> PROPOSAL (operator decision): re-anchor the kill-switch start-of-day NAV to the PRIOR day's end-of-day snapshot instead of the current evaluation instant. Today the once-daily cycle sets SOD = NAV at the same moment it evaluates (`paper_trader.py:1034`, `kill_switch.py:244`), so the 4% daily-loss leg compares NAV to itself and is structurally dead; only the 10% trailing-drawdown leg can fire. Re-anchoring SOD to yesterday's close would let the daily leg see a real overnight+intraday move within one cycle. The 4% daily and 10% trailing limits are UNCHANGED — this is an anchor fix, not a threshold change. Risk: a larger overnight gap could trip the daily leg on day 1; recommend a one-cycle dry-run logging the would-be daily_loss_pct under the new anchor before enabling. Reply "F-9: APPROVED" to schedule it as a 56.x follow-up, or leave it parked.

(The 06-05 scenario unit-test fix is NOT required: 55.1's audited verdict was CORRECTLY-DID-NOT-TRIP, so the criterion's IFF condition is false.)

## E. Test-quarantine audit trail (criterion 4)

Root-cause classification of the 13 observed failures (NOT blanket-skipped; the "16 env-coupled" framing in cycle_block_summary under-counted by mixing buckets):

| Bucket | Tests | Action |
|---|---|---|
| STALE assertions (2) | agent-map model pin 4-7→4-8; lock count 14→15 (`alerting.py:64` AlertDeduper is a real, reviewed, non-re-entrant lock) | **UPDATED** (real test regressions — skipping would hide drift) |
| Live-BQ/state probes (4) | full-path-proxy-7d; kill-switch-audit-log-rows; 2x BQ table freshness (paper_*, 24h SLA trips before the daily cycle) | `@pytest.mark.requires_live` + skipif `PYFINAGENT_LIVE_TESTS != "1"`, per-test reasons naming the exact dependency (NEW `pytest.ini` registers the marker) |
| Live-HTTP probe (1) | ticker-meta endpoint (flaky during the 18:00Z cycle window — the F-C pattern in test form) | `requires_live` + honest reason |
| Moved doc (7) | phase-23.2.16 shortlist tests | **REPOINTED** to `handoff/archive/phase-23.2.16/` (the archive-handoff hook moved it; doc still exists) |
| Test pollution (1-2) | rainbow-canary buffer partition (passes alone, failed in suite) | **ROOT-CAUSE FIXED**: `reset_buffer_for_test()` now also re-arms `_last_flush_ts` — the time-based flush trigger was ripe after ~2 min of suite runtime and drained the buffer mid-test (order-dependence eliminated, not skipped) |

## F. Do-no-harm statement

All 56.2 changes are observability/ops-layer: probe, guard, alert, metering, dead-code removal, timeout constant, test hygiene. NO change to screener/optimizer/portfolio-manager decision math; the conviction-fallback ranking is byte-identical (unit-tested); the prompt templates are untouched (F-8 escalated for exactly that reason). The backend on :8000 picks these up at its next restart (same operator window as the 56.1 deploy note).
