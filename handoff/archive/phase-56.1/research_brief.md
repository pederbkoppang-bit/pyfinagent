# Research Brief — phase-56.2 "Ops fixes" (Research Gate)

**Tier:** complex. **Date:** 2026-06-10. **Step:** 56.2 (phase-56, fix
work, finding-ID-driven). Inputs: 55.3 synthesis table
(`handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` §1, finding IDs
F-1..F-19); 55.2 ops+skills audit; 55.1 post-mortem.

This brief is **fix-design-first**: per-finding recommended option + file:line
anchor + test design, then the pytest failure inventory, then external sources
+ recency scan + query log + JSON envelope.

---

## PART A — INTERNAL FIX-DESIGN (file:line anchored)

### Shared infrastructure discovered (reused by F-4 + F-5)

**Slack alert routing primitive — `raise_cron_alert` / `raise_cron_alert_sync`**
(`backend/services/observability/alerting.py:119` async, `:185` sync). This is
the canonical "alert to Slack from a backend service" path. It:
- routes through `backend.tools.slack.send_notification` (webhook, NOT the
  AsyncApp-coupled slack_bot process — that coupling was the phase-23.2.18 bug);
- is dedup-aware (`AlertDeduper`, default 3 occurrences in 5 min, repeat 1h);
- **P0/critical bypasses dedup and always fires** (`:71`);
- is **fail-open** — never raises; logs WARNING if `slack_webhook_url` unset.

`kill_switch.pause()` already uses `raise_cron_alert_sync(source="kill_switch",
error_type="auto_pause_<trigger>", severity="P1", title=..., details={...})` at
`kill_switch.py:169-179`. **F-4 and F-5 should reuse this exact primitive** —
the autonomous_loop is async, so call the awaitable `raise_cron_alert(...)`
directly; no new notifier needed.

`_route_exception_to_p1` lives in `backend/slack_bot/scheduler.py` (the slack_bot
process side); it is NOT the right entry point from backend services — use
`alerting.raise_cron_alert` instead (decoupled, webhook-based).

---

### F-4 — claude-CLI rail-health detect + alert (HIGH; fix-in-56.2)

**Finding:** the `claude` CLI OAuth session failed unattended during the away
week; the .env key was valid throughout (direct-API probes succeeded minutes
before the failed Approve). The key-scrub (`claude_code_client.py:159-162`,
phase-38.13.1) is a CORRECT billing guard — do NOT un-scrub (55.2 F-A1). The
gap: no health-check distinguished "rail down" from "no work".

**Where the rail is invoked:** `claude_code_invoke()` at
`claude_code_client.py:79`; the autonomous-loop caller is `_run_claude_analysis`
at `autonomous_loop.py:1573-1592` (gated by
`settings.paper_use_claude_code_route`).

**Recommended option — cheap CLI probe at cycle start, alert-once on failure.**
Add a `claude_code_health_probe()` helper to `claude_code_client.py` that runs a
**non-LLM, free** subprocess check. Options assessed:
- `claude auth status`-style subprocess (or `claude --version` with an auth
  precheck) — does NOT consume tokens, just checks the OAuth/keychain session.
  CHEAPEST and FREE. **RECOMMENDED.** (External B-1 validates synthetic/liveness
  probes that exercise the real auth path.)
- A 1-token `claude --print` invoke — costs a token + latency; rejected (the
  constraint says the probe must be free/cheap; a 1-token call is neither
  strictly free nor deterministic re: model-default ceilings).

**Wiring:** call the probe **at cycle start in `run_paper_trading_cycle`**
(before the analysis dispatch ~`autonomous_loop.py:744`), guarded by
`if getattr(settings, "paper_use_claude_code_route", False):` so it is a no-op
on the direct-API rail. On probe failure, fire
`await raise_cron_alert(source="claude_code_rail", error_type="rail_down",
severity="P1", title="Claude Code CLI rail unhealthy -- approve-flow +
conviction overlay degraded", details={...})`. Non-blocking: the cycle continues
(the lite-analyzer already fails-soft to `text=""`); the alert is the operator's
away-week signal. Probe wrapped in its own try/except (fail-open) so a probe bug
never breaks a cycle.

**Why cycle-start (not watchdog/scheduler):** the watchdog (`scheduler.py`) runs
in the **slack_bot process**, which does NOT use the claude-code rail; the rail
is exercised only inside the backend autonomous cycle. Probing where the rail is
actually used avoids a false "healthy" from a different process. (External B-1:
"test the path you actually use" — synthetic-transaction monitoring beats
shallow health endpoints.)

**Test design (unit):** `test_phase_56_2_rail_health.py` —
(a) monkeypatch `subprocess.run` so the probe command returns non-zero -> assert
`claude_code_health_probe()` returns False;
(b) monkeypatch the probe to return False + assert `raise_cron_alert` is called
with `severity="P1"` and `source="claude_code_rail"` (patch alerting,
assert_called_once); (c) probe returns True on exit-0. No live CLI, no network,
no tokens.

**Decision-impact:** NONE. Observability-only addition; does not touch the US
momentum core decision math, does not change any trade, does not un-scrub the
key. Do-no-harm clean.

---

### F-5 — degraded-scoring guard (HIGH; in 56.2 criteria) + F-7 conviction sentinel

**Finding (F-5):** 11 rows scored `0.0/HOLD` (UPPERCASE tell) published silently
by the digest. `formatters.py:37` (`format_analysis_result`) defaults a missing
score to 0. No output assertion distinguishes "scored 0" from "scoring failed".

**Root-cause trace (CODE-CONFIRMED):** in `_run_claude_analysis`
(`autonomous_loop.py:1601-1607`), when the rail returns `text=""` (rail down),
the JSON regex fails to match and `analysis` defaults to
`{"action":"HOLD","confidence":0,"score":5,...}` (score **5**, UPPERCASE HOLD).
The `0.0` the digest shows is `formatters.py:37`'s missing-field default for
rows where `final_score`/`final_weighted_score` is absent. The reliable tell is
**confidence==0 + UPPERCASE recommendation + the parse-fail score sentinel**.
The degraded path emits a value indistinguishable from a real neutral.

**Recommended option — cycle-level zero-score assertion in autonomous_loop.**
After the analysis gather (`autonomous_loop.py:820` `candidate_analyses` +
`:827` `holding_analyses`), compute the count of analyses whose `final_score ==
0` OR (`confidence == 0` AND the reason string marks a parse/rail failure). If
**ALL** analyses in the cycle are degraded, OR `N >= 3` degraded, fire
`await raise_cron_alert(source="autonomous_loop", error_type="degraded_scoring",
severity="P1", title="Cycle scoring degraded -- N/M analyses scored 0 (scoring
backend likely down)", details={...})` and stamp `summary["degraded"] = True`.
Do NOT publish the cycle's recommendations as confident neutrals.

This is the right layer (not `formatters.py`): the formatter is downstream and
per-row; the cycle-level guard sees the aggregate ("all 11 scored 0") which is
the actual signal. (External B-2: fail-loud output assertions belong at the
producer, where the invariant "a cycle cannot legitimately score every name 0"
is checkable; silent defaults are the anti-pattern.)

**F-7 — conviction fallback sentinel (HIGH; in 56.2):** the 55.3 finding cites
`meta_scorer.py:254 _fallback_all emits conviction 10.00`. **CORRECTION FROM
CODE:** `_fallback_all` (`meta_scorer.py:249-256`) sets
`conviction_score = _fallback_conviction(c)` which is
`max(1, min(10, round(composite_score)))` (`:138-142`), NOT a hardcoded 10.00.
It emits 10 only when `composite_score >= 9.5`. So the real failure mode is:
**the fallback inherits the raw composite score, so a high-momentum name keeps
its max-aggression rank even though the damping LLM leg is dead** — the damping
is *silently removed*. That is the F-7 substance.

**Consumer check (decision-impact — REQUIRED by constraints):** the only
consumer is `autonomous_loop.py:698-708`:
`candidates = await meta_score_candidates(...)`. The candidates come back
**sorted desc by conviction_score** (`meta_scorer.py:240/256`), and downstream
the loop takes `new_candidates[:settings.paper_analyze_top_n]` (`:723`) — i.e.
**conviction_score determines top-K analyze order/selection.**

Therefore changing the fallback from `round(composite_score)` to a flat neutral
(e.g. 5.0) **WOULD change live top-K ordering** when the LLM leg is down: every
candidate would collapse to the same neutral, so the secondary sort key / input
order would decide selection instead of the composite re-rank. **This is a
live-selection behavior change and per the constraints needs a flag or operator
sign-off.**

**Recommended option (do-no-harm safe):** do NOT silently flip the fallback
value (that changes selection). Instead:
1. Keep `_fallback_conviction` as-is for ORDERING (preserves current selection
   behavior — byte-identical top-K), BUT
2. Tag the fallback path: set `summary["meta_scorer_degraded"] = True` when
   `meta_score_candidates` fell back for all candidates, and have the F-5 guard
   treat that as a degraded signal -> Slack alert. This makes the silent
   damping-removal LOUD without changing the trade.
3. The *actual* neutral-sentinel redesign (emit conviction 5.0 + skip-overlay
   flag) is a **behavior change -> escalate to a config-gated flag in phase-57**
   (it belongs with the binding-RiskJudge FEATURE which already reworks the
   decision-consumption layer). Present as an operator decision.

**Test design (unit):** `test_phase_56_2_degraded_guard.py` —
(a) cycle result where all analyses have `final_score==0` -> assert
`raise_cron_alert` called with `error_type="degraded_scoring"` and
`summary["degraded"] is True`;
(b) N=3 zeros among 6 -> alert fires; N=2 -> no alert (threshold boundary);
(c) meta_scorer fallback-all path sets `meta_scorer_degraded` and is picked up
by the guard;
(d) **decision-invariant test (do-no-harm)**: `meta_score_candidates` fallback
returns the SAME top-K order as today (byte-identity guard) — proves no live
selection change.

---

### F-6 — llm_call_log instrumentation (HIGH; fix-in-56.2)

**Finding:** `llm_call_log` has ZERO rows for 6 of 7 away-week cycles —
`log_llm_call` is absent from `claude_code_client.py` (CLI rail) + the lite
analyzer; `cycle_id`/`ticker` columns exist but are NULL.

**The logger is ready-made.** `log_llm_call` (`api_call_log.py:203`) already
accepts `provider, model, agent, latency_ms, ttft_ms, input_tok, output_tok,
ok, ticker, cycle_id` and **auto-populates `cycle_id`/`session_cost_usd` from
autonomous_loop module state** (`:237-248`) — so the CLI caller only needs to
pass `provider`, `model`, `agent`, `latency_ms`, `ok`, `ticker`. Fail-open
(`:276`, never raises).

**The CLI envelope carries the data.** `claude_code_invoke` returns
`duration_ms`, `usage.input_tokens`, `usage.output_tokens`, `total_cost_usd`
(`claude_code_client.py:110-114`, already logged at `:226-231`).

**Recommended option — log at the CALLER, not inside `claude_code_invoke`.**
`claude_code_invoke` is a generic module-level function with no ticker context;
the caller `_run_claude_analysis` knows the ticker. Mirror the GeminiClient
pattern (`llm_client.py:1065-1082`): after each `claude_code_invoke` returns
(trader call `autonomous_loop.py:1580`, risk-judge call `:1636`), call
`log_llm_call(provider="claude-code", model=model_name, agent="lite_trader"|
"lite_risk_judge", latency_ms=envelope.get("duration_ms"),
input_tok=(envelope.get("usage") or {}).get("input_tokens",0),
output_tok=...output_tokens, ok=True, ticker=ticker)`. On `ClaudeCodeError`,
log with `ok=False` (records "the rail fired and failed" — ties into F-4).

**Lite Gemini path (`_run_gemini_analysis`):** routes through
`make_client(...).generate_content(...)` (`autonomous_loop.py:1775/1795`), which
is `GeminiClient.generate_content` — **already instrumented** at
`llm_client.py:1065-1082` IF the caller passes `_role`/`_ticker` in
generation_config. **Gap:** `_run_gemini_analysis` does NOT pass them (`:1798`,
`:1835`), so those rows log NULL agent/ticker. **Fix:** add
`"_role": "lite_trader"`/`"lite_risk_judge"` and `"_ticker": ticker` to the two
`generation_config` dicts. Zero new log calls — the existing GeminiClient
instrumentation picks them up.

**Test design (unit):** `test_phase_56_2_llm_log.py` —
(a) monkeypatch `claude_code_invoke` to return a fake envelope + patch
`log_llm_call`; run `_run_claude_analysis` -> assert `log_llm_call` called twice
with `provider="claude-code"`, `ticker=<t>`;
(b) rail error -> `log_llm_call` called with `ok=False`;
(c) `_run_gemini_analysis` passes `_ticker`/`_role` into generation_config
(assert dict contents via a spy on `generate_content`).

**Table note:** the pytest run showed `api_call_log` table NotFound warnings —
that is a DIFFERENT table (`api_call_log`, the HTTP-call log) from `llm_call_log`
(LLM telemetry). `log_llm_call` writes to `llm_call_log` (`api_call_log.py:280+`
`flush_llm`). The instrumentation is fail-open regardless of table presence; the
table had rows historically so it exists.

---

### F-14 — dead approve button (LOW latent; register-or-remove)

**Finding:** `governance.py:166-175` defines `action_id: "approval_approve"` /
`"approval_deny"` buttons with **no `@app.action` handler** -> clicking no-ops
silently; the path "fails CLOSED on action by accident".

**CODE FINDING — the buttons are in fully DEAD code.** `send_approval_gate`
(`governance.py:136`) has **ZERO callers** anywhere in the repo (grep confirmed).
The approval-gate class is never invoked. So this is not a live fail-open risk
today; it is latent (would bite if someone wires `send_approval_gate` without
adding handlers).

**Recommended option — REMOVE the buttons (not register).** Since the gate is
never sent, registering handlers wires a path with no producer. Minimal safe
change: delete the `actions` block (`governance.py:160-178`) and keep the
approval text as an informational section, OR add a docstring noting the gate is
unwired pending a producer. If the operator wants the gate live later, that is a
phase-57 feature (wire producer + handler + test together).

**Alternative (if operator prefers keeping the buttons):** register
`@app.action("approval_approve")`/`("approval_deny")` following the
`app_home.py:397-411` pattern — handler calls `await ack()` FIRST (Bolt
ack-within-3s discipline, external B-4), then posts "Approval recorded (no-op:
governance gate not wired to an executor)". Makes the failure LOUD not silent.

**Recommendation: REMOVE** (dead code; do-no-harm; no behavior to preserve). If
kept, the ack+post handler is the safe shape. No test required for removal; if
registering, a unit test asserting `ack` is called.

---

### F-8 — RiskJudge prompt/context (MEDIUM-HIGH) — RECOMMEND ESCALATE

**Finding:** the lite RiskJudge SYSTEM prompt (`autonomous_loop.py:1434`) cites a
phantom "10% of portfolio in one sector" cap vs the config
`paper_max_per_sector_nav_pct` (30%), and the template (`:1441-1453`) injects NO
live sector breakdown -> the judge reasons "no portfolio sector breakdown
provided".

**Blast-radius assessment (do-no-harm):** changing `_LITE_RISK_JUDGE_SYSTEM`
(`:1429`) and `_LITE_RISK_JUDGE_TEMPLATE` (`:1441`) **alters LLM behavior on the
lite path** — the RiskJudge `decision`/`recommended_position_pct` feed
`risk_assessment` which sizes positions (`portfolio_manager.py`). This is the
lite path (NOT the US momentum-core math) — but it is still a decision-affecting
prompt change. Per the constraints ("if risky, escalate to phase-57"), and
because **F-8 is already in the phase-57 binding-RiskJudge FEATURE spec** (55.3
§2.6: "inject the live portfolio sector breakdown + the real configured caps
into the judge's prompt/context"), the prompt+context rework belongs there with
its own regression fixture + event-study.

**Recommended: ESCALATE to phase-57 (do NOT fix in 56.2).** Rationale: any doubt
-> escalate (constraint). The fix is non-trivial (need to thread live sector
weights into the per-ticker call, which has no portfolio context today) and is
decision-affecting. A cheap-and-safe subset that COULD land in 56.2 if the
operator insists: fix only the **phantom number** — make the system prompt read
the configured cap instead of the hardcoded "10%". Even that is a prompt change;
recommend bundling into phase-57 for a clean measured rollout. **Default:
ESCALATE.**

---

### Watchdog bounded tweak (F-C family, LOW; WONTFIX-acceptable)

(Pending: digest cron times + scheduler probe lines — read next.)

---

### F-9 — kill-switch SOD anchor (MEDIUM) — OPERATOR PROPOSAL ONLY (no code)

(Pending: confirm `paper_trader.py` SOD anchor + `update_sod_nav` call site —
read next.)

---

### Test quarantine — `requires_live` marker

(Pending: full failure classification — see PART C.)

---

## PART C — PYTEST FAILURE INVENTORY

**13 failures observed** (725 passed, 2 skipped, 8 xfailed, 1 xpassed; 115s).
Classification in progress.

Failed tests:
1. `test_agent_map_live_model.py::test_endpoint_injects_live_model_field`
2. `test_phase_23_2_10_watchdog_no_fire_7d.py::test_..._watchdog_log_present_and_fresh`
3. `test_phase_23_2_12_layer1_pipeline_active.py::test_..._at_least_one_full_proxy_in_last_7d`
4. `test_phase_23_2_14_no_reentrant_locks.py::test_..._threading_lock_count_matches_roster`
5-11. `test_phase_23_2_16_shortlist_doc_presence.py::` (7 tests — moved doc)
12. `test_phase_23_2_5_kill_switch_no_false_fires.py::test_..._historical_false_fires_documented`
13. `test_rainbow_canary.py::test_canary_snapshot_from_buffer_partitions_by_source`

(Per-test classification next read.)

---

## PART B — EXTERNAL RESEARCH

(In progress — sources, recency scan, query log, JSON envelope at tail.)
