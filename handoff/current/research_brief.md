# Research Brief — phase-56.2 "Ops fixes" (Research Gate)

**Tier:** complex. **Date:** 2026-06-10. **Step:** 56.2 (phase-56, fix work,
finding-ID-driven). Inputs: 55.3 synthesis table
(`handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` §1, IDs F-1..F-19);
55.2 ops+skills audit; 55.1 post-mortem.

Fix-design-first: per-finding recommended option + file:line anchor + test
design, then the pytest failure inventory + quarantine design, then external
sources + recency scan + query log + JSON envelope.

---

## PART A — INTERNAL FIX-DESIGN (file:line anchored)

### Shared infra discovered (reused by F-4 + F-5)

**Slack alert primitive — `raise_cron_alert` (async, `alerting.py:119`) /
`raise_cron_alert_sync` (`:185`).** Canonical "alert to Slack from a backend
service". Routes through `backend.tools.slack.send_notification` (webhook, NOT
the AsyncApp-coupled slack_bot process — that coupling was the phase-23.2.18
bug). Dedup-aware (`AlertDeduper`: 3 occurrences / 5 min, repeat 1h); **P0/
critical bypasses dedup, always fires** (`:71`); **fail-open** (never raises;
WARNs if `slack_webhook_url` unset). `kill_switch.pause()` already uses
`raise_cron_alert_sync(source="kill_switch", error_type="auto_pause_...",
severity="P1", ...)` at `kill_switch.py:169-179`. **F-4 + F-5 reuse this
exact primitive** — the autonomous_loop is async, so call the awaitable
`raise_cron_alert(...)`; no new notifier needed. (`_route_exception_to_p1`
at `scheduler.py:26` is the slack_bot-process side — NOT the entry point from
backend services.)

---

### F-4 — claude-CLI rail-health detect + alert (HIGH; fix-in-56.2)

**Finding:** the `claude` CLI OAuth session failed unattended away-week; the
.env key was valid throughout. Key-scrub (`claude_code_client.py:159-162`) is a
CORRECT billing guard — do NOT un-scrub (55.2 F-A1). Gap: no health-check
distinguished "rail down" from "no work".

**Rail invocation sites:** `claude_code_invoke()` (`claude_code_client.py:79`);
caller `_run_claude_analysis` (`autonomous_loop.py:1573-1592`), gated by
`settings.paper_use_claude_code_route`.

**Recommended — cheap CLI probe at cycle start, alert-once on failure.** Add
`claude_code_health_probe()` to `claude_code_client.py` running a **free,
non-LLM** subprocess (`claude auth status` / equivalent; checks OAuth/keychain,
consumes no tokens). RECOMMENDED over a 1-token `--print` invoke (rejected: not
strictly free, non-deterministic re: model-default ceilings; constraint = probe
must be free/cheap). Wire at cycle start in `run_paper_trading_cycle`
(~`autonomous_loop.py:744`), guarded `if getattr(settings,
"paper_use_claude_code_route", False):` (no-op on direct rail). On failure:
`await raise_cron_alert(source="claude_code_rail", error_type="rail_down",
severity="P1", title="Claude Code CLI rail unhealthy ...", details={...})`.
Non-blocking (lite-analyzer already fails-soft to `text=""`); own try/except so
a probe bug never breaks a cycle.

**Why cycle-start (not watchdog/scheduler):** the watchdog runs in the
slack_bot process, which does NOT use the claude-code rail; the rail is
exercised only inside the backend cycle. Probe where the rail is actually used
(external B-1: synthetic-transaction / "test the path you use" beats shallow
health endpoints).

**Test (unit) `test_phase_56_2_rail_health.py`:** (a) monkeypatch
`subprocess.run` non-zero -> probe returns False; (b) probe False -> assert
`raise_cron_alert` called once w/ `severity="P1"`, `source="claude_code_rail"`;
(c) exit-0 -> True. No live CLI/network/tokens. **Decision-impact: NONE**
(observability-only; no trade/core-math change; no un-scrub). Do-no-harm clean.

---

### F-5 degraded-scoring guard (HIGH; in 56.2 criteria) + F-7 conviction sentinel

**F-5 finding:** 11 rows `0.0/HOLD` (UPPERCASE tell) published silently by the
digest; `formatters.py:37` defaults missing score to 0. No assertion
distinguishes "scored 0" from "scoring failed".

**Root-cause (CODE-CONFIRMED):** `_run_claude_analysis` (`autonomous_loop.py:
1601-1607`) — rail returns `text=""` -> JSON regex fails -> `analysis =
{"action":"HOLD","confidence":0,"score":5,...}` (score 5, UPPERCASE HOLD). The
digest's `0.0` is `formatters.py:37`'s missing-`final_weighted_score` default.
Reliable tell = **confidence==0 + UPPERCASE recommendation + parse-fail
reason**. The degraded path emits a value indistinguishable from a real neutral.

**Recommended — cycle-level zero-score assertion in autonomous_loop.** After the
gather (`autonomous_loop.py:820` `candidate_analyses` + `:827`
`holding_analyses`), count analyses with `final_score == 0` OR (`confidence ==
0` AND reason marks parse/rail failure). If ALL degraded OR `N >= 3` -> `await
raise_cron_alert(source="autonomous_loop", error_type="degraded_scoring",
severity="P1", title="Cycle scoring degraded -- N/M scored 0 ...", ...)` and
stamp `summary["degraded"] = True`. Do NOT publish recs as confident neutrals.
Right layer (not `formatters.py`): the cycle-level guard sees the aggregate
("all 11 scored 0") which is the actual invariant (external B-2: fail-loud
output assertions belong at the producer; silent defaults = anti-pattern).

**F-7 conviction sentinel (HIGH; in 56.2).** 55.3 cites `meta_scorer.py:254
_fallback_all emits conviction 10.00`. **CODE CORRECTION:** `_fallback_all`
(`:249-256`) sets `conviction_score = _fallback_conviction(c)` =
`max(1,min(10,round(composite_score)))` (`:138-142`) — NOT a hardcoded 10.00;
emits 10 only when composite >= 9.5. Real failure mode: **fallback inherits the
raw composite, so a high-momentum name keeps its max-aggression rank while the
damping LLM leg is dead** — damping is *silently removed*.

**Consumer / decision-impact (REQUIRED by constraints):** sole consumer
`autonomous_loop.py:698-708`; candidates returned **sorted desc by
conviction_score** (`meta_scorer.py:240/256`); downstream
`new_candidates[:settings.paper_analyze_top_n]` (`:723`). So conviction_score
**determines top-K selection.** Flipping the fallback to a flat neutral (5.0)
**WOULD change live top-K ordering** when the LLM leg is down (all collapse to
the same value -> secondary key/input order decides). **This is a live-selection
behavior change -> needs a flag or operator sign-off** per constraints.

**Recommended (do-no-harm safe):** do NOT silently flip the fallback value.
(1) Keep `_fallback_conviction` as-is for ORDERING (byte-identical top-K). (2)
Tag the path: `summary["meta_scorer_degraded"] = True` when fallback fired for
all candidates; the F-5 guard treats it as a degraded signal -> Slack alert
(makes silent damping-removal LOUD without changing the trade). (3) The actual
neutral-sentinel redesign (conviction 5.0 + skip-overlay flag) is a behavior
change -> **escalate to a config-gated flag in phase-57** (belongs with the
binding-RiskJudge FEATURE that reworks decision-consumption). Operator decision.

**Test (unit) `test_phase_56_2_degraded_guard.py`:** (a) all `final_score==0`
-> alert `error_type="degraded_scoring"` + `summary["degraded"] is True`;
(b) N=3/6 zeros -> fires, N=2 -> no fire (threshold boundary); (c) meta_scorer
fallback-all sets `meta_scorer_degraded` + guard picks it up; (d) **do-no-harm
invariant**: fallback returns SAME top-K order as today (byte-identity).

---

### F-6 — llm_call_log instrumentation (HIGH; fix-in-56.2)

**Finding:** `llm_call_log` ZERO rows for 6/7 away-week cycles; `log_llm_call`
absent from `claude_code_client.py` + lite analyzer; cycle_id/ticker NULL.

**Logger is ready-made.** `log_llm_call` (`api_call_log.py:203`) accepts
`provider, model, agent, latency_ms, ttft_ms, input_tok, output_tok, ok,
ticker, cycle_id`; **auto-populates cycle_id/session_cost_usd from
autonomous_loop module state** (`:237-248`); fail-open (`:276`). The CLI
envelope carries `duration_ms`, `usage.input_tokens/output_tokens`,
`total_cost_usd` (`claude_code_client.py:110-114`).

**Recommended — log at the CALLER (not inside `claude_code_invoke`, which has
no ticker context).** Mirror the GeminiClient pattern (`llm_client.py:1065-
1082`): after each `claude_code_invoke` (trader `autonomous_loop.py:1580`,
risk-judge `:1636`), call `log_llm_call(provider="claude-code", model=
model_name, agent="lite_trader"|"lite_risk_judge", latency_ms=envelope.get
("duration_ms"), input_tok=..., output_tok=..., ok=True, ticker=ticker)`. On
`ClaudeCodeError` -> log with `ok=False` (records rail-fired-and-failed; ties
to F-4).

**Lite Gemini path:** `_run_gemini_analysis` routes through
`make_client(...).generate_content` = `GeminiClient.generate_content`, which is
**already instrumented** at `llm_client.py:1065-1082` IF the caller passes
`_role`/`_ticker` in generation_config. **Gap:** `_run_gemini_analysis` omits
them (`:1798`, `:1835`) -> NULL agent/ticker. **Fix:** add `"_role"` +
`"_ticker": ticker` to the two `generation_config` dicts. Zero new log calls.

**Test (unit) `test_phase_56_2_llm_log.py`:** (a) patch `claude_code_invoke`
(fake envelope) + patch `log_llm_call`; `_run_claude_analysis` -> assert called
twice w/ `provider="claude-code"`, `ticker`; (b) rail error -> `ok=False`;
(c) `_run_gemini_analysis` passes `_ticker`/`_role` (spy on `generate_content`).

**Table note:** the pytest `api_call_log` NotFound warnings are a DIFFERENT
table (HTTP-call log) from `llm_call_log` (LLM telemetry, `flush_llm`
`api_call_log.py:280+`). Instrumentation is fail-open regardless; `llm_call_log`
had rows historically so it exists.

---

### Criterion 2 — Slack "Approve" -> "Missing API key for provider anthropic"

**Root cause (CODE-CONFIRMED, distinct path from F-4).** The Slack approve
flow: a message in `#ford-approvals` -> `commands.py:184` `handle_any_message`
-> ticket ingested (`ticket_ingestion.py`, classification is **keyword-only,
no LLM** `:23-53`) -> **`ticket_queue_processor._invoke_agent`
(`ticket_queue_processor.py:156-180`) runs the assigned agent via the DIRECT
Anthropic SDK** (`import anthropic`; `agent_model_map` all claude-*; `client =
anthropic.Anthropic(api_key=api_key)` `:180`). `:176-178`: `api_key =
settings.anthropic_api_key.get_secret_value() or os.getenv('ANTHROPIC_API_KEY')`
then `if not api_key: raise ValueError("ANTHROPIC_API_KEY not found ...")`.

**Key insight:** this path **does NOT honor `settings.paper_use_claude_code_
route`** — so even with the autonomous loop on the Max-subscription CLI rail,
the Slack approve->ticket->agent path tries the (credit-exhausted/keyless)
direct API. The "Missing API key for provider anthropic" is the SDK/wrapper
surfacing of the keyless direct call. **This is the criterion-2 transcript
blocker.**

**Recommended options (operator picks; both satisfy criterion 2):**
1. **Make the processor honor the CLI rail** — when
   `settings.paper_use_claude_code_route`, route `_invoke_agent` through
   `claude_code_invoke` instead of `anthropic.Anthropic`. Mirrors
   `_run_claude_analysis`'s rail switch. Restores the Approve path on the Max
   rail. (Cleanest; ~20 LOC; reuses claude_code_client.) RECOMMENDED if the
   operator wants the Approve flow live unattended.
2. **Escalate as a one-line operator action** — if the direct .env key is valid
   now (away-week was a transient OAuth issue), criterion 2 is met by capturing
   a transcript showing "Approve" succeeds with the current key, plus a note
   that the processor uses the direct rail (so it depends on the .env key being
   present, NOT the CLI rail). This is the "residual escalated with a one-line
   operator action" branch the criterion allows.

**Test (if option 1):** unit test asserting `_invoke_agent` calls
`claude_code_invoke` (not `anthropic.Anthropic`) when the route flag is set;
patch both, assert the right one fires. Capture the end-to-end transcript for
the live_check regardless of option.

---

### F-14 — dead approve button (LOW latent; register-or-remove)

**Finding:** `governance.py:166-175` defines `approval_approve`/`approval_deny`
buttons with **no `@app.action` handler** -> silent no-op; "fails CLOSED by
accident". **CODE FINDING:** `send_approval_gate` (`governance.py:136`) has
**ZERO callers** repo-wide -> fully DEAD code; not a live risk today, latent.

**Recommended — REMOVE the buttons** (delete the `actions` block
`governance.py:160-178`, keep the text as an informational section). Registering
handlers would wire a path with no producer. **Alternative (if operator keeps
them):** register `@app.action("approval_approve")`/`("approval_deny")` per the
`app_home.py:397-411` pattern — `await ack()` FIRST (Bolt 3s ack discipline,
external B-4), then post "Approval recorded (no-op: gate not wired to an
executor)". Default: **REMOVE** (dead code; do-no-harm; no behavior to
preserve). No test for removal; if registering, assert `ack` is called.

---

### F-8 — RiskJudge prompt/context (MEDIUM-HIGH) — RECOMMEND ESCALATE

**Finding:** lite RiskJudge SYSTEM prompt (`autonomous_loop.py:1434`) cites a
phantom "10% of portfolio in one sector" vs config
`paper_max_per_sector_nav_pct` (30%); template (`:1441-1453`) injects NO live
sector breakdown -> judge reasons "no portfolio sector breakdown provided".

**Blast radius (do-no-harm):** changing `_LITE_RISK_JUDGE_SYSTEM` (`:1429`) /
`_LITE_RISK_JUDGE_TEMPLATE` (`:1441`) **alters LLM behavior on the lite path** —
the judge `decision`/`recommended_position_pct` size positions
(`portfolio_manager.py`). Lite path, NOT US momentum-core math — but still
decision-affecting. F-8 is **already in the phase-57 binding-RiskJudge FEATURE
spec** (55.3 §2.6). **Recommended: ESCALATE to phase-57** (constraint = any
doubt escalate; fix is non-trivial — needs live sector weights threaded into a
per-ticker call with no portfolio context today). Cheap subset if operator
insists: fix only the phantom number (system prompt reads the configured cap) —
still a prompt change; recommend bundling into phase-57. **Default: ESCALATE.**

---

### Watchdog bounded tweak (F-C family, LOW; WONTFIX-acceptable)

**Layout:** morning digest 8:00 ET (`settings.morning_digest_hour=8`), evening
digest 17:00 ET (`evening_digest_hour=17`), watchdog every 15 min
(`watchdog_interval_minutes=15`). Digest probes use `httpx.AsyncClient(timeout=
30.0)` (`scheduler.py:399`, `:435`); the **watchdog** probe uses `timeout=10.0`
(`:485`) on `GET /api/health`, with state-transition gating (`:469-522`) —
alerts only on None->False / True->False / False->True.

**55.2 root cause:** event-loop starvation during the 18:00Z cycle made the 10s
`/api/health` probe time out -> True->False transition -> a "1/3 FAIL" alert,
but **the backend was never down** (the cycle's blocking work starved the loop;
the watchdog and the cycle share the backend's event loop... actually the
watchdog runs in the slack_bot process and hits the backend over HTTP, so the
starvation was on the BACKEND side answering `/api/health`). LOW severity.

**Recommended — smallest bounded change (pick one):**
1. **Raise the watchdog probe timeout 10s -> 30s** (`scheduler.py:485`) so a
   transiently busy backend during the cycle window doesn't read as "down".
   One-line; mirrors the digest 30s. SMALLEST.
2. **Add a cycle-running guard** — the backend already offloads cycle blocking
   work via `asyncio.to_thread` (phase-23.1.23, e.g. `autonomous_loop.py:720`,
   `:837`) specifically to keep `/api/health` responsive; the residual
   starvation is narrow. A guard that skips/relaxes the watchdog while a cycle
   is running is more code for little gain.
**Recommendation: option 1 (raise to 30s)** OR document WONTFIX (55.2 explicitly
marks F-C WONTFIX-acceptable). If touched, no new test needed (timeout is a
config constant); the existing watchdog tests cover the gating logic.

---

### F-9 — kill-switch SOD anchor (MEDIUM) — OPERATOR PROPOSAL ONLY (no code)

**Confirmed mechanism (CODE):** `evaluate_breach` (`kill_switch.py:244-248`):
`daily_loss_pct = (sod - current_nav)/sod`. SOD is set at `paper_trader.py:
1034-1035`: `if snap.sod_nav is None or sod_date != today: update_sod_nav(nav)`.
Under once-daily cadence the day's ONLY cycle sets `sod_nav = nav` at the same
instant it then evaluates -> `sod == current_nav` -> `daily_loss_pct ≈ 0` ->
**the daily-loss leg can never fire**; only the trailing-peak leg (`peak_nav`,
ratcheted across days, `:212-217`) has teeth. (06-05 verdict stands: CORRECTLY-
DID-NOT-TRIP — true day -2.82% < 4%, trailing 3.26% < 10%.)

**Draft operator-proposal text (no code change in 56.2):**
> PROPOSAL (operator decision): re-anchor the kill-switch start-of-day NAV to
> the PRIOR day's end-of-day snapshot instead of the current evaluation instant.
> Today the once-daily cycle sets SOD = NAV at the same moment it evaluates, so
> the 4% daily-loss leg compares NAV to itself and is structurally dead; only
> the 10% trailing-drawdown leg can fire. Re-anchoring SOD to yesterday's close
> would let the daily leg detect an overnight + intraday drop within one cycle.
> The 4% daily limit and 10% trailing limit are UNCHANGED — this is an anchor
> fix, not a threshold change. Risk: a larger overnight gap could trip the daily
> leg on day 1 of a re-anchor; recommend a one-cycle dry-run logging the
> would-be daily_loss_pct under the new anchor before enabling. This is the F-9
> finding; any threshold change remains a separate operator decision.

No code lands in 56.2; criterion 4 requires this be **presented as an operator
decision (proposal text), never auto-applied.** Kill-switch unit-test fix is
NOT required (55.1 ruled CORRECTLY-DID-NOT-TRIP; IFF condition false).

---

## PART C — PYTEST FAILURE INVENTORY + QUARANTINE DESIGN

**Full suite:** `13 failed, 725 passed, 2 skipped, 8 xfailed, 1 xpassed`
(115s). **No `pytest.ini` / `conftest.py` / pyproject pytest config exists** —
so the `requires_live` marker needs a NEW `pytest.ini` (register marker +
`addopts`) and a skipif helper keyed on an env var (`PYFINAGENT_LIVE_TESTS`).

**The 16-env-coupled framing in cycle_block_summary under-counts: of the 13
observed, 2 are STALE assertions (not env), 2 are TEST-POLLUTION (pass alone,
fail in suite — NOT live-BQ), 2 are live-BQ probes, 7 are the moved doc.** A
blanket `requires_live` skip would WRONGLY hide the 2 stale + 2 pollution
failures. Classify by ROOT CAUSE, not by "it failed in the suite":

| # | Test | Isolation | Root cause | Bucket | Fix |
|---|------|-----------|-----------|--------|-----|
| 1 | `test_agent_map_live_model::test_endpoint_injects_live_model_field` | FAIL alone | asserts `claude-opus-4-7`; live model is `claude-opus-4-8` (2026-05-28 bump) | **B STALE** | update assertion 4-7 -> 4-8 |
| 2 | `test_phase_23_2_14_no_reentrant_locks::...lock_count_matches_roster` | FAIL alone | found 15 locks, expects 14; the 15th = `alerting.py:64` (AlertDeduper, added post-audit) | **B STALE** | bump `EXPECTED_LOCK_COUNT` 14 -> 15 + note the new site |
| 3 | `test_phase_23_2_12_layer1_pipeline_active::...full_proxy_in_last_7d` | FAIL alone | live BQ: 0 full-path rows (cost>0.05) in last 7d (away-week was lite) | **A LIVE-BQ** | `requires_live` skipif |
| 4 | `test_phase_23_2_5_kill_switch_no_false_fires::...false_fires_documented` | FAIL alone | live BQ: 0 historical `drawdown_breach` rows (expects 5-20) | **A LIVE-BQ** | `requires_live` skipif |
| 5 | `test_phase_23_2_10_watchdog_no_fire_7d::...log_present_and_fresh` | **PASS alone** | test pollution (shared module state in full suite) | **D POLLUTION** | NOT live — isolate state OR `requires_live` is WRONG; see below |
| 6 | `test_rainbow_canary::...partitions_by_source` | **PASS alone** | test pollution (shared buffer/singleton state) | **D POLLUTION** | NOT live — isolate state |
| 7-13 | `test_phase_23_2_16_shortlist_doc_presence` (7 tests) | FAIL alone | expects `handoff/current/phase-23.2.16-shortlist.md`; archived to `handoff/archive/phase-23.2.16/phase-23.2.16-shortlist.md` by the archive-handoff hook | **C MOVED DOC** | repoint `SHORTLIST_DOC` (`:18`) to the archive path OR skip w/ reason |

**Quarantine design (durable, the verification command needs exit 0):**

1. **New `pytest.ini`** at repo root — register the marker:
   ```ini
   [pytest]
   markers =
       requires_live: test depends on live BigQuery / live-system state; skipped unless PYFINAGENT_LIVE_TESTS=1
   ```
2. **Marker + skipif** on the 2 live-BQ tests (#3, #4): decorate with
   `@pytest.mark.requires_live` AND a module/test-level
   `pytest.mark.skipif(os.getenv("PYFINAGENT_LIVE_TESTS") != "1", reason="live
   BQ probe: needs <table> rows in last 7d; set PYFINAGENT_LIVE_TESTS=1 to run
   against prod")`. Per-test reason strings naming the exact live dependency.
3. **Stale assertions (#1, #2): UPDATE, do NOT skip** — bump 4-7->4-8 and
   14->15 (the new lock is legitimate; document `alerting.py:64` in the roster
   comment). These are correctness regressions of the test, not env-coupling.
4. **Moved doc (#7-13): repoint** `SHORTLIST_DOC` to
   `handoff/archive/phase-23.2.16/phase-23.2.16-shortlist.md` (the archive
   path), OR mark the 7 tests `requires_live`-adjacent with a reason "fixture
   doc archived by archive-handoff hook; historical artifact". Repointing is
   cleaner (the doc still exists, just moved). RECOMMENDED: repoint.
5. **Pollution (#5, #6): the hard ones.** A `requires_live` skip is the WRONG
   tool (they pass in isolation, have no live dep). Two honest options:
   (a) **Fix the shared state** — add `reset_default_deduper()` /
   kill-switch-state reset / buffer-clear fixtures (autouse) so the suite is
   order-independent. The deduper already exposes `reset_default_deduper()`
   (`alerting.py:222`); the canary likely needs a buffer reset.
   (b) If (a) is out of 56.2 scope, mark them `@pytest.mark.requires_live` with
   an HONEST reason "flaky under full-suite ordering (shared module singleton);
   tracked for state-isolation fix" — this is a quarantine, not a live-dep
   claim, and the reason string says so. RECOMMENDED: try (a) first
   (`reset_default_deduper` autouse fixture is ~5 LOC); fall back to (b)-with-
   honest-reason if the canary state proves deeper.

**After quarantine the full suite must exit 0** (criterion 4 + the immutable
verification command). The 2 stale + 7 doc are deterministic fixes; the 2
live-BQ are skipped by default; the 2 pollution need a reset fixture or honest
quarantine. **Do NOT claim a green suite by blanket-skipping** — the operator
memory `project_dod_production_ready_gate.md` flags "16 env-coupled failures =
watermelon risk"; classify honestly (the table above is the audit trail).

---

## PART B — EXTERNAL RESEARCH

### Read in full (7; >=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://docs.slack.dev/tools/bolt-python/concepts/acknowledge/ | 2026-06-10 | Official doc | WebFetch full | "Actions ... must always be acknowledged using the `ack()` function"; "you only have 3 seconds to respond before Slack registers a timeout error"; "call `ack()` right away before initiating any time-consuming processes". -> **F-14**: if buttons are kept, the handler must `ack()` FIRST. |
| https://docs.pytest.org/en/stable/how-to/skipping.html | 2026-06-10 | Official doc | WebFetch full | `@pytest.mark.skipif(<cond>, reason=...)` evaluated at collection time is the conditional-skip idiom; "Always include a `reason`"; xfail `strict=False` is the default and "allows expected failures without breaking the test suite". -> **PART C**: `requires_live` = skipif on env var w/ per-test reason; do NOT xfail the stale/doc ones. |
| https://aipatternbook.com/silent-failure | 2026-06-10 | Authoritative blog (agentic patterns) | WebFetch full | "A function that 'fills in' a missing field with '' or 0 or null produces a result that looks legitimate. The downstream code can't distinguish 'this field was empty in the source' from 'this field was lost on the way here.'"; remedy = "Output assertions ... Default value detection: Flag missing inputs rather than masking with empty strings/zeros"; "A loud failure is a problem the team has. A silent failure is a problem the team doesn't yet know it has, which is strictly worse." -> **F-5/F-6/F-7** core justification. |
| https://arxiv.org/abs/2603.09947 | 2026-06-10 | Preprint (peer-adjacent, 2026) | WebFetch full (HTML abstract chain after PDF returned binary) | Confidence Gate Theorem: distinguishes **structural uncertainty** (missing data) from **contextual uncertainty** (drift). "Structural uncertainty yields consistent abstention improvements across all tested domains"; provides "a clean negative result against using exception labels derived from residuals, which degrade under distribution shift". -> **F-7**: a DEAD LLM rail is *structural* (the signal is entirely absent) -> abstention/neutral is the supported response; but the *value* must be chosen carefully (don't let a residual/composite fallback masquerade as conviction). |
| https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | 2026-06-10 | Authoritative blog (2026) | WebFetch full | Layered defense: collector heartbeat + per-service freshness + external dead-man switch. "Detecting idle vs failure ... drops below 10% of historical rates indicate partial failure rather than idle state." Heartbeat = "I am alive; if the signal stops, something is wrong." -> **F-4**: the rail-health probe must distinguish "rail down" from "no work" (the exact away-week gap); volume/explicit-probe beats inferring from absence. |
| https://greatexpectations.io/blog/why-data-validation-is-critical-to-your-pipelines/ | 2026-06-10 | Industry (data-quality vendor) | WebFetch full | Post-transformation validation: "confirm that the transformation's output conforms to what downstream consumers are expecting"; Write-Audit-Publish — validate "before data passes into the gold (consumption) layer". -> **F-5**: assert the cycle output BEFORE publishing to the digest (the consumption layer), not after. |
| https://www.index.dev/blog/avoid-silent-failures-python | 2026-06-10 | Authoritative blog | WebFetch full | "Always catch specific exceptions instead of using bare except"; "fallback mechanisms should be *intentional and observable*, not silent masking of failures"; "Degraded paths must be detectable through proper logging". -> **F-5/F-7**: a fallback is fine, but it must set an observable flag (`summary["degraded"]`/`meta_scorer_degraded`) + alert, not silently substitute. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.checklyhq.com/blog/heartbeat-monitoring-with-checkly/ | Vendor blog | OneUptime covered the heartbeat pattern in full; redundant |
| https://blog.ediri.io/how-to-set-up-a-dead-mans-switch-in-prometheus | Blog | Prometheus-specific; pyfinagent has no Prometheus |
| https://robotalp.com/blog/top-8-heartbeat-monitoring-tools-2026/ | Vendor listicle | 2026 recency hit, but tool-listing not design guidance |
| https://docs.pytest.org/en/stable/explanation/flaky.html | Official doc | Flaky-test explanation; the skipping how-to (read in full) was the actionable one |
| https://docs.slack.dev/tools/bolt-python/concepts/actions/ | Official doc | The acknowledge concept (read in full) carried the load-bearing 3s rule |
| https://www.technetexperts.com/python-sentinel-flags-best-practice/amp/ | Blog | Sentinel-flag mechanics; the abstention paper + silent-failure source covered the design choice |
| https://arxiv.org/pdf/2504.13966 | Preprint | Adversarial reject-option; tangential to our structural-uncertainty case |
| https://learn.microsoft.com/en-us/azure/databricks/ldp/expectations | Official doc | Databricks-specific expectations API; GX blog covered the producer-side assertion principle |
| https://www.geeksforgeeks.org/machine-learning/the-reject-option-pattern-recognition-and-machine-learning/ | Tutorial | Reject-option primer; the 2026 Confidence-Gate paper superseded it for ranked systems |

### Recency scan (2024-2026) — MANDATORY, performed

Searched 2024-2026 literature across all five focus areas. **Findings:**
- **NEW + directly load-bearing:** *The Confidence Gate Theorem* (arXiv:2603.09947,
  **2026**) is brand-new and squarely on F-7 — it formalizes when a **ranked /
  top-K** system (exactly pyfinagent's conviction selector) should abstain, and
  its structural-vs-contextual distinction tells us a *dead rail* is structural
  uncertainty where abstention/neutral is supported, while warning (clean
  negative result) against letting a residual-derived fallback pose as a real
  confidence signal. This SUPERSEDES the older reject-option primers for our case.
- **NEW (2026):** OneUptime's heartbeat/dead-man-switch design post (**Feb 2026**)
  gives the current best-practice layered-probe pattern + the explicit
  "idle vs failure" detection rule that F-4 needs.
- **NEW (2026):** the "Silent Failure" agentic-patterns entry is current and
  agent-system-specific (it names the exact `return [] / fill with 0` anti-pattern
  that produced F-5).
- **Stable canonical (year-less, still authoritative):** the Slack Bolt `ack()`
  3-second rule and pytest `skipif`/`xfail` semantics are unchanged in the current
  docs; no 2024-2026 work supersedes them.
- **No relevant new finding** that would change the F-4/F-5/F-6 *recommended
  options* beyond confirming them; the 2026 sources reinforce rather than overturn
  the fix designs.

### Query log (3-variant discipline)

| Focus | Current-frontier (2026) | Last-2-yr (2025) | Year-less canonical |
|---|---|---|---|
| Rail health-check / dead-man | "...monitoring best practice" (hit 2026 OneUptime/Robotalp) | (covered in same pass) | "health check heartbeat synthetic probe dead man switch" (year-less; hit Grafana/cloudonaut canon) |
| pytest quarantine | "...best practice 2025" | 2025 in query | "pytest skip xfail markers" (hit official docs canon) |
| Slack Bolt ack | (docs are version-current) | — | "Slack Bolt Python app.action ack ... best practice" (year-less; hit official docs) |
| Fail-loud / sentinels | "5 silent failure modes 2026" (hit DEV 2026 + aipatternbook) | — | "fail loud silent default anti-pattern sentinel value" (year-less; hit Medium/Ronald-Haring canon) |
| Neutral fallback / abstain | "Confidence Gate Theorem 2026" (arXiv:2603.09947) | — | "abstention reject option classifier confidence fallback" (year-less; hit JMLR/GeeksforGeeks canon) |
| Output assertions | (GX current) | — | "data pipeline output assertions great expectations validation guard" (year-less) |

### Consensus vs debate (external)

- **Consensus (strong):** silent defaulting of a missing/failed value to 0/None/
  neutral is an anti-pattern; the fix is an *observable* degraded flag + alert at
  the PRODUCER, before the consumption layer (Silent-Failure, index.dev, GX all
  agree). This is unanimous and directly backs F-5/F-6/F-7.
- **Consensus:** `ack()` within 3s is mandatory for any Slack action handler
  (official). Backs F-14's "if kept, ack-first" branch.
- **Consensus:** conditional skip via `skipif(env, reason=...)` with a reason is
  the idiom; xfail is for known-failing-not-flaky (official). Backs PART C's
  "skip live-BQ, UPDATE stale, repoint doc — don't blanket-xfail".
- **Debate / nuance (F-7):** the abstention literature is NOT a blanket "always
  abstain". The 2026 Confidence-Gate paper shows abstention helps for *structural*
  uncertainty (our dead-rail case) but a residual/observation-count-derived
  confidence "performs no better than random" under drift — i.e. the *value* you
  abstain TO matters. This is why the brief recommends NOT silently flipping the
  conviction fallback to a flat neutral (which changes selection) and instead
  flagging+alerting now, deferring the true neutral-sentinel redesign to a gated
  phase-57 change with measurement.

### Pitfalls (from literature)

1. **Empty/zero indistinguishable from failure** (Silent-Failure) — exactly F-5's
   `final_score=0.0` and F-6's "0 rows". Fix = flag, don't default.
2. **Inferring health from absence** (OneUptime) — "no work" looks like "rail
   down"; F-4 must probe explicitly, not infer.
3. **Fallback value masquerading as signal** (Confidence-Gate clean negative
   result) — F-7: a composite-derived conviction fallback is a residual-style
   signal; do not treat it as a real confidence, and do not silently change the
   selection it drives.
4. **ack timeout** (Slack) — F-14: a handler that does work before `ack()` times
   out; ack first.
5. **Blanket xfail as permanent quarantine** (pytest) — "rather dangerous to use
   permanently"; PART C uses skipif-with-reason for live deps and FIXES the stale/
   doc/pollution ones rather than hiding them.

### Application to pyfinagent (external -> file:line)

- F-4 rail probe -> `claude_code_client.py` new `claude_code_health_probe()` +
  `autonomous_loop.py:~744` cycle-start call + `alerting.py:119`
  `raise_cron_alert` (OneUptime idle-vs-failure; explicit probe).
- F-5 degraded guard -> `autonomous_loop.py:820/827` post-gather assertion +
  `alerting.py:119` (Silent-Failure "flag don't default"; GX validate-before-
  consumption-layer).
- F-6 instrumentation -> `claude_code_invoke` callers `autonomous_loop.py:1580/
  1636` + `log_llm_call` (`api_call_log.py:203`) + `_run_gemini_analysis`
  `_role`/`_ticker` at `:1798/:1835` (Silent-Failure "0 rows" anti-pattern).
- F-7 sentinel -> `meta_scorer.py:138-142/249-256` consumer `autonomous_loop.py:
  698-723` (Confidence-Gate structural-uncertainty + value-matters; flag now,
  gate the value-change to phase-57).
- F-14 -> `governance.py:160-178` remove, or `app_home.py:397`-style handler with
  ack-first (Slack 3s rule).
- PART C -> new `pytest.ini` + `skipif(PYFINAGENT_LIVE_TESTS)` on the 2 live-BQ
  tests; repoint `test_phase_23_2_16` `:18`; bump `test_agent_map`/`23_2_14`
  constants; `reset_default_deduper()` (`alerting.py:222`) autouse for pollution.

---

## Research Gate Checklist

Hard blockers — `gate_passed` is true only if all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read)
- [x] 10+ unique URLs total incl. snippet-only (16 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set (arXiv via HTML
      abstract chain after PDF returned binary, per research-gate.md)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (claude_code_client,
      meta_scorer, autonomous_loop x2, governance, formatters, scheduler,
      kill_switch, alerting, api_call_log, llm_client, ticket_ingestion,
      ticket_queue_processor, commands, app_home + the 13 failing tests)
- [x] Contradictions / consensus noted (F-7 abstention nuance)
- [x] All claims cited per-claim with file:line or URL

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 9,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
