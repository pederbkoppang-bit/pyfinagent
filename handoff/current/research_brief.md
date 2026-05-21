# Research Brief — phase-33.1

**Tier:** simple (internal-only)
**Step:** phase-33.1 — Post-cron observation of first post-phase-32 autonomous cycle
**Cycle under inspection:** cycle_id `8df751b3`, fired 2026-05-21T18:00:00Z (= 20:00 CEST), completed 18:05:21Z (321 sec), n_trades=0
**Read-only:** no code edits, no mutating BQ/Alpaca/LLM calls

---

## Executive Summary

The 14:00 ET / 18:00 UTC cron fired on schedule and the cycle completed
cleanly (no exceptions / no error_count). However, the phase-33.0
NOT_READY blockers were NOT cleared before the cron — **kill-switch
remained active** (paused 2026-05-19, no resume since) and **Anthropic
credit remained unfunded** (100+ `credit balance is too low` errors
during analysis). Consequently the cycle short-circuited at Step 5.5
(`kill-switch active -- skipping decide/execute`), wrote a single
`$CYCLE / HOLD` placeholder signal, took the EOD snapshot, and ended.
**Steps 5.6 (stop-loss + phase-30.2 backfill), 6 (decide), 7 (execute),
and the phase-32 ratchet / HWM-trail / company-name-backfill paths
never ran in this cycle.** Phase-32 features cannot be verified live
until the operator resumes the kill-switch and funds Anthropic credits.

---

## Topic 1 — Cycle log location

**File:** `/Users/ford/.openclaw/workspace/pyfinagent/backend.log`
(uvicorn-served backend, 253 MB, updated continuously).

**Important caveats for downstream probe steps:**
- This file is at the **repo root**, NOT `handoff/logs/backend.log`
  (that file does not exist). Search code that assumes
  `handoff/logs/backend.log` will return empty.
- Timestamps are in **CEST (UTC+2)**, not UTC. Today's 18:00 UTC cron
  appears as `20:00 CEST`.
- The literal string `cycle_id` never appears in the log. There is no
  `8df751b3` substring anywhere — the cycle is delimited by the
  job-runner bookends and by `Paper trading: Step N` markers.

**Today's cycle window:** backend.log lines **1689002–1691384**.

Bookends:
- Start L1689002 (20:00:01 CEST): `[autonomous_loop] Loaded best params (Sharpe 1.1704633657934074)`.
- Cron-job start L1689001 (20:00:00 CEST): `[base] Running job "Paper trading daily run (trigger: cron[day_of_week='mon-fri', hour='14', minute='0'], next run at: 2026-05-22 14:00:00 EDT)"`.
- End L1691383–L1691384 (20:05:21 CEST): `[paper_trading] Scheduled paper trading result: running` / `[base] Job "Paper trading daily run ..." executed successfully`.

**Cycle metadata also persisted at:**
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/cycle_history.jsonl` — final line: `{"cycle_id": "8df751b3", "started_at": "2026-05-21T18:00:00.415298+00:00", "completed_at": "2026-05-21T18:05:21.983315+00:00", "duration_ms": 321568, "status": "running", "n_trades": 0, "error_count": 0, ...}`. NB: status stayed `"running"`, never flipped to `"completed"` despite duration_ms being final — same pattern as several prior cycles (435644e4, 9fdcc2df). Likely tracker bug; not a runtime error.
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/.cycle_heartbeat.json` — last write: `{"cycle_id": "8df751b3", "event": "end", "updated_at": "2026-05-21T18:05:21.984129+00:00"}`.

---

## Topic 2 — Risk Judge prompt records

**Write site:** `backend/agents/llm_client.py:1645-1669` (Anthropic
direct path) and `:2011-2038` (advisor path) call
`backend.services.observability.api_call_log.log_llm_call`, which
buffers rows for `pyfinagent_data.llm_call_log`.

**Schema (per `backend/services/observability/api_call_log.py:181-197`):**
`ts, provider, model, agent, latency_ms, ttft_ms, input_tok,
output_tok, cache_creation_tok, cache_read_tok, request_id, ok,
ticker, cycle_id, session_cost_usd`. The `agent` field is
best-effort; for Risk Judge calls this is typically `risk_debate`
(per `llm_client.py:5` module docstring: "orchestrator, debate, and
risk_debate" all flow through `LLMClient`).

**There is no per-prompt local file log** for Risk Judge — prompts
land in BQ `llm_call_log` only. (Token counts and request_id are
captured; the prompt body itself is NOT persisted.)

**Today's cycle:** **zero Risk Judge invocations** — the cycle
short-circuited at Step 5.5 (kill-switch) before reaching the
decide-trades step where Risk Judge would fire. Confirmed by `grep -i
"risk_judge\|risk judge\|risk-judge\|riskjudge"` against the cycle
window L1689000–L1691400: no matches. (The only `risk_judge` matches
in backend.log are from cycles on 2026-04-26 02:24 CEST and earlier.)

The downstream probe step should NOT expect to find
`cycle_id='8df751b3'` rows with `agent='risk_debate'` in
`pyfinagent_data.llm_call_log`. It SHOULD find rows tagged with
analysis-stage agents (`orchestrator`, debate skills) where the
Claude calls failed with HTTP 400 credit-balance errors — those rows
will have `ok=false` and per-ticker `ticker` tags.

---

## Topic 3 — Cycle summary trace (phase-32 markers)

Steps actually emitted by today's cycle (backend.log line numbers):

| Step | Time (CEST) | Line | Marker |
|------|-------------|------|--------|
| Step 1 — Screening | 20:00:01 | 1689003 | `Paper trading: Step 1 -- Screening universe` |
| Step 1 result | 20:00:34 | 1689009 | `Screening complete: 501/503 passed basic filters` |
| Step 3 — Analyze | 20:00:36 | 1689014 | `Paper trading: Step 3 -- Analyzing 3 new + 11 re-evals (lite_mode=False)` |
| Step 5 — Mark-to-market | 20:02:57 | 1691270 | `Paper trading: Step 5 -- Mark to market` |
| Step 5.5 — Kill switch HALT | 20:04:12 | 1691328 | `Paper trading: kill-switch active -- skipping decide/execute` |
| Signal log (HOLD placeholder) | 20:04:13 | 1691329 | `Logged 1 signal(s) to BQ signals_log for 2026-05-21` |
| Cycle end | 20:05:21 | 1691383 | `Scheduled paper trading result: running` |

**Phase-32 markers searched and NOT found in today's cycle:**

| Marker | Expected step | Found in cycle window? |
|--------|---------------|------------------------|
| `phase-32.1: ratchet fired` | Step 5.6 area | NO |
| `phase-32.2: trail fired` | Step 5.6 area | NO |
| `phase-30.2: backfill_missing_stops` | Step 5.6 | NO |
| `phase-32.4: backfilled company_name` | Step 5.7-ish | NO |
| `phase-32.3: portfolio_sector_exposure` | Step 6 (Risk Judge input) | NO |
| `Step 5.6 -- Stop-loss enforcement` | Step 5.6 | NO |
| `Step 6 -- Deciding trades` | Step 6 | NO |
| `Step 7 -- Executing N trades` | Step 7 | NO |
| `Step 8 -- Final snapshot` | Step 8 | NO |

The kill-switch branch in `backend/services/autonomous_loop.py:746-760`
explicitly bypasses everything after Step 5.5: it logs the empty-
records HOLD placeholder, takes a final mark_to_market, calls
`save_daily_snapshot(trades_today=0, ...)`, and returns. Phase-32's
ratchet, trail, sector-exposure-injection, and company-name backfill
all live downstream of this branch and were therefore never exercised.

**Side-channel observation — company_name fetches DID happen during
Step 3 analysis** (lines 1689332, 1689335, 1689483, 1689742, 1689849,
1689945, 1690137, 1690292, 1690463, 1690643, 1690794, 1691006,
1691035, 1691230 — 14 tickers): the SEC ingestion agent fetched
`company_name` metadata for each ticker during analysis. This is the
*orchestrator's* per-ticker SEC lookup, **not** phase-32.4's
`backfill_missing_company_names` step (which writes to the positions
table during Step 5.7). Don't conflate.

---

## Topic 4 — Anomaly scan

**Errors / warnings observed in today's cycle:**

1. **Anthropic API authentication / credit failure — SEVERE, BLOCKING.**
   First occurrence L1689574 at 20:01:27 CEST. Every per-ticker Full
   orchestrator call fails with HTTP 400: `Your credit balance is too
   low to access the Anthropic API. Please go to Plans & Billing to
   upgrade or purchase credits.` The fallback to lite Claude
   analyzer ALSO fails with the same error (e.g. L1689587, L1689593,
   L1689595, L1689983, L1690101, L1690321, L1690488, L1690510,
   L1690823, L1690866, L1691078, L1691215, L1691255, L1691269). All
   14 analyzed tickers (STX, CIEN, AMD, GEV, MU, KEYS, COHR, ON,
   INTC, DELL, GLW, LITE, SNDK, WDC) failed both paths. **The
   phase-33.0 Anthropic-credit blocker was NOT cleared by the
   operator before the 14:00 ET cron.**

2. **Kill switch ACTIVE — SEVERE, BLOCKING.** L1691328 at 20:04:12
   CEST: `Paper trading: kill-switch active -- skipping decide/
   execute`. Per `handoff/kill_switch_audit.jsonl` the most recent
   `pause` event was 2026-05-19T19:34:17 (trigger `manual`), and the
   most recent `resume` event was 2026-05-07T18:06:11. **There has
   been NO resume between 2026-05-19 and 2026-05-21. The phase-33.0
   kill-switch blocker was NOT cleared by the operator either.**

3. **`promoted_strategies` table 404 — MEDIUM, expected.** L1689001
   (20:00:01): `Promoted strategy BQ unavailable, falling back to
   optimizer_best: 404 Not found: Table
   sunny-might-477607-p8:pyfinagent_data.promoted_strategies`.
   Fail-open; fallback worked.

4. **`cycle_health` BQ queries fail repeatedly — MEDIUM, non-blocking.**
   `bq_max_event_age(...)` queries fail at 20:00:37, 20:01:19,
   20:01:20, 20:02:20, 20:03:21, 20:04:22, 20:05:23 (and continue
   afterwards) with `400 SAFE with function timestamp is not
   supported.` This is a pre-existing health-probe bug, not introduced
   by phase-32, but worth flagging as observed today. Tables affected:
   `historical_prices.ingested_at`,
   `historical_fundamentals.ingested_at`,
   `historical_macro.ingested_at`, `signals_log.recorded_at`. The
   `SAFE.TIMESTAMP(...)` BQ idiom is invalid; needs `TIMESTAMP(...)`
   or `SAFE_CAST(... AS TIMESTAMP)`.

5. **`Failed download: ['ACN']: TypeError("'NoneType' object is not
   subscriptable")`** — MEDIUM, fail-open. L1689006-1689009 at
   20:00:34. Single-ticker yfinance failure inside the screener;
   501/503 still passed.

6. **`get_kill_switch_state: BQ portfolio fetch timed out after 5s`**
   — L1613409, 20:53:02 — POST-cycle, unrelated to 8df751b3.

7. **No TRACEBACK in the cycle window** — the Python traceback module
   was not invoked; all errors are caught by fail-open handlers.

**Status of phase-32 code in this cycle:** unobservable — the
short-circuit at Step 5.5 means every phase-32 marker we sought is
absent NOT because the code is broken, but because the code path
was never entered. **Today's run does not validate or invalidate
phase-32 features.** A new cycle is needed with kill-switch
resumed and Anthropic credit funded.

---

## Topic 5 — n_trades=0 root cause hypothesis

**Root cause: KILL-SWITCH ACTIVE at Step 5.5.**

Evidence (single load-bearing line):

> `backend.log:1691328` (20:04:12 CEST = 18:04:12 UTC):
> `20:04:12 W [autonomous_loop] Paper trading: kill-switch
> active -- skipping decide/execute`

This is the exact branch in
`backend/services/autonomous_loop.py:746-760` that, when the kill-
switch is paused or breached, logs the warning, appends
`kill_switch_halted` to the cycle's `summary["steps"]`, writes a
single HOLD-placeholder signal to BQ, takes a final mark-to-market,
calls `save_daily_snapshot(trades_today=0, ...)`, and `return`s
without reaching Steps 5.6, 6, 7, 8.

**Why the kill-switch is paused: operator paused it on 2026-05-19T19:34:17
and never resumed.** Per `handoff/kill_switch_audit.jsonl` tail, the
last 3 events for kill-switch state are:
- 2026-05-19T19:33:49: `pause` (trigger `phase-30-overnight-remediation`)
- 2026-05-19T19:34:17: `pause` (trigger `manual`)
- (no `resume` event since)

(Subsequent entries are `sod_snapshot` and `peak_update` — bookkeeping,
not state changes.)

**Secondary (not the root cause but a hard blocker on its own):** even
if the kill-switch had been resumed, every Claude analysis call failed
with `credit balance is too low` (100+ occurrences in today's cycle
window), so Step 6 (decide_trades) would have had 0 valid candidates
to feed Risk Judge, and Risk Judge itself would have failed at the
first call. **The Anthropic credit blocker (phase-33.0 NOT_READY #2)
is also still active and would independently force n_trades=0.**

**Eliminated alternative hypotheses:**
- "Risk Judge declined all candidates (REJECT)" — NO, Risk Judge never
  ran. Zero `risk_judge` lines in the cycle window.
- "Sector-cap pre-trade gate (Tech 49.54% > 30%) blocked all BUYs"
  — NO, the sector-exposure code lives in Step 6 (Risk Judge input
  preparation), which never ran. Cannot be the cause.
- "All positions kept their BUY recommendation, no sells" —
  NO, Step 5.6 stop-loss enforcement never ran; Step 6 decide-trades
  never ran; no sell decisions were even attempted.
- "Anthropic credit fail mid-cycle" — the credit failure is REAL
  but it happens BEFORE the kill-switch check (during Step 3
  per-ticker analysis at 20:01:27+). It is a co-blocker, not the
  short-circuit cause.

---

## Cross-check matrix: 9 contract categories

| # | Category | Evidence source |
|---|----------|-----------------|
| 1 | Cycle started + completed | `backend.log:1689001-1691384` + `handoff/cycle_history.jsonl` last line |
| 2 | Kill switch unpaused | **FAIL** — `handoff/kill_switch_audit.jsonl` last resume = 2026-05-07; last pause = 2026-05-19; today line 1691328 confirms still active |
| 3 | Anthropic credit funded | **FAIL** — `backend.log:1689574` and 100+ subsequent `credit balance is too low` errors during Step 3 analysis |
| 4 | Risk Judge invocations | **NOT RUN** — zero `risk_judge` matches in cycle window (1689000-1691400); never reached Step 6 |
| 5 | Sector exposure injected (phase-32.3) | **NOT RUN** — sector-exposure injection lives in Step 6 Risk Judge prep; never reached |
| 6 | Stop backfill (phase-30.2) | **NOT RUN** — Step 5.6 never executed; no `phase-30.2` marker in cycle window |
| 7 | Breakeven ratchet (phase-32.1) | **NOT RUN** — `phase-32.1: ratchet fired` absent from cycle window |
| 8 | HWM trail (phase-32.2) | **NOT RUN** — `phase-32.2: trail fired` absent from cycle window |
| 9 | Company-name backfill (phase-32.4) | **NOT RUN** — `phase-32.4: backfilled company_name` absent. (Note: SEC `company_name` metadata fetches DID occur in Step 3 analysis at L1689332+, but those are orchestrator side-channel SEC lookups, not phase-32.4 backfill writes to positions.) |

---

## Recency scan

Transitively inherited from phase-31.0 research brief (no fresh
external scan required for internal-only observation step). Confirmed
phase-33.1 is a read-only observation step; no external SDK/API
changes since last brief.

---

## Internal files inspected

1. `/Users/ford/.openclaw/workspace/pyfinagent/handoff/logs/` (directory listing)
2. `/Users/ford/.openclaw/workspace/pyfinagent/handoff/.cycle_heartbeat.json`
3. `/Users/ford/.openclaw/workspace/pyfinagent/handoff/cycle_history.jsonl`
4. `/Users/ford/.openclaw/workspace/pyfinagent/backend.log` (lines 1689000–1691400, full cycle window)
5. `/Users/ford/.openclaw/workspace/pyfinagent/handoff/kill_switch_audit.jsonl` (last 20 events)
6. `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/observability/api_call_log.py` (lines 181-280, llm_call_log writer)
7. `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/autonomous_loop.py` (lines 720-780, 1820-1860 — Step 5.5 short-circuit + signal-log site)
8. `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/llm_client.py` (grep targets; lines 1645-1669, 2011-2038 — `log_llm_call` call sites for Risk Judge / orchestrator)
9. `/Users/ford/.openclaw/workspace/pyfinagent/.mcp.json` (BQ MCP availability check — not present in this session)

---

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
