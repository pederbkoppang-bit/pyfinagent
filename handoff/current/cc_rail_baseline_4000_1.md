# CC-rail baseline -- step 4000.1 (pre-registered 2026-08-06, BEFORE any 4000.3 window)

All measurements taken 2026-08-06 by Main against the RUNNING backend and BQ via
ADC. Captures are verbatim command output; section (a)'s fenced block is a
projection of the full GET body and its exact producing command is stated
in-section (Q/A cycle-1 fix). Contract: handoff/current/contract_4000.1.md.
Research brief: handoff/current/research_brief_4000.1.md (gate_passed=true).

## Section (a) -- live flag state

Captured 2026-08-06T10:33:35Z via localhost curl (DEV_LOCALHOST_BYPASS rail;
NOTE the GET is cached ~300s server-side per api_cache.py:136, so reads can lag
a PUT by up to 5 minutes). The fenced block below is a 2-key PROJECTION of the
full 45-key GET body (use the trailing slash: the un-slashed /api/settings
307-redirects). Exact producing command, regenerable:

```sh
echo "== GET /api/settings (captured $(date -u '+%Y-%m-%dT%H:%M:%SZ')) =="; \
curl -s -m 10 http://localhost:8000/api/settings/ | python3 -c "import json,sys; \
s=json.load(sys.stdin); print(json.dumps({k:s[k] for k in \
('paper_use_claude_code_route','paper_analyze_top_n') if k in s}))"
```

Output, verbatim:

```
== GET /api/settings (captured 2026-08-06T10:33:35Z) ==
{"paper_use_claude_code_route": true, "paper_analyze_top_n": 5}
```

paper_use_claude_code_route is ALREADY True: the phase-78.1 steady state (the
flag=false direction is 78.1's documented one-flag revert, harness_log:28513-28523).
The 4000.3 smoke is therefore VERIFY-AND-CHARACTERIZE, not flip-and-revive.
`claude --version` at baseline time (F1 bisect anchor): `2.1.223 (Claude Code)`.

## Section (b) -- backend process age vs phase-78.2

```
uvicorn worker pid 60478 lstart: ons.  5 aug. 17.38.35 2026  (2026-08-05 17:38:35 +0200)
phase-78.2 commits: 5e51f4a9 2026-07-25 15:35:19 +0200 (thread explicit --model)
                    a75c209f 2026-07-25 16:29:31 +0200 (close-out)
                    acf89271 2026-07-25 16:33:38 +0200 (same-day tail, Q/A cycle-1 addition)
backend cwd (P3 overhead anchor): /Users/ford/.openclaw/workspace/pyfinagent
```

Does the running process predate the --model change? **NO** -- the process
started 11 days after 78.2 landed. No restart is required inside 4000.3.

## Section (c) -- CC-rail row-selection rule (E1 rule, executable WHERE clause)

A row in `sunny-might-477607-p8.pyfinagent_data.llm_call_log` (time column `ts`)
IS a CC-rail (flat-fee) row iff:

```sql
provider = 'claude-code' OR agent = 'cc_rail' OR agent LIKE 'cc_rail:%'
```

and is Anthropic-direct (metered) iff:

```sql
provider = 'anthropic' AND (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

Derivation: the RAIL clause is the exact logical complement (De Morgan) of the
spend-breaker exclusion at backend/services/observability/spend.py:228-230; the
METERED clause is that exclusion further narrowed to provider='anthropic'
(spend's `provider != 'claude-code'` also spans gemini/openai rows -- Q/A
cycle-1 precision fix). THREE writer shapes exist and provider ALONE cannot
separate the rails (two writers use provider='anthropic'):
W1 backend/agents/claude_code_client.py:656-668 (provider='anthropic',
agent='cc_rail:<role>' or BARE 'cc_rail' when the role is unset); W2
backend/services/autonomous_loop.py:2347-2358 (provider='claude-code',
agent='lite_trader'/'lite_risk_judge'; NOTE the directory prefix -- a second,
620-line backend/autonomous_loop.py exists and has no line 2347); W3
backend/services/ticket_queue_processor.py:228-238. NEVER simplify the agent clause to a
'cc_rail%' prefix -- spend.py:37-38 rejects it on purpose ('cc_railway' hazard).
Since 78.2 the `model` column carries the RESOLVED model (resolve_rail_model,
claude_code_client.py:649-655) with a loud WARN on mismatch.

## Section (d) -- E8 trade-cadence baseline (the 4000.5 comparator)

Rule (stated once, both halves computed under it): source table
`sunny-might-477607-p8.financial_reports.paper_trades`; `created_at` is a STRING
(migrate_paper_trading.py:68) so the window filter uses
`SAFE_CAST(created_at AS TIMESTAMP)`; window = trailing 28 days from query time
(2026-08-06); a TRADE is any row in-window; a CLOSED ROUND TRIP is a DISTINCT
`round_trip_id` among in-window rows with `action='SELL' AND round_trip_id IS
NOT NULL`. Query text, verbatim:

```sql
WITH t AS (SELECT SAFE_CAST(created_at AS TIMESTAMP) pts, action, round_trip_id
  FROM `sunny-might-477607-p8.financial_reports.paper_trades`)
SELECT FORMAT_DATE('%G-W%V', DATE(pts)) wk, COUNT(*) trades,
  COUNTIF(action='BUY') buys, COUNTIF(action='SELL') sells,
  COUNT(DISTINCT IF(action='SELL' AND round_trip_id IS NOT NULL, round_trip_id, NULL)) rt_closed
FROM t WHERE pts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 28 DAY)
GROUP BY wk ORDER BY wk LIMIT 10
```

Result (2026-08-06):

```
2026-W28: trades=2 (2 BUY),          rt_closed=0
2026-W29: trades=1 (1 SELL),         rt_closed=1
2026-W30: trades=1 (1 BUY),          rt_closed=0
2026-W31: trades=3 (1 BUY, 2 SELL),  rt_closed=2
TOTALS:   7 trades / 3 closed round trips in 28d  =  1.75 trades/week, 0.75 RT/week
```

Cross-check: all-time closed RTs by this rule = 32; the dedicated
`financial_reports.paper_round_trips` table holds n=32 (first_exit 2026-05-14,
last_exit 2026-07-27) -- the two rules RECONCILE exactly. At 0.75 RT/week the
100-round-trip validation sample is ~90 weeks away; that number is the phase's
reason to exist.

## Section (e) -- chosen bounded entrypoint for the smoke

POST /api/analysis/ (backend/api/analysis.py:350, `start_analysis`) with ONE
ticker. It is ASYNC: returns a task id; poll GET /api/analysis/{analysis_id}
(analysis.py:384). Flows through _run_sync_analysis -> orchestrator ->
make_client -> the ClaudeCodeClient seam (llm_client.py:2108-2131). The lite
path is REJECTED as entrypoint (reachable only via a full 2h autonomous cycle
that trades). CAVEAT (research finding, binding on 4000.2): rail-calls-per-
analysis is NOT derivable from code -- only claude-* pinned roles route to the
rail (llm_client.py:2110) -- so 4000.2's call counter must be able to abort
MID-analysis and the dry pass must record the observed per-analysis count
before any live window.

## Section (f) -- pre-registered checks E1-E7 (selection rule + expected pass shape)

- E1 rail-used -- rule: the Section-(c) WHERE clause over llm_call_log rows in
  the 4000.3 window with model LIKE 'claude%'. Pass shape: N-of-N rows match
  (both halves under the one rule), N > 0.
- E2 metered-dark -- rule: count of metered-complement rows (Section (c) second
  clause) in-window == 0 AND the spend metric (spend.py::fetch_llm_spend
  semantics, which EXCLUDE rail rows) shows zero in-window delta. NEVER
  SUM(session_cost_usd) -- it is a per-session gauge.
- E3 model-truth -- rule: for >=1 sampled window call, parse the CLI envelope's
  RAW modelUsage map (iterate ALL keys; keys can share one canonicalModel and
  resolve_rail_model collapses them last-wins, claude_code_client.py:274-291).
  Pass shape: (i) abs(total_cost_usd - sum(costUSD over ALL entries)) < 1e-6;
  (ii) resolved model == llm_call_log.model for that call == the configured
  tier for the calling role.
- E4 output-integrity -- rule: the analysis completes; result is
  schema-compliant; NO final_synthesis.error / synthetic 0.0-HOLD; the persisted
  analysis row exists. Pass shape: all four shown with captures.
- E5 guard-health -- rule: rail-guard state at window end (breaker not open,
  smoke-window consecutive_failures == 0) PLUS a characterization of the
  STANDING failure rate: baseline 7d measured 2026-08-06 = 615 rail calls, 449
  ok (27% fail; by model: sonnet-4-6 105/482 fail, opus-4-8 50/115 fail 43%,
  haiku-4-5 11/18 fail 61%; failed-call mean latencies 116-150s vs
  claude_code_timeout_s=150 -> timeout-shaped). Pass shape: the window's own
  failures explained + the standing rate restated against this baseline.
- E6 quota-math -- rule: per-call tokens INCLUDING cache_creation from the RAW
  modelUsage sum, extrapolated to a 13-ticker cycle at current cadence. Pass
  shape: TWO figures with the estimation basis stated: (i) %-of-weekly-Max-pool
  per day, (ii) %-of-$100 Agent-SDK monthly credit under the PAUSED policy
  (support.claude.com/en/articles/15036540, F0). Context: overhead dominates --
  a 9-input-token probe booked 45,580 cache-creation tokens ($0.1537 would-be)
  because `claude -p` without --bare loads the repo cwd context every call (P3).
- E7 UI -- rule: 'no UI claims' unless a UI claim is made; then a Playwright
  capture behind the NextAuth wall. Expected: no UI claims.

## Discovered-defects register (queue at the 4000.1 flip; NOT fixed in this step)

- D1 attribution loss: ALL 605 overlay rail rows in 7d log bare agent='cc_rail'
  -- call sites never pass the role, so per-overlay health is unmeasurable
  (claude_code_client.py:659 fallback). Own step.
- D2 rail failure rate: 27% of rail calls fail at the timeout ceiling (see E5
  baseline). Characterization belongs to 4000.3 E5; any code fix is its own step.
- D3 cost under-count: resolve_rail_model collapses duplicate canonicalModel
  keys (P1) -- any consumer totalling the collapsed map under-counts; plus
  spend.py:30 cites a stale line (:504 -> now :659). Own step.
- W1/W2 vendor watch (for 4000.4): F0 paused `-p` billing policy; F1 --bare
  becoming the -p default (rail would lose credentials). Watch, not code.
