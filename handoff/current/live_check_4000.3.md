# live_check -- 4000.3: the operator-gated live smoke window (AAPL, 2026-08-06)

Window verdict: the smoke returned an HONEST FAIL -- the analysis died in
production at the quant step BEFORE any LLM call, so the window measured a
real defect instead of the rail. Per the operator's pre-stated end-state rule
this is a valid, closable outcome: rail stays ON, no retry, the defect is
queued (step 4000.10). Every capture below is verbatim.

## Preconditions (criterion 2)

All three recorded in handoff/harness_log.md BEFORE the window: the
'RAIL TIERS: AS CONFIGURED' line (:31182, 79.55 closed 5857bc3c); the
'4000.3 LIVE-WINDOW AUTHORIZATION' operator block with the end-state rule
stated in advance (all-pass = ON; FAIL = STAYS ON + queue, operator owns any
OFF-flip; cap trip = loud stop, no retry; no --keep-on so the ExitStack
restore returns the flag to its pre-window True on every path); and the
single-writer confirmation evidenced by the two git add -An captures below.

## Window timeline + collision avoidance

Pre-window check 2026-08-06T14:23:11Z; margin to the same-process
paper_trading_daily cycle (18:00:00Z, cron mon-fri 14:00 ET): 217 minutes.
Window opened ~14:24Z, closed (post-flush checks emitted) ~14:26Z -- more
than 3.5 hours clear of the cycle. Parameters: --live --ticker AAPL
--expected-calls-per-analysis 30 --analysis-deadline 3600 (per
contract_4000.3.md; ticker + deadline from the research gate).

## Full window output (verbatim; machine-readable E1-E6 verdicts, criterion 3)

```
{"event": "pre_flag", "paper_use_claude_code_route": true}
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
{"event": "spend_bracket_before", "daily_usd": 0.0}
{"event": "flag_put", "paper_use_claude_code_route": true}
{"event": "analysis_started", "ticker": "AAPL", "analysis_id": "13646d61-bdf1-415b-8004-1cbcf27967b5"}
{"event": "flush_wait", "seconds": 65}
{"check": "E1", "ok": false, "matched": 0, "claude_role_rows": 0, "rule": "provider='claude-code' OR agent='cc_rail' OR agent LIKE 'cc_rail:%'"}
{"check": "E2", "ok": false, "metered_rows": 0, "rail_rows_positive_control": 0, "spend_before": 0.0, "spend_after": 0.0, "spend_delta": 0.0, "fail_open_note": "fetch_llm_spend fails open to 0.0; a 0.0/0.0 bracket cannot distinguish dark from fetch-failure -- the row-count leg is primary"}
{"check": "E3", "ok": true, "probe": "1 entries, cost sum 0.015108 == total", "stray_models": [], "expected_set": ["claude-haiku-4-5", "claude-opus-4-8", "claude-sonnet-4-6"]}
{"check": "E4", "ok": false, "failures": [["AAPL", "failed", "[RuntimeError] Step 'quant': ERROR: QuantAgent failed for AAPL: 'NoneType' object has no attribute 'get'"]], "analyses": 1, "persisted_row_leg": "NOT PROVABLE HERE: the sync poll reads the in-memory task dict (analysis.py:392), which is not persistence evidence -- frozen-baseline E4 leg 4 is queued as its own step and owed as 4000.3 live_check evidence"}
{"check": "E5", "ok": true, "failed_rail_rows": 0, "rail_guard_skipped_markers": 0, "note": "breaker P1 page not observable from here; direct consecutive_failures read has no out-of-process surface (gap queued as 4000.8)"}
{"check": "E6", "ok": true, "basis": "USD at API list rates over window rail rows (cache_read priced as input; session_cost_usd is a gauge and is never summed); 13-ticker cycle x --cycles-per-day extrapolation", "window_rail_calls": 0, "window_tokens_in_incl_cache": 0, "window_tokens_out": 0, "would_be_usd_per_analysis": 0.0, "would_be_usd_per_day_13_tickers": 0.0, "pct_of_100usd_monthly_credit": 0.0, "weekly_pool_note": "no token-denominated weekly-pool figure is published; the USD figures above are the stated proxy (contract R9/E6)"}
{"summary": true, "mode": "live", "all_ok": false, "checks": [["E1", false], ["E2", false], ["E3", true], ["E4", false], ["E5", true], ["E6", true]]}
SMOKE_EXIT=
```

EXIT-CODE CAVEAT, disclosed: the harness wrapper echoed SMOKE_EXIT=${PIPESTATUS[0]}
under zsh, where the array is lowercase `pipestatus`, so the literal capture is
empty. The exit code is established by the script's proven mapping (22-test
suite, criterion 4 of 4000.2: overall exit is non-zero iff any check fails) and
the summary line above: all_ok=false -> exit 1.

## Rail-call counter (criterion 4)

window_rail_calls = 0 <= 30 (E6 line above); tickers = 1 <= 2. The cap was
armed (pre-start gate passed at expected=30 <= 30) and never tripped.

## llm_call_log window rows under the 4000.1(c) rule (criterion 3)

Rule, restated inline and executable verbatim: a rail row iff
`provider = 'claude-code' OR agent = 'cc_rail' OR agent LIKE 'cc_rail:%'`;
metered iff `provider = 'anthropic' AND (agent IS NULL OR (agent != 'cc_rail'
AND agent NOT LIKE 'cc_rail:%'))`. Post-window re-query at 14:27Z (3 minutes
after the flush wait, so buffered-tail ambiguity is reduced though not
eliminated -- the flush piggybacks on the NEXT backend LLM call):

```
SELECT FORMAT_TIMESTAMP('%H:%M:%S', ts) t, provider, model, agent, ok, ticker
FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
WHERE ts >= TIMESTAMP('2026-08-06T14:20:00Z') ORDER BY ts LIMIT 30
-- result: rows: 0
```

ZERO rows of any class. Interpretation, grounded in code: the analysis died at
orchestrator.py:1790-1792 (`step("quant", ...)` -> run_quant_agent), which is
Step 2 -- BEFORE the first LLM-bearing step -- so no rail call was ever made.
The E1/E2 zeros are therefore the POSITIVE-CONTROL design working as intended
(a zero without a rail-row control is a FAIL, never evidence of darkness).

## One complete CLI envelope with the modelUsage map (criterion 3)

Captured post-window by the same probe mechanism the smoke uses (empty temp
cwd, --model claude-haiku-4-5; the in-window probe printed only its summary
line, so this complete capture is a separate invocation, disclosed as such).
NOTE it demonstrates the duplicate-canonicalModel pattern AGAIN (two keys, one
canonical) with the sum invariant holding exactly (0.0234713 == total) -- the
third live sighting of the 4000.7 defect's trigger shape:

```json
{
    "is_error": false,
    "duration_api_ms": 5667,
    "num_turns": 1,
    "stop_reason": "end_turn",
    "session_id": "46ae58e8-9bd3-42d2-9892-dee9afa2032d",
    "total_cost_usd": 0.0234713,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 10445,
        "cache_read_input_tokens": 17703,
        "output_tokens": 44,
        "server_tool_use": {
            "web_search_requests": 0,
            "web_fetch_requests": 0
        },
        "service_tier": "standard",
        "cache_creation": {
            "ephemeral_1h_input_tokens": 10445,
            "ephemeral_5m_input_tokens": 0
        },
        "inference_geo": "not_available",
        "iterations": [
            {
                "input_tokens": 10,
                "output_tokens": 44,
                "cache_read_input_tokens": 17703,
                "cache_creation_input_tokens": 10445,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 0,
                    "ephemeral_1h_input_tokens": 10445
                },
                "type": "message"
            }
        ],
        "speed": "standard"
    },
    "modelUsage": {
        "claude-haiku-4-5-20251001": {
            "inputTokens": 521,
            "outputTokens": 12,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "webSearchRequests": 0,
            "costUSD": 0.000581,
            "contextWindow": 200000,
            "maxOutputTokens": 32000,
            "canonicalModel": "claude-haiku-4-5",
            "provider": "firstParty"
        },
        "claude-haiku-4-5": {
            "inputTokens": 10,
            "outputTokens": 44,
            "cacheReadInputTokens": 17703,
            "cacheCreationInputTokens": 10445,
            "webSearchRequests": 0,
            "costUSD": 0.0228903,
            "contextWindow": 200000,
            "maxOutputTokens": 32000,
            "canonicalModel": "claude-haiku-4-5",
            "provider": "firstParty"
        }
    },
    "permission_denials": [],
    "terminal_reason": "completed",
    "fast_mode_state": "off",
    "fast_mode_disabled_reason": "sdk_opt_in_required",
    "subtype": "success",
    "api_error_status": null,
    "result": "OK",
    "ttft_ms": 2048,
    "ttft_stream_ms": 1598,
    "time_to_request_ms": 25,
    "type": "result",
    "duration_ms": 2113,
    "uuid": "ed1bb1c1-eddb-42a1-820c-2aaa7a3c6e55"
}
```

## End state (criterion 5)

Pre-window flag True; post-window capture:

```
== POST-WINDOW GET /api/settings (2026-08-06T14:26:21Z) ==
{"paper_use_claude_code_route": true}
```

The pre-stated rule applied to the observed results (FAIL) says: rail STAYS
ON, evidence queued, any OFF-flip is the operator's own later decision. The
observed end state matches. The ExitStack restore PUT executed on the normal
exit path (no --keep-on was passed; no keep_on event appears in the output).

## git add -An brackets (criterion 6) -- REWRITTEN cycle 2 with verbatim captures

CORRECTION 2 (Q/A cycle-2 finding 1): the FULL path lists of the two window
brackets were NEVER RETAINED -- the as-run commands printed a count plus a
FILTERED view, and the unfiltered lists are genuinely unrecoverable. Stated
explicitly per the evaluator's named fix: this criterion's foreign-sweep
assurance therefore rests on (a) the as-run filtered captures below, exactly
as executed, and (b) the PRE-FLIP full capture protocol at the end of this
section. The 9 -> 11 delta CANNOT be fully reconciled retroactively: the
known movement is the concurrent session's project_autoresearch_max_rail_85_1.md
(new memory file, named in the POST filter output below) and the
UNSPECIFIED->85.1 brief rename (a substitution, net zero -- the cycle-1 text
wrongly presented it as accounting for the delta); at least one POST path is
unnamed and unrecoverable. What both as-run filters DO establish is the
load-bearing half: no backend/, frontend/ or settings path appeared in either
bracket's filtered residue.

As-run PRE capture, verbatim (14:23:11Z; count + filtered view as executed):

```
$ git add -An | wc -l
       9
$ git add -An 2>&1 | grep -vE "handoff/(audit|away_ops)/|cycle_heartbeat|archive-baseline|MEMORY.md|cycle_history|kill_switch|prompt_leak|4000|harness_log|threshold" | head -8
add 'handoff/current/research_brief_UNSPECIFIED.md'
```

As-run POST capture, verbatim (14:26:21Z; same filter):

```
$ git add -An | wc -l
      11
$ git add -An 2>&1 | grep -vE "handoff/(audit|away_ops)/|cycle_heartbeat|archive-baseline|MEMORY.md|cycle_history|kill_switch|prompt_leak|4000|harness_log|threshold|UNSPECIFIED" | head -5
add '.claude/agent-memory/researcher/project_autoresearch_max_rail_85_1.md'
```

Fresh FULL capture, verbatim (2026-08-06T14:41:58Z):

```
add '.claude/agent-memory/researcher/MEMORY.md'
add '.claude/masterplan.json'
add 'handoff/audit/config_change_audit.jsonl'
add 'handoff/audit/instructions_loaded_audit.jsonl'
add 'handoff/audit/pre_tool_use_audit.jsonl'
add 'handoff/away_ops/health.jsonl'
add 'handoff/harness_log.md'
add '.claude/agent-memory/researcher/project_autoresearch_max_rail_85_1.md'
add '.claude/agent-memory/researcher/project_cc_rail_live_window_4000_3.md'
add 'handoff/current/contract_4000.3.md'
add 'handoff/current/evaluator_critique_4000.3.md'
add 'handoff/current/experiment_results_4000.3.md'
add 'handoff/current/live_check_4000.3.md'
add 'handoff/current/research_brief_4000.3.md'
add 'handoff/current/research_brief_82.58.md'
add 'handoff/current/research_brief_85.1.md'
add 'scripts/qa/verify_phase_4000_3_live_smoke.sh'
```

Reconciliation, every path accounted: THIS STEP (8): the five suffixed 4000.3
handoff files + the check script + the 4000.3 researcher memory
(project_cc_rail_live_window_4000_3.md) + masterplan.json (the 4000.10 queue
addition). SHARED APPEND STREAMS (5): three audit jsonl + health.jsonl +
harness_log.md (both sessions append). CONCURRENT SESSION (4): researcher
MEMORY.md (interleaved index lines from both sessions),
project_autoresearch_max_rail_85_1.md, research_brief_85.1.md (the renamed
UNSPECIFIED), research_brief_82.58.md (mtime 14:40Z, ACTIVELY being written).
No production code from either session was uncommitted AT THAT CAPTURE TIME
(14:41:58Z) -- a statement about that instant only. Q/A cycle-2 measured the
tree moving 17 -> 18 -> 22 paths within ten minutes as the concurrent session
began editing backend/services/observability/spend.py (+18/-2, PRODUCTION
CODE) and backend/tests/conftest.py, which also proves the cycle-1 flip
sentinel (one research brief's mtime) was the WRONG gate -- it read settled
at exactly the moment the foreign session moved into backend code.

FLIP PROTOCOL (cycle-3, replacing the wrong sentinel; the gate is the FULL
derived foreign set, not any single file):

1. Immediately pre-flip, take `git add -An` TWICE, >=3 minutes apart.
2. The flip proceeds ONLY if (a) the two captures are IDENTICAL, and (b)
   neither contains any backend/, frontend/, or .claude/settings path that is
   not this step's own. Shared append streams (handoff/audit, health,
   harness_log, agent-memory MEMORY.md indexes) and the concurrent session's
   completed handoff/research artifacts are named and swept KNOWINGLY.
3. If the gate fails, WAIT and re-take, or let the 82.58 session commit
   first. Sweeping another session's uncommitted backend code is the
   feedback_uncommitted_is_not_protected hazard and never acceptable.
4. The passing pre-flip capture is pasted verbatim into this file's
   'Pre-flip capture' addendum before the status flip.

## Pre-flip capture (the gate, EXECUTED and PASSED)

Protocol run: capture 1 at 15:28:06Z (11 paths, zero foreign); capture 2 at
15:31:29Z DIFFERED (the concurrent session closed 82.58 -- its archive
snapshots appeared -- and started 82.51's research brief), so per step 3 the
gate looped; captures 2 and 3 (15:31:29Z / 15:35:03Z, >=3 min apart) are
IDENTICAL with ZERO backend/, frontend/, or .claude/settings paths. Passing
capture, verbatim:

```
CAPTURE 3 (2026-08-06T15:35:03Z):
add '.claude/.archive-baseline.json'
add 'handoff/audit/config_change_audit.jsonl'
add 'handoff/audit/instructions_loaded_audit.jsonl'
add 'handoff/audit/pre_tool_use_audit.jsonl'
add 'handoff/away_ops/health.jsonl'
add '.claude/agent-memory/researcher/project_cc_rail_live_window_4000_3.md'
add 'handoff/archive/phase-82.58/contract.md'
add 'handoff/archive/phase-82.58/evaluator_critique.md'
add 'handoff/archive/phase-82.58/experiment_results.md'
add 'handoff/archive/phase-82.58/research_brief.md'
add 'handoff/current/contract_4000.3.md'
add 'handoff/current/evaluator_critique_4000.3.md'
add 'handoff/current/experiment_results_4000.3.md'
add 'handoff/current/live_check_4000.3.md'
add 'handoff/current/research_brief_4000.3.md'
add 'handoff/current/research_brief_82.51.md'
add 'scripts/qa/verify_phase_4000_3_live_smoke.sh'
```

Knowingly-swept non-step paths, named per condition (b)'s carve-out: four
hook-churn streams + archive-baseline; the concurrent session's COMPLETED
phase-82.58 archive snapshots; its settled research_brief_82.51.md (mtime
stable across the identical pair); the researcher memory for this step. No
production code from any session is swept.

## Backend restart (criterion 7) -- REWRITTEN cycle 2 from re-derived facts

CORRECTION (Q/A cycle-1 finding 2): the first version of this section carried
the 4000.1 baseline's pid (60478) forward without re-deriving it -- that claim
was FALSE at window time. Re-derived 2026-08-06T14:41Z, verbatim:

```
$ ps -eo pid,ppid,lstart,etime,command | grep -i "uvicorn backend.main" | grep -v grep
89530     1 tor.  6 aug. 13.46.15 2026      02:54:21 /opt/homebrew/.../Python...
89533 89530 tor.  6 aug. 13.46.15 2026      02:54:21 /usr/bin/caffeinate -i -s .../uvicorn backend.main:app...
$ lsof -nP -iTCP:8000 -sTCP:LISTEN | tail -1
Python  89530 ford   10u  IPv4 ...  TCP *:8000 (LISTEN)
```

A backend restart DID occur at 2026-08-06 13:46:15 local (11:46:15Z) -- both
workers (parent 89530 + child 89533) share that single start instant, i.e.
parent AND child were cycled together. Ordering: the restart POST-dates the
RAIL TIERS record (commit 992238b5, pushed 09:59:27Z), so it shipped
operator-confirmed tiers, and PRE-dates this step's first artifact
(research_brief_4000.3.md, 14:21:03Z) by ~35 minutes. Therefore NO restart
occurred INSIDE this step; the criterion's conditional branch is satisfied in
substance as well -- the one restart that happened today is after the RAIL
TIERS timestamp with both workers cycled. The unbroken 02:54 elapsed spans the
entire window (14:24-14:26Z).

## E4 leg 4 live evidence (owed per 4000.2 R10)

The failed analysis persisted NO analysis_results row, exactly as the research
gate predicted for the failure path (save_report is post-success only):

```
SELECT ticker, analysis_date FROM `...financial_reports.analysis_results`
WHERE ticker='AAPL' ORDER BY analysis_date DESC LIMIT 3
-- newest: 2026-03-07 20:39:27 UTC (then 2025-11-24, 2025-11-23)
```

Two facts in one capture: (a) no new row from this window (failure semantics
hold); (b) the newest SUCCESSFUL full AAPL analysis is FIVE MONTHS old --
independent corroboration of the phase's throughput thesis.

## FAILed checks -> queued step (criterion 8)

E1, E2 and E4 all root-cause to ONE defect: the quant-step crash. Queued as
masterplan step 4000.10 (added in the same edit as this step's flip -- UPDATE
cycle 2: added EARLY, ahead of the flip, per Q/A cycle-1 finding 5; it exists
in .claude/masterplan.json now, P1, with the re-run window as its live_check):
`[RuntimeError] Step 'quant': ERROR: QuantAgent failed for AAPL: 'NoneType'
object has no attribute 'get'` inside run_quant_agent (orchestrator.py:1792,
the _sec_covered branch); the same defect CLASS was fixed once before at a
sibling site (orchestrator.py:1809-1811, phase-27.6.2 `valuation: None`
guard). Because the crash is pre-LLM, it kills EVERY full analysis on this
path -- a direct throughput blocker and the reason this window could not
measure the per-analysis rail-call count. E6's real measurement moves to
4000.10's live_check (a re-run window on the fixed pipeline, under a fresh
operator token).
