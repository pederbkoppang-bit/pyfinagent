# live_check 76.9.2 — verbatim live evidence (2026-07-24)

## 1. Criterion 5 LIVE — dead rail fails LOUDLY on the REAL script (run first, so the
##    fail-state increment is later reset by the successful run)

```
$ export AUTORESEARCH_USE_MAX_RAIL=1; export AUTORESEARCH_MAX_RAIL_URL=http://127.0.0.1:59999
$ bash scripts/autoresearch/run_nightly.sh; echo "rc=$?"
rc=78
$ tail -2 handoff/autoresearch.log
[2026-07-24T17:01:26+02:00] START nightly autoresearch
[2026-07-24T17:01:26+02:00] END nightly autoresearch FAIL rc=78 (max-rail preflight: bridge http://127.0.0.1:59999 down; NOT falling back to metered API)
$ fail_state: {'consecutive_fails': 2}   # incremented through the 75.11 seam (was 1)
```

run_memo NEVER executed (no memo, no LLM traffic) — silent metered fallback is
structurally impossible: the exit precedes the run_memo invocation.

## 2. MODEL_MAP probe — the silent-downgrade trap is closed (LIVE, via the durable bridge)

```
$ curl -s -X POST http://127.0.0.1:18797/v1/messages ... "model":"claude-opus-4-8" ...
   "Reply with exactly your model family name..."
text: Opus
```

Before this step, claude-opus-4-8 was NOT in MODEL_MAP and silently ran as sonnet.
The proxy was kickstarted after the .js edit (`node --check` clean; /health OK).

## 3. Durable bridge — repo script, not the scratchpad copy

```
$ .venv/bin/python scripts/ops/anthropic_max_bridge.py &   # pid 85602
$ curl -s http://127.0.0.1:18797/health
{"ok":true,"proxy":"claude-code-cli"}    <- relayed through bridge -> proxy chain
```

(The session scratchpad bridge was stopped first — port freed; every subsequent
request in this file traverses the REPO-versioned script.)

## 4. Criterion 1 — real run_memo through the DURABLE routing

Flag path proven on the REAL script (verbatim log):

```
[2026-07-24T17:41:48+02:00] START nightly autoresearch
[2026-07-24T17:41:48+02:00] max-rail ON -- routing via http://127.0.0.1:18797 (dummy key, $0 metered)
```

PROCESS DISCLOSURE (honest record): the first two in-session full runs were KILLED
by the harness's ~10-minute background-task ceiling after the pipeline stalled in
gpt_researcher RETRIEVER fetches (external weather — S2/DDG rate-limited after a
full day of runs from this IP; the stall is upstream of the bridge: proxy per-call
cap is 180s and the bridge logged no hung request). No state damage either time
(no partial memo, fail-state untouched, bridge healthy). Third attempt runs
DETACHED (setsid nohup) with a Monitor on the log; final outcome appended below in
section 7. The $0-leakage property is independent of retriever weather: the dummy
key (`max-rail-dummy-key`) makes any api.anthropic.com call fail 401, and the
16:12 pre-hardening test already proved the identical chain end-to-end (rc=0,
15,809-char memo, 4m40s).

## 5. Mutation matrix — 6 planned, M4/M5/M6 executed + verbatim; M1 live (section 1);
##    M2/M3 executed after the detached run ends (mutating a script bash is
##    executing corrupts the running process) — results appended in section 7.

| # | Mutation | Result |
|---|---|---|
| M1 | wrong port in rail URL (LIVE, real script) | rc=78 loud, run_memo never ran — section 1 |
| M4 | bridge aggregation neutered (`elif False`) | `2 failed` (unit + e2e aggregation tests) |
| M5 | **FIXTURE**: stub upstream serves JSON instead of SSE | `1 failed` (aggregation e2e flips — fixture load-bearing) |
| M6 | **STUB**: fixture run_memo stops recording env | `2 failed` (flag-off + flag-on env asserts) |

Pre/post SHA-256 identical (bridge / test file / run_nightly.sh: `OK OK OK`).

## 6. Suite + immutable command (verbatim)

```
11 passed in 3.73s        (backend/tests/test_phase_76_9_2_max_bridge.py)
IMMUTABLE exit=0          (bash -n run_nightly.sh + ast.parse run_memo.py)
All checks passed! lint exit=0   (robust xargs form, reference/ excluded)
```

## 7. Detached-run outcome + M2/M3 — appended on completion
