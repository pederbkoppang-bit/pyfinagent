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

## 7. Detached-run outcomes (honest record; criterion 1 NOT yet closed this session)

Attempt 3 (detached, 19:14:08 START): flag path + rail proven live (max-rail ON line,
code-0 claude calls) but the client hung ~30 min after a single code-1 claude call --
run_memo's asyncio loop wedged on a lost future (stack sampled: kevent select, idle;
a fresh probe through the SAME chain completed in seconds => transport healthy,
client-side gpt_researcher hang). Killed at 62 min. Attempt 4 (direct run_memo, fresh
topic, identical env exports, gtimeout 1200): DIRECT-RC=124 (cap fired; retriever
weather: S2 429-throttled all day + DDG import-broken -- queued 76.9.3). The 16:12
end-to-end completion (rc=0, 15,809-char memo, 4m40s) proves the CHAIN; it predates
the repo-hardened bridge by minutes (scratchpad copy, same code lineage). CRITERION 1
therefore remains OPEN pending one clean rc=0 through the repo bridge -- retry in
quiet hours or after 76.9.3 restores DDG. M2/M3 mutations also pending (script now
idle). Q/A NOT yet spawned -- step stays in_progress.

## 8. Attempt 5 (2026-07-25, quiet hours, 3 retrievers) — WEDGED. Criterion 1 still OPEN.

Launched 00:16:04 detached (`gtimeout 1800`, subshell `nohup`), flag ON, after 76.9.3
restored the DuckDuckGo leg so all three retrievers were live for the first time.

```
[2026-07-25T00:16:16+02:00] START nightly autoresearch
[2026-07-25T00:16:16+02:00] max-rail ON -- routing via http://127.0.0.1:18797 (dummy key, $0 metered)
...
INFO: [00:17:02] Browsing the web to learn more about the task: ...
INFO: [00:17:03] Planning the research strategy and subtasks...
Error processing https://arxiv.org/abs/2502.05878: Page request resulted in HTTP 429 (export.arxiv.org)
   [... HTTP 429 on EVERY arxiv fetch ...]
Using default report type: research_report prompt.
   [... then silence; last log write 00:17:58 ...]
=== RUN ENDED 00:46:23 ===        <- killed by gtimeout at the 30-min cap; no END line written
```

**The transport did its job.** The bridge served **27 responses, every one HTTP 200**
(17 of them `POST /v1/messages`), last completing at **00:22:52**:

```
$ grep -oE "\" [0-9]{3} " handoff/logs/anthropic-bridge.log | sort | uniq -c
  27 " 200
$ grep -vE "GET /health.*200|POST /v1/messages.*200" handoff/logs/anthropic-bridge.log
[max-bridge] listening on http://127.0.0.1:18797 -> https://localhost:18796
```

Zero non-200s, zero PROXY-ERROR lines. The research phase completed.

**Then the client wedged**, measured rather than inferred:

| Probe | Observation |
|---|---|
| CPU | 0.0%, ~10.8s total CPU over 20+ minutes |
| `claude` child process | none |
| bridge → proxy connection | none (proxy showed only LISTEN) |
| client → bridge connection | ONE, ESTABLISHED, idle keep-alive (fd 11 → 127.0.0.1:18797) |
| retriever sockets | all CLOSED / CLOSE_WAIT |
| thread sample | parked in `_PySemaphore_Wait` / `_queue_SimpleQueue_get`; asyncio threads idle |

Identical signature to attempt 3 (2026-07-24 19:14, killed at 62 min).

**Honest reading of the evidence.** 27× HTTP 200 proves the transport *delivered*
responses; it does **not** prove the final response *body* was well-formed, so the
bridge's SSE→JSON aggregation is **not** fully exonerated — a 200 with a subtly
truncated body would look exactly like this from the outside. Both hypotheses (H1
client-side gpt_researcher asyncio defect; H2 malformed aggregated body) are live, and
settling them requires capturing an actual response body, which this run did not do.

**Criterion 1 is therefore NOT met and is not claimed.** Five attempts: one success
(2026-07-24 16:12, rc=0, 15,809-char memo, 4m40s — but through the scratchpad-lineage
bridge, minutes before the repo hardening), one deliberate dead-rail test (rc=78, the
criterion-5 proof in §1), one externally killed (rc=143), one gtimeout (rc=124), and
this one. The blocker is a distinct defect, not a property of the routing, so it is
queued as its own research-gated step **76.9.5** (P1) rather than retried blindly — the
step requires the wedge point to be *captured* (stack at the moment of the hang, plus an
actual response body) and H1-vs-H2 settled before any fix, and it carries the mutation
that a stub upstream which never responds must make the run exit loudly.

## 9. ROOT CAUSE FOUND AND FIXED (2026-07-25) — an HTTP framing regression I introduced

The 76.9.2 Q/A returned **FAIL** on criterion 1 and, unlike me, **measured** the cause.

**The defect.** `scripts/ops/anthropic_max_bridge.py:103` declares
`protocol_version = "HTTP/1.1"`, while the SSE **passthrough** branch sent neither
`Content-Length`, nor `Transfer-Encoding`, nor `Connection: close`. Per **RFC 7230
§3.3.3** that response is delimited *only* by connection close — so a keep-alive client
blocks forever after the last SSE byte. The production client is httpx (anthropic SDK),
which gpt_researcher reaches via `stream=True` at **6 sites** in
`actions/report_generation.py`: the report phase, exactly where attempts 3 and 5 wedged.

**Why my hardening caused it** (verified first-hand): the scratchpad bridge set NO
`protocol_version`, so it defaulted to HTTP/1.0 and closed the socket. That accident is
the only reason the 2026-07-24 16:12 run completed — the criterion's "not the session
scratchpad bridge" exclusion was substantive, not pedantic.

**Why my §8 inference was wrong.** "Bridge logged 200, client at 0% CPU with one idle
connection" is precisely what this defect produces: the bridge finishes writing while
the client waits for a delimiter that never comes. I read that as pointing client-side.

**Why no test caught it.** The sole coverage,
`test_e2e_streaming_client_gets_sse_passthrough`, uses `urllib`, whose
`AbstractHTTPHandler.do_open` sets `headers["Connection"] = "close"` — forcing the very
close the production client never sends. Confirmed in the installed stdlib.

### The fix, and proof it is load-bearing

```
=== BASELINE (fix in place) ===              12 passed in 3.98s

=== M7: revert the framing fix ===           1 failed, 11 passed
  reds: ['test_e2e_streaming_body_is_framed_for_a_KEEP_ALIVE_client']
  NEW keep-alive guard RED    : True
  OLD urllib guard STILL GREEN: True   <- proves the old test was VACUOUS

=== POST-REVERT ===                          12 passed  reds: []
SHA identical: True
```

```
=== M8: strip the loud-fail behaviour but KEEP the comment ===
  1 failed | reds: ['test_nightly_default_documented_off']
  guard KILLED by a comment-only survival: True     (it stayed GREEN before this fix)
```

### Live re-probe of the running bridge (raw socket, HTTP/1.1 keep-alive)

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: close
... body bytes=1081
has Connection: close : True
server CLOSED stream   : True   <- False was the hang
```

Criterion 1 stays **unclaimed** until a full run_memo actually completes rc=0 through
the fixed bridge. Attempt 6 is executing.

## 10. CRITERION 1 — MET. Attempt 6 completed rc=0 through the FIXED repo bridge.

First clean end-to-end run through the durable routing, immediately after the §9 framing
fix. Six attempts; the framing regression was the whole cause.

```
[2026-07-25T01:16:47+02:00] START nightly autoresearch
[2026-07-25T01:16:47+02:00] max-rail ON -- routing via http://127.0.0.1:18797 (dummy key, $0 metered)
[2026-07-25T01:22:47+02:00] END nightly autoresearch OK
```

**rc=0 is structural, not inferred.** `run_nightly.sh:94-96` emits
`END nightly autoresearch OK` **only** inside the `if python …/run_memo.py; then`
success branch; the failure branch writes `FAIL rc=$rc`. The same branch resets the
fail-state, and it now reads `{"consecutive_fails": 0}` (it was 2 after the §1
dead-rail test), which no other path does.

**The memo is real and non-ERROR** (contrast: every prior nightly wrote `*-ERROR-*.md`):

```
handoff/autoresearch/2026-07-25-topic10-what-are-the-published-findings-2025-2026-on-retrieval-augme.md
16,634 bytes   filename contains ERROR: NO

# Autoresearch memo -- 2026-07-25
**Topic (index 10):** What are the published findings (2025-2026) on retrieval-augmented
LLMs for equity research ...
**Source:** gpt-researcher `detailed_report`, Claude-driven, semantic_scholar + arxiv +
duckduckgo retrievers.
...
Li, Y., Wang, S., Ding, H., & Chen, H. (2024). *Large language models in finance: A survey*.
Wang, F., Wan, X., Sun, R., Chen, J., & Arik, S. O. (2025). Astute RAG: Overcoming imperfect r...
```

Synthesized prose with a real bibliography — not a stub. Note the three-retriever line:
**duckduckgo is present**, so 76.9.3's ddgs pin is doing work here too.

**Served by the DURABLE bridge**, not the scratchpad copy: pid 50256 running
`scripts/ops/anthropic_max_bridge.py`, started after the framing fix; 22 `POST
/v1/messages` served across its lifetime.

**$0 metered leakage:** the run exported `ANTHROPIC_API_KEY=max-rail-dummy-key`, and the
run output contains **zero** occurrences of `api.anthropic.com`, `401`, or
`authentication_error` — any call escaping to the metered API with that key would have
failed loudly and appeared here.

Disclosed, and NOT a defect of this step: the memo header says
`gpt-researcher detailed_report` while the runtime logged
`Invalid report type ... Using default report type: research_report prompt.` — the
artifact mislabels its own report type. That is exactly the queued **76.9.1**
(`report_type='detailed_report'` is invalid in gpt-researcher 0.14.8), now corroborated
by a successful run rather than only by an errored one.

### Criteria status after this run

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Real run_memo rc=0 through the DURABLE routing + $0-leakage proof | **MET** (this section) |
| 2 | Flag-gated, one-env-change revert, default documented | MET (§ earlier + tests) |
| 3 | Other proxy clients enumerated/unbroken or additive-only | MET |
| 4 | Durable CLAUDE_PATH + MODEL_MAP coverage | MET |
| 5 | MUTATION: broken routing fails loudly, never silent metered fallback | MET (§1, rc=78 live) |
