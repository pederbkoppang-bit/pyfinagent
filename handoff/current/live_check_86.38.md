# Live check -- step 86.38

**Step**: `86.38` -- the deep pipeline is quota-killed and the book is trading
off the fallback. **Captured**: 2026-08-11T07:08:59Z, tree `40030783`,
by Main (`pyfinagent-51`), Opus 5 / effort max.

The masterplan names three required items. All three are below, plus what the
research gate REFUTED.

---

## A. The verbatim 429 body (required item 1)

**It was neither truncated nor unread.** Captured verbatim from `backend.log`:

```
{"timestamp": "2026-08-10 20:46:27,916", "level": "WARNING", "module": "autonomous_loop", "message": "Full orchestrator failed for HUM: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource exhausted. Please try again later. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}} -- falling back to lite Claude analyzer"}
```

The body is **complete**, and it carries **no discriminator**. That is by design:
Google's own guidance is that the HTTP code never conveys the cause -- the same
condition surfaces as 429 REST, `ResourceExhausted` gRPC, 403 on GCE and 5XX
under Provisioned Throughput -- and that classification must be done out of band
from `serviceruntime.googleapis.com/quota/rate/net_usage`, never by parsing the
error string.

**The step's question also contains a false premise.** It asks whether this is
"a per-minute rate limit, a daily quota, or a billing state". **Vertex generative
AI has no per-day quota**: every enforced dimension is per-minute (TPM/RPM), and
per-day limits belong to the AI Studio free tier, a different product. Under
Dynamic Shared Quota (Gemini 2.0+, and this book runs `gemini-2.5-flash`) there
is no project-level number at all, which is the documented explanation for a 429
arriving while usage looks trivial. The real trichotomy is **rate / DSQ capacity
/ model-or-billing state**.

**Consequence, stated as a design finding rather than a fix**: no logging change
at the call site can answer this question. Only the quota metric can, and reading
it is a GCP-console/monitoring action outside this step.

---

## B. Per-cycle full-versus-lite over >=10 cycles, with the command (required item 2)

```
$ python scripts/qa/derive_lite_fallback_census_86_38.py
========================================================================================
COVERAGE -- every fallback line must be accounted for, or no census is printed
========================================================================================
  backend.log.20260612T104931Z.gz        raw=  416  parsed=  416  ok
  backend.log.20260706T225648Z.gz        raw=   12  parsed=   12  ok
  backend.log.20260724T064045Z.gz        raw=    5  parsed=    5  ok
  backend.log.20260729T171222Z.gz        raw=    1  parsed=    1  ok
  backend.log.20260804T182713Z.gz        raw=    3  parsed=    3  ok
  backend.log.20260810T064130Z.gz        raw=    2  parsed=    2  ok
  backend.log                            raw=    3  parsed=    3  ok
  total accounted: 442

========================================================================================
PER-DAY full-pipeline vs lite-fallback  (JSON-format era only -- see UNDATED below)
========================================================================================
date           full   lite   lite%  causes
----------------------------------------------------------------------------------------
2026-07-24        3      1     25%  code defect: QuantAgent NoneType x1
2026-07-30        3      0      0%  
2026-07-31        4      3     43%  code defect: QuantAgent NoneType x3
2026-08-03        8      0      0%  
2026-08-04       11      0      0%  
2026-08-05        6      2     25%  code defect: QuantAgent NoneType x2
2026-08-06       11      0      0%  
2026-08-07       10      0      0%  
2026-08-09        8      0      0%  
2026-08-10        3      3     50%  429 RESOURCE_EXHAUSTED (quota) x3
----------------------------------------------------------------------------------------
TOTAL            67      9   11.8%

  days covered: 10

========================================================================================
UNDATED -- legacy plain-text log lines carrying a time but no date
========================================================================================
  These are REAL events, excluded from the per-day table only because the
  line format has no date. They are NOT dropped and NOT zero.
     199  backend.log.20260612T104931Z.gz :: config: GITHUB_TOKEN unset (legacy, resolved)
      88  backend.log.20260612T104931Z.gz :: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request
      62  backend.log.20260612T104931Z.gz :: timeout
```

**The instrument asserts its own coverage, and that is not decoration.** A first
version parsed only the JSON log format and silently dropped **416 events** from
the June-era log, which uses an older ANSI plain-text format -- the zero it
reported for June was a property of the parser, not of the system. The script now
counts the raw population per file, parses with both readers, and **WITHHOLDS the
census entirely on any shortfall**.

**What the table says, and what it refutes:**

- **67 full-pipeline completions vs 9 lite fallbacks over 10 days -- 11.8%.** The
  step text inferred from `llm_call_log` row counts that "the deep path has been
  mostly dead", while honestly flagging the inference as unconfirmed. Confirmed
  against the orchestrator's own accounting: **the deep path completed 88.2%**.
  The inference is REFUTED.
- **The 429 is a single-day event** (2026-08-10, 3 tickers). It is not chronic.
- **6 of the 9 fallbacks are a different defect entirely** --
  `QuantAgent failed for X: 'NoneType' object has no attribute ...`, a code bug,
  not quota. That is more of the degradation than the 429 is, and it is queued as
  its own step rather than absorbed here.

---

## C. The measured date of the last paper_trades row (required item 3)

```
TABLE: sunny-might-477607-p8.financial_reports.paper_trades
COLUMNS: ['trade_id', 'ticker', 'action', 'quantity', 'price', 'total_value', 'transaction_cost', 'reason', 'analysis_id', 'risk_judge_decision', 'created_at', 'round_trip_id', 'holding_days', 'realized_pnl_pct', 'mfe_pct', 'mae_pct', 'capture_ratio', 'signals']
time column chosen: created_at
QUERY: SELECT MAX(created_at) AS last_trade, COUNT(*) AS n FROM `sunny-might-477607-p8.financial_reports.paper_trades`
last_trade = 2026-07-31T18:47:37.309178+00:00 | total rows = 65
```

**2026-07-31T18:47:37Z**, 65 rows total. The step text's "last trade 2026-07-31"
is confirmed independently.

### And the correlation the step assumed does NOT hold

The step was filed on the premise that the zero-trade outcome is explained by the
degraded path. **Cross-referencing B and C refutes that**: 2026-08-03 through
2026-08-09 ran **54 full-pipeline analyses with ZERO fallbacks** and still
produced no trades. Whatever stopped the book trading, it is not the fallback.

**This is the single most important line in this step**, because the standing
framing pointed the next investigator at the wrong subsystem.

---

## D. Was the fallback silent to the operator? (the step's item (c))

**Partly -- RECORDED but not PAGED, and the distinction is the defect.**

RECORDED: the lite analysers stamp `_path: "lite"`, the fallback site stamps
`_fallback_reason`, and `_persist_analysis` writes both into the persisted
`full_report`. A BQ reader can already tell a lite row from a full one.

NOT PAGED: `_fallback_rate_check` fires only when the fraction **strictly
exceeds** its threshold. The 2026-08-10 cycle produced 3 full-pipeline Critic
verdicts and 3 fallbacks; `3/6 = 0.500` does not exceed `0.500`.

```
grep -c "Fallback-rate alarm fired"   backend.log   -> 0
grep -c "Fallback-rate alarm errored" backend.log   -> 0
POSITIVE CONTROL (literal injected into a pipe)     -> 1
```

The alarm **evaluated and did not fire** -- the zero is a real non-firing, not a
never-evaluated block, and the pattern is proven matchable rather than assumed so.

**HONEST LIMIT.** The predicate's denominator is
`len(candidate_analyses) + len(holding_analyses)`, which I did **not** measure
directly. I measured six analyses (3 full + 3 lite). With `n_fallback = 3` and
no firing, that is consistent with `n_total >= 6`. **The exact denominator is
unconfirmed, so "it missed by exactly one ticker" is NOT claimed** -- only that
it did not fire.

---

## E. What was changed, and what was deliberately not

```
$ bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(...);print(\"parsed\")"'
parsed
EXIT=0
```

```
$ python -m pytest backend/tests/test_phase_86_38_degradation_visibility.py \
                   backend/tests/test_phase_60_1_deep_pipeline.py -q
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
29 passed, 1 warning in 5.47s
EXIT=0
```

**Changed**: the degradation fields are now recorded on EVERY cycle via the
extracted seam `_degradation_summary_fields`, and persisted under a new
`degradation` key on the cycle record, kept separate from the phase-66.2
`funnel` because they answer different questions. `_intended_path` was
**removed** -- write-only (1 write, 0 reads repo-wide, recall-tested at 13 hits
for the sibling `_fallback_reason`) and redundant with `_fallback_reason`, so
wiring it would have created a second source of truth for one fact.

**NOT changed, deliberately**: the RiskJudge, every risk threshold, position
sizing, and any gate. `_fallback_rate_check` is untouched and paging behaviour
is byte-identical. The strict `>` is an operator decision (ask below), and
mutation cell M6 PROVES it is pinned by loosening it to `>=` and watching the
boundary test go red.

---

## F. Mutation matrix -- and the cell that caught ME

```
$ python scripts/qa/mutation_matrix_86_38.py
==============================================================================
phase-86.38 mutation matrix
==============================================================================
  backend/services/autonomous_loop.py  sha256=5b4d2680146f790c
  backend/services/cycle_health.py  sha256=49367c4b3090ad99

CONTROL -- the whole module must be GREEN before any mutation
  control rc=0  GREEN

  M1 KILLED  | seam reports nothing for every cycle (the original defect, relocated)
          test_the_recorded_fields_are_produced_by_a_seam_that_can_be_EXECUTED went RED (rc=1)
  M1b KILLED  | seam still correct but no longer wired into the cycle
          test_the_seam_is_actually_wired_into_the_cycle went RED (rc=1)
  M2 KILLED  | record the rate but hardcode that it never paged
          test_the_recorded_fields_are_produced_by_a_seam_that_can_be_EXECUTED went RED (rc=1)
  M3 KILLED  | restore the write-only _intended_path field
          test_intended_path_is_gone_and_not_merely_unused went RED (rc=1)
  M4 KILLED  | stop persisting degradation under the key readers look for
          test_degradation_is_persisted_on_a_quiet_cycle went RED (rc=1)
  M5 KILLED  | fold degradation into the 66.2 funnel instead of keeping it separate
          test_degradation_defaults_empty_and_breaks_no_existing_caller went RED (rc=1)
  M6 KILLED  | loosen the alarm to >= (paging behaviour MUST be pinned)
          test_the_2026_08_10_boundary_does_not_page went RED (rc=1)

[integrity] tracked sources unchanged: True
    backend/services/autonomous_loop.py  5b4d2680146f790c
    backend/services/cycle_health.py  49367c4b3090ad99

ALL 7 MUTANTS KILLED -- every guard in this matrix can fail.
EXIT=0
```

**The first run of this matrix scored 5 of 6, and the survivor was my own test.**
Cell M1 disabled the recording entirely (`if _n_fb_total:` -> `if False:`) and
the guard SURVIVED, because that guard asserted only that
`summary["fallback_rate"]` appeared BEFORE `if _fb_fire:` **in the source
text** -- and flipping a condition moves no text. A guard that cannot fail when
its subject is broken does not count.

The fix was not to strengthen the string match. The logic was **extracted into a
seam a test can execute**, and the matrix now attacks it from **both ends**: M1
mutates the seam's behaviour, M1b unwires it from the call site. Killing only one
would have left the other untested.

Every cell asserts its anchor exists before applying and scores **BROKEN**, not
KILLED, on a no-op replace. The matrix refuses to run at all if the target files
are already dirty, because then "restored" could not be distinguished from
"clobbered someone else's edit" -- and it re-verifies both files' sha256 after
every cell and at exit.

---

## G. What is NOT in force, and what I could not verify

- **COMMITTED BUT NOT IN FORCE -- MEASURED AGAINST THE RUNNING PROCESS, not
  asserted from the file.** The rule in this project is that a committed change
  is not a live change, and the way that rule gets broken is by reading the file
  instead of the process. So:

```
$ pgrep -f "uvicorn backend.main:app"          -> 66306
$ ps -o lstart= -p 66306                    -> man. 10 aug. 21.33.01 2026  
$ git log -1 --format=%ad --date=iso fd419038  -> 2026-08-11 09:04:35 +0200
```

  **The running process started BEFORE the GENERATE commit**, so it holds the
  pre-change `autonomous_loop` module in `sys.modules`. Nothing in this step is
  live until the session-end restart.

  Corroborated independently from the data side: the most recent row in
  `handoff/cycle_history.jsonl` (cycle `a5654ab9`, completed
  2026-08-10T19:15:34Z, 75.6 min, 0 trades) has **no `degradation` key at all** --
  which is exactly what the pre-change code writes, and what the post-change code
  cannot write. The next book cycle is 20:00 CEST.
- **The exact alarm denominator is unmeasured** (section D).
- **I did not read the GCP quota metric.** Classifying the 429 as rate vs DSQ vs
  billing requires `serviceruntime.googleapis.com/quota/rate/net_usage`, which
  is a console/monitoring action outside this step's scope.
- **I did not fix the QuantAgent `NoneType` defect** -- 6 of 9 fallbacks. It is
  queued as its own research-gated step.
- **No UI claim is made**, and no frontend file is touched, so no Playwright
  capture was taken.
