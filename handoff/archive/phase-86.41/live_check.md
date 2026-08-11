# live_check -- step 86.41

Required by `verification.live_check`: the re-derived census output including its
coverage assertion; the identified call site or an explicit statement that it
could not be identified; the absent-field-versus-logic-error adjudication with its
evidence; and the mutation output.

Captured 2026-08-11 by Main (`pyfinagent-06`). All output verbatim.

---

## 1. Re-derived census, WITH the coverage assertion

```
$ source .venv/bin/activate && python scripts/qa/derive_lite_fallback_census_86_38.py
EXIT CODE: 0

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
PER-DAY full-pipeline vs lite-fallback  (JSON-format era only)
========================================================================================
date           full   lite   lite%  causes
----------------------------------------------------------------------------------------
2026-07-24        3      1     25%  remote QuantAgent crash after SEC.gov 429 on the CIK map (upstream) x1
2026-07-30        3      0      0%  
2026-07-31        4      3     43%  remote QuantAgent crash after SEC.gov 429 on the CIK map (upstream) x3
2026-08-03        8      0      0%  
2026-08-04       11      0      0%  
2026-08-05        6      2     25%  remote QuantAgent crash after SEC.gov 429 on the CIK map (upstream) x2
2026-08-06       11      0      0%  
2026-08-07       10      0      0%  
2026-08-09        8      0      0%  
2026-08-10        3      3     50%  429 RESOURCE_EXHAUSTED (quota) x3
----------------------------------------------------------------------------------------
TOTAL            67      9   11.8%

  days covered: 10
```

**STATED LIMIT.** The coverage assertion proves **line counting**, not
**attribution**. The instrument counted all 442 events correctly while filing an
entire family under the wrong cause -- that misclassification is the finding of
this step. A green coverage line is not evidence the classification is right.

**STRONGER LIMIT, found by the cycle-1 Q/A and verified here.** The assertion is
**vacuous even against dropped rows** -- the thing it was written for.
`raw += 1` (`:274`) and `per_file_parsed += 1` (`:289`) are in the same
`if FALLBACK_MARK` branch with no path incrementing one without the other, so
`parsed == raw` is **structurally guaranteed**. Reproducing the historical
read-level JSON-only filter (the defect that actually dropped the 416) leaves the
assertion **GREEN at exit 0 while 433 of 442 events vanish**. It cannot see a
filter applied before the counting loop.

Criterion 2's literal requirements are still met and the population is
corroborated by two independent greps, so **the numbers stand**. What was wrong
was the artifact's claim about what the guard protects.

## 2. Call site -- IDENTIFIED

**`/workspace/main.py:89`, in `get_cik`, inside the REMOTE Quant Agent Cloud
Function.** Identified from a traceback, not inferred from ticker names. One
frame after a SEC 429 exhausts the CIK-map retry ladder and the fetcher returns
`None` instead of raising.

**The failing call is not in this repository**, so "fix the NoneType" is not a
change available here.

## 3. Absent upstream field vs logic error -- ADJUDICATED

**ABSENT UPSTREAM FIELD.**

Evidence that distinguishes them:

1. **Correlation with an upstream rate limit -- TOTAL, not partial.** **17 of 17
   distinct events** are preceded within a measured 25-line window by a SEC 429
   on `www.sec.gov/files/company_tickers.json`; **zero are not**. A logic error
   does not correlate with a third party's rate limiter, and certainly not at
   100%. (Corrected from "17 of 34 raw events" — 34 was a LINE count; see the
   denominator correction below.)
2. **Provider split, 0 Vertex.** 10 events carry `Failed to fetch CIK map: 429
   ... www.sec.gov`; 7 carry `Quant: SEC 429 rate-limit on CIK map, retrying`;
   **0 are Vertex**. Two providers, two remedies.
3. **Absence on healthy days.** 7 of the 10 census days carry zero such events. A
   logic error would not switch off.
4. **Corroborated in our own source.** `orchestrator.py` documents this upstream
   at class level: *"Cloud Function ... fetches
   https://www.sec.gov/files/company_tickers.json ... observed returning 429
   under 8 concurrent calls (cycle d73f5129)"*.

**DENOMINATOR -- CORRECTED. The dispute was a CATEGORY ERROR and is resolved.**

An earlier revision of this file called 34 and 18 "both defensible". **34 is a
count of LINES, not events.** Every occurrence emits exactly two log lines 17
lines apart (one `orchestrator` INFO, one `autonomous_loop` WARNING) — measured
across all 42 retained logs, per-file counts `12,4,4,2,6,6`, **every one even**,
every consecutive same-ticker gap exactly **17**.

Collapsing the pairs:

```
DISTINCT EVENTS (2-line pairs collapsed): 17
distinct tickers: 13  ['AAPL','COHR','CRWD','DDOG','DELL','DVA','INTC','MU',
                       'NTAP','PANW','SNDK','STX','WDC']
cue split: {'failed_to_fetch_cik_429': 10, 'sec429_retrying': 7}
events with NO 429 cue in a 25-line window: 0
=> attribution: 17 of 17 events carry a SEC 429 cue (100%)
```

**Neither "94%" nor "50%" was correct.** 94% (17/18) carried one phantom event;
50% (17/34) counted every event twice. **The correct figure is 100% — every
distinct occurrence carries an upstream SEC 429 cue**, which strengthens the
criterion-3 verdict rather than weakening it.

## 4. Mutation output

```
$ source .venv/bin/activate && python scripts/qa/mutation_matrix_86_41.py

[control] unmutated suite green: True

[M1-no-guard] KILLED  (remove the try/except entirely (the pre-86.41 state))
    expected red: ['test_quant_failure_does_not_abort_the_ticker']
    actual  red: ['test_quant_failure_does_not_abort_the_ticker', 'test_quant_failure_is_reported_as_degraded_and_names_the_cause', 'test_degraded_reason_does_not_impersonate_the_non_us_branch']

[M2-plain-append] KILLED  (setdefault -> plain [] append (the KeyError defect, real))
[M3-step-inside-try] KILLED  (move step() back inside the try (the over-broad-catch defect, real))
    expected red: ['test_healthy_quant_is_untouched_by_the_guard']
    actual  red: ['test_healthy_quant_is_untouched_by_the_guard']
[M4-impersonate-non-us] KILLED  (reuse phase-60.1's non-US reason for a rate-limit failure)
[M5-default-reason-drift] KILLED  (change the phase-60.1 default reason string)

[restore] target byte-identical to pre-mutation: True

==========================================================================
  KILLED           M1-no-guard              remove the try/except entirely (the pre-86.41 state)
  KILLED           M2-plain-append          setdefault -> plain [] append (the KeyError defect, real)
  KILLED           M3-step-inside-try       move step() back inside the try (the over-broad-catch defect, real)
  KILLED           M4-impersonate-non-us    reuse phase-60.1's non-US reason for a rate-limit failure
  KILLED           M5-default-reason-drift  change the phase-60.1 default reason string
==========================================================================
post-restore suite green: True 
RESULT: all 5 cells KILLED, control green, target restored.
```

**M3 SURVIVED on the first run** -- a real hole, not a formality. The healthy-path
test read only the first `quant/completed` event, whose message is identical
whether the guard stayed out of the way or fired and overwrote a good report.
Closed with a yfinance call counter plus a positive control so the empty-list
assertion cannot pass vacuously. The re-run above is post-fix.

## 5. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'
parsed
exit=0
```

It parses a file this step deliberately did not modify. Green reaches
**criterion 1 only**; it cannot see the guard, the tests, or the matrix.

## 6. NOT IN FORCE

The running backend imported the pre-change module. `73dcf2c8` and `678d979f`
are on origin but **not executing**. Restarts batch to session end; the book runs
at 20:00 CEST. This is a live-code capture, **not** a live-behaviour capture, and
is labelled as such.
