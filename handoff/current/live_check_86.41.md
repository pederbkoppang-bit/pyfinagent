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

1. **Correlation with an upstream rate limit.** 17 of 34 raw events are preceded
   within a measured 25-line window by a SEC 429 on
   `www.sec.gov/files/company_tickers.json`. A logic error does not correlate
   with a third party's rate limiter.
2. **Provider split, 0 Vertex.** 10 events carry `Failed to fetch CIK map: 429
   ... www.sec.gov`; 7 carry `Quant: SEC 429 rate-limit on CIK map, retrying`;
   **0 are Vertex**. Two providers, two remedies.
3. **Absence on healthy days.** 7 of the 10 census days carry zero such events. A
   logic error would not switch off.
4. **Corroborated in our own source.** `orchestrator.py` documents this upstream
   at class level: *"Cloud Function ... fetches
   https://www.sec.gov/files/company_tickers.json ... observed returning 429
   under 8 concurrent calls (cycle d73f5129)"*.

**DENOMINATOR RULE, stated as the step demands:** 34 = raw wrapper events, no
dedup, across all 42 retained logs. 18 = the deduped count from the filing
session, whose dedup rule I did not reproduce. Same numerator (17). **"94%" and
"50%" are both defensible and neither may be quoted without its rule.**

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
