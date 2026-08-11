# Experiment results -- step 86.41

**Step**: `86.41` (phase-86, P2, `harness_required: true`) | **Phase**: GENERATE
**Date**: 2026-08-11 | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Contract**: `handoff/current/contract_86.41.md` (committed `b8d2ea96`, BEFORE any code)

---

## 1. Headline

**The step's premise was refuted by its own research gate, and the refutation
held under GENERATE.** The QuantAgent NoneType crash is not a competing cause to
the 429 -- it is **downstream of a different provider's 429**, raised in a
**remote Cloud Function that is not in this repository**.

What survived is narrower and genuinely ours:
`backend/agents/orchestrator.py` had **exactly one unguarded sub-agent call**,
and `backend/services/autonomous_loop.py:2201` converted that single failure into
a **whole-ticker lite fallback**. That asymmetry is fixed.

**The guard I wrote shipped with two defects of its own.** Both were invisible to
`ast.parse` and to 22 pre-existing green tests, and both were found only by
driving the real pipeline. They are recorded in full in §5 because they are the
most useful thing this step produced.

## 2. Immutable success criteria -- VERBATIM from `.claude/masterplan.json`

Copied verbatim from `verification.success_criteria`. Note the key is
`success_criteria`, **not** `criteria`; my first probe read `criteria` and
reported an empty list, which would have been a false claim that this step had no
criteria. Corrected before use.

> 1. The failing `.get` call site is IDENTIFIED from source or from a reproduction, not inferred from the ticker name -- and if it cannot be identified, the step says so explicitly and states what evidence would be needed rather than guessing.
> 2. The population is RE-DERIVED with scripts/qa/derive_lite_fallback_census_86_38.py (or a successor) rather than copied from this step text, and the instrument's coverage assertion is shown passing -- an earlier version of that parser silently dropped 416 events and reported a zero that was a property of the parser.
> 3. The step states whether the None is an ABSENT UPSTREAM FIELD or a LOGIC ERROR, with the evidence that distinguishes them. 'Handle the None' without answering this is a rejected outcome.
> 4. Any fix is mutation-tested: revert it at the call site and show the guard goes red. A guard whose mutant survives does not count, and a guard whose control was already red scores nothing.
> 5. The fallback mechanism shipped by 86.38 is NOT modified: prove with a diff that `_fallback_rate_check`, `_degradation_summary_fields` and the record-always call site are byte-identical.
> 6. It is NOT claimed that this defect caused the trade drought. If the step finds evidence bearing on the drought, that evidence is filed as its own step rather than folded in here.

## 3. Criterion-by-criterion

### Criterion 1 -- call site IDENTIFIED (from a traceback, not inferred)

**`get_cik`, in `/workspace/main.py`, inside the remote Quant Agent Cloud
Function.** One frame after a SEC 429 exhausts its CIK-map retry ladder, the
fetcher returns `None` instead of raising.

> **THE LINE NUMBER IS DEPLOYMENT-VERSION-DEPENDENT -- do not cite it as fixed.**
> An earlier revision said `/workspace/main.py:89`. Measured across the retained
> logs: `line 89, in get_cik` appears **14x** (JSON-era, 2026-07-24 onward) and
> `line 79, in get_cik` appears **20x** (pre-JSON era) -- so the address I quoted
> is not even the more common one. **The stable identifier is the FUNCTION
> `get_cik`**, which is invariant across both deployments. Flagged by the cycle-2
> Q/A; re-measured here. This is the same "re-derive the line number before citing
> it again" trap CLAUDE.md warns about twice, committed anyway.

**"Fix the NoneType" is not a change available in this repository.** The step's
own instruction was to find the site rather than guess it; the finding is that
the site is not ours.

### Criterion 2 -- population RE-DERIVED, coverage assertion shown passing

Command: `python scripts/qa/derive_lite_fallback_census_86_38.py` -- **exit 0**.

```
COVERAGE -- every fallback line must be accounted for, or no census is printed
  backend.log.20260612T104931Z.gz        raw=  416  parsed=  416  ok
  backend.log.20260706T225648Z.gz        raw=   12  parsed=   12  ok
  backend.log.20260724T064045Z.gz        raw=    5  parsed=    5  ok
  backend.log.20260729T171222Z.gz        raw=    1  parsed=    1  ok
  backend.log.20260804T182713Z.gz        raw=    3  parsed=    3  ok
  backend.log.20260810T064130Z.gz        raw=    2  parsed=    2  ok
  backend.log                            raw=    3  parsed=    3  ok
  total accounted: 442

TOTAL            67      9   11.8%      (10 days)
```

**A PASSING COVERAGE ASSERTION PROVES LINE COUNTING, NOT ATTRIBUTION.** This
distinction is the whole finding of the step: the instrument counted all 442
events correctly *while* classifying an entire family into the wrong bucket. A
green coverage line is **not** evidence the classification is right.

> **CORRECTION (cycle 1 Q/A finding C1, verified independently by Main).** An
> earlier revision of this section said *"The 416-event assertion protects
> against a parser that drops rows."* **That is MEASURED FALSE and is
> withdrawn.**
>
> `raw += 1` (`derive_lite_fallback_census_86_38.py:274`) and
> `per_file_parsed[...] += 1` (`:289`) sit in the **same `if FALLBACK_MARK`
> branch**, with the intervening `if day: / else:` at `:284-288` rejoining
> before `:289`. **No path increments one without the other**, so `parsed ==
> raw` is structurally guaranteed, not verified.
>
> The Q/A demonstrated this by reproducing the historical defect shape — a
> read-level JSON-only filter, which is what actually dropped the 416 — and the
> assertion **stayed green while 433 of 442 events vanished**, exit 0, census
> still printed.
>
> So the assertion protects against a drop occurring *between the two counters*
> (structurally impossible today) and is **blind to the read-level filter that
> caused the very incident it was written for**. The criterion-2 requirements are
> still met and the population is independently corroborated by two greps, so
> **the numbers stand** — what was wrong was my claim about the guard. This is
> the "assert the property, not a proxy" defect, in my own artifact, in a
> paragraph warning about exactly that.

The classifier has since been corrected (by the session that filed the step); the
run above shows the corrected labels -- `remote QuantAgent crash after SEC.gov
429 on the CIK map (upstream)`.

**DENOMINATOR -- THE WHOLE DISPUTE WAS A CATEGORY ERROR, AND IT IS NOW RESOLVED.**

> **CORRECTION (cycle 1 Q/A finding D1, re-measured independently by Main).** An
> earlier revision presented "34 raw events" against a deduped 18 and called both
> defensible. **34 is a count of LINES, not events**, and the residual "17 with
> no 429 in window" was `34 - 17` — an arithmetic leftover describing **no real
> population at all**.
>
> **Every occurrence emits exactly TWO log lines, 17 lines apart.** Measured
> across all 42 retained logs: per-file line counts `12,4,4,2,6,6` — **every one
> even**, grouped by file+ticker into **17 groups of exactly 2**, every
> consecutive same-ticker gap exactly **17**.
>
> **Correction to my own description of the pair** (cycle-2 Q/A precision point,
> re-measured here): I wrote that the pair is always "one `orchestrator` INFO,
> one `autonomous_loop` WARNING". That holds for **16 of 17**. The exception is
> AAPL on 2026-08-06, whose second line is `module: analysis` and reads
> `Analysis failed for AAPL: [RuntimeError] Step 'quant'` — a **hard analysis
> failure, not a lite fallback**. The pairing itself is proved by the group
> structure and the uniform gap, so the count of 17 is unaffected; the module
> names were asserted rather than measured.

Re-measured at the EVENT level, collapsing the pairs:

| quantity | value |
|---|---|
| matching LINES, all 42 retained logs | 34 |
| **distinct EVENTS** (pairs collapsed) | **17** |
| events preceded by a SEC 429 within a 25-line window | **17** |
| events with **no** 429 cue | **0** |
| cue split | `Failed to fetch CIK map: 429` = **10**, `SEC 429 rate-limit ... retrying` = **7** (10+7=17) |
| distinct tickers | **13** (AAPL COHR CRWD DDOG DELL DVA INTC MU NTAP PANW SNDK STX WDC) |

**Attribution is 17 of 17 — 100%.** Neither "94%" nor "50%" was correct: 94%
(17/18) was close but carried one phantom event; 50% (17/34) double-counted every
event as two. **The correct statement is that every distinct occurrence carries an
upstream SEC 429 cue.**

This **strengthens** criterion 3 rather than weakening it. It is recorded as a
correction rather than quietly fixed because the step's own title says RE-DERIVE
EVERY NUMBER, and I carried the 10/7/17 split from a peer session without
re-deriving it — the precise failure the step was written to prevent, committed by
the session re-deriving it.

### Criterion 3 -- ABSENT UPSTREAM FIELD, not a logic error

**Verdict: absent upstream field.**

Distinguishing evidence: the failure **correlates with an upstream rate limit**.
The CIK map is absent because SEC.gov refused it with a 429; the remote function
converts that refusal into a `None` return rather than an exception, and the next
frame dereferences it. A logic error would not correlate with an upstream 429,
and would not disappear when the upstream is healthy (7 of the 10 days in the
census carry zero such events).

Corroborating, from the code under change -- `orchestrator.py` already documents
this exact upstream at class level: *"Cloud Function ... fetches
https://www.sec.gov/files/company_tickers.json ... observed returning 429 under 8
concurrent calls (cycle d73f5129)"*.

Provider split, **re-derived by me at the event level** (not inherited): of **17
distinct events**, **0 are Vertex**; **10** carry `Failed to fetch CIK map: 429
... www.sec.gov`, **7** carry `Quant: SEC 429 rate-limit on CIK map, retrying`,
and **0 have no 429 cue**. 10 + 7 = 17, so the split is exhaustive with no
residual.

The earlier revision of this line read *"of 34 raw events ... 17 have no 429 in
window"*, inherited from the filing session. That residual was `34 - 17` over a
LINE count and described no real population — **every event is attributed**. See
the denominator correction under criterion 2.

### Criterion 4 -- mutation-tested

`python scripts/qa/mutation_matrix_86_41.py` -- **5 cells, all KILLED, control
green first, target restored byte-identical.**

```
[control] unmutated suite green: True

  KILLED   M1-no-guard              remove the try/except entirely (the pre-86.41 state)
  KILLED   M2-plain-append          setdefault -> plain [] append (the KeyError defect, real)
  KILLED   M3-step-inside-try       move step() back inside the try (the over-broad-catch defect, real)
  KILLED   M4-impersonate-non-us    reuse phase-60.1's non-US reason for a rate-limit failure
  KILLED   M5-default-reason-drift  change the phase-60.1 default reason string

[restore] target byte-identical to pre-mutation: True
post-restore suite green: True
RESULT: all 5 cells KILLED, control green, target restored.
```

**M3 SURVIVED on the first run.** That was a real hole in my suite, not a
formality -- see §5.3. The runner also rejects a cell whose anchor does not match
(`ANCHOR-MISS`) or matches more than once (`ANCHOR-AMBIGUOUS`), because a
no-match replacement mutates nothing and leaves the suite green, which is
indistinguishable from a surviving mutant.

### Criterion 5 -- 86.38's mechanism NOT modified (proved three ways)

```
diff b8d2ea96..HEAD -- backend/services/autonomous_loop.py: (EMPTY)
  across 17 commits, including the peer session's

WHOLE-FILE sha256[:16] of backend/services/autonomous_loop.py:
  b8d2ea96 (base) : b1c38453bee0be23
  HEAD            : b1c38453bee0be23   IDENTICAL, 0-byte diff
  -> subsumes all three regions: if the whole file is byte-identical, every
     region in it is, with no extraction rule to get wrong.

backend/tests/test_phase_86_38_degradation_visibility.py: 9 passed
```

> **WHY THE THREE REGION HASHES WERE REPLACED** (cycle-2 Q/A note 3). An earlier
> revision quoted `fd034fae2f837117` / `7e6de86233adedf9` / `c8b0daf5d7531713`
> for the three protected regions. Those values are **not independently
> reproducible as quoted**, because I never stated the region-extraction rule --
> the Q/A's own extractor returned different values for the same function names.
> That is the "state the normalization rule with the ratio" defect applied to a
> hash. The whole-file identity above needs **no** rule, and is strictly
> stronger: byte-identity of the file implies byte-identity of every region in
> it. The original claim was true; it was just unverifiable by anyone else.

The empty whole-file diff is the stronger claim: across every commit in the
window, from either session, nobody touched that file at all.

### Criterion 6 -- NO drought claim

**No claim is made that this defect caused the trade drought, and none is
implied.** The contrary evidence is on record: 86.38 measured the 2026-08-03..09
window at 48 full-pipeline analyses with zero trades, and I separately confirmed
16 cycles / 0 trades in BigQuery since 2026-07-31. I withdrew an earlier 429-based
drought claim of my own this morning on this same pipeline; I am not replacing it
with a NoneType-based one. The drought's cause is **open**.

**FORWARD POINTER (cycle-2 Q/A precision point 2).** "Belongs to its own step" was
a promise with no address a reader could follow. The step is **86.47**, to be
filed at the close of this one (masterplan is not edited mid-EVALUATE: the
auto-commit hook's `git add -A` would sweep this step's in-flight files under
another step's name — the cross-attribution hazard that has fired three times in
two days).

The evidence 86.47 must start from, measured today and **deliberately not folded
in here**: last trade **2026-07-31 (NTAP)**; 8 trades across 21 weekdays before
it; over the 7 weekdays since, expected **2.67** trades at the measured rate of
0.381/weekday, giving **P(zero) = 6.95%** under Poisson. That does **not** clear a
5% bar, so the zero-trade run is **not decisive evidence** of a broken funnel — it
removes the drought as *proof* of breakage without showing the funnel is healthy.
The rule matters as much as the number: rate measured over **weekdays only**
(the book runs mon-fri), first-to-last observed trade, projected forward.

## 4. What changed

| File | Change |
|---|---|
| `backend/agents/orchestrator.py` | Guard the one unguarded sub-agent call; `_quant_from_yfinance` gains an optional `reason` param (default byte-identical) |
| `backend/tests/test_phase_86_41_quant_isolation.py` | NEW -- 7 tests that drive the real `run_full_analysis` |
| `scripts/qa/mutation_matrix_86_41.py` | NEW -- 5-cell mutation matrix |

Commits: `73dcf2c8` (guard), `678d979f` (tests + matrix + both fixes).

The fallback reuses the yfinance-only quant path **phase-60.1 already built and
proved for the same failing stage** -- its docstring: *"the quant Cloud Function
hard-aborts at its SEC-CIK stage"*. It is given a **deliberately distinct reason
string**: reusing 60.1's "non-US listing" text would relabel a transient rate
limit as a permanent listing-coverage fact, which is precisely the wrapper-string
collapse that made the census read 6 code defects where there were 0.

## 5. THE TWO DEFECTS IN MY OWN GUARD

Both were present in `73dcf2c8`, which passed `ast.parse` and 22 pre-existing
tests. Neither is detectable by reading the diff.

### 5.1 `KeyError` -- the guard was worse than useless

`report["skipped_stages"]` is created at `:1761` **only under `if not
_sec_covered`**. My guard runs in the `_sec_covered` branch, so the plain
`.append` raised `KeyError` -- aborting the ticker **in exactly the case the
guard exists to protect**. Fixed with `setdefault`. Pinned by M2.

### 5.2 Over-broad catch -- silently overwriting a GOOD report

`step()` was inside the `try`. `step()` invokes a **caller-supplied progress
callback**; a raising SSE emitter was therefore caught, mislabelled a quant
failure, and the fallback silently replaced a perfectly good quant report. The
`try` now contains the sub-agent call alone, with the healthy report in an `else`.
Pinned by M3.

### 5.3 And my mutation matrix found a hole in my own suite

M3 survived the first run. The healthy-path test read only the **first**
`quant/completed` event, which says `"Financial data collected"` whether the
guard stayed out of the way **or** fired and overwrote the report -- the control
answer and the mutant's answer coincided, so the cell could not discriminate.

Closed by asserting on a **yfinance call counter** (empty on the healthy path),
plus a **positive control** asserting the counter records exactly one call on the
degraded path -- without which an always-empty counter would make the healthy
assertion pass vacuously forever.

The suite also asserts its own precondition (`_is_sec_covered("AAPL") is True`);
without it, every test would exercise the non-US `else` branch and prove nothing
about the guard.

## 6. Verbatim command output

```
$ bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());print(\"parsed\")"'
parsed
exit=0
```

**What the immutable command does NOT prove:** it parses
`autonomous_loop.py` -- a file this step deliberately did **not** modify. Green
here is necessary and reaches **criterion 1 only**; it cannot see the guard, the
tests, or the mutation matrix. Recording that limit rather than presenting the
green as broad evidence.

```
$ python -m pytest backend/tests/test_phase_86_41_quant_isolation.py \
                   backend/tests/test_phase_60_1_deep_pipeline.py -q
29 passed, 1 warning in 6.31s
```

## 7. NOT IN FORCE -- the honest status

**The running backend imported the pre-change `orchestrator` module.** These
commits are on disk and on origin but are **NOT executing**. Restarts batch to
session end per the standing operator instruction, and the book runs at 20:00
CEST. Nothing in this step reaches tonight's cycle unless the operator restarts.

Any future claim that a cycle "was protected by this guard" must first check
`ps -eo pid,lstart` for the backend against the commit time of `73dcf2c8`.

## 8. Residual risk, disclosed

- **If yfinance ALSO fails**, the fallback itself raises and behaviour is
  unchanged from today (whole-ticker lite fallback). The guard strictly improves
  the common case; it does not make the path total. Not fixed here because it
  would need its own failure-injection test.
- **The remote Cloud Function is unfixed** and will keep returning `None` after a
  SEC 429. This step isolates the blast radius; it does not remove the cause.
- **The dedup rule behind "18" is not reproduced by me.** Any future quote of a
  ratio from this family must state its rule.
