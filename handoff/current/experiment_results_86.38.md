# Experiment results -- step 86.38

**Step**: `86.38` (phase-86, P1) | **Phase**: GENERATE
**Driver**: Main (`pyfinagent-51`), Opus 5 / effort max, 2026-08-11
**Contract**: `handoff/current/contract_86.38.md` (`cef76c3b`), written after the
research gate PASSED and before any production code. Ordering is provable from
git: gate + census instrument `c116e63a` -> contract `cef76c3b` -> code
`fd419038`.

---

## 0. The headline: the step's own premise was wrong, twice

This step was filed on the theory that the book stopped trading because the deep
pipeline was quota-killed. **Measurement refutes the framing**, and that refutation
is the most valuable output here -- a wrong diagnosis, confidently held, would
have sent the next investigator at the wrong subsystem.

| the step assumed | measured |
|---|---|
| the 429 might be "a per-minute limit, a **daily quota**, or a billing state" | **Vertex generative AI has NO per-day quota.** Every enforced dimension is per-minute. Per-day belongs to AI Studio, a different product. Under Dynamic Shared Quota there is no project-level number at all |
| "the error body was **truncated** in the log line and was not read" | The body is **complete** in the log. It simply carries no discriminator -- which Google documents as by design |
| "the deep path has been **mostly dead**" (flagged as unconfirmed) | **67 of 76 analyses (88.2%) completed the full pipeline** over 10 days. The 429 is a **one-day** event |
| the zero-trade outcome is explained by the degraded path | **2026-08-03..09 ran 54 full-pipeline analyses with ZERO fallbacks and still produced no trades.** The drought does NOT correlate with degradation |

The step was right about one thing and it is the thing worth fixing: **the
degradation was invisible to the operator.**

---

## 1. Files changed

| file | change |
|---|---|
| `backend/services/autonomous_loop.py` | new seam `_degradation_summary_fields`; record-always at the call site; `_intended_path` REMOVED; `_degradation` passed to the cycle record |
| `backend/services/cycle_health.py` | `record_cycle_end(degradation=...)` + a `degradation` key on the persisted row |
| `backend/tests/test_phase_86_38_degradation_visibility.py` | **new** -- 7 tests |
| `scripts/qa/mutation_matrix_86_38.py` | **new** -- 7 cells, all killed |
| `scripts/qa/derive_lite_fallback_census_86_38.py` | **new** (pre-contract) -- the 10-day census, with a coverage assertion |
| `handoff/current/{contract,live_check,experiment_results}_86.38.md` | artifacts |

Scope derived from `git status --porcelain` over explicit paths, never a
directory glob -- a peer session is live in this repository.

---

## 2. What changed, and the two decisions inside it

**RECORD ALWAYS, PAGE ONLY ABOVE THRESHOLD.** Every degradation field was
previously set inside the `if _fb_fire:` branch, so a cycle that degraded *below*
the paging threshold left no trace anywhere an operator looks. The fields are now
produced unconditionally by `_degradation_summary_fields` and persisted under a
new `degradation` key on the cycle record.

**DECISION 1 -- the alarm is NOT re-tuned.** `_fallback_rate_check` is untouched
and paging behaviour is byte-identical. Its strict `>` means the measured
incident (`3/6 = 0.500`) did not page, and changing that to `>=` would have
paged. **I did not make that change**: it is an alarm-sensitivity decision, it is
already pinned by `test_phase_60_1_deep_pipeline.py`, and the research found no
external best practice favouring either (Google's SRE Workbook uses both).
**Mutation cell M6 proves the pin is real** by loosening it and watching the
boundary test go red. It is raised as an operator ask instead.

**DECISION 2 -- `degradation` is a NEW key, not a widening of `funnel`.** The
phase-66.2 funnel answers "how many candidates survived each stage"; this answers
"was the pipeline that judged them the real one". Conflating them is how the
fallback rate ended up with no home in the first place. Cell M5 pins the
separation.

**DECISION 3 -- `_intended_path` was REMOVED, not wired.** It was write-only (1
write, 0 reads repo-wide; the search's recall validated by returning 13 hits for
the sibling `_fallback_reason`) and never reached the persisted `full_report`.
But the decisive reason is that it is **redundant**: the set it marked is exactly
the set carrying `_fallback_reason`, which already has a consumer and IS
persisted. Wiring it would have created two sources of truth for one fact.

---

## 3. Criterion by criterion

The step's criteria are its three `live_check` items.

| # | required | evidence | status |
|---|---|---|---|
| 1 | the verbatim 429 body, **or an explicit statement it could not be captured** | live_check A -- captured verbatim. And the premise corrected: the body was never truncated, and no per-day quota exists for this product | MET |
| 2 | the per-cycle full-vs-lite table over >=10 cycles **with the command that produced it** | live_check B -- 10 days, 67 vs 9, command shown, instrument asserts its own coverage | MET |
| 3 | the measured date of the last `paper_trades` row | live_check C -- **2026-07-31T18:47:37Z**, 65 rows, query shown | MET |

Non-scope honoured: no risk threshold, no position sizing, no gate, no paid tier.

---

## 4. The mutation matrix caught MY test, and that is the story

The first matrix run scored **5 of 6**. The survivor was cell M1, which disabled
the recording entirely (`if _n_fb_total:` -> `if False:`) -- and my guard
**SURVIVED**, because it asserted only that `summary["fallback_rate"]` appeared
*before* `if _fb_fire:` **in the source text**. Flipping a condition moves no
text.

**The fix was not a better string match.** The logic was extracted into
`_degradation_summary_fields`, a seam a test can EXECUTE, and the matrix now
attacks it from both ends: **M1** mutates the seam's behaviour, **M1b** unwires
it from the call site. Killing only one would have left the other untested.

Final: **7 cells, 7 killed**, both target files' sha256 verified unchanged after
every cell and at exit. The matrix refuses to run at all if the target files are
dirty, because then "restored" could not be distinguished from "clobbered someone
else's edit".

Also pinned, because absence must not be ambiguous: a **healthy** cycle records
`0/6`, not nothing. Absence of a rate now means "analysed nothing", never
"nothing went wrong".

---

## 5. What I did NOT do

- **Did not touch the RiskJudge, any threshold, position sizing, or any gate.**
  The step text is emphatic and correct: the gate behaved correctly on the
  evidence it was given.
- **Did not change `>` to `>=`.** Operator ask.
- **Did not read the GCP quota metric**, so the 429 remains unclassified between
  rate / DSQ-capacity / billing. Google's guidance is that it cannot be
  classified from the error string, so this is a stated limit, not an omission.
- **Did not fix the `QuantAgent NoneType` defect** -- 6 of the 9 fallbacks over
  ten days, i.e. more of the degradation than the 429. Queued as its own
  research-gated step per the standing rule.
- **Did not restart the backend.** This is committed but **NOT IN FORCE**; the
  running process still holds the pre-change module. Next book cycle 20:00 CEST.

---

## 7. CYCLE 2 -- two findings from a dropped Q/A, both confirmed

The cycle-1 Q/A dropped at 162,182 tokens without returning, then a second run
dropped at 180,539. Neither is a verdict. But the second got far enough to find
two real defects, and **its write-first record survived both times** -- the
durability change shipped by the peer's 86.36 earlier the same day.

**F1 -- THE WIRING HAD NO GUARD, and it is the half that reaches storage.**
Deleting `degradation=_degradation,` from the `record_cycle_end(...)` call left
the entire suite GREEN (7 passed). I reproduced it. Under that mutant every
future cycle persists `degradation: {}` -- the exact defect this step exists to
remove, returning silently behind a green suite. I had guarded
`summary -> _degradation_summary_fields` and left `_degradation ->
record_cycle_end` uncovered. **Guarding one end of a two-ended wire is not
guarding the wire.**

Fixed by extracting `_degradation_record()` + `DEGRADATION_RECORD_KEYS` so the
persisted key SET is behaviourally testable, plus a call-site pin and cells
MX/MY. Matrix now **9 cells, 9 killed**.

**And the first version of that call-site pin was itself defeated -- by my own
prose.** It asserted `"degradation=_degradation," in inspect.getsource(al)`, and
MX still passed, because the docstring of `_degradation_record` QUOTES that
literal while explaining the defect. A grep cannot distinguish a call site from a
sentence about a call site. The guard is now an **AST walk** for the
`record_cycle_end` Call requiring a `degradation` keyword fed by `_degradation`;
docstrings are not Call nodes, so prose cannot satisfy it.

**F2 -- THE HONEST LIMIT LIVED ONLY IN THE HANDOFF ARTIFACT.** `live_check`
section D correctly states that the alarm's denominator
(`len(candidate_analyses)+len(holding_analyses)`) was NOT measured and that
"missed by exactly one ticker" is NOT claimed. But four other places asserted the
boundary as measured cause: the `autonomous_loop` call-site comment, the
`cycle_health` comment, the test module docstring, and a test docstring reading
"this is the measured case". **The version that survives in production source is
the one a future reader finds; a disclosure that lives only in a handoff artifact
is not a disclosure.** All corrected.

A fifth site survived my own first sweep, and the sweep was wrong in both
directions at once: it missed the seam docstring's `(3/6 = 0.500, no page)` and
simultaneously reported `autonomous_loop.py` as having zero qualifiers, because
my corrected text spans a comment line break that a flat grep cannot see. Fixed
with a wrap-normalising sweep that now reports every file clean.
