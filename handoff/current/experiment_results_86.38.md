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
| the zero-trade outcome is explained by the degraded path | **2026-08-03..09 ran 48 full-pipeline analyses across the FIVE fallback-free days in 2026-08-03..09 (the window's full total is 54 full / 2 lite -- 2026-08-05 had two QuantAgent-NoneType fallbacks) and still produced no trades.** The drought does NOT correlate with degradation |

The step was right about one thing and it is the thing worth fixing: **the
degradation was invisible to the operator.**

---

## 1. Files changed

| file | change |
|---|---|
| `backend/services/autonomous_loop.py` | new seam `_degradation_summary_fields`; record-always at the call site; `_intended_path` REMOVED; `_degradation` passed to the cycle record |
| `backend/services/cycle_health.py` | `record_cycle_end(degradation=...)` + a `degradation` key on the persisted row |
| `backend/tests/test_phase_86_38_degradation_visibility.py` | **new** -- 9 tests (7 at cycle 1, +2 at cycle 2 for the second seam) |
| `scripts/qa/mutation_matrix_86_38.py` | **new** -- 9 cells, all killed (7 at cycle 1, +MX/MY at cycle 2) |
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
and paging behaviour is byte-identical. Its strict `>` means the 2026-08-10 cycle
did not page; the TICKER ratio was 3/6, though the alarm's own denominator was
not measured (section 7 F2), so it is not claimed that `>=` would have paged. **I did not make that change**: it is an alarm-sensitivity decision, it is
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

## 3. Criterion by criterion (the SIX immutable criteria, not the live_check)

**This section previously mapped the three `live_check` items and called them
"the step's criteria". That was the same breach section 3 of the contract
carried**: the live_check is an ADDITIONAL obligation, not the criteria. The six
below are `verification.success_criteria` from `.claude/masterplan.json`.

| # | criterion (abridged) | evidence | status |
|---|---|---|---|
| 1 | the 429 is CHARACTERISED from primary evidence, not guessed: capture the full error body (the lo... | live_check A. Body captured verbatim, 420 chars, complete JSON. **I decline to state WHICH quota and say why**: the body carries no discriminator by Google's design, and Vertex genAI has no per-day quota at all, so the criterion's own trichotomy is mis-stated. Criterion 1 permits 'say so and state what was done instead' -- done, and closed with ASK #2 rather than a guess. | MET |
| 2 | the degradation RATE is derived over at least the last 10 completed cycles: for each, how many t... | live_check **B2**, derived PER-CYCLE from `cycle_history.jsonl` with the command shown. 10 attributable cycles (the floor, exactly). 88 of 98 terminal cycles are reported **UNATTRIBUTABLE**, never zero. NOT derived from llm_call_log. **Section B's per-DAY table is retained but is NOT the criterion's evidence** -- that substitution was the cycle-1 FAIL. | MET |
| 3 | the correlation with the trade drought is either demonstrated or explicitly refuted: the last pa... | live_check B2. **REFUTED.** 9 of 10 attributable cycles zero-trade; 3 degraded, **6 completely clean**. The one trading cycle (`1326ca36`) was 43% degraded. NOTE, carried from the Q/A: 'the correlation runs the wrong way' is an **n=1 over-read** -- the refutation stands independently on the six clean cycles and is stated that way. | MET |
| 4 | operator visibility is stated as a fact: does a 429 fallback surface anywhere the operator sees ... | live_check D + section 2. **LOG-AND-PAGE-ONLY, confirmed by grep over `backend/api`, `backend/slack_bot`, `frontend/src`: ZERO consumers.** That is the finding; the remedy makes it durable per cycle without changing when it pages. | MET |
| 5 | any remedy is fail-safe and does not touch the risk judge, position sizing, or any gate threshol... | section 2 DECISION 1. `_fallback_rate_check` sha256 identical; the change is an additive kwarg + key. No risk threshold, position sizing or gate touched. Cell M6 pins the strict `>` by loosening it and watching the boundary test go red. Nothing makes the lite path more likely to trade. | MET |
| 6 | if the conclusion is that the correct action is an operator decision (paid tier, quota increase,... | section 8 + `handoff/current/operator_asks_2026-08-11.md`. **ASK #2** filed with three options, costs, and a recommendation (read the free metric, then accept; explicitly NOT the paid tier). Criterion 6 blesses this as a valid close. | MET |

**The `live_check` obligation is tracked separately** and is also met: the
verbatim 429 body (A), the per-cycle table with its command (B2), and the
measured last `paper_trades` date (C).

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

Cycle-1 final was 7 cells / 7 killed; **cycle 2 is 9 cells, 9 killed** after the
second seam was guarded (section 7). Both target files' sha256 verified unchanged after
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

---

## 8. CRITERION 6 -- the conclusion IS an operator decision. Numbered ask below.

Criterion 6 says plainly that if the correct action is an operator decision,
"the step closes with that stated plainly and a numbered ask -- that is a valid
outcome and is preferable to a speculative code change." **That is where this
step lands on the 429**, and stating it is the deliverable rather than a
substitute for one.

### OPERATOR ASK #2 -- classify the 429, or accept lite-on-quota-exhaustion

**What I established.** The 429 body is captured verbatim and is COMPLETE (it was
never truncated -- the step text was wrong about that). It carries **no
discriminator**. Google documents this as by design: the same condition surfaces
as 429 REST, `ResourceExhausted` gRPC, 403 on GCE and 5XX under Provisioned
Throughput, and classification must be done **out of band** from
`serviceruntime.googleapis.com/quota/rate/net_usage`. Vertex generative AI has
**no per-day quota at all** -- per-day belongs to AI Studio, a different product
-- so the step's own trichotomy was mis-stated. The real one is **rate limit /
Dynamic-Shared-Quota capacity / model-or-billing state**.

**Why I cannot close it myself.** Reading that quota metric is a GCP
console/monitoring action outside this step's surface, and the remedies
(requesting a quota increase, enabling Provisioned Throughput, or a paid tier)
are all spend decisions. The standing constraint is **$0 metered**, so none of
them is mine to take.

**The three options, with what each costs and what it buys:**

| option | cost | buys |
|---|---|---|
| A. Read `quota/rate/net_usage` in the console for 2026-08-10 18:00-19:15Z | free, ~5 min of operator time | the actual classification -- rate vs DSQ capacity |
| B. Accept lite-on-quota-exhaustion as designed behaviour | free | nothing changes; the fallback already degrades gracefully and is now RECORDED per cycle |
| C. Provisioned Throughput / paid tier | **metered spend** | removes DSQ contention |

**My recommendation is A, then B.** The 429 was a **single-day event** in ten
cycles, the deep path ran at 88.2%, and -- decisively -- the per-cycle evidence
in live_check B2 shows degradation does NOT explain the trade drought (six
completely clean cycles produced zero trades; the one cycle that DID trade was
43% degraded). **Spending money to fix a fallback that is not causing the problem
would be the wrong trade**, and C should not be considered until the drought's
real cause is known.

**This ask does not block the step's code.** Everything shipped here is
observability: the degradation is now recorded on every cycle rather than only
when it pages, and nothing touches a gate, a threshold, or the risk judge.
