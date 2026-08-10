# Experiment results -- step 86.25

**Step**: `86.25` (phase-86, P2, `harness_required: true`)
**Phase**: GENERATE
**Date**: 2026-08-10
**Driver**: Main (session `pyfinagent-06`), Opus 5 / effort max

---

## 0. What was built

The outcome-row vocabulary is now resolved **at the boundary**. A new
`recommendation_vocab.resolve_outcome_recommendation()` takes **only** an
analyst-recommendation candidate and returns either the canonical
recommendation (including a real `HOLD`) or the out-of-domain sentinel
`UNKNOWN_RECOMMENDATION`. Both call sites that used to hand a
non-recommendation value to a parameter typed for a recommendation now go
through it. The `"HOLD"` coercion is deleted.

## 1. Files changed (EXPLICIT LIST, derived from `git status --porcelain`)

| File | Change |
|---|---|
| `backend/services/recommendation_vocab.py` | + `UNKNOWN_RECOMMENDATION`, + `resolve_outcome_recommendation()` |
| `backend/services/autonomous_loop.py` | S1 call site + import; the `"HOLD"` coercion deleted |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | S2 call site + import; a stale comment that blessed the defect corrected |
| `backend/tests/test_phase_86_25_outcome_tracker_vocabulary_boundary.py` | **new**, 16 tests |
| `backend/tests/test_phase_35_1_learn_loop_writer.py` | one test **rewritten** (below) |
| `scripts/qa/measure_86_25_join_hitrate.py` | **new** (committed earlier, `64d20023`) |
| `.claude/masterplan.json` | queues **86.35**; no status flipped |

## 2. The design changed because P1 measured first

The contract's §2 chose "(A) where the lookup resolves, (C) where it does not".
**P1 measured the hit-rate before any code was written, and (A) is reachable for
ZERO of 32 rows** -- full detail in `contract_86.25.md` §6. Summary:

| path to an analyst recommendation | result |
|---|---|
| `analysis_results` has an `analysis_id` column | **NO** (91 cols; identity is `(ticker, analysis_date)`) |
| SELL row carries its own `analysis_id` | **0/32** (BUY rows: 33/33) |
| SELL -> BUY leg via `round_trip_id` | **0/32** (`round_trip_id`: 32/32 SELL, **0/33 BUY** -- one-sided) |
| anchor resolving to a recommendation | **0/32** |

So the shipped design is **(C) for 100% of the live population**, with the (A)
shape retained and tested but **labelled as covering zero real rows** rather
than presented as coverage.

## 3. Criterion-by-criterion

| # | Criterion (abridged) | Evidence | Status |
|---|---|---|---|
| 1 | REPRODUCE FIRST: a row scored `directionally_correct=False` for a trade whose return proves the call was right | `TestReproduceTheScoringDefect`, driving the REAL `evaluate_recommendation` with only its price source stubbed | MET |
| 2 | distribution RE-DERIVED from the table | `measure_86_25_join_hitrate.py` Q1/Q2 -- reproduces the step text exactly | MET |
| 3 | producer of the three 'SELL' rows DETERMINED | `nightly_outcome_rebuild` (S2), confirmed by Main in source: `risk_judge_decision or action` with 32/32 empty approvals | MET |
| 4 | `APPROVE_*` NOT mapped onto buy/sell | `TestBoundaryNeverInventsADirection`, parametrised over the FULL measured value set | MET |
| 5 | 'direction unknown' distinguishable from 'scored incorrect' in what is persisted | asserted at the **write chokepoint**, not the returned dict | MET, **premise corrected** |
| 6 | mutation-test every new guard, incl. reverting the fix at the call site | 3 cells, **all KILLED** | MET |

### Criterion 1 -- and the two seams fail in OPPOSITE directions

This is the finding the step text did not have:

- **S1's `"HOLD"`** is *not* directional, so a correct sell scores
  `directionally_correct=False` -- a **false negative**.
- **S2's `"SELL"`** *is* directional, so a losing long scores
  `directionally_correct=True` -- a **false positive that credits the system
  for a call nobody made**. This is the worse of the two, and it is the one
  that ran **ungated** in production (cron 04:00 UTC, not behind
  `paper_learn_loop_enabled`).

Both are reproduced through the real scorer.

### Criterion 5 -- the premise is wrong and the answer says so

`directionally_correct` is **never persisted**: `save_outcome` writes nine
columns and that is not one of them, and it has **no production consumer**. So
"whatever is persisted" is the `recommendation` column. The test asserts against
the write chokepoint that the persisted value is `UNKNOWN`, is not `"HOLD"`, and
that `directionally_correct` is absent from the persisted keys -- so if that
ever changes, the criterion's answer must be revisited.

### Criterion 6 -- mutation results

| cell | reverts | verdict |
|---|---|---|
| S1 | the `autonomous_loop` call site to `risk_judge_decision or "HOLD"` | **KILLED** (`test_phase_86_25_empty_risk_judge_decision_becomes_UNKNOWN_not_HOLD`) |
| S2 | the `nightly_outcome_rebuild` call site to `risk_judge_decision or action` | **KILLED** (2 named assertions) |
| V1 | the resolver's sentinel to the in-domain `"HOLD"` | **KILLED** (2 named assertions) |

Control green first; the tree was restored and `git diff --stat` shows only the
intended edits. **Each call site is a separate cell** -- one cell covering both
would hide a one-seam fix.

## 4. A test that pinned the defect, rewritten not deleted

`test_phase_35_1_empty_risk_judge_decision_coerced_to_hold` asserted
`rec_arg == "HOLD"` and called the coercion something the dispatcher "MUST" do.
Avoiding a crash on the empty string was a real need; spelling the marker
`"HOLD"` was the wrong way to meet it, and the test froze it. It is **rewritten**
(`test_phase_86_25_empty_risk_judge_decision_becomes_UNKNOWN_not_HOLD`), still
drives the REAL `_learn_from_closed_trades`, and still asserts the empty string
never reaches the tracker. It is the behavioural coverage of the S1 call site --
mutation cell S1 kills through it.

## 5. Verbatim output

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or autonomous_loop or learn_loop"'
92 passed, 3319 deselected, 1 warning in 6.26s
exit=0

$ python -m pytest backend/tests/test_phase_86_25_outcome_tracker_vocabulary_boundary.py -q
16 passed in 1.44s

$ uvx ruff check --select F821,F401,F811 <4 git-derived files>
All checks passed!   ruff=0
```

## 6. DISCOVERED DEFECTS -- queued, not fixed here

**86.35 (P2) -- the scorer raises `TypeError` on every real row.**
`evaluate_recommendation` parses the anchor with `fromisoformat` and subtracts it
from a **naive** `now()`. The comment above it claims "rec_date from
fromisoformat is naive" -- **measured FALSE**: `analysis_id` is empty on 32/32
SELLs so the anchor is always `created_at`, and **32/32 SELL rows carry a
tz-AWARE `created_at`**, so the subtraction raises for every candidate row. It is
swallowed by a broad per-ticker `try`. **The learn loop does not merely mislabel
outcomes on the S1 path -- it never scores them at all.** The two defects stack:
86.25 fixes the label, 86.35 fixes whether the scorer is reached. This is why
`TestReproduceTheScoringDefect` uses a deliberately NAIVE anchor, stated in the
test itself.

**`round_trip_id` is one-sided** (32/32 SELL, 0/33 BUY), so a round trip cannot
be reconstructed from `paper_trades`. Broader than this step; to be queued with
counts re-derived by the executor.

## 7. Scope bounds and what I cannot verify

- **S2 is LIVE** (ungated cron 04:00 UTC). Row COUNT is unchanged -- `UNKNOWN` is
  truthy so the skip does not fire and no close is dropped -- only the label
  changes, from a fabricated direction to an honest absence. The next 04:00 UTC
  run will write `UNKNOWN` where it used to write `SELL`.
- **S1 is DARK** (`paper_learn_loop_enabled` defaults False) *and* additionally
  unreachable today because of 86.35.
- **I did not back-fill or delete the three existing mislabelled rows.**
- **The (A) branch is tested but runs for no live row.** Stated, not hidden.
- **No flag was flipped and no `.env` was touched.**
- **The running backend has not been restarted**, so these changes are committed
  but NOT in force in the live process; restarts are batched to session end.

---

# CYCLE 2 -- three accuracy defects in my own claims, all confirmed and fixed

**Cycle-1 verdict: CONDITIONAL** (`wf_dd580823-63b`). It found **all 6 criteria
MET** and the shipped behaviour "correct and fail-safe on both seams", and
capped the verdict on three defects in the ARTIFACTS rather than the code. It
re-derived every one of my P1 numbers independently and they all stood; it also
ran 8 mutation cells of its own, 3 of them mine re-killed.

**I reproduced all three findings before fixing them.**

## W1 -- I gave the right conclusion with the wrong mechanism

I wrote, in two production comments and twice in this file, that the (A) branch
covers zero rows **because the anchor is unreachable** (analysis_id 0/32,
round_trip_id one-sided). Those measurements are real. **They are not the
operative cause.**

Measured by me after the finding: `analyst_recommendation` **is not a column of
`paper_trades` at all** — the table has 18 columns and that is not one of them —
and `_production_fns.LEDGER_FETCH_SQL` selects ten named columns, none of them
this one. So the code reads a dict key **no producer emits**: the branch is dead
**by construction**, not by a missing join.

Why this mattered enough to be a finding: my wording told the executor of the
queued `round_trip_id` step that this path would self-heal once the anchor
landed. **It would not.** Making it resolve needs a *producer* change. The Q/A
also showed the consequence directly — its cells M5 (hardcode the literal
`UNKNOWN`) and M8 (read `risk_judge_decision` through the resolver) both
**SURVIVED**, and both survive precisely because that key can never be present.

Corrected in `autonomous_loop.py`, `nightly_outcome_rebuild.py` and here. The
call is kept because it is the right boundary *shape*; the comments now say
plainly that it does nothing today and why.

## W2 -- the verification command did not cover the new guards

`-k "outcome_tracker or autonomous_loop or learn_loop"` collected **0 of the 16
new tests** — the file was named `test_phase_86_25_outcome_vocabulary_boundary.py`
(no `tracker`) and matched none of the three terms.

> A self-inflicted note, because it is the same class twice in one cycle: the
> bulk rename above was first applied with a blanket `sed` over the handoff
> files, which rewrote **this very sentence's historical filename** into the new
> one and made the sentence describe a state that never existed. Caught by
> re-reading the line after the edit. A global replace does not know which
> occurrences are history. So "92 passed, exit=0" was true but did not
exercise the S2 call-site or resolver guards at all; only the rewritten test in
`test_phase_35_1_learn_loop_writer.py` fell inside it.

**Fixed by bringing the tests into scope rather than by adding a sentence**: the
file is renamed `test_phase_86_25_outcome_tracker_vocabulary_boundary.py`, so the
existing immutable filter matches it. No criterion was amended.

```
before: 92 passed, 3319 deselected   (0 of the new suite collected)
after : 108 passed, 3303 deselected  (16 of the new suite collected)
```

And the mutants now die **inside** the immutable command:

| cell | before rename | after rename |
|---|---|---|
| S2 (revert the call site) | killed only by the excluded file | **KILLED — 3 failed / 105 passed** |
| V1 (sentinel -> `"HOLD"`) | killed only by the excluded file | **KILLED — 10 failed / 98 passed** |

## W3 -- I inflated the research-gate numbers ~2x

Contract §1 claimed **14** sources read in full and **44** URLs collected. The
brief's own envelope says **7**, **22** snippet-only, **29** collected, and its
tally line at `:141` says so in words. Both my figures were roughly double, in
the direction that makes the gate look stronger — **in the one table sitting
directly above my sentence "Every load-bearing internal claim below was
nonetheless re-verified by Main against source."** That sentence was not true of
that table.

I did not take those numbers from anywhere; I wrote them without checking. The
gate still PASSES on the real numbers (7 >= 5, 29 >= 10, recency scan
performed), so it is a transcription defect and not a gate failure — but it is
the same class as the "SIX not two" error earlier today: a number stated with
confidence and no derivation behind it. Corrected in place with the correction
disclosed rather than silently overwritten.

## Cycle-2 evidence

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or autonomous_loop or learn_loop"'
108 passed, 3303 deselected, 1 warning in 5.56s
exit=0
```

## What is still true and unchanged

The shipped behaviour did not change in cycle 2. No code path was altered — the
edits are two comment blocks, one file rename, and one contract table. Both
seams remain fixed, `APPROVE_*` still never becomes a direction, row count is
still unchanged, and the three original mutation cells still kill.

## Still open, unchanged

86.35 (the scorer's `TypeError` on every real row) and the one-sided
`round_trip_id` finding remain queued and unfixed here. The (A) branch remains
dead — now for the correctly-stated reason.
