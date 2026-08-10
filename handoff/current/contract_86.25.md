# Contract -- step 86.25

**Step**: `86.25` (phase-86, P2, `harness_required: true`)
**Phase**: PLAN
**Date**: 2026-08-10
**Driver**: Main (session `pyfinagent-06`), Opus 5 / effort max

> Line numbers below were re-derived 2026-08-10 and WILL go stale. Grep the
> symbol. This project staled three citations inside one step on this same day.

---

## 1. Research gate summary

**PASSED** -- run `wf_a3511e6a-c28`, tier `moderate`, brief
`handoff/current/research_brief_86.25.md` (38,615 chars, `STATUS: COMPLETE`).
**Not re-run this session**: the gate is satisfied and re-spawning would burn
~200K tokens to re-derive a brief already on disk. Every load-bearing internal
claim below was nonetheless **re-verified by Main against source**, not taken on
report.

| Field | Value |
|---|---|
| sources read in full | 14 |
| URLs collected | 44 |
| recency scan | performed (2024-2026) |
| internal files inspected | full writer/caller enumeration |

### Findings that decide the design

**F1 -- THE HEADLINE MISMATCH IS LATENT, NOT ACTIVE.** Measured
(`risk_judge_decision`, n=65): `''` 46, `APPROVE_REDUCED` 15, `REJECT` 3,
`APPROVE_HEDGED` 1. Split by action: **every SELL row is empty, 32 of 32** -- and
`_learn_from_closed_trades` reads SELL rows *only*. So the fetch returns `''` on
100% of live candidates, the empty-coercion rewrites it to `"HOLD"`, and the
APPROVE/BUY vocabulary collision **never fires today**. It would fire the moment
one SELL row carried a populated `risk_judge_decision`. The step must fix the
latent collision *and* the degenerate that is actually running.

**F2 -- `directionally_correct` IS NEVER PERSISTED.** Re-derived by Main:
`outcome_tracker.py` computes it (`:66`) and returns it in the dict (`:77`), but
`bigquery_client.save_outcome` writes **nine** columns and that is not one of
them. Grep for consumers outside tests returns only the tracker itself and a
measurement script -- **no production consumer at all.** This reframes criterion
5: the thing that IS persisted and IS wrong is the `recommendation` column,
which today receives the literal `"HOLD"` on rows where no recommendation was
ever known.

**F3 -- THE COERCION IS PEP 661's NAMED ANTI-PATTERN.** `recommendation = "HOLD"`
(`autonomous_loop.py:3416-3417`) uses an **in-domain value as the missing-data
marker**. A downstream reader cannot distinguish "the analyst said hold" from
"we had nothing". That is the defect, independent of the vocabulary collision.

**F4 -- THE DEFECT HAS TWO SEAMS, AND THE STEP TEXT NAMES ONLY ONE.** Verified by
Main in source:
- **S1** `autonomous_loop.py:3412` -- `risk_judge_decision` -> `recommendation`.
  Behind `paper_learn_loop_enabled`, whose `settings.py` default is **False**.
- **S2** `nightly_outcome_rebuild.py` -- `recommendation = t.get("risk_judge_decision") or t.get("action")`.
  This passes an **ACTION** where a recommendation is expected, and it is
  **NOT gated by that flag**: it runs on cron `hour=4` UTC. The brief's
  provenance work (I5) determines this is the seam that wrote the three live
  `'SELL'`-spelled rows -- the `or`-fallback firing deterministically on the
  measured 32/32 empty `risk_judge_decision`.

Fixing S1 alone would leave the live producer intact. That is the
"guards stop one seam short" failure and this contract refuses it.

**F5 -- DO NOT WIDEN THE VOCABULARY.** Resolve at the boundary
(parse-don't-validate): hand `evaluate_recommendation` either a real analyst
recommendation or an explicit unknown -- never a risk-approval token wearing a
recommendation's parameter name. `recommendation_vocab.is_directional` already
exists for the discrimination and is already live at `backend/api/portfolio.py:145`.

**F6 -- THE JOIN HIT-RATE IS UNMEASURED, and the brief says so.** Option (A)
requires resolving `analysis_id -> analysis_results.recommendation`. The anchor
at `autonomous_loop.py:3409` is `analysis_id or created_at`, i.e. already
ambiguous. **A contract choosing (A) must measure the hit-rate first.** This one
does, as step P1, before any code is written.

---

## 2. Hypothesis

The learn loop scores every fallback-path outcome with no direction, and it does
so through **two** independent seams that each hand a non-recommendation value
to a parameter typed for a recommendation. Resolving the vocabulary **at the
boundary** -- looking up the real analyst recommendation where it resolves, and
recording an explicit out-of-domain unknown where it does not -- removes the
wrong label without inventing a mapping, and makes "we did not know" readable in
the persisted row instead of being spelled `"HOLD"`.

**Design: (A) where the lookup resolves, (C) where it does not, never (B)
silently.** A wrong label is worse for a learning signal than an absent one; a
silently-dropped event is worse than a recorded unknown.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. REPRODUCE FIRST: drive the real fallback path and show a row scored directionally_correct=False for a trade whose realised return proves the call was right.
2. The `risk_judge_decision` distribution is RE-DERIVED from the table, not copied from this step text.
3. The producer of the three existing 'SELL'-spelled outcome_tracking rows is DETERMINED, or the step states explicitly what was checked and why it remains undetermined.
4. APPROVE_* is NOT mapped onto a buy or sell intent.
5. 'Direction unknown' is distinguishable from 'scored incorrect' in whatever is persisted -- or the step justifies why it need not be.
6. Mutation-test every new guard, including reverting the fix at the call site; a guard whose mutant survives does not count.

**Verification command** (immutable):
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or autonomous_loop or learn_loop"'`

**live_check**: `live_check_86.25.md` -- verbatim reproduce-then-fix output
driving the real fallback path; the re-derived `risk_judge_decision`
distribution; the adjudication of the three SELL rows' provenance; and the
before/after label for a trade whose return proves the direction.

### A note on criterion 5's premise

Criterion 5 says "in whatever is persisted". **`directionally_correct` is not
persisted at all** (F2). The criterion is answered on the column that *is*:
`recommendation`. The answer must state that finding rather than quietly
answering a different question, and must cover the in-memory tri-state as the
secondary half.

---

## 4. Plan

**P1 -- MEASURE BEFORE BUILDING (F6).** A committed script that re-derives the
`risk_judge_decision` distribution (criterion 2) **and** measures the
`analysis_id -> analysis_results.recommendation` join hit-rate over the 32 SELL
rows. The hit-rate decides how much of the population option (A) can serve; the
remainder is option (C)'s population. Both numbers reported, neither assumed.

**P2 -- REPRODUCE (criterion 1).** Drive the REAL fallback path against a trade
whose realised return proves the call was right, and capture
`directionally_correct=False`. Reproduce before fixing; a fix whose defect was
never demonstrated is not a fix.

**P3 -- RESOLVE AT THE BOUNDARY, BOTH SEAMS (F4).** In `_learn_from_closed_trades`
(S1) and in `nightly_outcome_rebuild`'s row builder (S2): look up the real
analyst recommendation; if it resolves, pass it; if not, pass an explicit
out-of-domain unknown. **Delete the `"HOLD"` coercion** (F3) -- an in-domain
missing-data marker. `recommendation` is REQUIRED at the destination so SQL NULL
is unavailable; the marker must be a sentinel that `is_directional` already
rejects.

**P4 -- NO MAPPING (criterion 4).** `APPROVE_*` is never turned into buy or sell
intent, and `recommendation_vocab` is not widened. Guarded by a test that feeds
every measured `risk_judge_decision` spelling through the boundary and asserts
none produces a directional intent.

**P5 -- MUTATION MATRIX (criterion 6).** Hermetic, mini-repo pattern. Cells must
include: reverting the fix at EACH call site independently (S1 and S2 separately
-- a single cell covering both would hide a one-seam fix); restoring the `"HOLD"`
coercion; widening the vocabulary to accept `APPROVE_REDUCED`; and neutralising
the unknown-marker so it collapses back onto a directional value.

**P6 -- S2 IS LIVE; S1 IS DARK.** Any change to `nightly_outcome_rebuild` alters
an ungated cron job. The contract requires the before/after behaviour of S2 to
be demonstrated against a fixture, and an explicit statement of what the next
04:00 UTC run will do differently.

### Explicitly NOT doing

- **Not** mapping `APPROVE_*` onto buy/sell (criterion 4, F5).
- **Not** widening `recommendation_vocab` (F5).
- **Not** dropping the event silently (option B).
- **Not** adding a `directionally_correct` column to `outcome_tracking` -- that
  is a schema migration, out of scope; the step states the finding instead.
- **Not** flipping `paper_learn_loop_enabled`. Flag promotions are operator-gated.
- **Not** back-filling or deleting the three existing rows.

### Risk

S1 is behind a flag defaulting False; **S2 is not gated and runs nightly**. The
fix direction is fail-safe on both (an unknown marker is non-directional, so no
outcome can be newly scored as a correct call). No money path is touched: this
is an evaluation path, and `directionally_correct` has no production consumer.

---

## 5. References

- `handoff/current/research_brief_86.25.md` (gate PASSED, `wf_a3511e6a-c28`)
- PEP 661 (sentinels); parse-don't-validate; Google Rules of ML #29/#30/#34;
  `draft-thomson-postel-was-wrong-03` (leniency as the mechanism of decay);
  noisy-label canon (demote to unlabelled, `arXiv:2404.04159`)
- `backend/services/autonomous_loop.py::_learn_from_closed_trades`;
  `backend/services/outcome_tracker.py::evaluate_recommendation`;
  `backend/db/bigquery_client.py::save_outcome`;
  `backend/slack_bot/jobs/nightly_outcome_rebuild.py`;
  `backend/services/recommendation_vocab.py::is_directional`

---

## 6. P1 RESULT -- the design changes, and this is why the measurement came first

`scripts/qa/measure_86_25_join_hitrate.py` plus three follow-up probes, run
2026-08-10 against live BigQuery. **The contract's own design assumption in §2
did not survive contact with the data, and the amendment is recorded here rather
than quietly applied.**

### Criterion 2 -- distribution RE-DERIVED (reproduces the step text exactly)

| `risk_judge_decision` | n |
|---|---|
| `''` (empty) | 46 |
| `APPROVE_REDUCED` | 15 |
| `REJECT` | 3 |
| `APPROVE_HEDGED` | 1 |
| **total** | **65** |

Non-empty spellings: `APPROVE_HEDGED`, `APPROVE_REDUCED`, `REJECT`. **On the
BUY/HOLD/SELL scale: NONE.** Split by action:

| action | n | empty `risk_judge_decision` |
|---|---|---|
| BUY | 33 | 14/33 (42%) |
| **SELL** | **32** | **32/32 (100%)** |

F1 confirmed: the learn loop reads SELL rows only, and every one is empty.

### Option (A) is reachable for ZERO of the 32 rows -- measured, not assumed

The brief said a contract choosing (A) must measure the hit-rate first. It is
**0/32 by every available path**:

| path | result |
|---|---|
| `analysis_results` has an `analysis_id` column to join on | **NO** -- 91 columns, identity is `(ticker, analysis_date TIMESTAMP)`. My first query assumed this column and failed loudly, which is how the assumption surfaced. |
| SELL row carries its own `analysis_id` | **0/32.** BUY rows carry it **33/33**. The SELL writer drops it. |
| SELL -> BUY leg via `round_trip_id` | **0/32.** `round_trip_id` is populated on **32/32 SELLs and 0/33 BUYs** -- the link is ONE-SIDED and cannot be traversed. |
| the anchor then resolving to `analysis_results.recommendation` | **0/32.** |

Also established: `analysis_id` is an **ISO timestamp**, not an opaque id
(`'2026-04-26T21:12:16.666983+00:00'`), so the join would work *if the anchor
were present*; and `evaluate_recommendation` performs **no join at all** today --
it uses `analysis_date` solely for `holding_days` via `fromisoformat`.

### AMENDED DESIGN

**Option (C) for 100% of the current population.** Option (A) is retained only as
the shape the code takes *if* an anchor is ever present -- it must not be the
guarded path, because guarding it would mean tests that stub a lookup which
resolves for no real row. That is precisely the failure the brief named: *"an
(A)-shaped fix guarded only by tests that stub the lookup would be a fix for a
population that does not exist."*

Consequences for the plan:
- P3 stands, but the **unknown marker is the primary path**, not the fallback.
- The mutation matrix must include a cell proving the unknown path is what
  actually executes on the measured data -- an (A)-branch that never runs cannot
  be credited.
- The `"HOLD"` coercion deletion (F3) becomes the highest-value single change,
  because it is what runs on 32 of 32 rows today.

### DISCOVERED DEFECT -- queue, do not fix here

**`round_trip_id` is one-sided: 32/32 on SELLs, 0/33 on BUYs.** A round trip
therefore cannot be reconstructed from the table, which breaks any
entry-to-exit attribution, not just this step's lookup. Independent of 86.25's
vocabulary defect and out of its scope. To be queued as its own research-gated
step with the counts re-derived by the executor.
