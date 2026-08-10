# phase-86.22 -- GENERATE

**Step:** 86.22 -- the recommendation-vocabulary split is cross-module, and on
the learn-loop side it poisons the signal at scale rather than dropping one row.
**Contract:** `handoff/current/contract_86.22.md`
**Research:** `handoff/current/research_brief_86.22.md` (gate PASSED, `wf_e6a9d91d-dda`)

---

## 1. What the defect actually is

phase-86.20 fixed ONE consumer -- `portfolio_manager`'s trade gate. This step is
the class. The same `financial_reports.analysis_results.recommendation` string is
read by six other sites in **three mutually incompatible dialects**:

| dialect | expression | breaks on |
|---|---|---|
| TITLE-CASE exact | `rec in ("Strong Buy","Buy")`, no folding at all | every UPPERCASE row |
| UPPER_SNAKE | `rec.upper() in ("STRONG_BUY","BUY")` -- folds case, not separator | the spaced `Strong Buy` |
| SUBSTRING | `"STRONG_BUY" in rec_label`, first clause wins | grades by clause order |

They fail in **opposite directions**, which is why neither was noticed: the
dialect that drops `Strong Buy` is the one that handles `BUY`, and vice versa.

## 2. Measured, re-derived (NOT taken from the step text)

`scripts/qa/measure_vocabulary_impact_86_22.py` re-queries the column and
recomputes every figure. Distribution, n=543 total:

```
value                   n   genuine   canonical
HOLD                  275        49   -
Hold                  115        66   -
BUY                    91        91   BUY
Buy                    39        30   BUY
Sell                   16         8   SELL
Strong Buy              5         1   BUY
N/A                     2         0   -
```

Per-consumer divergence from the shared vocabulary:

| consumer | dialect | rows classified differently |
|---|---|---|
| `outcome_tracker:57-58` | title-case | **91 / 543 (16.8%)** |
| `memory:229-230` | title-case | **91 / 543 (16.8%)** |
| `bias_detector:119,128` | upper-snake | 5 / 543 (0.9%) |
| `api/portfolio:140-142` | upper-snake | 5 / 543 (0.9%) |
| `conflict_detector:121+` | substring | **0 / 543 (0.0%)** -- see the caveat below |

**The 0.0% needs stating carefully, or it reads as "no defect".** It measures
*intent* only. `"STRONG BUY"` fails conflict_detector's `"STRONG_BUY"` clause but
then matches the `elif "BUY"` clause, so the intent comes out right -- and the
**threshold** comes out wrong: the strictest check (7.0) silently becomes the
loosest (5.5) for exactly the highest-conviction calls. My measurement script
does not measure thresholds, so 0.0% is a statement about intent and nothing
else. `test_conflict_detector_grades_a_strong_buy_at_the_STRICTER_threshold`
covers the part the number does not.

## 3. Already-persisted rows: measured, not assumed

The step asks whether a wrong reflection has already been persisted. Measured:

- `agent_memories` -- **0 rows.** No reflection has been written yet.
- `outcome_tracking` -- **3 rows**, and it has **no `directionally_correct`
  column at all** (schema: ticker, analysis_date, recommendation,
  price_at_recommendation, current_price, return_pct, holding_days,
  beat_benchmark, evaluated_at). The label was never persisted, so there is
  nothing to backfill.

Those 3 rows are nonetheless a real instance of the defect:

```
ticker  rec      return_pct   before -> after
AMD     SELL      -11.3160    False  -> True
PANW    SELL      -10.9368    False  -> True
MU      SELL       -7.2643    False  -> True

rows whose directionally_correct label CHANGES: 3/3
```

Three correct sell calls, each scored **directionally wrong** by the pre-fix
expression, because `"SELL"` is not in `("Strong Sell","Sell")`.

## 4. What was built

**One** shared vocabulary, extending 86.20's canonicaliser rather than minting a
second one (two normalisers that disagree would be this defect again):

- `backend/services/recommendation_vocab.py` -- added `BUY_INTENT`,
  `SELL_INTENT`, `is_buy_intent()`, `is_sell_intent()`, `is_directional()`.
  `HOLD` is in neither intent set: a considered hold and an unparseable value
  are different facts, and collapsing them is how the drift stayed invisible.

Six consumers migrated to it:

| file | site | risk profile |
|---|---|---|
| `backend/services/outcome_tracker.py` | :57-58 | **learning** -- writes the label |
| `backend/agents/memory.py` | :228-231 | **learning** -- renders it into a persisted reflection |
| `backend/agents/bias_detector.py` | :119, :128, :154-155 | analysis |
| `backend/api/portfolio.py` | :138-142 | reporting |
| `backend/agents/conflict_detector.py` | :121/:131/:140 | reporting; thresholds 7.0/5.5/6.0 preserved exactly |
| `backend/slack_bot/formatters.py` | `_rec_color` | display; **no callers** (verified repo-wide) |
| `backend/agents/skill_optimizer.py` | :243 `debate_consensus` | learning -- scores debate agents (added cycle 2) |

`backend/agents/skill_optimizer.py:243` was **also migrated in cycle 2**. It
was originally excluded on the grounds that it reads a schema-enforced
`Literal`; the cycle-1 Q/A disproved that reason -- the value comes from
`debate_consensus`, selected from `financial_reports.analysis_results`, the same
persisted table as `recommendation`. A `Literal` in the producer says nothing
about the persisted string (`api/models.py` names a member `STRONG_BUY` whose
VALUE is `"Strong Buy"`). Measured distribution of `debate_consensus`: `''` 487,
NULL 51, `'HOLD'` 4, `'BUY'` 1 -- so the old expression was correct in EFFECT
and hid no live defect. The ARGUMENT was wrong, and an allow-list entry is worth
exactly what its argument is worth, so the site was migrated instead of
re-worded. That makes **seven** migrated consumers, not six.

**No set was widened.** `"Accumulate"`, `"Overweight"`, `"BUYING"`,
`"NOT A BUY"`, `"Strong Buy!"` and `"N/A"` all remain non-directional.

## 5. Two things I got wrong, and how they were caught

**(a) My consumer population was wrong, because my detector was blind.** The
first derivation used a regex, then an AST scan with two rules. It reported a
confident population of 10 sites across 4 files -- and silently **missed
`conflict_detector.py` entirely**, because a substring test is a different AST
shape, not a different spelling. Adding rule R3 raised the pre-fix population to
**17 offender sites** and surfaced a **sixth consumer I had not migrated**
(`slack_bot/formatters.py`). A guard that covers one shape of a defect is a
guard against an instance, not against the class.

**(b) A surviving mutant showed my negative set had a hole.** Cell D4 removes
R2's requirement that the literals be recommendation-shaped, and it **survived**
-- because every known-negative I had written failed R2's *first* condition, so
the second was never exercised. Two negatives were added
(`recommendation_status in ("pending","done")`) and D4 now dies.

Both were found by the harness, not by inspection. Neither was in the fix.

## 6. Verification -- verbatim

**Immutable command** (`.claude/masterplan.json`, unmodified):

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q \
    -k "outcome_tracker or bias_detector or conflict_detector or portfolio_manager"'
188 passed, 3097 deselected, 1 warning in 6.92s
immutable-command exit=0
```

**Detector validation -- recall AND precision both gate the exit code:**

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py --validate
recall 9/9   precision 10/10
Method validated in BOTH directions.
```

**Criterion 4 -- no second vocabulary survives (before vs after):**

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py --against-git-rev 4b7dab7b
population at git rev 4b7dab7b: 23 in-scope site(s)
NOT on the allow-list: 17

$ python scripts/qa/derive_recommendation_consumers_86_22.py
population in the WORKING TREE: 2 in-scope site(s)
NOT on the allow-list: 0        (exit 0)

NOTE the rev is PINNED. The artifact first recorded `--against-git-rev HEAD`,
which was true when captured and false ten minutes later: the auto-changelog
hook lands its own commit on top of every fix, so neither `HEAD` nor `HEAD~1`
is the pre-fix tree. `4b7dab7b` is. The cycle-1 Q/A caught this and confirmed
the recorded output reproduces exactly once the rev is pinned.
```

**Mutation matrix -- 11 cells, both the vocabulary and its detector:**

```
$ python scripts/qa/mutation_matrix_86_22.py
BASELINE (un-mutated): GREEN   46 passed
V1..V7  vocab     killed
D1..D4  detector  killed
RESTORED (un-mutated): GREEN   46 passed
  backend/services/recommendation_vocab.py unchanged: True (71a82b632375ff0e7f983104dddb55b5)
  scripts/qa/derive_recommendation_consumers_86_22.py unchanged: True (4bf4c85c2115e3e301333ec050a24480)
11 killed / 0 survived of 11
```

The matrix's claim is scoped exactly as printed: every guard **in this matrix**
can fail. It says nothing about guards it does not mutate.

## 7. Full-suite delta -- one break was mine

Baseline (tonight, pre-86.22): **14** failures. After the change: **17**. Diffed
rather than counted:

| new failure | mine? |
|---|---|
| `test_phase_82_12_string_column_guards.py::test_classified_line_numbers_still_point_at_a_row_read` | **YES** |
| `test_phase_82_0_macro_ingestion.py::test_ingested_rows_carry_a_vintage` | no -- midnight rollover |
| `test_phase_82_0_macro_ingestion.py::test_macro_end_date_is_severed_from_backtest_end_date` | no -- midnight rollover |

Mine: my comment block shifted `outcome_tracker.py` past the registry's +/-6
tolerance (`:100` -> read now at `:109`). Fixed by **re-deriving** the numbers,
not by widening the tolerance. My first patch applied a uniform +9 and got the
second reference wrong (`:113` vs the real `:112`) -- corrected against the file.

The macro pair asserts `'2026-08-09' == '2026-08-10'`: the clock rolled over
mid-session and something computes "today" once while the test computes it
afresh. Not mine, not in scope, and **queued as its own step** rather than
disclosed in prose -- it will fail every night at midnight, and it may be a real
staleness in a long-running backend rather than only a test artifact.

## 8. Not claimed

- **No lost trade and no lost P&L.** These are evaluation, analytics,
  bias-detection, reporting and display paths. None places an order. The money
  path was 86.20.
- **No claim about `conflict_detector`'s threshold impact beyond 5 rows.**
- **No backfill was performed**, because there is nothing to backfill: the label
  was never persisted and `agent_memories` is empty.
- The exact producer of the three `SELL`-spelled `outcome_tracking` rows is
  **not determined**. `analysis_results` contains only `Sell` (n=16) and
  `paper_trades.risk_judge_decision` contains only
  `APPROVE_REDUCED`/`REJECT`/`APPROVE_HEDGED`/empty -- neither yields `SELL`.
  Stated as unresolved rather than guessed.

## 9. Defect found in passing -- queued, not fixed here

`autonomous_loop.py:3412` does `recommendation = trade.get("risk_judge_decision",
"HOLD")` and passes it to `evaluate_recommendation(...)` as the *recommendation*.
Measured, `risk_judge_decision` holds `APPROVE_REDUCED` (n=15), `REJECT` (n=3),
`APPROVE_HEDGED` (n=1) and empty (n=46) -- an **approval** vocabulary, never a
recommendation. Both before and after the fix these canonicalise to neither buy
nor sell, so behaviour is unchanged and fail-safe, but every outcome row written
by that fallback path is scored with no direction at all. Out of scope for
86.22; filed as its own step.

## 10. Files changed

```
backend/services/recommendation_vocab.py                 (+ intent predicates)
backend/services/outcome_tracker.py                      (migrated)
backend/agents/memory.py                                 (migrated)
backend/agents/bias_detector.py                          (migrated, 3 sites)
backend/agents/conflict_detector.py                      (migrated, thresholds preserved)
backend/api/portfolio.py                                 (migrated)
backend/slack_bot/formatters.py                          (migrated _rec_color)
backend/tests/test_phase_86_22_...vocabulary.py          (NEW, 46 tests)
backend/tests/test_phase_82_12_string_column_guards.py   (line registry re-derived)
scripts/qa/derive_recommendation_consumers_86_22.py      (NEW, detector + validation)
scripts/qa/measure_vocabulary_impact_86_22.py            (NEW, re-derives every number)
scripts/qa/mutation_matrix_86_22.py                      (NEW, 11 cells)
```

---

# CYCLE 2 -- the cycle-1 Q/A returned FAIL, and it was right

Verdict transcribed verbatim in `handoff/current/evaluator_critique_86.22.md`.
Two BLOCKs, both correct, and both were **guard defects, not fix defects** --
the product code was right and the evidence that it was right did not exist.

## The finding that mattered

The Q/A did the thing I had not: it reverted **each migrated file** to its
pre-fix source and re-ran the suite. Four of six survived:

```
outcome_tracker   reverted -> 46 passed  (SURVIVED)
memory            reverted -> 46 passed  (SURVIVED)
api/portfolio     reverted -> 46 passed  (SURVIVED)
slack/formatters  reverted -> 46 passed  (SURVIVED)
bias_detector     reverted ->  2 failed  (killed)
conflict_detector reverted ->  1 failed  (killed)
```

Every assertion I had written tested `is_buy_intent` -- the shared vocabulary --
and **nothing tested that a consumer actually calls it**. Including both
learning-path consumers, the ones that make this step P1.

The worst of it was a test I had named "the load-bearing behavioural
assertion". It recomputed `directionally_correct` **in the test body** from
`is_buy_intent`, then asserted `t is not None and Settings is not None`. A
tautology and an import check, wearing the costume of behaviour. It has been
deleted, not repaired.

## What changed in cycle 2

1. **Seven consumer-driving tests**, calling the real functions:
   `outcome_tracker.evaluate_recommendation` (driven with the literal `BUY`,
   asserting `directionally_correct is True` on a +12% return),
   `memory.generate_reflection` (asserting the PROMPT carries
   `Directionally correct: YES`, plus the LLM-failure fallback string),
   `api.get_portfolio_performance` (async, driven end to end),
   `_rec_color`, and `skill_optimizer`.
2. **Seven per-site mutation cells (S1-S7)** that revert each consumer to
   `4b7dab7b` and re-run the suite. This is the axis criterion 8 names and the
   cycle-1 matrix never ran. **All seven now die.**
3. **`skill_optimizer` migrated** rather than defended (see section 4).
4. The `--against-git-rev HEAD` anchor **pinned to `4b7dab7b`**, and the stale
   `19` corrected to the measured `17`.

## A fixture that could not have failed

`test_api_portfolio_accuracy_...` first used a WINNING `Strong Buy`. Excluded
from the denominator it reads 1/1 = 100%; included, 2/2 = 100%. Identical either
way -- the test could not tell the fix from the defect. The fixture now makes
that position a LOSER, so the readings are 100% (pre-fix) vs 50% (post-fix), and
mutant S4 dies. **An excluded row only becomes visible when it would have counted
against the score.**

## Cycle-2 verification

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q \
    -k "outcome_tracker or bias_detector or conflict_detector or portfolio_manager"'
200 passed, 3097 deselected, 1 warning in 8.63s        exit=0

$ python scripts/qa/mutation_matrix_86_22.py
BASELINE (un-mutated): GREEN   58 passed
V1..V7 vocab killed | D1..D4 detector killed | S1..S7 per-site killed
RESTORED (un-mutated): GREEN   58 passed
  recommendation_vocab.py unchanged: True (71a82b632375ff0e7f983104dddb55b5)
  derive_recommendation_consumers_86_22.py unchanged: True (ac9983a21f9ed57360ad2bf27aa211a2)
18 killed / 0 survived of 18 cells (11 vocab+detector, 7 per-site)

$ python scripts/qa/derive_recommendation_consumers_86_22.py --validate
recall 9/9   precision 10/10

$ python scripts/qa/derive_recommendation_consumers_86_22.py
population in the WORKING TREE: 2 in-scope site(s)
NOT on the allow-list: 0                               exit=0
```

## Full suite, cycle 2

**14 failed / 3265 passed** -- the same COUNT as the pre-86.22 baseline, with
membership differing by one in each direction:

- gone: `test_phase_82_54_cost_budget_columns::test_production_sql_dry_runs_valid`
  (a BQ dry-run, network-dependent);
- new: `test_phase_86_2_replay_poison_row::test_c1_c2_a_poison_row_first_...`,
  which fails with `daily anchor STALE (sod_date='2026-08-09')` -- the third
  midnight-rollover casualty tonight, and it greps zero of the modules this step
  touches;
- the two `test_phase_82_0_macro_ingestion` rollover failures from cycle 1 have
  **resolved on their own**, because both sides of their comparison now compute
  `2026-08-10`.

That churn is the point: three separate tests broke at midnight and one healed
itself six minutes later. **Queued as its own step** -- a suite that changes
colour with the wall clock cannot be a gate.

## Still not claimed

Unchanged from cycle 1: no lost trade, no lost P&L, no backfill performed
(nothing to backfill), and the producer of the three `SELL`-spelled
`outcome_tracking` rows remains **undetermined**.
