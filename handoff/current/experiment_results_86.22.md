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

`backend/agents/skill_optimizer.py:244` is deliberately **not** migrated: it
reads the schema-enforced `Literal` in `agents/schemas.py`, so the producer
cannot emit another spelling. Migrating it would add a dependency without
removing a risk.

**No set was widened.** `"Accumulate"`, `"Overweight"`, `"BUYING"`,
`"NOT A BUY"`, `"Strong Buy!"` and `"N/A"` all remain non-directional.

## 5. Two things I got wrong, and how they were caught

**(a) My consumer population was wrong, because my detector was blind.** The
first derivation used a regex, then an AST scan with two rules. It reported a
confident population of 10 sites across 4 files -- and silently **missed
`conflict_detector.py` entirely**, because a substring test is a different AST
shape, not a different spelling. Adding rule R3 raised the pre-fix population to
**19 offender sites** and surfaced a **sixth consumer I had not migrated**
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
$ python scripts/qa/derive_recommendation_consumers_86_22.py --against-git-rev HEAD
population at git rev HEAD: 23 in-scope site(s)
NOT on the allow-list: 19

$ python scripts/qa/derive_recommendation_consumers_86_22.py
population in the WORKING TREE: 6 in-scope site(s)
NOT on the allow-list: 0        (exit 0)
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
