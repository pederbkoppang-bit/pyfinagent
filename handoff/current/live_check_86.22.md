# live_check -- phase-86.22

Every block below is captured stdout, not transcribed. Regenerate with the
commands shown. Captured 2026-08-10 01:47:27 CEST.

## A. Derived call-site population -- the false-negative check the step names

The step names `outcome_tracker:57` and `bias_detector:119` as the two
false negatives that must be shown. Both appear below, in opposite dialects.

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py --against-git-rev HEAD
population at git rev HEAD: 23 in-scope site(s)

file                                      line  function                rule                                        tested expression           members
-------------------------------------------------------------------------------------------------------------------------------------------------------
backend/agents/bias_detector.py            119  _check_tech_bias        R1 strong-conviction token                  recommendation.upper()      ['STRONG_BUY', 'BUY']
backend/agents/bias_detector.py            128  _check_tech_bias        R1 strong-conviction token                  recommendation.upper()      ['STRONG_BUY', 'BUY']
backend/agents/bias_detector.py            154  _check_confirmation_bi  R1 strong-conviction token                  rec                         ['STRONG_BUY', 'BUY']
backend/agents/bias_detector.py            155  _check_confirmation_bi  R1 strong-conviction token                  rec                         ['STRONG_SELL', 'SELL']
backend/agents/conflict_detector.py        121  _check_recommendation_  R3 SUBSTRING test against a canonical token rec_label                   ['STRONG_BUY']
backend/agents/conflict_detector.py        131  _check_recommendation_  R3 SUBSTRING test against a canonical token rec_label                   ['BUY']
backend/agents/conflict_detector.py        140  _check_recommendation_  R3 SUBSTRING test against a canonical token rec_label                   ['SELL']
backend/agents/memory.py                   229  generate_reflection     R1 strong-conviction token                  original_recommendation     ['Strong Buy', 'Buy']
backend/agents/memory.py                   230  generate_reflection     R1 strong-conviction token                  original_recommendation     ['Strong Sell', 'Sell']
backend/agents/skill_optimizer.py          244  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_BUY', 'BUY']  (allowed)
backend/agents/skill_optimizer.py          247  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_SELL', 'SELL']  (allowed)
backend/agents/skill_optimizer.py          252  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_BUY', 'BUY']  (allowed)
backend/agents/skill_optimizer.py          255  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_SELL', 'SELL']  (allowed)
backend/api/portfolio.py                   140  get_portfolio_performa  R1 strong-conviction token                  rec                         ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']
backend/api/portfolio.py                   142  get_portfolio_performa  R1 strong-conviction token                  rec                         ['BUY', 'STRONG_BUY']
backend/services/outcome_tracker.py         57  evaluate_recommendatio  R1 strong-conviction token                  recommendation              ['Strong Buy', 'Buy']
backend/services/outcome_tracker.py         58  evaluate_recommendatio  R1 strong-conviction token                  recommendation              ['Strong Sell', 'Sell']
backend/slack_bot/formatters.py            169  _rec_color              R3 SUBSTRING test against a canonical token action_upper                ['STRONG_BUY']
backend/slack_bot/formatters.py            169  _rec_color              R3 SUBSTRING test against a canonical token action_upper                ['STRONG BUY']
backend/slack_bot/formatters.py            171  _rec_color              R3 SUBSTRING test against a canonical token action_upper                ['BUY']
backend/slack_bot/formatters.py            173  _rec_color              R3 SUBSTRING test against a canonical token action_upper                ['SELL']
backend/slack_bot/formatters.py            295  _signal_emoji           R3 SUBSTRING test against a canonical token action_upper                ['BUY']  (allowed)
backend/slack_bot/formatters.py            297  _signal_emoji           R3 SUBSTRING test against a canonical token action_upper                ['SELL']  (allowed)

NOT on the allow-list: 17
  !! backend/agents/bias_detector.py:154 (_check_confirmation_bias)  rec  ['STRONG_BUY', 'BUY']
  !! backend/agents/bias_detector.py:155 (_check_confirmation_bias)  rec  ['STRONG_SELL', 'SELL']
  !! backend/agents/bias_detector.py:119 (_check_tech_bias)  recommendation.upper()  ['STRONG_BUY', 'BUY']
  !! backend/agents/bias_detector.py:128 (_check_tech_bias)  recommendation.upper()  ['STRONG_BUY', 'BUY']
  !! backend/agents/conflict_detector.py:121 (_check_recommendation_alignment)  rec_label  ['STRONG_BUY']
  !! backend/agents/conflict_detector.py:131 (_check_recommendation_alignment)  rec_label  ['BUY']
  !! backend/agents/conflict_detector.py:140 (_check_recommendation_alignment)  rec_label  ['SELL']
  !! backend/agents/memory.py:229 (generate_reflection)  original_recommendation  ['Strong Buy', 'Buy']
  !! backend/agents/memory.py:230 (generate_reflection)  original_recommendation  ['Strong Sell', 'Sell']
  !! backend/api/portfolio.py:140 (get_portfolio_performance)  rec  ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']
  !! backend/api/portfolio.py:142 (get_portfolio_performance)  rec  ['BUY', 'STRONG_BUY']
  !! backend/services/outcome_tracker.py:57 (evaluate_recommendation)  recommendation  ['Strong Buy', 'Buy']
  !! backend/services/outcome_tracker.py:58 (evaluate_recommendation)  recommendation  ['Strong Sell', 'Sell']
  !! backend/slack_bot/formatters.py:171 (_rec_color)  action_upper  ['BUY']
  !! backend/slack_bot/formatters.py:173 (_rec_color)  action_upper  ['SELL']
  !! backend/slack_bot/formatters.py:169 (_rec_color)  action_upper  ['STRONG_BUY']
  !! backend/slack_bot/formatters.py:169 (_rec_color)  action_upper  ['STRONG BUY']
```

### After the migration

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py
population in the WORKING TREE: 6 in-scope site(s)

file                                      line  function                rule                                        tested expression           members
-------------------------------------------------------------------------------------------------------------------------------------------------------
backend/agents/skill_optimizer.py          244  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_BUY', 'BUY']  (allowed)
backend/agents/skill_optimizer.py          247  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_SELL', 'SELL']  (allowed)
backend/agents/skill_optimizer.py          252  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_BUY', 'BUY']  (allowed)
backend/agents/skill_optimizer.py          255  analyze_agent_performa  R1 strong-conviction token                  consensus                   ['STRONG_SELL', 'SELL']  (allowed)
backend/slack_bot/formatters.py            316  _signal_emoji           R3 SUBSTRING test against a canonical token action_upper                ['BUY']  (allowed)
backend/slack_bot/formatters.py            318  _signal_emoji           R3 SUBSTRING test against a canonical token action_upper                ['SELL']  (allowed)

NOT on the allow-list: 0
exit=0
```

### The method itself -- recall AND precision, both gate the exit code

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py --validate
METHOD VALIDATION -- recall AND precision, both gate the exit code

KNOWN POSITIVES (must be flagged)
  OK    outcome_tracker (title-case)               R1 strong-conviction token
  OK    outcome_tracker sell leg                   R1 strong-conviction token
  OK    memory.py                                  R1 strong-conviction token
  OK    bias_detector (upper-snake)                R1 strong-conviction token
  OK    api/portfolio.py                           R1 strong-conviction token
  OK    a plain-token recommendation site (R2 only) R2 tested expression names a recommendation
  OK    conflict_detector strong-buy (R3 substring) R3 SUBSTRING test against a canonical token
  OK    conflict_detector buy (R3 substring)       R3 SUBSTRING test against a canonical token
  OK    conflict_detector sell (R3 substring)      R3 SUBSTRING test against a canonical token

KNOWN NEGATIVES (must NOT be flagged)
  OK    order side                                 -
  OK    signal action                              -
  OK    lite-analyzer action                       -
  OK    attribution side                           -
  OK    unrelated literals                         -
  OK    already migrated                           -
  OK    signal-sentiment substring                 -
  OK    bearish substring                          -
  OK    recommend-named field, unrelated literals  -
  OK    recommend-named field, workflow literals   -

recall 9/9   precision 10/10

Method validated in BOTH directions.
```

## B. Re-derived per-value matched/missed table, per consumer

The step supplies a distribution and then says, in capitals, RE-DERIVE.
This script queries the column and recomputes every number from that query;
nothing is carried over from the step text.

```
$ python scripts/qa/measure_vocabulary_impact_86_22.py
==============================================================================
1. DISTRIBUTION of financial_reports.analysis_results.recommendation
   (re-derived; the step's own numbers are deliberately not reused)
==============================================================================

value                   n   genuine   canonical
------------------------------------------------------------
HOLD                  275        49   -
Hold                  115        66   -
BUY                    91        91   BUY
Buy                    39        30   BUY
Sell                   16         8   SELL
Strong Buy              5         1   BUY
N/A                     2         0   -
TOTAL                 543

buy-intent rows (canonical): 135 across 3 spelling(s): ['BUY', 'Buy', 'Strong Buy']

==============================================================================
2. PER-CONSUMER, PER-VALUE  MATCHED / MISSED
==============================================================================

outcome_tracker:57-58  (title-case)
  value                 n  before(buy,sell)   after(buy,sell)   verdict
  ----------------------------------------------------------------------
  HOLD                275  (False, False)     (False, False)    same
  Hold                115  (False, False)     (False, False)    same
  BUY                  91  (False, False)     (True, False)     MISSED
  Buy                  39  (True, False)      (True, False)     same
  Sell                 16  (False, True)      (False, True)     same
  Strong Buy            5  (True, False)      (True, False)     same
  N/A                   2  (False, False)     (False, False)    same
  --> rows this consumer classified DIFFERENTLY from the shared vocabulary: 91 / 543 (16.8%)

memory:229-230         (title-case)
  value                 n  before(buy,sell)   after(buy,sell)   verdict
  ----------------------------------------------------------------------
  HOLD                275  (False, False)     (False, False)    same
  Hold                115  (False, False)     (False, False)    same
  BUY                  91  (False, False)     (True, False)     MISSED
  Buy                  39  (True, False)      (True, False)     same
  Sell                 16  (False, True)      (False, True)     same
  Strong Buy            5  (True, False)      (True, False)     same
  N/A                   2  (False, False)     (False, False)    same
  --> rows this consumer classified DIFFERENTLY from the shared vocabulary: 91 / 543 (16.8%)

bias_detector:119,128  (upper-snake)
  value                 n  before(buy,sell)   after(buy,sell)   verdict
  ----------------------------------------------------------------------
  HOLD                275  (False, False)     (False, False)    same
  Hold                115  (False, False)     (False, False)    same
  BUY                  91  (True, False)      (True, False)     same
  Buy                  39  (True, False)      (True, False)     same
  Sell                 16  (False, True)      (False, True)     same
  Strong Buy            5  (False, False)     (True, False)     MISSED
  N/A                   2  (False, False)     (False, False)    same
  --> rows this consumer classified DIFFERENTLY from the shared vocabulary: 5 / 543 (0.9%)

api/portfolio:140-142  (upper-snake)
  value                 n  before(buy,sell)   after(buy,sell)   verdict
  ----------------------------------------------------------------------
  HOLD                275  (False, False)     (False, False)    same
  Hold                115  (False, False)     (False, False)    same
  BUY                  91  (True, False)      (True, False)     same
  Buy                  39  (True, False)      (True, False)     same
  Sell                 16  (False, True)      (False, True)     same
  Strong Buy            5  (False, False)     (True, False)     MISSED
  N/A                   2  (False, False)     (False, False)    same
  --> rows this consumer classified DIFFERENTLY from the shared vocabulary: 5 / 543 (0.9%)

conflict_detector:121+ (substring)
  value                 n  before(buy,sell)   after(buy,sell)   verdict
  ----------------------------------------------------------------------
  HOLD                275  (False, False)     (False, False)    same
  Hold                115  (False, False)     (False, False)    same
  BUY                  91  (True, False)      (True, False)     same
  Buy                  39  (True, False)      (True, False)     same
  Sell                 16  (False, True)      (False, True)     same
  Strong Buy            5  (True, False)      (True, False)     same
  N/A                   2  (False, False)     (False, False)    same
  --> rows this consumer classified DIFFERENTLY from the shared vocabulary: 0 / 543 (0.0%)

==============================================================================
3. directionally_correct  BEFORE -> AFTER  (outcome_tracker)
==============================================================================

`directionally_correct = (is_buy and return>0) or (is_sell and return<0)`.
The label flips ONLY where the before/after intent differs, so the delta is
bounded by the MISSED count above and its sign depends on the realised return.
Both outcomes are reported rather than assumed.

rows where outcome_tracker's intent changes: 91 / 543
  of these, the label was previously FALSE regardless of return,
  because neither leg matched. After the fix the label is decided by
  the return: a winning call reads correct, a losing one reads wrong.
    BUY            n=91    genuine=91    now scored as BUY

==============================================================================
4. HAS A WRONG REFLECTION ALREADY BEEN PERSISTED?  (measure, not assume)
==============================================================================
  agent_memories       rows = 0
  outcome_tracking     rows = 3
```

### Caveat on conflict_detector's 0.0%

That figure measures INTENT only. `"STRONG BUY"` fails the `"STRONG_BUY"`
clause and then matches the `elif "BUY"` clause, so the intent is right
while the THRESHOLD is wrong -- the strictest check (7.0) becomes the
loosest (5.5) for exactly the highest-conviction calls. 0.0% is a statement
about intent and nothing else.

## C. Reproduce-then-fix, BOTH directions

The two reproduce tests hold the pre-86.22 expressions verbatim and assert
they FAIL on the spelling each dialect cannot see. They are the "before"
half; the parametrised intent tests immediately after are the "after" half.

```
$ python -m pytest backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py -v -k "REPRODUCE or HOLD_is_recognised or conflict_detector"
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_REPRODUCE_title_case_consumer_misses_the_upper_literal PASSED [ 20%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_REPRODUCE_upper_snake_consumer_misses_the_producer_spelling PASSED [ 40%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_HOLD_is_recognised_but_NOT_directional PASSED [ 60%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_conflict_detector_grades_a_strong_buy_at_the_STRICTER_threshold PASSED [ 80%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_conflict_detector_does_not_grade_a_SELL_as_a_BUY PASSED [100%]
======================= 5 passed, 41 deselected in 0.02s =======================
```

## D. directionally_correct -- the before/after delta on REAL persisted rows

`outcome_tracking` has no `directionally_correct` column (schema below), so
the label was never persisted and there is nothing to backfill. The three
rows that ARE persisted are nonetheless real instances of the defect.

```
outcome_tracking columns: ['ticker', 'analysis_date', 'recommendation', 'price_at_recommendation', 'current_price', 'return_pct', 'holding_days', 'beat_benchmark', 'evaluated_at']
directionally_correct present: False

ticker  rec      return_pct   before -> after
------------------------------------------------------
AMD     SELL       -11.3160   False  -> True
PANW    SELL       -10.9368   False  -> True
MU      SELL        -7.2643   False  -> True

labels that CHANGE: 3/3   (three correct sell calls, each previously scored wrong)
agent_memories rows = 0  -> no reflection persisted yet
```

## E. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or bias_detector or conflict_detector or portfolio_manager"'
188 passed, 3097 deselected, 1 warning in 6.92s
exit=0
```

## F. Mutation matrix -- vocabulary AND detector

```
$ python scripts/qa/mutation_matrix_86_22.py
phase-86.22 mutation matrix
  target backend/services/recommendation_vocab.py  md5=71a82b632375ff0e7f983104dddb55b5
  target scripts/qa/derive_recommendation_consumers_86_22.py  md5=4bf4c85c2115e3e301333ec050a24480
BASELINE (un-mutated): GREEN
  46 passed in 2.30s
id   kind      result    mutation
----------------------------------------------------------------------------------------------------
V1   vocab     killed    stop folding the separator (revert to the pre-86.20 behaviour)
               proves: the whole defect -- 'Strong Buy' must not fall back to UNKNOWN
V2   vocab     killed    fold whitespace only -- hyphen and underscore stop being separators
               proves: 'STRONG-BUY' / 'Strong_Buy' are in the parametrised buy set
V3   vocab     killed    WIDEN is_buy_intent to any recognised value (HOLD becomes a buy)
               proves: the fix must not become an over-permissive gate
V4   vocab     killed    put HOLD in BUY_INTENT
               proves: a considered HOLD is not a directional call
V5   vocab     killed    alias is_sell_intent to the buy set (direction inverted)
               proves: sell must not be graded as buy -- the substring defect's core
V6   vocab     killed    accept non-strings by coercing with str()
               proves: a dict or enum reaching the gate is a caller bug, not a token
V7   vocab     killed    make is_directional() true for HOLD as well
               proves: 'unparseable' and 'considered hold' must stay distinguishable
D1   detector  killed    delete rule R3 (the substring shape becomes invisible again)
               proves: recall -- this is the exact blindness that missed conflict_detector
               false positives: []
D2   detector  killed    delete rule R1 (strong-conviction tokens no longer in scope)
               proves: recall -- R1 is what catches a site with an unhelpful variable name
               false positives: []
D3   detector  killed    flag EVERY literal membership test (perfect recall, no precision)
               proves: precision -- a detector that flags everything is not a detector
               false positives: ['order side', 'signal action', 'lite-analyzer action', 'attribution side']
D4   detector  killed    drop the R2 requirement that the literals be recommendation-shaped
               proves: precision -- `mode in ('fast','slow')` must not enter the population
               false positives: ['recommend-named field, unrelated literals', 'recommend-named field, workflow literals']
RESTORED (un-mutated): GREEN
  46 passed in 2.20s
  backend/services/recommendation_vocab.py unchanged: True (71a82b632375ff0e7f983104dddb55b5)
  scripts/qa/derive_recommendation_consumers_86_22.py unchanged: True (4bf4c85c2115e3e301333ec050a24480)
11 killed / 0 survived of 11
Every guard IN THIS MATRIX can fail. That is the scope of this claim:
it says nothing about guards the matrix does not mutate.
```

## G. Full-suite delta -- one break was mine, and it is fixed

```
baseline (pre-86.22, tonight): 14 failures
after 86.22:                   16 failures

NEW vs baseline:
FAILED backend/tests/test_phase_82_0_macro_ingestion.py::test_ingested_rows_carry_a_vintage
FAILED backend/tests/test_phase_82_0_macro_ingestion.py::test_macro_end_date_is_severed_from_backtest_end_date

Both are a midnight rollover: they assert '2026-08-09' == '2026-08-10'.
The clock turned over mid-session. Not caused by this change; queued as
its own step because it will fail every night at midnight.

A THIRD new failure appeared before this capture and WAS mine --
test_phase_82_12_string_column_guards::test_classified_line_numbers_...
My comment block shifted outcome_tracker.py past the registry's +/-6
tolerance. Fixed by re-deriving the line numbers from the file.
```

---

# CYCLE 2 -- captured after the cycle-1 FAIL

Captured 2026-08-10 02:31:21 CEST. Every block is stdout.

## H. The rev is now PINNED (cycle-1 Q/A finding)

Section A recorded `--against-git-rev HEAD`, true when captured and false
minutes later: the auto-changelog hook commits on top of every fix, so
neither HEAD nor HEAD~1 is the pre-fix tree. The pre-fix tree is 4b7dab7b.

```
$ python scripts/qa/derive_recommendation_consumers_86_22.py --against-git-rev 4b7dab7b
population at git rev 4b7dab7b: 23 in-scope site(s)
NOT on the allow-list: 21

$ python scripts/qa/derive_recommendation_consumers_86_22.py
population in the WORKING TREE: 2 in-scope site(s)
NOT on the allow-list: 0
```

## I. PER-SITE mutation -- the axis criterion 8 names

The cycle-1 matrix never reverted a fixed SITE. The Q/A ran that axis and
found four of six migrations unguarded. All seven now die:

```
BASELINE (un-mutated): GREEN
  58 passed in 2.29s
id   kind      result    mutation
----------------------------------------------------------------------------------------------------
V1   vocab     killed    stop folding the separator (revert to the pre-86.20 behaviour)
               proves: the whole defect -- 'Strong Buy' must not fall back to UNKNOWN
V2   vocab     killed    fold whitespace only -- hyphen and underscore stop being separators
               proves: 'STRONG-BUY' / 'Strong_Buy' are in the parametrised buy set
V3   vocab     killed    WIDEN is_buy_intent to any recognised value (HOLD becomes a buy)
               proves: the fix must not become an over-permissive gate
V4   vocab     killed    put HOLD in BUY_INTENT
               proves: a considered HOLD is not a directional call
V5   vocab     killed    alias is_sell_intent to the buy set (direction inverted)
               proves: sell must not be graded as buy -- the substring defect's core
V6   vocab     killed    accept non-strings by coercing with str()
               proves: a dict or enum reaching the gate is a caller bug, not a token
V7   vocab     killed    make is_directional() true for HOLD as well
               proves: 'unparseable' and 'considered hold' must stay distinguishable
D1   detector  killed    delete rule R3 (the substring shape becomes invisible again)
               proves: recall -- this is the exact blindness that missed conflict_detector
               false positives: []
D2   detector  killed    delete rule R1 (strong-conviction tokens no longer in scope)
               proves: recall -- R1 is what catches a site with an unhelpful variable name
               false positives: []
D3   detector  killed    flag EVERY literal membership test (perfect recall, no precision)
               proves: precision -- a detector that flags everything is not a detector
               false positives: ['order side', 'signal action', 'lite-analyzer action', 'attribution side']
D4   detector  killed    drop the R2 requirement that the literals be recommendation-shaped
               proves: precision -- `mode in ('fast','slow')` must not enter the population
               false positives: ['recommend-named field, unrelated literals', 'recommend-named field, workflow literals']
per-SITE cells -- revert each migrated consumer to 4b7dab7b
id   result    consumer                                    guard that must catch it
--------------------------------------------------------------------------------------------------------------------
S1   killed    backend/services/outcome_tracker.py         test_outcome_tracker_evaluate_recommendation_IS_DRIVEN_wit
S2   killed    backend/agents/memory.py                    test_memory_generate_reflection_IS_DRIVEN_and_the_PROMPT_c
S3   killed    backend/agents/bias_detector.py             test_bias_detector_fires_on_every_strong_buy_spelling
S4   killed    backend/api/portfolio.py                    test_api_portfolio_accuracy_DENOMINATOR_includes_every_buy
S5   killed    backend/agents/conflict_detector.py         test_conflict_detector_grades_a_strong_buy_at_the_STRICTER
S6   killed    backend/slack_bot/formatters.py             test_slack_formatter_rec_color_handles_BOTH_dialects
S7   killed    backend/agents/skill_optimizer.py           test_skill_optimizer_consensus_uses_the_shared_vocabulary
RESTORED (un-mutated): GREEN
  58 passed in 2.34s
  backend/services/recommendation_vocab.py unchanged: True (71a82b632375ff0e7f983104dddb55b5)
  scripts/qa/derive_recommendation_consumers_86_22.py unchanged: True (ac9983a21f9ed57360ad2bf27aa211a2)
18 killed / 0 survived of 18 cells (11 vocab+detector, 7 per-site)
Every guard IN THIS MATRIX can fail. That is the scope of this claim:
it says nothing about guards the matrix does not mutate.
```

## J. The consumers are now DRIVEN, not re-implemented

```
$ python -m pytest ...test_phase_86_22... -v -k "IS_DRIVEN or api_portfolio or skill_optimizer or rec_color"
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_outcome_tracker_evaluate_recommendation_IS_DRIVEN_with_literal_BUY PASSED [ 20%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_memory_generate_reflection_IS_DRIVEN_and_the_PROMPT_carries_the_label PASSED [ 40%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_api_portfolio_accuracy_DENOMINATOR_includes_every_buy_spelling PASSED [ 60%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_slack_formatter_rec_color_handles_BOTH_dialects PASSED [ 80%]
backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py::test_skill_optimizer_consensus_uses_the_shared_vocabulary PASSED [100%]
======================= 5 passed, 53 deselected in 1.65s =======================
```

## K. Immutable command, cycle 2

```
200 passed, 3097 deselected, 1 warning in 8.63s
exit=0
```
