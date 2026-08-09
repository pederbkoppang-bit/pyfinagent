# Experiment results -- phase-86.20

**Step:** 86.20 (P1) -- the trade gate and the analyzer speak different
recommendation vocabularies.
**Contract:** `handoff/current/contract_86.20.md` (written BEFORE any code).
**Research:** `handoff/current/research_brief_86.20.md` (gate PASSED, `wf_66bcd575-e9a`).

---

## 1. What was built

`.upper()` folds CASE but never the SEPARATOR. The full-pipeline producer emits
spaced title case, so `"Strong Buy"` became `"STRONG BUY"` and matched none of
the three membership sets -- dropped by `continue` with **no log line at all**.
Plain `"Buy"` worked by accident, which is why it survived: the mismatch
destroyed the HIGHEST-conviction spelling while letting the medium one through.

**Two files, and the split between them is the design.**

| File | Role |
|---|---|
| `backend/services/recommendation_vocab.py` (NEW) | `canonical_recommendation()` -- a FINITE MAPPING onto the closed scale `{STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}`, or `None` for UNKNOWN. Folds case and treats space / hyphen / underscore as one separator. **No substring matching, no prefix matching, no synonym expansion, no other punctuation stripped.** |
| `backend/services/portfolio_manager.py` | `_resolve_rec()` replaces the three `.upper()` read sites (holding re-eval, held-position row, buy candidate). |

**The RESOLUTION is flag-gated; the OBSERVABILITY is not.** That split is
deliberate and is the most important design decision here:

- `paper_recommendation_vocab_fix_enabled` defaults **False**, read via
  `getattr(settings, ..., False)` so flag-absent is byte-identical to flag-OFF.
  Arming it changes what the book does, so it is an operator decision.
- The two WARNING paths ship **live in both states**, because they change no
  decision. With the fix DARK, `"Strong Buy"` is still dropped -- but it is no
  longer dropped **silently**. That is what converts this defect from invisible
  into counted, and it is what the operator's arming decision should rest on.

Two loud cases, and they are different facts:
- **UNRECOGNISED** -- outside the closed scale entirely (e.g. `"N/A"`).
- **VOCABULARY MISMATCH** -- a recognised intent whose legacy spelling the gates
  reject. That is this defect, counted per occurrence.

**UNKNOWN is fail-safe by construction:** it resolves to a sentinel that is in
none of the three sets, so it is not a buy, not a sell and not a downgrade. It
is deliberately NOT collapsed to `HOLD` -- `HOLD` is in `_DOWNGRADE_RECS`, so
that collapse would **sell a position on a parse failure** (mutant M8).

## 2. Files changed

```
$ git status --porcelain | awk '{print $NF}' | grep '\.py$'
backend/config/settings.py
backend/services/portfolio_manager.py
backend/services/recommendation_vocab.py
backend/tests/test_phase_86_20_portfolio_manager_recommendation_vocabulary.py
scripts/qa/measure_86_20_recommendation_population.py
scripts/qa/mutation_matrix_86_20.py
```

The test module is named `..._portfolio_manager_...` **on purpose**: the step's
immutable command selects on `-k "portfolio_manager or decide_trades"`, and a
module named only for the phase would have fallen OUTSIDE the frozen scope. It
was renamed after measuring that the command collected 14 tests and none of the
new ones; it now collects 70 (see §5).

## 3. Criterion 1 -- REPRODUCE FIRST, recorded verbatim

Two candidates identical in every field EXCEPT the recommendation spelling, so
the difference is isolated to the vocabulary and to nothing else. Both reproduce
tests carry an explicit **control** assertion first, so neither can be vacuous:
if `BUY` failed to produce an order the test fails loudly rather than
"passing" for the wrong reason.

Run against the **un-fixed** tree:

```
backend/tests/..._recommendation_vocabulary.py::test_REPRODUCE_producer_strong_buy_yields_no_buy_while_gate_buy_does PASSED [ 50%]
backend/tests/..._recommendation_vocabulary.py::test_REPRODUCE_producer_strong_sell_is_not_sold_at_all PASSED [100%]
============================== 2 passed in 0.27s ===============================
```

The defect at source, printed rather than asserted in prose:

```
_BUY_RECS       = ['BUY', 'STRONG_BUY']
_SELL_RECS      = ['SELL', 'STRONG_SELL']
_DOWNGRADE_RECS = ['HOLD', 'SELL', 'STRONG_SELL']
'Strong Buy'    .upper()='STRONG BUY'    buy=False sell=False downgrade=False
'Buy'           .upper()='BUY'           buy=True  sell=False downgrade=False
'BUY'           .upper()='BUY'           buy=True  sell=False downgrade=False
'Strong Sell'   .upper()='STRONG SELL'   buy=False sell=False downgrade=False
'Sell'          .upper()='SELL'          buy=False sell=True  downgrade=True
'HOLD'          .upper()='HOLD'          buy=False sell=False downgrade=True
'N/A'           .upper()='N/A'           buy=False sell=False downgrade=False
```

**The reproduce tests still pass with the fix ON DISK**, because the flag is
OFF -- which is the evidence that OFF is byte-identical legacy behaviour rather
than a claim about it.

## 4. Criterion 2 -- the population, RE-DERIVED

Generated by `scripts/qa/measure_86_20_recommendation_population.py`, which
resolves the table **through the settings** rather than hardcoding it, so a
dataset rename cannot silently point the measurement somewhere else. Bounded,
cache-disabled, read-only.

```
phase-86.20 criterion 2 -- recommendation population
table (resolved from settings, not hardcoded): sunny-might-477607-p8.financial_reports.analysis_results
_BUY_RECS=['BUY', 'STRONG_BUY'] _SELL_RECS=['SELL', 'STRONG_SELL'] _DOWNGRADE_RECS=['HOLD', 'SELL', 'STRONG_SELL']

value             n genuine  maxgen   legacy           BUY  SELL  DOWN    canonical       BUY  SELL  DOWN   VERDICT
-------------------------------------------------------------------------------------------------------------------
HOLD            275      49    6.00   'HOLD'         False False  True    HOLD          False False  True   unchanged
Hold            115      66    7.33   'HOLD'         False False  True    HOLD          False False  True   unchanged
BUY              91      91    8.00   'BUY'           True False False    BUY            True False False   unchanged
Buy              39      30    8.80   'BUY'           True False False    BUY            True False False   unchanged
Sell             16       8    5.47   'SELL'         False  True  True    SELL          False  True  True   unchanged
Strong Buy        5       1    8.36   'STRONG BUY'   False False False    STRONG_BUY     True False False   FIXED: now reaches BUY
N/A               2       0       -   'N/A'          False False False    None          False False False   UNRECOGNISED (no gate, either way)

rows whose BUY gate is FIXED       : 5
rows whose SELL gate is FIXED      : 0
rows whose DOWNGRADE gate is FIXED : 0
rows UNRECOGNISED by the closed scale: 2

NOTE: reaching a gate is NOT trading. Risk Judge sizing, sector caps
and available cash all sit downstream. No lost-trade or lost-P&L claim
is licensed by this table.
```

**Read the sell-side row honestly: `Strong Sell` does not appear in this corpus
at all.** The sell-side gap is REAL IN CODE and is the fail-dangerous half --
`"Strong Sell"` matches neither `_SELL_RECS` nor `_DOWNGRADE_RECS`, so such a
position would not be exited by either branch and only the stop-loss could close
it -- but it has **zero occurrences today**, so it is a LATENT defect, not a
live one. The buy side is the live half: 5 rows, 1 genuine.

**Correction to the step text, measured rather than inherited.** The step says
the 8.36 `Strong Buy` is "HIGHER THAN ANY ROW THAT DID MATCH (max BUY score
8.0)". That is **false as written**: `Buy` also matches after `.upper()` and
reaches **8.80** on genuine rows (see the `maxgen` column). The true, narrower
statement is that 8.36 is higher than any row spelled with the literal uppercase
`BUY`, and that `Strong Buy` is the highest-scoring genuine recommendation that
FAILS to reach the gate.

**No lost-trade or lost-P&L claim is made.** Reaching the buy-candidate stage is
not trading: Risk Judge sizing, sector caps and available cash all sit
downstream.

## 5. Criteria 3, 4, 5 -- variants, both directions, and no widening

- **Criterion 3** -- six spacing/punctuation variants of one intent
  (`Strong Buy`, `strong buy`, `STRONG-BUY`, `Strong_Buy`, `Strong  Buy`,
  `  Strong Buy  `) all fold to `STRONG_BUY`, and five reach the buy stage
  end-to-end with the fix armed. `Buy` -- safe before -- is asserted unchanged,
  as are `BUY`/`Hold`/`HOLD`/`Sell`/`SELL`/`Strong Sell`/`STRONG_SELL`.
- **Criterion 4** -- SELL and DOWNGRADE are covered, not just BUY. Four
  `Strong Sell` spellings exit via `sell_signal`, and a
  `Strong Buy -> Hold` downgrade exits via `signal_downgrade`. **That downgrade
  test is also the phase-61.2 interaction**: `paper_trader` persists the
  recommendation verbatim, so a full-path position row carries `"Strong Buy"`,
  which resolved to `"STRONG BUY"` and could never satisfy
  `old_rec in _BUY_RECS` -- the rule 61.2 exists to revive was structurally
  dead for exactly those rows even with its own flag ON.
- **Criterion 5** -- `HOLD`, `Hold`, `Sell`, `N/A`, `""`, `None` and
  `Accumulate` are each asserted to produce **no** buy order with the fix armed.
  At unit level `Accumulate`, `Overweight`, `Outperform`, `Strong Buy!`,
  `BUYING`, `STRONG`, `NOT A BUY`, `N/A`, `""`, `"   "`, `None`, `123`,
  `{"a": 1}` and `["BUY"]` all resolve to UNKNOWN. **`BUYING` and `NOT A BUY`
  are in that list specifically because a substring matcher would admit both** --
  that is a real sibling defect, not a hypothetical (filed as 86.22).
  An unrecognised value on a HELD position is also asserted not to sell.

## 6. Criterion 6 -- the silent skip is now observable

Four tests, and the pair that matters is the pair that distinguishes the cases:

- an UNRECOGNISED recommendation is logged distinctly;
- a **recognised** non-buy (`HOLD`) is NOT logged as unrecognised, and produces
  no mismatch line either -- an alarm that fires on every hold is one an
  operator trains themselves to ignore;
- the live defect is loud **with the flag OFF**, naming both the canonical token
  and what the legacy gate saw;
- a genuinely absent recommendation on a legacy position row stays quiet.

The live-defect line, as emitted:

```
WARNING  backend.services.portfolio_manager: phase-86.20: recommendation VOCABULARY MISMATCH 'Strong Buy' -- canonical STRONG_BUY, but the legacy gate sees 'STRONG BUY' (buy candidate ticker=AAA). vocab_fix_enabled=False
```

## 7. Criterion 7 -- MUTATION TEST, both files

`scripts/qa/mutation_matrix_86_20.py`. **It never writes to the repository**:
each mutant is applied in memory inside a throwaway subprocess that registers
the mutated module in `sys.modules` before pytest imports anything, and both
targets' digests are asserted unchanged across the run. Anchors must match
exactly once (a no-match `str.replace` is indistinguishable from success), the
baseline must be GREEN first, and a restored run closes the transcript.

```
phase-86.20 criterion 7 -- mutation matrix (in-memory; repo never written)
  backend/services/recommendation_vocab.py  md5 ba5efe75d056d0dde15d6c7584ac01a5
  backend/services/portfolio_manager.py  md5 b43237b0855de5c1785cb32f637ac720

[baseline] un-mutated tree: 56 passed in 0.53s
  KILLED  | M1 [recommendation_vocab.py]: revert the separator fold (case-only, i.e. pre-86.20 behaviour)
           proves: criterion 3 + the whole fix -- the separator gap must reopen
           tests : test_already_working_spellings_are_unchanged[Strong, test_armed_downgrade_from_a_spaced_prior_recommendation_exits, test_armed_every_strong_buy_spelling_reaches_the_buy_stage[STRONG-BUY], test_armed_every_strong_buy_spelling_reaches_the_buy_stage[Strong (+9 more)
           result: 14 failed, 42 passed in 0.59s
  KILLED  | M2 [recommendation_vocab.py]: drop the closed-scale check (return whatever was folded)
           proves: criterion 5 anti-widening -- UNKNOWN must not become a token
           tests : test_an_unrecognised_recommendation_is_logged_distinctly, test_unrecognised_values_are_never_guessed_into_an_intent[, test_unrecognised_values_are_never_guessed_into_an_intent[Accumulate], test_unrecognised_values_are_never_guessed_into_an_intent[BUYING] (+7 more)
           result: 11 failed, 45 passed in 0.59s
  KILLED  | M3 [recommendation_vocab.py]: match by SUBSTRING instead of by exact membership
           proves: criterion 5 -- 'NOT A BUY' and 'BUYING' must never be a BUY
           tests : test_unrecognised_values_are_never_guessed_into_an_intent[BUYING], test_unrecognised_values_are_never_guessed_into_an_intent[NOT, test_unrecognised_values_are_never_guessed_into_an_intent[Strong
           result: 3 failed, 53 passed in 0.56s
  KILLED  | M4 [recommendation_vocab.py]: also strip non-separator punctuation (widen the fold)
           proves: criterion 5 -- 'Strong Buy!' must stay UNRECOGNISED, not be guessed
           tests : test_unrecognised_values_are_never_guessed_into_an_intent[Strong
           result: 1 failed, 55 passed in 0.58s
  KILLED  | M5 [recommendation_vocab.py]: accept non-strings by str()-ing them
           proves: a dict or enum reaching the gate is a caller bug, not a token
           tests : test_unrecognised_values_are_never_guessed_into_an_intent[123], test_unrecognised_values_are_never_guessed_into_an_intent[value12], test_unrecognised_values_are_never_guessed_into_an_intent[value13]
           result: 3 failed, 53 passed in 0.57s
  KILLED  | M6 [portfolio_manager.py]: ignore the flag -- always use the canonical resolution
           proves: the change is genuinely DARK; OFF must stay byte-identical
           tests : test_REPRODUCE_producer_strong_buy_yields_no_buy_while_gate_buy_does, test_REPRODUCE_producer_strong_sell_is_not_sold_at_all, test_armed_the_same_downgrade_does_NOT_fire_with_the_flag_off, test_the_live_defect_is_loud_even_with_the_flag_OFF
           result: 4 failed, 52 passed in 0.36s
  KILLED  | M7 [portfolio_manager.py]: ignore the flag the other way -- always legacy (fix never applies)
           proves: the armed path is really wired to the flag
           tests : test_armed_downgrade_from_a_spaced_prior_recommendation_exits, test_armed_every_strong_buy_spelling_reaches_the_buy_stage[STRONG-BUY], test_armed_every_strong_buy_spelling_reaches_the_buy_stage[Strong, test_armed_every_strong_buy_spelling_reaches_the_buy_stage[strong (+3 more)
           result: 7 failed, 49 passed in 0.37s
  KILLED  | M8 [portfolio_manager.py]: resolve UNRECOGNISED to HOLD instead of a non-matching sentinel
           proves: HOLD is in _DOWNGRADE_RECS -- this would SELL on a parse failure
           tests : test_armed_an_unrecognised_holding_is_not_sold_either[Accumulate], test_armed_an_unrecognised_holding_is_not_sold_either[N/A], test_armed_an_unrecognised_holding_is_not_sold_either[Overweight]
           result: 3 failed, 53 passed in 0.35s
  KILLED  | M9 [portfolio_manager.py]: drop the UNRECOGNISED warning (silent again)
           proves: criterion 6 -- an unparseable recommendation must be loud
           tests : test_an_unrecognised_recommendation_is_logged_distinctly
           result: 1 failed, 55 passed in 0.34s
  KILLED  | M10 [portfolio_manager.py]: drop the VOCABULARY MISMATCH warning (the live defect goes silent)
           proves: criterion 6 -- with the fix DARK, the drop must still be visible
           tests : test_the_live_defect_is_loud_even_with_the_flag_OFF
           result: 1 failed, 55 passed in 0.34s
  KILLED  | M11 [portfolio_manager.py]: fire the mismatch warning on EVERY value (alarm becomes noise)
           proves: an alarm that fires on every HOLD is one an operator trains away
           tests : test_a_recognised_non_buy_is_NOT_logged_as_unrecognised
           result: 1 failed, 55 passed in 0.35s
  KILLED  | M12 [portfolio_manager.py]: remove the quiet path for a genuinely absent recommendation
           proves: legacy position rows carry none; alarming on them is noise
           tests : test_an_absent_recommendation_on_a_legacy_position_row_is_quiet
           result: 1 failed, 55 passed in 0.36s
[restored] un-mutated tree: 56 passed in 0.54s
[integrity] both targets' md5 unchanged: True
ALL 12 MUTANTS KILLED -- every guard IN THIS MATRIX can fail.
```

M1 is the "revert the normalisation" cell criterion 7 names explicitly. M6 and
M7 are the two directions of the flag, so "flag-gated" is a measured property
rather than a claim. M8 is the one worth reading twice: resolving UNKNOWN to
`HOLD` instead of a non-matching sentinel would **sell a position on a parse
failure**, because `HOLD` is in `_DOWNGRADE_RECS`.

## 8. Verification

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q -k "portfolio_manager or decide_trades"
70 passed, 3106 deselected, 1 warning in 6.20s
```

exit **0**. It collected **14** before this step's tests existed; the 56 new
tests are inside the frozen selector (see §2).

Wider regression over everything that imports `portfolio_manager`:
`-k "portfolio_manager or decide_trades or autonomous_loop or paper_trader or
kill_switch"` -> **294 passed, 1 skipped, exit 0**.

Lint gate (`qa.md` §1a) on a git-DERIVED file set, not a hand-typed one --
`ruff check --select F821,F401,F811` over the 6 changed `.py` files:
`All checks passed!`, exit 0.

Runtime smoke: `import backend.services.portfolio_manager,
backend.services.recommendation_vocab` -> OK; `/api/health` -> **200**.

## 9. NOT IN FORCE -- stated plainly

The running backend (pid 6644, started 18:56:00) imported `portfolio_manager`
long before this change, and CPython serves an imported module from
`sys.modules` without re-reading the file. **So none of this is active on the
live process yet, including the observability.** The behaviour change is DARK
regardless (flag False), but the WARNING lines will not appear in the live log
until a restart. A restart is authorised for this session and is recorded as
pending rather than claimed as done.

## 10. What I could NOT verify

- **No live cycle has exercised this path.** All evidence is in-process against
  the real `decide_trades` with dict fixtures. The next scheduled cycle will
  exercise it only after a restart, and only the observability half while the
  flag stays dark.
- **The sell-side fix is unexercised by real data**: `Strong Sell` has zero rows
  in the corpus (§4), so that half is proven only against fixtures.
- **Whether arming the flag would have changed any actual trade is UNKNOWN and
  deliberately unclaimed.** Five candidate rows would newly reach the buy stage;
  what Risk Judge, the sector caps and available cash would then have done is
  not modelled here.
- **The producer is still unconstrained.** `agents/schemas.py` types the
  synthesis `action` as a free-text `str` whose vocabulary lives only in the
  prompt description, so a new spelling can appear at any time. This step
  detects that loudly but does not prevent it; pinning the field to a `Literal`
  is the durable fix and is deferred to its own step.
- **Seven other consumers of this column remain broken**, in both directions --
  filed as **86.22** with the measured enumeration, not fixed here.
