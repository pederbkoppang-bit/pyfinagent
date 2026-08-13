# Experiment results — step 86.58

**Step:** 86.58 — the `signal_downgrade` SELL rule is structurally dead on held positions
**Date:** 2026-08-13
**Contract:** `handoff/current/contract_86.58.md`
**Research gate:** PASSED (`wf_58d0341b-6cb`) — brief at `handoff/current/research_brief_86.58.md`

**Outcome: the step's four measurable criteria are satisfied by MEASUREMENT, and NO
production code was changed.** That is the honest deliverable: criterion 4 forbids
promoting the flags, and the flags are the fix. What this step produces is proof the
rule is dead, the derived population, the quantified blast radius, and a recorded
operator recommendation.

---

## What was built

| Path | What |
|---|---|
| `scripts/qa/drive_86_58_dead_downgrade.py` | Re-runnable driven proof of criterion 1, with a positive control and two negative-control cells |
| `handoff/current/contract_86.58.md` | Contract; criteria copied programmatically from the masterplan |
| `handoff/current/research_brief_86.58.md` | Research gate brief (31,129 chars, 6 sources, 38 URLs) |

**No file under `backend/` was modified.** Verified below.

---

## Criterion 1 — PROVEN BY DRIVING, not by reading source

Command:
```
source .venv/bin/activate && PYTHONPATH=/Users/ford/.openclaw/workspace/pyfinagent \
  python scripts/qa/drive_86_58_dead_downgrade.py
```

Verbatim output (exit 0):

```
phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
phase-86.20: UNRECOGNISED recommendation 'swap_buy' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
=== FLAG STATE (asserted, not assumed) ===
  paper_position_recommendation_fix_enabled = False
  paper_recommendation_vocab_fix_enabled = False

=== VOCABULARY, read from source ===
  _BUY_RECS       = ['BUY', 'STRONG_BUY']
  _DOWNGRADE_RECS = ['HOLD', 'SELL', 'STRONG_SELL']
  _SELL_RECS      = ['SELL', 'STRONG_SELL']
  DERIVED: sell_signal fires first and `continue`s, so the ONLY fresh
           recommendation that can ever REACH signal_downgrade is:
           _DOWNGRADE_RECS - _SELL_RECS = ['HOLD']

=== CELLS ===
  A  pos.recommendation='new_buy_signal', fresh='HOLD'  -> []
     signal_downgrade present? False   (claim: False)
  B  pos.recommendation='BUY',            fresh='HOLD'  -> [('NTAP', 'SELL', 'signal_downgrade')]
     signal_downgrade present? True   (control: True)
  C  pos.recommendation='swap_buy',       fresh='HOLD'  -> []
     signal_downgrade present? False   (claim: False)
  D  pos.recommendation='BUY',            fresh='SELL'  -> [('NTAP', 'SELL', 'sell_signal')]
     reason is 'sell_signal', NOT 'signal_downgrade' -> ['sell_signal']

=== VERDICT ===
  CONTROL GREEN: the harness CAN observe signal_downgrade firing.
  A (new_buy_signal) dead: True
  C (swap_buy)       dead: True

  PROVEN BY DRIVING: a held row whose recommendation field carries an order
  reason cannot be sold by signal_downgrade, while an otherwise identical
  row carrying 'BUY' is sold. The rule is structurally dead on the real data.
```

**Cell B is the anti-vacuity control.** Without it, cells A and C report only silence,
and a harness that can never observe the rule firing proves nothing by not observing it.
The script makes B's greenness a precondition: if B is red it prints
`every other cell is UNSCORABLE`.

**Cell D exists to prevent a false pass.** `sell_signal` at `:254` fires first and
`continue`s, so a fresh `SELL` never reaches `signal_downgrade`. Testing with `SELL`
would go green regardless of the held row's value and would prove nothing.

**Derived, not assumed:** `_DOWNGRADE_RECS - _SELL_RECS = {HOLD}`. **`HOLD` is the only
input that can ever reach the rule.** So `portfolio_manager.py:208-218`'s warning that
reviving it makes "HOLDs trigger signal_downgrade SELLs of healthy positions" is not
describing a side effect — it is the rule's **entire reachable domain**.

**Stop-loss confounder controlled:** `stop_loss_price` is set to 100.00 against a
`current_price` of 201.91, so the stop-loss branch (which precedes the rule and
`continue`s) cannot pre-empt it. Otherwise cell B would have gone green for the wrong reason.

---

## Criterion 2 — the population, DERIVED with the rule stated

**Population rule, taken from source and not assumed** (`portfolio_manager.py:60-64`):
the closed vocabulary the SELL path reads is
`_BUY_RECS ∪ _DOWNGRADE_RECS = {BUY, STRONG_BUY, HOLD, SELL, STRONG_SELL}`.
Anything else in `paper_positions.recommendation` is off-vocabulary for that rule.

```sql
SELECT IFNULL(recommendation,'<NULL>') AS rec, COUNT(*) AS n,
       COUNTIF(UPPER(IFNULL(recommendation,'')) IN
               ('BUY','STRONG_BUY','HOLD','SELL','STRONG_SELL')) AS in_closed_set
FROM `sunny-might-477607-p8.financial_reports.paper_positions`
GROUP BY rec ORDER BY n DESC
```

Result:

```
OFF-VOCAB  'new_buy_signal'  n=1  tickers=1
TOTAL 1 rows; in closed set 0 (0.0%); OFF-VOCAB 1 (100.0%)
```

**Currently held rows carrying a reason-shaped value: 1 of 1 (100%).
Carrying a member of the closed set: 0 of 1.**

`paper_positions` is a current-state table, so n=1 is the whole population — complete
but weak on its own. Strengthened from history:

`paper_round_trips.exit_reason`, all 32 completed round trips:
`stop_loss_trigger` 16 (50.0%) · `swap_for_higher_conviction` 13 (40.6%) ·
`sell_signal` 3 (9.4%) · **`signal_downgrade` 0 (0.0%)**.

**The zero is positive-controlled.** `sell_signal` is the adjacent branch in the same
function writing the same column, and it fired 3 times — so the column is populated by
rule-generated reasons and the absence is about `signal_downgrade` specifically, not
about an unwritten column. **The rule has never fired in the book's entire history.**

`paper_trades.reason` over 65 rows, for context on which vocabulary actually populates
these fields: `new_buy_signal` 20 · `stop_loss_trigger` 16 · `swap_buy` 13 ·
`swap_for_higher_conviction` 13 · `sell_signal` 3.

---

## Criterion 3 — flag-ON blast radius, measured, non-live

Measured **in-process only**; no flag was changed and nothing was written.

The book holds **1** position. With the fix flags ON, that row would carry its analysis
recommendation (`BUY`) instead of the order reason. Cell B proves `BUY` + fresh `HOLD`
produces `('NTAP','SELL','signal_downgrade')`.

NTAP's holding re-evals since 2026-07-24:

```
date        rec    score  empty_summary
2026-08-12  HOLD    0.0   True
2026-08-11  BUY     8.0   False
2026-08-10  Hold    6.08  True
2026-08-09  Hold    6.58  True
2026-08-09  HOLD    0.0   True
2026-08-08  HOLD    0.0   True
2026-08-05  HOLD    0.0   True
2026-07-31  BUY     8.0   False
2026-07-29  HOLD    0.0   True
```

**7 of 9 are an empty-summary HOLD; 5 carry `final_score = 0.0`** — the 86.69
placeholder defect.

**Blast radius: 1 of 1 currently-held positions (100%) would become a
`signal_downgrade` SELL candidate**, and on the most recent re-eval (2026-08-12,
`HOLD`, score 0.0, empty summary) the triggering verdict would be a **fabricated
placeholder, not a judgment.**

**Promoting the 06-8 flags today would sell the book's only position on an empty
analysis.** That is the measured recommendation for the ask.

---

## Criterion 4 — no flag promoted

Neither flag was touched. Both read `False` from `Settings()` and the driven script
**asserts** it rather than assuming, aborting if either is not `False`.

Recorded for **ask 06-8**: do **not** promote until 86.69 closes. The interaction is not
theoretical — while 81.2% of analyses are empty `HOLD` placeholders, promoting the fix
converts a dead SELL rule into one that fires on fabricated verdicts. The correct order
is 86.69 first, then re-measure this blast radius, then decide.

---

## Criterion 5 — no guard added, so nothing to mutate

This step adds no guard, so the criterion's condition ("any guard added") is not met.
This is stated rather than silently skipped.

**Why no guard:** the research gate established that the single-boundary module already
exists (`backend/services/recommendation_vocab.py`, 209 lines) and that the real defect
is the unguarded **write** seam at `paper_trader.py:452` (`_pos_rec = reason`, no parse
step). Adding a guard there changes SELL behaviour on the live book and belongs with the
flag decision, not ahead of it. The module's own comment at `:95-105` names the trap —
*"A caller that unwraps them back into a literal set has undone the point"* — and
`portfolio_manager` is that caller (imports only `canonical_recommendation` at `:16`,
hand-writes `_BUY_RECS` at `:60-64`). **That is 86.63's boundary-guard work, not a sixth
site patch here.**

---

## Criterion 6 — the 86.20 log line is preserved, and proven alive

- Present in source: `1` occurrence of `UNRECOGNISED recommendation` in
  `backend/services/portfolio_manager.py`.
- **Fired twice** in the driven run above (cells A and C), verbatim in the captured output.

Not weakened, not quieted, not reworded. The loudness that surfaced this defect is intact.

---

## Immutable verification command

```
bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/portfolio_manager.py\").read()); print(\"parses\")'
```

Result: `parses` (exit 0). Note this command only proves the file parses — it is not
evidence for any criterion above, which is why every criterion carries its own command.

---

## Scope honesty

- **No production code changed.** `git diff --stat` for `backend/` across this step's
  commits is empty; the only added file is `scripts/qa/drive_86_58_dead_downgrade.py`.
- **Protocol order was breached and is disclosed in the contract**: criteria 1 and 2 ran
  before the contract was written, while the research gate was still running. A
  file-mtime check would pass and would not catch it.
- **Criterion 5 is not-applicable, not satisfied.** Q/A should confirm that reading is right.
- What I could not verify: **all three relevant flags are ABSENT from `GET /api/settings/`**
  (45 keys; positive control — 15 `paper_*` keys ARE exposed). They are writable via
  `settings_api.py:261-266` but not readable, so their **live** values in the running
  process are unverified. The driven test asserts the values from a fresh `Settings()`,
  which is the defaults path, not the running process.
