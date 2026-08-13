# Experiment results — step 86.58

**Step:** 86.58 — the `signal_downgrade` SELL rule is structurally dead on held positions
**Date:** 2026-08-13
**Contract:** `handoff/current/contract_86.58.md`
**Research gate:** PASSED (`wf_58d0341b-6cb`) — brief at `handoff/current/research_brief_86.58.md`

> **CYCLE-2 CORRECTION (Q/A `wf_b127735e-55b` returned FAIL).** Three published
> claims did not survive re-measurement and are corrected below, each marked
> **CORRECTED**. The headline error: I reported a flag-ON blast radius of
> **1 of 1 (100%)** without ever running with the flags on. Measured with them
> genuinely on, it is **0 of 2**. The old text is not silently patched — what was
> wrong is stated next to what replaced it.

**Outcome: the step's measurable criteria are satisfied by MEASUREMENT, and NO
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

Verbatim output, REGENERATED after the cycle-2 script rewrite (exit 0):

```
phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
phase-86.20: UNRECOGNISED recommendation 'swap_buy' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
paper_position_recommendation_fix_enabled is ON while paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions. Enable the integrity flag first (phase-61.2 interaction hazard).
phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
paper_position_recommendation_fix_enabled is ON while paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions. Enable the integrity flag first (phase-61.2 interaction hazard).
paper_position_recommendation_fix_enabled is ON while paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions. Enable the integrity flag first (phase-61.2 interaction hazard).
phase-86.20: UNRECOGNISED recommendation 'swap_buy' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
phase-86.20: recommendation VOCABULARY MISMATCH 'Strong Buy' -- canonical STRONG_BUY, but the legacy gate sees 'STRONG BUY' (held position row ticker=NTAP). vocab_fix_enabled=False
paper_position_recommendation_fix_enabled is ON while paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions. Enable the integrity flag first (phase-61.2 interaction hazard).
phase-86.20: recommendation VOCABULARY MISMATCH 'Strong Buy' -- canonical STRONG_BUY, but the legacy gate sees 'STRONG BUY' (held position row ticker=NTAP). vocab_fix_enabled=True
=== FLAG STATE ===
  OFF cells use the real defaults:
    paper_position_recommendation_fix_enabled = False
    paper_recommendation_vocab_fix_enabled = False
  ON cells use an in-process override (no .env write, no promotion):
    paper_position_recommendation_fix_enabled = True
    paper_recommendation_vocab_fix_enabled = True

=== VOCABULARY (from source) ===
  _BUY_RECS=['BUY', 'STRONG_BUY'] _DOWNGRADE_RECS=['HOLD', 'SELL', 'STRONG_SELL'] _SELL_RECS=['SELL', 'STRONG_SELL']
  DERIVED: sell_signal fires first and continues, so the ONLY fresh rec that can
           reach signal_downgrade is _DOWNGRADE_RECS - _SELL_RECS = ['HOLD']

=== FLAGS OFF ===
  A  pos='new_buy_signal'   fresh='HOLD' -> []  fired=False
  B  pos='BUY'              fresh='HOLD' -> [('NTAP', 'SELL', 'signal_downgrade')]  fired=True
  C  pos='swap_buy'         fresh='HOLD' -> []  fired=False
  D  pos='BUY' fresh='SELL' -> [('NTAP', 'SELL', 'sell_signal')]  (sell_signal PRE-EMPTS; testing with SELL would prove nothing)

=== FLAGS ON (the condition criterion 3 names) ===
  E  pos='new_buy_signal'   fresh='HOLD' -> []  fired=False
  F  pos='BUY'              fresh='HOLD' -> [('NTAP', 'SELL', 'signal_downgrade')]  fired=True
  G  pos='swap_buy'         fresh='HOLD' -> []  fired=False

=== DISCRIMINATION CONTROL: pos='Strong Buy', fresh='HOLD' ===
  flags OFF -> []  fired=False
  flags ON  -> [('NTAP', 'SELL', 'signal_downgrade')]  fired=True

=== VERDICT ===
  Controls GREEN: B (OFF) and F (ON) both fire; the discrimination control
  is dead OFF and live ON, so the probe demonstrably reads flag state.
  A new_buy_signal OFF dead: True   E new_buy_signal ON dead: True
  C swap_buy       OFF dead: True   G swap_buy       ON dead: True

  MEASURED: a held row carrying an order REASON is dead under BOTH flag
  states. Flipping the flag does NOT revive the rule for rows already on
  disk, because flag-ON _resolve_rec maps the value to _UNRECOGNISED_REC
  (in none of the three sets) and the field is written only by execute_buy.
  Blast radius at PROMOTION TIME is therefore 0 for existing rows;
  exposure begins at the next execute_buy that rewrites the field.
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

**CORRECTED — re-derived at publication time, 2026-08-13T20:24:53Z:**

```
CURRENTLY HELD ROWS: 2
  NTAP   rec='new_buy_signal'  qty=5.346643  entry=2026-07-31T18:47:37Z  in_closed_set=False
  DELL   rec='new_buy_signal'  qty=4.806437  entry=2026-08-13T19:31:19Z  in_closed_set=False

  off-vocabulary: 2 of 2 = 100.0%   in closed set: 0 of 2
```

**Currently held rows carrying a reason-shaped value: 2 of 2 (100%).
Carrying a member of the closed set: 0 of 2.**

**What was wrong and why:** this section originally published "TOTAL 1 rows" and
"the book holds 1 position". DELL entered at **19:31:19Z — eight minutes before
the artifact was written** — and I had personally recorded that trade in
`q1_binding_constraint_86.59.md` in the same hour. Knowing a fact is not the same
as re-deriving the count that depends on it. **The proportion survives and
strengthens** (1/1 → 2/2, still 100% off-vocabulary, still 0 in the closed set);
the cardinality did not reproduce. The query and method were correct; the
publication time was not.

Strengthened from history:

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

## Criterion 3 — flag-ON blast radius, MEASURED WITH THE FLAGS ON (CORRECTED)

**What was wrong.** The first version of this section asserted **1 of 1 (100%)** and
concluded *"Promoting the 06-8 flags today would sell the book's only position on an
empty analysis."* **Both halves are false.** The driven script
`assert`ed both flags were `False` and aborted otherwise, then hand-set
`pos.recommendation='BUY'` as a stand-in for the post-fix value — so the production
flag-read executed in **zero cells**. An assertion pinning the subject to its default
is the signature of a proxy.

**What the corrected measurement shows.** The script now ENTERS the condition via
`Settings().model_copy(update={both flags: True})` — in-process only, no `.env`
write, no promotion, no live-book contact:

```
=== FLAGS OFF ===
  A  pos='new_buy_signal'   fresh='HOLD' -> []                                        fired=False
  B  pos='BUY'              fresh='HOLD' -> [('NTAP','SELL','signal_downgrade')]      fired=True
  C  pos='swap_buy'         fresh='HOLD' -> []                                        fired=False
  D  pos='BUY' fresh='SELL' -> [('NTAP','SELL','sell_signal')]   (sell_signal PRE-EMPTS)

=== FLAGS ON ===
  E  pos='new_buy_signal'   fresh='HOLD' -> []                                        fired=False
  F  pos='BUY'              fresh='HOLD' -> [('NTAP','SELL','signal_downgrade')]      fired=True
  G  pos='swap_buy'         fresh='HOLD' -> []                                        fired=False

=== DISCRIMINATION CONTROL: pos='Strong Buy', fresh='HOLD' ===
  flags OFF -> []                                         fired=False
  flags ON  -> [('NTAP','SELL','signal_downgrade')]       fired=True
```

**Three controls, all green.** B fires with flags OFF and F with flags ON, so neither
half is vacuous. The **discrimination control** is the one that was missing before:
`'Strong Buy'` is dead flag-OFF and fires flag-ON, which proves the probe actually
**reads flag state** rather than ignoring it. Without that cell, E and G being empty
would be indistinguishable from the override never taking effect.

**MEASURED BLAST RADIUS AT PROMOTION TIME: 0 of 2 currently-held rows.**

**Mechanism, verified in source — two things I never checked:**

1. Flag-ON `_resolve_rec` maps `'new_buy_signal'` to `_UNRECOGNISED_REC`
   (`'__UNRECOGNISED__'`), which is a member of **none** of `_BUY_RECS`,
   `_SELL_RECS` or `_DOWNGRADE_RECS`. The flag does not translate an order reason
   into a recommendation; it classifies it as unrecognised.
2. `_pos_rec` is written **only** by `execute_buy` (`paper_trader.py:488`, `:512`);
   the partial-sell path at `:676` preserves the stored value. **Flipping a flag does
   not rewrite rows already on disk.**

**So the hazard is real but DEFERRED, not immediate.** Exposure begins at the next
`execute_buy` that rewrites the field — and the fix is not inert: production does
pass `analysis_recommendation` through `portfolio_manager.py:578,:918` →
`autonomous_loop.py:251,:1768`.

## Criterion 4 — no flag promoted

Neither flag was touched. Both read `False` from `Settings()` and the driven script
**asserts** it rather than assuming, aborting if either is not `False`.

**CORRECTED recommendation for ask 06-8.** The original justification rested on the
refuted 1-of-1 immediate blast radius and is withdrawn.

The corrected position: promotion has **zero immediate effect on the 2 currently-held
rows**, so it is *not* an emergency and it would *not* sell the book on an empty
analysis today. The hazard is **deferred** — it begins at the next `execute_buy` that
rewrites `paper_positions.recommendation`, after which a held row would carry a real
`BUY` and a fresh `HOLD` would fire `signal_downgrade`.

That still argues for sequencing **86.69 before promotion**, but on a narrower and
honest basis: once the field starts carrying real recommendations, a fabricated
placeholder `HOLD` (81.2% of analyses in the POST-break regime) becomes a live SELL
trigger. The recommendation is unchanged; the reason it rests on is now one that
reproduces.

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
bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/portfolio_manager.py\").read()); print(\"parses\")"'
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
