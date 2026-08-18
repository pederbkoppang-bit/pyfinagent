# BUDGET EXHAUSTED -- step 86.47 -- OPERATOR DECISION REQUIRED

- attempts used : 5 / 5
- tokens used   : 0 / 1,200,000
- verdicts seen : 4  (so 1 attempt(s) produced NO verdict and cost tokens anyway)
- outcome mix   : {'FAIL': 2, 'CONDITIONAL': 2, 'NO_VERDICT': 1}

## THIS IS NOT A PASS AND NOT A FAIL

The step is NOT verified. No verdict is implied by exhaustion, and none
may be inferred from it. The loop stopped because it hit a cost ceiling,
which says nothing about whether the work is correct.

## What the operator must decide

1. RAISE the budget and continue, if the remaining work is bounded and known.
2. PARK the step with a written disposition (this project's existing vocabulary).
3. SPLIT it: close the verified part, queue the residuals as their own steps.

## Per-attempt record

- attempt 1: FAIL
- attempt 2: CONDITIONAL
- attempt 3: CONDITIONAL
- attempt 4: FAIL
- attempt 5: NO_VERDICT
## How to proceed (operator)

A further attempt requires an AUDITED extension row:

    python3 scripts/harness/attempt_gate.py --operator-extend 86.47 --by 1 --reason "<why another attempt is warranted>"

The denial itself is NOT a verdict: the step remains exactly as the
last Q/A left it.

*(written 2026-08-18T02:22:39Z by attempt_gate.py at the deny)*

---

## Main's disposition (2026-08-18, written at the deny)

**PARK.** Option 2 of the three above. I am not requesting the extension,
and the reason is that I cannot honestly argue the remaining work is
"bounded and known" (option 1's own test) after the cycle-4 FAIL found an
**analytical** error rather than a bookkeeping one.

### What the budget actually bought

| Attempt | Verdict | What it closed |
|---|---|---|
| 1 | FAIL | Headline "NO GATE IS AT FAULT" was wrong. The refusal signal I had called *underivable* was readable all along — I had probed a JSON **object** with `JSON_VALUE`, which returns NULL; `JSON_QUERY` reads 382/526. |
| 2 | CONDITIONAL | 9 of 15 mutants survived: conclusions were **hardcoded prose**, so the census printed its verdict regardless of the numbers above it. |
| 3 | CONDITIONAL | 4 of 17 still survived; `N_INVARIANTS` was a literal and `n_an` was never cross-checked against `len(WINDOW)`. |
| 4 | FAIL | I collapsed **two independent gates into one** — see below. |
| 5 | *(no verdict — rail drop)* | — |

### The cycle-4 finding, stated plainly because it is the useful output

Section 6b claimed the one BUY that reached the risk gate was refused "on
the same sector-concentration ground as section 4". **It was not.** The
judge's recorded `reasoning` names `projected_dd_over_cap`, **projected_dd
22.5% against a 10% cap**, and files concentration separately under
*"CORROBORATING DOWNSIDE (independent of the gate)"*.

So **two independent gates bind in the post-break era** — a portfolio
**sector** cap and a projected-**drawdown** cap — and the drawdown one is
the more general: the judge records the formula as ~0.5× annualized vol,
so it *"trips for ANY realized vol above ~20%"*, which is most of a
technology book. **That is a materially better answer to "why did the book
stop trading" than the one it replaces**, and it is the strongest single
result this step produced.

Manufacturing a single cause is the exact failure mode this step was
written to prevent, and I committed it — in the very section added to fix
the previous cycle's miss.

### State of the tree at the park

The cycle-5 corrections **are applied and are UNEVALUATED**. No Q/A has
seen them. Specifically: the two-gates correction with the verbatim quote,
`reasoning` added to the printed query (so the claim finally has a
predicate), the post-break-null conclusion made conditional, six new
guards, the false `_p0` code citation in `live_check` removed, and all four
verdicts transcribed into `evaluator_critique_86.47.md` — closing a
five-file breach of mine, since cycles 2 and 3 were ledgered but never
transcribed.

`python scripts/qa/drought_census_86_47.py --verify` → **OK: all 48
invariants hold**. That is a self-check, not a verdict, and it does not
substitute for one.

### What is SAFE about parking here

**Nothing shipped.** The only file this step authored is
`scripts/qa/drought_census_86_47.py`, a read-only census. No production
file was modified, no gate loosened, no flag promoted, no restart pending.
A park costs the project the *answer*, not the *engine*.

### Recommended split, if the operator prefers option 3

The parts I would close as verified are criteria 1, 3 and 4 — the
re-derived base rate with its normalisation rule, the proof that
`risk_judge_decision` is 18/580 in `analysis_results` versus 19/34 BUYs in
`paper_trades`, and the lite/full partition. The residual is criterion 2's
stated-reason element, which is what cycles 1 and 4 both landed on.

### Handed off regardless of disposition — this one is money-path

Four BUY trades executed while a REJECT verdict was on record: 2026-06-02
HPE, 2026-06-03 DELL, 2026-06-09 066570.KS (all `swap_buy`), and 2026-08-13
DELL (`new_buy_signal`, empty column, but the analysis carried REJECT@0%
53 minutes earlier). This step asserts **no mechanism** for it and changed
nothing — criterion 5 forbids this step touching a gate. It belongs to
**86.74**, and it should not wait on 86.47's disposition.
