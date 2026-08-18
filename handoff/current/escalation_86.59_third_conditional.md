# ESCALATION -- step 86.59 -- THIRD CONSECUTIVE CONDITIONAL -- OPERATOR DECISION

**Sequence:** `[CONDITIONAL, CONDITIONAL, CONDITIONAL]`
(`wf_5a3bc88c-4e1`, `wf_d1d01d57-0f6`, `wf_2cc6808c-bea`)
**Attempts used:** 3 of 5 -- **the budget is NOT the binding constraint.**
**Binding constraint:** CLAUDE.md F1 3rd-CONDITIONAL rule. The next Q/A pass
**must return FAIL regardless of evidence.**

## Why I parked instead of spawning a fourth

A fourth spawn cannot return PASS. Spending an attempt to obtain a verdict the
rule has already determined would burn tokens for no information. The standing
instruction is to park rather than iterate, so the step is `pending` and the
fix is committed.

**This is decision 5 from `goal_next_2026-08-19.md`, arriving for a third step.**
86.108 and 86.110 both parked on this rule last session. The difference here is
worth stating plainly, because it cuts the *other* way:

> **86.108 and 86.110 parked with every criterion MET.** 86.59 did **not**.
> Each of the three CONDITIONALs found a **real, blocking, reproducible defect**
> that I had shipped. The rule is not obstructing a converged step here -- it is
> stopping one that genuinely needed three rounds.

## What each cycle actually found -- none of it was bookkeeping

| cycle | finding | proved how |
|---|---|---|
| 1 | `panel_is_us_only` was a literal `True`; `baseline_arm_is_the_unflagged_ranking` was `len(x)==len(set(x))` on a set-derived list | AST + **poisoned the baseline arm and the run stayed green** while every criterion-4 delta collapsed to zero |
| 2 | the replacement asserted the arm **definition**, which a downstream injection does not touch; the §8 evidence block was **spliced from two runs**; `_PREDICATE_FIXTURE` had no cell | injected at the `replay_session` seam -- **min_k's delta flipped +2.1pp to -2.1pp** |
| 3 | the cycle-2 fix guarded the **wrong variable**: `base` feeds only the min_k arm, while `arms["baseline"]` -- the row every delta is subtracted from -- came from a structurally identical sibling call | injected at the arms loop -- **same sign flip**, and a `w=0.05` variant read every delta *exactly as published* while the baseline's top-sector share moved 0.72 to 0.64 |

**Four appearances of one lesson in one step**: a value check, a definition
check, a behavioural check on the wrong variable, and an overclaim about what
the check covered. Each fix relocated the seam instead of closing the class.

## State at the park -- the residual IS fixed, it is just unevaluated

The cycle-3 blocker is closed and verified by execution:

- the oracle is computed **once per cycle, before the arms loop**, and **both**
  unflagged slates must equal it (`baseline_ROW_matches_an_unflagged_direct_call`,
  cell M22);
- both of the evaluator's exact injections now **KILL** at the published
  `--cycles 20`, control GREEN first and reproducing 15.8%/12 with min_k at
  +2.1pp, disk md5 unchanged;
- matrix **23/23 KILLED**, coverage **24/24**, 0 SURVIVED, 0 UNSCORABLE;
- the false claim (*"an injection anywhere in the replay path makes these
  diverge"*) is narrowed in the shipped code comment and in both artifacts;
- the contract's abandoned P3/P4/P6 is now disclosed as a **deviation**.

**No Q/A has seen any of that.** `--verify` green is a self-check, not a verdict.

## The product, which survives the park

Three independent evaluators re-derived **every** published number and all three
reproduced them exactly: rho 0.9622/0.9319, one-sided top-10 turnover 15.8%/day,
3-of-19 zero-turnover sessions, 12 distinct names + IT 72.0% with counts
{20,72,8}, flag arms 15.8/28.4/22.1/17.9 with deltas +12.6/+6.3/+2.1pp, sigmas
10.646/19.850/30.441 at 2.86x, effective shares 22.6/37.0/40.4, multidim
50/10139, fidelity 80%, 18 live distinct tickers, dedup 47,880/200,875.

**Three findings that do not depend on this step's disposition:**

1. **The step's premise is partially refuted.** rho 0.9622 with 15.8% daily
   turnover is highly persistent but **not frozen**, and the live system
   analysed **18** distinct tickers over the same 20 sessions, not the "8 across
   8 cycles" the step text reports.
2. **The diversity mitigation already exists and is switched off.** All three
   dark flags move the slate; `min_k=3` cuts IT concentration 72% to 60% for
   +2.1pp turnover. Four numbered operator asks are recorded in
   `experiment_results_86.59.md`.
3. **Two system defects were filed rather than absorbed**: **86.116** (P1, 38.0%
   of `historical_prices` rows are duplicate keys, nothing under `backend/`
   de-duplicates, harm is positional) and **86.117** (declared 40/35/25 measures
   as effective 22.6/37.0/40.4), the latter BLOCKED-BY the former.

## What the operator can decide

1. **Authorise one attempt** knowing it returns FAIL by rule, purely to reset
   the counter, then a fifth to grade the fix on its merits. Costs 2 of the 2
   remaining attempts.
2. **Accept on the evaluators' own finding** that the PRODUCT is sound, nothing
   ships to production, and the blocker is fixed -- and flip. Main cannot do
   this without a PASS, which is why the step sits at `pending`.
3. **Leave parked.** The measurement is committed and re-runnable; the operator
   asks stand on numbers three evaluators reproduced.
4. **Revisit the rule itself** (decision 5). Three steps have now parked on it.
   86.59 is evidence it works as intended; 86.108 and 86.110 are evidence it
   also catches converged steps. Those are different cases and the rule cannot
   currently tell them apart.

**Nothing shipped, so a park costs the answer, not the engine.** Zero production
files across all four step commits.
