# Experiment results -- step 86.74

**Step:** falsy-zero check inverts a 0% REJECT into the 10%-NAV default.
**Date:** 2026-08-14. **Verification command:** GREEN, `34 passed`.

---

## 0. Headline

**DELL's exact case is now blocked, with the shipped production flag state (OFF).**
Driven through the real `decide_trades`, a nested `REJECT / 0%` verdict produces
**no order** with `paper_risk_judge_shape_fix_enabled` both OFF and ON.

**And the diagnosis had to be corrected mid-cycle.** Fixing the falsy-zero alone
did **not** fix DELL -- see §2, which is the most important section here.

---

## 1. What was changed

| File | Change |
|---|---|
| `backend/services/portfolio_manager.py` | 3-state `PositionVerdict`; `_resolve_position_pct`; `_sizing_pct` chokepoint; nested-first resolution made unconditional; `_extract_position_pct` fixed |
| `backend/services/autonomous_loop.py` | `_persist_analysis` now passes `risk_judge_decision`, `risk_level`, `recommended_position_pct` |
| `backend/services/signal_attribution.py` | nested-first judge resolution; RiskJudge row emitted when `pos_pct is not None` |
| `backend/agents/risk_debate.py` | completion log line carries `ticker=` |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | 9 -> 31 tests, 17 -> 51 asserts |
| `scripts/qa/mutation_matrix_86_74.py` | **new** -- 6-cell mutation harness |

---

## 2. THE CORRECTION -- the falsy-zero fix alone did NOT fix DELL

This is the finding of the cycle, and it inverts the step's own premise.

After fixing `_extract_position_pct`, `test_full_path_reject_not_blocked_even_binding_on`
**still passed** -- i.e. a REJECT still bought. The reason is that **two distinct
defects were being conflated**:

| defect | what it is | who fixes it |
|---|---|---|
| **falsy-zero** | a **visible** `0.0` collapses to `None` via `if pct:` | the helper fix |
| **nesting** | with the flag OFF the verdict is **not visible at all** | previously flag-gated |

With `shape_fix` OFF the full-path judge is nested under `risk_assessment["judge"]`,
so `_resolve_position_pct` **correctly** returned `ABSENT` and the 10% default was
legitimately reached. **The falsy-zero fix never got a chance to fire on DELL's
input.** DELL was a *nesting* casualty, not (only) a falsy-zero casualty.

Criterion 3 requires the default be reachable only from "a genuinely absent
verdict". A nested 0% verdict is **present**. So the nested-first resolution was
made **unconditional** (`portfolio_manager.py`, `_rj_view`). It can only ever
reveal a verdict that was already there; it never invents one.

**Had I stopped at the falsy-zero, this step would have shipped a P0 "fix" that
left the reported incident live**, with a green suite.

---

## 3. Criterion-by-criterion

### C1 -- fixed AT THE HELPER, holds in BOTH flag states ✅

`_resolve_position_pct` uses `is not None` on **both** sources (the second kept
the defect under *every* flag setting -- research finding R1). No flag is read by
the helper at all. Proven by `TestHelperDistinguishesZeroFromAbsent` (7 tests) and
by `TestRejectBindsInBothFlagStates`, parametrised over `[False, True]`.

### C2 -- a REJECT binds, driven through the real path ✅

`decide_trades` is driven end-to-end (not a stub). Mutation **M2** restores
`or 10.0` and the assertion **goes red** -- so the guard can fail.

### C3 -- the default set, DERIVED from source ✅

Enumeration rule: every `ast.BoolOp(Or)` whose right operand is the constant
`10.0` in `portfolio_manager.py`. **Pre-fix: 4 sites** -- `:507` (flag-guarded),
`:800`, `:853`, `:878` (**unguarded under every flag state**). The research brief
found `:878`; `:800` and `:853` are additional.

**Post-fix: 0.** All four route through `_sizing_pct`, whose branches are:
`SIZE -> the pct` / `UNPARSEABLE -> 0.0` / `ABSENT -> the default`. The default is
reachable from ABSENT and only ABSENT.

The in-suite check is AST-based, **not grep** -- grep matched my own explanatory
comments containing the phrase `or 10.0`, i.e. the probe matched its own
documentation. It carries a positive control that a synthetic `or 10.0` *is*
detected, so `offenders == []` cannot pass vacuously.

### C4 -- verdict persisted per ticker ✅ (root cause was NOT where the step assumed)

Baseline **reproduced exactly**:

```sql
SELECT COUNT(*) total, COUNTIF(risk_judge_decision != '') dec_pop,
       COUNTIF(risk_level != '') lvl_pop, COUNTIF(recommended_position_pct IS NOT NULL) pct_pop
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE DATE(analysis_date) BETWEEN '2026-07-20' AND '2026-08-13'
-- total_rows=129  decision=0  risk_level=0  pct=0   (2026-07-20..2026-08-13)
```

**The cause is a second write path**, exactly as the step's audit_basis
suspected ("at least two write paths and only one was traced"):
`tasks/analysis.py:273,302,303` **does** pass all three -- but that is the
**API-triggered** path. The **autonomous loop** uses
`autonomous_loop.py::_persist_analysis`, which called `bq.save_report(...)`
**without those three kwargs at all**, while `save_report` had accepted them the
whole time (`bigquery_client.py:119,148,149`). Now passed, nested-first.

**The verdict was never actually lost** -- it sits in the JSON blob at
`$.final_synthesis.risk_assessment.judge`. Confirmed for all six 2026-08-13
tickers, which also **retires this step's elimination-based attribution**:

| ticker | persisted decision | pct | incident's INFERRED verdict |
|---|---|---|---|
| HPE | APPROVE_HEDGED | 4 | APPROVE_HEDGED/4% ✓ |
| MRVL | REJECT | 0 | REJECT/0% ✓ |
| **DELL** | **REJECT** | **0** | **REJECT/0% ✓ (was by elimination)** |
| 009150.KS | REJECT | 0 | REJECT/0% ✓ |
| HPQ | APPROVE_REDUCED | 3 | APPROVE_REDUCED/3% ✓ |
| NTAP | REJECT | 0 | REJECT/0% ✓ |

**6 of 6 match.** The inference was correct and is now unnecessary.

**Post-fix populated share is NOT reported as a live number.** Criterion 4 asks
for the post-fix share against the 0-of-129 baseline; that requires an autonomous
cycle to run with the new code, and **the backend has not been restarted**
(restarts are batched to session end). The write is proven at the unit seam
(`TestVerdictIsPersistedPerTicker` + mutation M3), **not yet in BQ**. Reporting a
post-fix share now would be reporting a number I did not measure.

### C5 -- the log line carries its ticker ✅

`risk_debate.py` now logs `ticker={ticker}`. **This removes the elimination-based
attribution this step's own evidence relied on**: six concurrent debates on
2026-08-13 logged decision/risk_level/position/rounds and no ticker, so five were
paired by exact-second matching against BQ `analysis_date` and **DELL was
identified only by elimination against the one remaining completion**. Mutation
M6 removes the ticker again and the check goes red.

### C6 -- RiskJudge in `factors_json` regardless of pct ✅

`signal_attribution.py` read `risk_assessment` **top-level only**, so on the full
path both `decision` and `reasoning` were empty, the `if decision or reasoning:`
guard was False, and the RiskJudge row was **dropped entirely** -- the measured
DELL 3 agents/517 chars vs NTAP 4 agents/1232 chars gap. Fixed by nested-first
resolution plus `or pos_pct is not None`.

A stale comment there asserted *"`recommended_position_pct` is always > 0 by
construction"*. **That is false and was falsified in production** (DELL = 0);
corrected in place.

### C7 -- `paper_trades` swept ⚠️ PARTIAL, and reported as partial

**Enumeration rule:** population = every `paper_trades` row with
`UPPER(action)='BUY'`, all time = **34** (`COUNT(*)=66`, `COUNTIF(BUY)=34`,
`COUNT(DISTINCT trade_id)=66`). Joined to `analysis_results` on `ticker` AND
`|analysis_date - TIMESTAMP(analysis_id)| < 2s`. Flag = a completed verdict of
`REJECT` **or** `pct = 0` while a BUY executed.

```
INVERSION confirmed                 :  1   (DELL 2026-08-13, notional 2392.26, REJECT/0.0)
verdict permitted the buy           :  0
NO joinable verdict -> UNDETERMINED : 33
POSITIVE CONTROL: DELL detected     :  True
```

**The 1 is a measured 1; the 33 are NOT a measured zero.** 33 of 34 BUYs
(2026-04-26 .. 2026-07-31) have no joinable verdict row, so **the historical
sweep is inconclusive and I am not claiming DELL was the only occurrence.**
`analysis_results` holds 567 rows (2025-11-23..2026-08-13) of which 372 carry a
nested judge verdict, so the data broadly exists -- the join, not the data, is the
limit. **Closing this properly is queued as its own step rather than stretched
here.**

*(An earlier count of 35 BUYs was join fan-out from a same-day date join; the
correct population is 34, taken from the table directly.)*

### C8 -- flag-ON-only blindness closed ✅

```
test functions : 9  -> 31    (pytest -q prints this count)
assert stmts   : 17 -> 51    (grep -c 'assert ')
```

**The "9" in the criterion is the TEST count, not the assertion count** -- both
are reported with the rule so a net removal is visible in either denominator and
the two are never conflated.

**Two tests asserted the DEFECT and were rewritten, not deleted:**

- `test_full_path_sizes_at_10pct_default_and_empty_decision` required
  `abs(amount_usd - NAV*0.10) < 0.5`, commented *"10% NAV default (the bug)"*.
- `test_full_path_reject_not_blocked_even_binding_on` required
  `_buy(orders) is not None`, commented *"REJECT invisible top-level -> buys"*.

Both encoded the DELL defect as expected behaviour -- which is exactly why the
suite was green while the bug ran in production. The replaced text is quoted in a
comment block in the file so the inversion is visible in review.

### C9 -- mutation matrix ✅ 6/6 KILLED

Control observed **GREEN first**; all 4 subject files snapshotted up front and
restored byte-identically (sha256-verified).

| cell | subject | mutation | verdict |
|---|---|---|---|
| M1 | portfolio_manager | restore `if pct:` | **KILLED** |
| M2 | portfolio_manager | restore `or 10.0` at the sizing seam | **KILLED** |
| M3 | autonomous_loop | delete the persistence write | **KILLED** |
| M4 | portfolio_manager | default reachable from UNPARSEABLE | **KILLED** |
| M5 | signal_attribution | drop RiskJudge row when pct is 0 | **KILLED** |
| M6 | risk_debate | remove ticker from the log line | **KILLED** |

A cell whose target text is absent scores `NOT_APPLIED`, never KILLED -- a
no-match `str.replace` looks exactly like success.

### C10 -- nothing loosened, DELL untouched ✅

No threshold, gate or cap was weakened; **every change makes a buy strictly less
likely**. No `.env` write, no flag promotion, no manual cycle, no restart. **The
DELL position was not liquidated or resized** -- position remedy is operator work.

---

## 4. Deliberate behaviour change under flag-OFF -- declared

The `shape_fix` flag was documented "OFF -> byte-identical". **That is
deliberately no longer true**, and it is the point of the step: OFF is the
shipped production state and OFF is the broken one. A 0% REJECT now blocks, and
a nested verdict is now read, **in both flag states**. Declared in
`contract_86.74.md` §5a *before* the code was written.

This is **not** a flag promotion: no `.env` was touched and 79.1 remains the
operator's. The flag's sizing half is now vestigial.

---

## 5. Pre-existing failures -- NOT caused by this step

Two tests in adjacent suites fail:

```
test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
```

**Measured, not assumed.** Both were re-run with `portfolio_manager.py` reverted
to `HEAD` (via `git show HEAD:<path> >`, my copy restored byte-identically and
sha-verified): **both already failed at HEAD.** They are pre-existing and
unrelated. Queued as their own step rather than fixed inline.

## 6. What I could NOT verify

1. **The post-fix persisted share in BQ** (C4) -- needs an autonomous cycle after
   a restart; restarts are batched to session end. Proven at the unit seam only.
2. **33 of 34 historical BUYs** (C7) -- undetermined, not clean.
3. **Why `NTAP` carries `risk_judge_position_pct=4.0` from 2026-07-31** while its
   analysis row persisted no verdict -- untraced, as the step notes.
4. **Nothing was driven through a live browser or the running backend**; the
   running process still holds the pre-fix code.
