# Experiment results -- step 86.74

**Step:** falsy-zero check inverts a 0% REJECT into the 10%-NAV default.
**Date:** 2026-08-14. **Verification command:** GREEN, `41 passed`
(was `34 passed` at commit 9d14291e, before the cycle-1 corrections; the header
previously said `37` and was stale by two cycles -- re-measured, not carried
forward).

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
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | 9 -> 38 test functions, 17 -> 62 asserts (AST; see C8 -- an earlier `51` came from grep matching a comment, and `34/55` was stale by the cycle-4 swap-path additions) |
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

**Post-fix: 0.** All four route through `_sizing_pct`.

**CORRECTED after cycle 1 -- my first enumeration of this function was FALSE.**
I wrote *"the default is reachable from ABSENT and only ABSENT"* as prose. The
cycle-1 Q/A **executed** `_sizing_pct` over its state/pct grid and found **three**
default-yielding families, not one:

| family | pre-correction result | why it was wrong |
|---|---|---|
| `(ABSENT, any)` | default | legitimate |
| `(SIZE, pct=None)` | **default** | contradictory -- the state asserts a size was given |
| `(<unrecognised state>, any)` | **default** | worse: it **overrode an explicit 0.0** |

Both residual families were **unreachable in production** (`position_pct_state`
is written at exactly one site from `_verdict.kind`, which only ever holds the
three constants), so this was a **false claim, not a live defect** -- but
criterion 3 demands the set be *derived*, and mine was asserted.

**Fixed at the function**, so the enumeration is now true by construction rather
than by a reachability argument a future caller could invalidate: `SIZE` with no
number and any unrecognised state both **fail closed to 0.0**. The default is
returned for `ABSENT` and nothing else.

**And it is now derived, not asserted**:
`test_default_is_reachable_from_ABSENT_AND_NOTHING_ELSE` sweeps 6 states x 4 pct
values, asserts every default-yielding cell is a genuinely absent verdict, and
carries a vacuity check that the sweep found the legitimate cells at all.

*That sweep failed on its first run and caught a bug in itself*: it treated the
**return value** as a proxy for **which branch ran**, so `(SIZE, pct=10.0)` -- an
explicit 10% verdict -- looked like a default-reach. The probe now excludes
`pct == DEFAULT_POSITION_PCT` and asserts that exclusion holds.

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

**Post-fix populated share -- MEASURED 2026-08-14.**

*(This paragraph previously read "NOT reported as a live number ... the backend has
not been restarted ... not yet in BQ". Every clause of that is now false. It is
REPLACED, not annotated: the cycle-5 Q/A returned a BLOCK because live_check §3 had
been corrected while this file still carried the denial, so the two artifacts of one
step disagreed about the criterion the cycle exists to close. A correction that
lives in only one of two files is the same defect as a correction that merely
accompanies the old text.)*

The restart did happen -- in the prior session, not at this session's end:
commit **`d6a1500a`** (2026-08-14T15:52:58Z) *"session-end backend restart,
verified -- the 86.74 fix is now IN FORCE"*, matching the running process
**pid 85562, started 15:52:08Z**, which is **76 minutes after** the C4 fix commit
`9d14291e` (14:36:20Z). The ordinary scheduled cycle `68925781` then ran
**18:00:00Z -> 19:33:13Z**. No manual cycle and no restart were performed to obtain
this number.

```
BASELINE 2026-07-20..2026-08-13 : total=129  decision=0  risk_level=0  pct=0  ->  0 of 129 (0%)
POST-FIX 2026-08-14             : total=  6  decision=6  risk_level=6  pct=6  ->  6 of 6 (100%)
```

| ticker | `risk_judge_decision` | `pct` | `analysis_date` |
|---|---|---:|---|
| PANW | REJECT | 0 | 18:35:23Z |
| WDAY | REJECT | 0 | 18:36:27Z |
| HPE | REJECT | 0 | 18:37:54Z |
| STX | APPROVE_REDUCED | 2 | 19:02:32Z |
| MRVL | REJECT | 0 | 19:04:26Z |
| NTAP | APPROVE_REDUCED | 2 | 19:32:26Z |

Two distinct decisions and two distinct pcts, so the column carries real per-ticker
content rather than a literal. The unit seam (`TestVerdictIsPersistedPerTicker` +
mutation M3) still holds the regression guard; BigQuery now corroborates it
end-to-end. **NOT CLAIMED:** stability -- n = 6 rows, one cycle. Full detail and the
queries: `live_check_86.74.md` §3.

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

### C7 -- `paper_trades` swept ⚠️ PARTIAL (a cycle-4 "RESOLVED" claim was WRONG)

```
INVERSION (REJECT or 0% yet a BUY executed)  :  1   <- DELL, and only DELL
verdict PERMITTED the buy                    :  0
UNDETERMINED                                 : 14   <- was 33; see the second source below
POSITIVE CONTROL -- DELL detected            : True
```

**UNDETERMINED CORRECTED 33 -> 14 (cycle-6 Q/A).** The 33 was a property of the
ENUMERATION RULE, not of the data: every prior version read verdict-existence from
one source, the `analysis_results` JSON blob. `paper_trades.risk_judge_decision` is
a second, per-trade column, populated on **19 of 34** BUY rows, and it maps
**exactly** onto the 19 rows I had called "truncated / undetermined". The truncation
defect destroyed the blob copy; the per-trade column survived. Inversion stays **1**
— the 3 REJECTs are ~$240 notional against a ~$24k book (~1% of NAV) versus the
$2,392.26 that the 10% default produces. Detail, including the NAV-anchoring
residual: `live_check_86.74.md` §2h.

Population = 34 `paper_trades` BUYs, all time. Join = `ticker` +
`|analysis_date - TIMESTAMP(analysis_id)| < 2s`. Verdict read nested-first then
flat.

**What improved:** the 33 non-DELL BUYs carry a cause decomposition, and the two
halves have DIFFERENT outcomes -- **19** join but their `full_report_json` has **no
`final_synthesis` subtree** (truncated report), *and all 19 have their verdict in
`paper_trades.risk_judge_decision` anyway, so they are NOT undetermined*; **14**
have no row within 2s, nearest 15-20 DAYS away, from the window where
`analysis_results` holds zero rows (2026-04-20..2026-05-15; phase-24.2 F-2
"full pipeline previously evaporated without persistence", closed by 25.A2), and
those 14 are empty in BOTH sources. **Undetermined = 14, not 33.**

**What did NOT improve, and I claimed it had:** I asserted the 19 were a *measured
not-an-inversion* because the `risk_assessment` key was absent, so "no verdict
existed". **Refuted by one query** -- `final_synthesis` is absent in **19 of 19**,
so the report is truncated and a verdict may have existed and simply never been
persisted. "Key absent" supports *not persisted*, not *never existed*. **The 19
reverted to UNDETERMINED and C7 stays PARTIAL**, as the Q/A's CONDITIONAL had it.

**SUPERSEDED IN PART, and in the direction this paragraph's own reasoning
predicted.** The 19 did NOT stay undetermined: `paper_trades.risk_judge_decision`
holds a verdict for **19 of 19** of them, so "not persisted *here*" was exactly
right and the verdict was persisted somewhere else. **Undetermined is 14.** C7
still stays PARTIAL, on the strength of those 14 alone. See the C7 block above and
`live_check_86.74.md` §2h.

Recorded rather than deleted, because I committed and pushed the wrong claim before
running the check that kills it. Full detail: `live_check_86.74.md` §2c.

### C8 -- flag-ON-only blindness closed ✅

```
test functions (ast.FunctionDef test_*) :  9 -> 38    <- the criterion's "9"
assert stmts   (ast.Assert)             : 17 -> 62    <- the real count
'assert ' lines (grep -c)               : 17 -> 64    <- INFLATED by 2
```

**The "9" in the criterion is the TEST count, not the assertion count** -- both
are reported with the rule so a net removal is visible in either denominator and
the two are never conflated.

**RE-MEASURED cycle 5, and the staleness mattered.** This block read `34 / 55 / 56`
until now -- correct when written, stale by exactly the cycle-4 swap-path additions
(34+4=38, 55+7=62). The drift direction was UP, so no net removal was being hidden
*today*; but criterion 8's stated purpose is *"so a net removal is visible"*, and a
stale-LOW denominator is precisely what would let a future removal of up to 4 tests
and 7 asserts pass unnoticed. A number whose job is to be a tripwire has to be
re-derived every cycle, not carried forward. Re-measured with the same AST command
quoted above; the grep inflation is now **2**, not 1 (a second comment line has
since joined line 83).

**CORRECTED after cycle 1.** I first reported `17 -> 51` from `grep -c 'assert '`.
The cycle-1 Q/A re-derived it with the AST and found the grep count **inflated by
one**: line 83 is a *comment* -- `"...They now assert the corrected behaviour and"`
-- so the probe matched its own documentation. Exactly the trap I had explicitly
guarded against for `or 10.0` by using an AST check, and then walked into here.
The AST count is authoritative; the grep-only hit is named above so the
discrepancy is auditable rather than merely asserted.

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

| cell | subject | mutation | tests selected | verdict |
|---|---|---|---:|---|
| M1 | portfolio_manager | restore `if pct:` | 7 | **KILLED** |
| M2 | portfolio_manager | restore `or 10.0` at the sizing seam | 6 | **KILLED** |
| M3 | autonomous_loop | delete the persistence write | 3 | **KILLED** |
| M4 | portfolio_manager | default reachable from UNPARSEABLE | 9 | **KILLED** |
| M5 | signal_attribution | drop RiskJudge row when pct is 0 | 4 | **KILLED** |
| M6 | risk_debate | remove ticker from the log line | 1 | **KILLED** |

A cell whose target text is absent scores `NOT_APPLIED`, never KILLED -- a
no-match `str.replace` looks exactly like success.

**VACUITY HOLE FOUND BY THE CYCLE-1 Q/A AND CLOSED.** Cells were scored
`killed = rc != 0`, and **`pytest` exits 5 when `-k` selects nothing** -- so a
renamed or typo'd selector would have selected zero tests, exited 5, and been
scored **KILLED**: a cell that ran no assertions reported as proof the guard
works. My mutation harness could have certified itself.

Measured directly rather than reasoned about:

```
real selector  TestRejectBindsInBothFlagStates  -> 6 tests selected
bogus selector TestThisNameDoesNotExistAnywhere -> 0 tests selected, pytest exit 5
old rule `killed = rc != 0` would score the bogus cell : KILLED  (VACUOUS)
new rule `killed = rc == 1`                            : NOT killed  (correct)
```

Now every cell proves its selector is live (`--collect-only`) **before** its
verdict is believed, scores `UNSCORABLE` on an empty selection, and reports the
selected count -- shown in the table above, so no cell can be vacuous unseen.

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

**CORRECTED after cycle 1: I said "two"; the derived set is SEVEN.** The cycle-1
Q/A ran the whole tree (55 files) instead of the two suites I had picked by hand,
and found five more pre-existing failures -- **two of them in the very file I
cited**:

```
test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks   <- missed
test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants    <- missed
test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off                         <- missed
test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token        <- missed
test_phase_23_2_6_sector_cap_emit.py::..._skipping_buy_evidence                          <- missed
```

**The substance holds and the direction is unchanged: all seven are
pre-existing**, verified independently by the Q/A, and the code-attributable
failure set is **identical pre and post** 86.74 -- no regression. But **the SET
was hand-narrowed rather than derived**, which is the same
count-the-class-not-your-list failure this project has paid for repeatedly. My
original method (revert `portfolio_manager.py` via `git show HEAD:<path> >`,
re-run, restore byte-identically with sha256 verification) was sound; **the scope
I applied it to was not**.

Several of the newly-surfaced failures are `*_off_*` / `*_defaults_off` tests,
which is worth flagging: **86.74 deliberately changed flag-OFF behaviour** (§4).
Their pre-existence was verified against `HEAD`, so they are not mine -- but the
executor of the queued step should confirm that independently rather than
inheriting my claim.

Queued as its own step rather than fixed inline.

## 6. What I could NOT verify

*(Items 1 and 4 previously claimed the post-fix share was unmeasurable and that
"the running process still holds the pre-fix code". Both are now FALSE and are
replaced. A section titled "What I could NOT verify" that has gone stale is
dangerous in the OPPOSITE direction from an overclaim -- a reader could trigger a
manual cycle or a restart believing a shipped fix is not in force, which is the
exact action the batched-restart policy exists to prevent.)*

1. ~~The post-fix persisted share in BQ (C4)~~ -- **RESOLVED, see §C4 above.**
   Measured 6 of 6 (100%) against the 0-of-129 baseline, from scheduled cycle
   `68925781` on a process (pid 85562, 15:52:08Z) that post-dates the fix.
2. **14 of 34 historical BUYs** (C7) -- undetermined, not clean. **Still open**,
   and structurally so. *(This item read "33 of 34" until 2026-08-15; that number
   came from a single-source enumeration rule. The 19 "truncated report" rows have
   their verdict in `paper_trades.risk_judge_decision` -- 19 of 19 -- so only the
   14 with no `analysis_results` row within 2s are absent in both sources. The
   truncation itself is real and still firing, queued as D5.)*
3. **Why `NTAP` carries `risk_judge_position_pct=4.0` from 2026-07-31** while its
   analysis row persisted no verdict -- untraced, as the step notes.
4. ~~Nothing was driven through the running backend~~ -- **superseded.** The
   running process (pid 85562) holds **post-fix** code, and that is established by
   observation rather than by inference from the commit clock: the live
   `backend.log` carries six `Risk debate complete: ticker=` lines from the 18:00Z
   cycle matching the six BigQuery rows, and `/api/paper-trading/portfolio` answers
   on that same process.
5. **NEW, and previously undisclosed (cycle-5 Q/A, WARN):** criterion 6 asks that
   the RiskJudge contribution appear in `signals_log.factors_json` **for a gated
   buy**. Cycle `68925781` executed **`n_trades=0`**, so **no post-fix
   `signals_log` row for a gated buy exists yet** and C6 rests on the unit seam
   alone. The seam is proven (real `extract_all_signals` emits RiskJudge on DELL's
   nested REJECT/0% shape, with a discrimination control and a killing mutation);
   the end-to-end is not. This is the same shape C4 carried until today, and it is
   recorded here rather than graded silently. It clears on the first post-fix cycle
   that actually places a buy.

---

## 7. CYCLE 1 -- the Q/A rail DROPPED (no verdict) and I fixed what it found

**Run `wf_2e5ddb63-de9` -- 385,807 tokens, 2 agents, both returned empty.** Error:
`subagent completed without calling StructuredOutput (after in-conversation
nudge)`. Per CLAUDE.md an errored/empty rail return is **NO VERDICT, NEVER PASS**,
and it still costs an attempt. **86.74 was NOT graded by that run.**

**Write-first saved the work.** Both agents left records under
`.claude/agent-memory/qa/verdicts/`, and `qa_wip.py` correctly classified them
`INCOMPLETE: ... Treat as EVIDENCE FOR THE NEXT SPAWN ONLY.` Neither transcript
contained a verdict -- one assistant text block each ("I'll start by reading my
operating instructions"), then ~400KB of tool calls and silence. That is the
long-prompt drop shape; my cycle-1 prompt carried 10 criteria plus a 7-point
`extra`, and the next spawn goes out **leaner**.

### What the dropped Q/A CONFIRMED (re-derived independently, not taken from me)

- Immutable command reproduced: `34 passed`, exit 0. Lint clean over a scope
  **derived** from `git diff --name-only 9d14291e^ 9d14291e -- '*.py'` (6 files).
- Harness compliance 5/5, including contract-before-generate by file birth times
  (`research 10:24:44Z < contract 14:19:46Z < first code file 14:23:27Z`).
- `or 10.0` sites **4 -> 0**, re-derived by its own AST sweep. Test functions
  **9 -> 31** at that commit. Mutation matrix **6/6 KILLED**, re-run by the Q/A,
  with its own pre/post sha256 on all four subjects matching.
- The two rewritten tests are an **inversion, not a weakening**: "The new
  assertions are STRICTLY STRONGER: they forbid a buy the old ones REQUIRED."
  Verified from the diff rather than from my summary.
- Working tree dirty set identical to the session-start snapshot; nothing
  unintended in the commit.

### The four defects it found in MY work -- all now fixed

| # | finding | severity | fix |
|---|---|---|---|
| 1 | `_sizing_pct` default reachable from **three** families, not one; an unrecognised state **overrode an explicit 0.0** | WARN (unreachable in prod -> false claim, not live defect) | function now fails closed on both; claim is **derived by an exhaustive sweep** |
| 2 | assert count `51` came from grep matching a **comment** on line 83 | NOTE | AST is authoritative: **55**; grep-only hit named |
| 3 | "two adjacent failures" was hand-narrowed; the derived set is **seven** | WARN (scope honesty) | full set listed in §5 |
| 4 | mutation harness scored `killed = rc != 0`, and **pytest exits 5 on an empty `-k`** -- a typo'd selector would score KILLED | -- (found by its harness-vacuity probe) | selector proven live via `--collect-only`; `killed = rc == 1`; per-cell selected counts published |

Finding 4 is the one that matters most: **my own verification harness could have
certified itself.** It is measured now, not argued -- a bogus selector selects 0
tests and pytest exits 5, which the old rule scored as KILLED.

Fixing these and re-spawning a fresh Q/A on **changed evidence** is the
documented cycle-2 flow, not verdict-shopping: there is no prior verdict to shop
against, because the rail produced none.

---

## 8. CYCLE 3 -- a verdict arrived, and then MY OWN follow-up test found a live defect the verdict had missed

**The Agent-tool fallback returned `CONDITIONAL` (`ok: false`)** after two rail
drops -- the full verdict is transcribed verbatim in
`evaluator_critique_86.74.md` §0. 8 of 10 criteria MET on the evaluator's own
re-derivation; C4 and C7 cap it.

### 8a. Acting on its WARN uncovered a second money defect

The verdict carried an independent WARN:

> *"the AST seam scan matches only `ast.Constant==10.0` -- I verified a
> reintroduction written `or DEFAULT_POSITION_PCT` evades it, and sites
> 824/877/902 sit in `_compute_swap_candidates` which no test drives."*

Both halves are now closed, and **the second half paid off immediately**:

1. **The scan now catches both spellings** (`ast.Constant == 10.0`, `ast.Name`,
   and `ast.Attribute` named `DEFAULT_POSITION_PCT`), with a positive control for
   **each**. The evasion was reproduced first -- old scan vs
   `or DEFAULT_POSITION_PCT` returned `MISSED` -- because a hole should be
   observed open before it is closed.

2. **The untested swap path is now driven behaviourally** -- and doing so
   surfaced a **live defect that all three Q/A passes missed**:

```
the swap path sized a BUY from a 0% verdict: [('NEW', 0.0)]
```

### 8b. The defect: a REJECT could LIQUIDATE a holding

`_compute_swap_candidates` applied the `$50` floor **only inside `if _atomic:`**,
and production runs `paper_atomic_swap_enabled=False`. So on the swap path a
`0%` REJECT emitted:

- a **real SELL** of the displaced holding, and
- a **$0.00 BUY** that is a no-op.

Net effect: **-1 position, with the risk judge's REJECT silently liquidating a
holding.** Same falsy-zero family as the DELL inversion, pointing the opposite
way. Criterion 2 requires a 0% verdict to produce **no order**, and the swap path
is a buy path -- so **C2 was NOT actually met when the verdict was issued.**

**Why the evaluator missed it, stated plainly:** it drove `decide_trades` (the
main path) and marked C2 MET there. It explicitly flagged that sites 824/877/902
were undriven -- it identified the gap and classified it WARN, but did not drive
them itself. The gap was real and its own note is what led me to it.

**Fixed by moving the floor out of the `_atomic` branch**, so neither leg is
emitted when the sized BUY falls below `$50`. **Tightening only:** a legitimate
swap (3% of a $10k NAV = $300) is untouched; this can only ever suppress a
degenerate pair. Three behavioural tests now drive the real function -- including
an anti-vacuity test that **caught its own harness producing no swap at all**
(`paper_swap_max_per_cycle` defaults to `0` and short-circuits the function, so
every assertion would have passed on an empty list).

Verification command: **40 passed**. The four adjacent failures are the same
pre-existing ones from the set of 19; no new regressions.

### 8c. Status -- honest

**This fix landed AFTER the CONDITIONAL verdict, so it is UNGRADED.** The step was
already not closing (C4 and C7 are blockers no cycle-3 work can clear: C4 needs a
restart that policy defers, C7 is a join-coverage limit). A fresh Q/A would
predictably return CONDITIONAL again for the same two reasons, at ~200-400K
tokens, so one was **not** spawned.

**86.74 remains `pending`. The next session must grade the swap fix**, and can
close C4 once the backend restarts on this code.

> **BOTH CLAUSES NOW DISCHARGED (2026-08-14 evening / 2026-08-15).** The swap fix
> was graded: cycles 5 and 6 each graded the FULL step at HEAD, and both re-ran a
> mutation matrix covering the swap sizing sites (`824`/`877`/`902`, which live in
> `_compute_swap_candidates`) with no survivors. C4 is closed -- the restart landed
> `d6a1500a` 15:52:58Z and scheduled cycle `68925781` produced **6 of 6 (100%)**
> against the 0-of-129 baseline. Nothing here is outstanding; do not re-grade the
> swap fix as though it were ungraded.

---

## 9. CYCLE 4 -- the swap fix was graded, and the grade caught a PROXY assertion

The Agent-tool Q/A graded commit `76ac89ee` (scoped to it, not a re-grade of the
step): **CONDITIONAL**. Working record:
`.claude/agent-memory/qa/verdicts/verdict_wip_86.74__20260814T153803Z.md`.

### 9a. What it CONFIRMED -- re-derived, not taken on my word

- **The defect is real.** With the floor block excised in-memory, the real
  `_compute_swap_candidates` returns
  `[('SELL','OLD',None), ('BUY','NEW',0.0)]` on a 0% REJECT, against `[]` as
  shipped.
- **Tightening only.** 3% -> `$300` and ABSENT -> `$1000` both fire unchanged.
  The suppression set is exactly `nav*pct/100 < 50`, which is **parity with the
  identical `if buy_amount < 50` the MAIN buy path already carries at `:536`** --
  so this aligns the two paths rather than inventing a constraint. The atomic
  path is provably untouched: `:932` runs *after* `min(buy_amount, cash+freed)`,
  so the new check at `:919` can never reject what the later one would pass.
  **It found no swap that should fire and is now suppressed.**
- **C2 now holds behaviourally on the swap path.**
- **No new regressions** -- the four adjacent failures reclassified against the
  **parent** module, and the zero-buy-gap failure is `Expected 2 swap SELLs, got 1`
  with its swap BUYs at `$1000`, twenty times the floor: unrelated by mechanism.

### 9b. What CAPPED it -- I asserted a proxy for the harm

> *"the commit's own claim is 'emitting neither leg, so the SELL cannot orphan',
> and that property has no guard. I mutated to suppress ONLY the BUY append,
> leaving the SELL to orphan -- the exact net -1 harm -- and all three new tests
> PASSED (rc 0). The assertion filters `o.action == 'BUY'`, so the half that
> causes the loss is unmeasured."*

**The harm is the orphaned SELL, not the zero-dollar BUY**, and I asserted on the
BUY subset. Fixed exactly as named: assert the **whole order list** is empty, plus
a second test naming the orphaned SELL so a regression prints its own harm.

**Proven, not asserted:** the orphan mutant **SURVIVED** the old assertion
(rc 0, 3 passed) and is **KILLED** by the new one (rc 1). Added as permanent cell
**M7**, which needs **two** edits -- delete the early floor, re-apply it after the
SELL append -- because emitting the SELL and suppressing the BUY are two lines
apart. Matrix now **7/7 KILLED**, control green first, byte-identical restore.

### 9c. A comment of mine misdescribed production, in the dangerous direction

I wrote that `paper_swap_max_per_cycle` *"defaults to 0 and short-circuits the
whole function"*. **Production defaults it to 2** (`settings.py:378`) with
`paper_swap_enabled=True` (`:368`) -- **the swap path is LIVE BY DEFAULT.** The 0
is the `getattr` fallback for an attribute absent from my test's `SimpleNamespace`
stub. A reader could have concluded the swap path was dark when it is live, which
makes the orphan-SELL harm a **live** concern rather than a hypothetical.
Corrected in place.

The Q/A also checked whether I had over-configured my way into an unreachable
scenario and found the opposite: all four explicit values **match the production
defaults exactly**, so the scenario is production-representative -- *"stronger
evidence of live reachability than your own framing gave it."*

### 9d. Correction to commit `cba60c0b`'s message

That message states **"42 passed"**. The measured count is **41**
(`python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q`). The
commit is no longer the tip (the changelog hook committed above it) and a peer
session is active, so it was **not** rebased; **41 is the correct figure** and this
line supersedes it.

### 9e. Status

**86.74 still does NOT close.** This grade was scoped to the swap commit; C4 and
C7 are untouched by it and remain the step's blockers.

---

## 10. CYCLE 5 -- the cycle-4 work graded **PASS**, with two WARNs

Agent-tool Q/A, scoped to `cba60c0b^..HEAD`. Working record:
`.claude/agent-memory/qa/verdicts/verdict_wip_86.74__20260814T155443Z.md`.

**`ok: true, verdict: PASS`** — and it verified more than I had reported. It
rebuilt M7's two edits **in-memory from `MUTATIONS`** and drove the real function:

```
0% REJECT   -> [('SELL','OLD',None)]                              a GENUINE orphan
legitimate 3% -> [('SELL','OLD',None), ('BUY','NEW',300.0)]        FULL pair intact
```

That second line is the part I had not done: it proves **M7 isolates the orphan
rather than collapsing the harness**, so the kill is *attributable*. It also
reviewed the new multi-edit normaliser for a fail-open and confirmed it scores
`NOT_APPLIED` and continues (`mutation_matrix_86_74.py:207-211`) — **fails closed**.

### WARN 1 -- a count that did not reproduce, inside a comment written to fix a count that did not reproduce

My corrected comment said *"These four values"* while **five** kwargs are set, and
said `_settings()` *"OMITS them"* when `paper_swap_enabled` is **present-but-False**
at `test:26`. Both verified and fixed; the comment now distinguishes the two
reasons:

```
kwargs actually passed : 5
PRESENT in base (must be OVERRIDDEN) : ['paper_swap_enabled']
ABSENT  from base (must be SUPPLIED) : [max_per_cycle, min_delta_pct, churn_fix, atomic]
```

Now **machine-checked** rather than re-asserted — the assertion above parses the
call site and the `_settings` base and fails if either count drifts. Writing an
inaccurate comment *while fixing an inaccurate comment* is its own lesson.

### WARN 2 -- the evaluator disclosed the limit of its own grade

> *"I did NOT re-measure those numbers \[the C7 decomposition]. My item-3 grade
> rests on internal consistency and the soundness of the reasoning. Acceptable
> ONLY because the claim now standing is the CONSERVATIVE one — an unverified
> 'still undetermined' creates no risk. It would NOT be acceptable for a
> 'resolved'."*

Recorded verbatim because it is exactly right, and it means **the C7 decomposition
(19/14/0) is still only MY measurement.** The next session must re-derive it before
building on it.

> **DISCHARGED -- it is no longer a single-author measurement.** The cycle-6 Q/A
> wrote its own query from the stated enumeration rule and reproduced the split
> exactly (`34 / 1 inversion / 0 permitted / 19 truncated / 14 no-row / 0
> fs-present-no-risk-assessment`, summing to the full population, DELL positive
> control detected), and additionally ran a discrimination control Main had not
> (inverting the INVERSION predicate moves DELL into PERMITTED). It was re-derived
> a third time on 2026-08-15 against BigQuery, which also established that all 19
> truncated rows carry a verdict in `paper_trades.risk_judge_decision` -- the
> finding that cut undetermined from 33 to **14**. Three independent derivations
> now agree; re-deriving a fourth time is not required before building on it.

### Two scope corrections it made against me

1. **`cba60c0b..HEAD` EXCLUDES `cba60c0b`** — and the orphan guard and M7 live *in*
   that commit, so the range I gave was docs-only. It graded `cba60c0b^..HEAD`
   instead of grading the wrong thing.
2. **The tree MOVED during its evaluation** — `38ba13ad` and `a33a5117` (the C7
   retraction) landed after the range I gave, so my tasking message was stale
   against the tree. It graded `HEAD` and said so.

Both are mine. The second is the freeze-the-tree rule again: I retracted C7 *while*
an evaluator was reading. It was safe only because the movement made the claim
strictly **more conservative** and was fully recorded — but "safe because of which
direction it happened to move" is luck, not method.

### Standing

**This PASS is scoped to the cycle-4 work and does NOT close 86.74.** C4 is open,
C7 is PARTIAL — exactly as the earlier CONDITIONAL had them.

---

## Cycle-7 preparation -- a SEMANTIC sweep of every 86.74-touching artifact

**Why this section exists.** Cycles 5 and 6 each returned CONDITIONAL, and neither
found a code defect. Both capped on **prose**: a stale claim in a file that had not
been swept. Cycle 6's cap was `goal_next_2026-08-15.md`, the artifact BINDING on the
next session. The failure mode both times was the same and it is worth naming
precisely: **after fixing a claim I built the "did I get them all" probe out of the
exact phrasings I had just edited**, so it could only rediscover what was already
fixed.

**Method this time -- enumerate the CLAIM, not the wording.** I wrote down what
CHANGED, then triaged every artifact by meaning against it, by hand:

| # | The claim, in its current form |
|---|---|
| 1 | **C4 is SATISFIED.** 129 rows 0/0/0 (2026-07-20..08-13) vs **6 of 6 (100%)** on 08-14, from scheduled cycle `68925781` on post-fix `pid 85562`. No manual cycle, no second restart. |
| 2 | **C7 undetermined is 14, not 33.** `paper_trades.risk_judge_decision` is a second per-trade verdict source populated on 19 of 34 BUYs, mapping exactly onto the 19 "truncated" rows. **Inversion stays 1** (DELL). C7 stays PARTIAL on the strength of the 14. |
| 3 | **The restart is DONE** (`d6a1500a`, 15:52:58Z). "Committed but NOT IN FORCE / pid 27945" is false. |
| 4 | **The nested-first resolution is UNCONDITIONAL**, not flag-gated; all four sizing sites route through `_sizing_pct`. |

**Triage rule applied.** A statement is STALE if a reader acting on it *today* would
take a wrong action or record a wrong number. A statement is HISTORICAL-OK if it sits
in a verbatim verdict transcript, an append-only log, or a chronological session block
**and carries an in-place supersession marker at the point of the claim**. Verbatim
Q/A verdicts (`evaluator_critique_86.74.md`, `.claude/agent-memory/qa/verdicts/*`) and
`handoff/harness_log.md` were deliberately left untouched.

### Independent re-measurement that settled the arithmetic

Re-ran the enumeration rule against BigQuery on 2026-08-15 (third independent
derivation; the cycle-6 Q/A was the second):

```
AR_verdict_known                    :  1   pt_verdict populated:  0   (DELL 2026-08-13)
UNDET_truncated_no_final_synthesis  : 19   pt_verdict populated: 19   <- ALL recoverable
UNDET_no_row_within_2s              : 14   pt_verdict populated:  0   <- genuinely undetermined
                                      --
                                      34   = full UPPER(action)='BUY' population
```

`paper_trades.risk_judge_decision` distribution over the 34: 15 `APPROVE_REDUCED`,
3 `REJECT`, 1 `APPROVE_HEDGED`, **15 empty** -- and 15 empty = the 14 + DELL, which
closes the arithmetic. The 14 fall on exactly four dates, all inside the
`2026-04-20..05-15` window where `analysis_results` holds zero rows:
**04-26 (9: CIEN, DELL, GLW, INTC, LITE, ON, SNDK, TER, WDC), 04-27 (1: COHR),
04-28 (3: GEV, KEYS, MU), 05-01 (1: FIX)**.

### What the sweep FOUND -- 12 stale claims across 5 files

Nothing was found in code. Every finding is prose.

| file | what was stale | why it mattered |
|---|---|---|
| `live_check_86.74.md` | §2 headline fenced block still read `UNDETERMINED : 33`; §2a said the other 33 "cannot be answered"; §2b's table called all 19 truncated rows "unrecoverable"; §2g said "the 33 remain unrecoverable" | The correction lived at **§2h only** -- it ACCOMPANIED the stale text instead of REPLACING it, so §2's own summary contradicted its own §2h |
| `experiment_results_86.74.md` | §6 "What I could NOT verify" item **2** still said "33 of 34"; §C7's prose said "the 33 now carry a cause decomposition"; the C7 retraction's conclusion still read "the 19 revert to UNDETERMINED" | Items 1 and 4 of that same list HAD been corrected -- a partial sweep of a single list. The honest-limits section overstated the unknown by 19 rows |
| `experiment_results_86.74.md` | "the next session must grade the swap fix"; "the C7 decomposition is still only MY measurement -- the next session must re-derive it" | Both discharged (cycles 5+6 graded the full step; three derivations now agree). Directed redundant work |
| `queued_defects_from_86.74.md` | D2 tail asked "why **33** don't join" and set the deliverable as "resolve the 33"; its date list was labelled "Undetermined", summed to **31**, and its recoverable tail listed 17 of the actual 19 | **Three different numbers in one section** (14 / 33 / 31). Forward-looking: it tasks an executor with no session context |
| `day_report_2026-08-14.md` | §3 items 2/3/4, §4 "Pending restart", §5 D2/D3, §9's "C4 still will not be measurable today", §8a's labelled **"Corrected position: 33 UNDETERMINED"** | Chronological sections with **no in-place marker**, while §9a and Session 4 contradict them 150+ lines later. §8a is the worst: the sentence a reader trusts is literally labelled "Corrected position" |
| `.claude/agent-memory/researcher/project_risk_gate_veto_86_74.md` | headline: "the `or 10.0` idiom is at FOUR sites and the approved fix guards ONE" | A **memory**, recalled in future sessions. Measured today: all four sites now call `_sizing_pct`; raw `or 10.0` survives only in comments and `DEFAULT_POSITION_PCT`. A future researcher would have believed three sites were still unguarded |

### What was deliberately NOT changed

- `evaluator_critique_86.74.md` -- every `33` in it is inside a **verbatim Q/A
  verdict** (JSON `reason` / `notes` / `constraint`). Historical verdicts stand.
- `.claude/agent-memory/qa/verdicts/*` -- write-first crash-survival records.
- `handoff/harness_log.md` -- append-only cycle history.
- `.claude/masterplan.json` -- immutable criteria are never edited.
- `contract_86.74.md:71` ("the brief recorded this as UNMEASURED") -- resolved in
  place by §2b immediately below it.
- `settings.py:348` / `:352` -- verified already carrying the phase-86.74
  corrections; no further edit needed.

### Verification

Immutable command re-run after every edit above: **41 passed, exit 0**. No
production code was touched by this sweep -- it is documentation-only, which
`git diff --stat` on the commit confirms.

### The honest limit on this sweep

I can state the method and the findings; **I cannot prove absence.** Two cycles
running have each found prose defects in files I had not thought to sweep. What is
different this time is the enumeration rule: I triaged by MEANING against a written
claim-set rather than by grepping strings I had just edited, I swept the
forward-looking artifacts FIRST, and I re-measured the underlying numbers
independently rather than propagating them. That raises the floor; it does not
close the question.

---

## C6 re-measurement (2026-08-17, Main): the missing row is STARVED, not broken

Re-ran the C6 probe three days after the cycle-7 grade:

```
SELECT DATE(created_at), ticker, LENGTH(factors_json),
       STRPOS(factors_json,'RiskJudge')>0
FROM `sunny-might-477607-p8.financial_reports.signals_log`
WHERE created_at >= TIMESTAMP('2026-08-14') ORDER BY created_at LIMIT 50
-- 1 row: 2026-08-14  $CYCLE  fj_len=19  RiskJudge=False
```

Population rule: every signals_log row since the fix went live; the one row is
the same cycle-marker the cycle-7 Q/A measured. **Zero gated buys have occurred
in the three days since the fix** -- C6's missing artifact cannot exist until
the book produces one, and the drought's cause is owned by steps 86.38/86.47/
86.69 (the 86.69 empty-HOLD regression starves the BUY funnel upstream).
Decision under the operator's product-vs-evidence directive: no evaluation is
spawned against a criterion starved of its input (it would fail on C6
unchanged, ~200K tokens for no information). 86.74 stays open; its closing
rail-bound evaluation fires when the first organic gated buy writes the row.
