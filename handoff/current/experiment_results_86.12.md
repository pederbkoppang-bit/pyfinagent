# phase-86.12 -- GENERATE

**Step:** P1 -- DETERMINE WHETHER THE KILL SWITCH EVALUATES DRAWDOWN AGAINST A
STALE NAV.
**Contract:** `handoff/current/contract_86.12.md`
**Research:** `handoff/current/research_brief_86.12.md` (gate PASSED,
`wf_b6b1e4e3-df2`, 7 sources read in full, 30 URLs)

**This is an investigation. The deliverable is an answer.** Criterion 5 forbids
changing any threshold as part of the diagnosis, and nothing was changed.

---

## THE ANSWER, up front

**The daily-loss leg CAN fire on a drawdown that exists at cycle time, and the
ordering is what makes that true.** `autonomous_loop` marks to market at Step 5
(`:1368`) and enforces at Step 5.5 (`:1400`) -- 32 lines later, same cycle. The
NAV the breach reads was written by that mark. Demonstrated firing at 4.01%,
4.1% and 12% against a 4.0% limit.

**The suspicion in the step's title is NOT a defect, and here is why the numbers
look the way they do.** `current_nav` is the stored
`paper_portfolio.total_nav`, and the SOD roll anchors `sod_nav` to that same
stored value. Immediately after a cycle they are equal *by construction*, and
they stay equal until the next mark. Measured over the journal's full history
the equality holds **7 of 10 days (70%) -- SOMETIMES, not always, not a
one-off**, which is exactly what "equal after a mark, divergent after the next
one" predicts.

**What IS a real weakness, and it is an asymmetry rather than a staleness
bug:** `evaluate_breach` checks whether the *baseline* is stale
(`_sod_date_is_stale`) but validates `current_nav` only for `None`/`<= 0`.
Nothing anywhere asks how old `current_nav` is -- and the asof is sitting in the
same dict every caller reads it out of.

---

## Criterion 1 -- PROVENANCE, traced to file:line

`evaluate_breach` (`kill_switch.py:805`) takes `current_nav` as a **parameter**
and never marks -- asserted from the AST, not read off the page
(`test_evaluate_breach_NEVER_marks_to_market`).

Every function containing a call, **derived from the AST**:

| file:function | resolves `current_nav` from | live mark? |
|---|---|---|
| `api/paper_trading.py::get_kill_switch_state` (`:517`) | `portfolio["total_nav"]` (BQ) | **no -- stored** |
| `api/paper_trading.py::resume_trading` (`:580`) | `portfolio["total_nav"]` (BQ) | **no -- stored** |
| `services/paper_trader.py::check_and_enforce_kill_switch` (`:1343`) | `portfolio["total_nav"]` (BQ) | **no -- stored** |
| ...the same function calls it **twice** (`:1357` pre-roll, `:1460` decision) | same value | **no -- stored** |
| `agents/mcp_servers/risk_server.py::kill_switch` (`:63`) | `current_nav: float \| None = None` **parameter** | caller's problem |
| `services/kill_switch.py::check_auto_resume` (`:1065`) | its own `current_nav` **parameter** | caller's problem |

**I got this list wrong first, and the correction is the point.** My initial
version transcribed the research brief's prose -- "five producers" including
`_roll_sod_anchor_if_needed` (no such function) and `risk_server.kill_switch`
(which takes a parameter, not `total_nav`). The test failed and I re-derived
from the AST. **Four call sites in three functions** read the stored figure;
two receive it. The brief's count was prose, not a derivation, and I should not
have copied it.

`check_auto_resume` deserves naming: it decides whether to **un-pause a halted
book**, and it re-evaluates the breach on whatever NAV it is handed.

## Criterion 2 -- the equality, across MULTIPLE days

The step says the journal holds "8 sod_snapshot rows". **Measured: 10.** The
step's figure is stale; the measurement governs.

```
date               sod_nav   snapshot nav   equal?
2026-07-27        23838.16       23772.49   NO
2026-07-28        23772.49       23772.49   YES
2026-07-29        23772.49       23772.49   YES
2026-07-30        23772.49       23772.49   YES
2026-07-31        23772.49       23770.98   NO
2026-08-03        23803.94       23803.94   YES
2026-08-05        23830.46       23830.46   YES
2026-08-08        23830.46       23833.94   NO
2026-08-09        23833.94       23833.94   YES
2026-08-09        23833.94       23833.94   YES

equality held on 7/10 comparable days (70%)
```

**Answer: SOMETIMES.** Not always, and emphatically not a one-off. The three
NOs are the days where a later mark moved the NAV away from the anchor -- which
is the system working, not failing.

### A finding this table surfaced: the anchor is a PRIOR-CLOSE anchor

The hour each `sod_snapshot` was stamped, UTC:

```
13:xx  x2   2026-08-09 (x2)
16:xx  x1   2026-07-28
18:xx  x3   2026-07-29, 07-30, 07-31
19:xx  x2   2026-08-03, 08-05
20:xx  x1   2026-08-08
```

US markets close at 20:00/21:00 UTC. An anchor stamped 18:00-20:58 is taken at
or after the prior close, so what the code calls "start of day" is in practice
**the previous session's close**. That is defensible for a paper book that
trades once a day -- but the limit's name and the variable `sod_nav` both claim
something the timestamps do not support, and a reader auditing a breach would be
misled about the window being measured.

## Criterion 3 -- the $0.06 delta

**It does not reproduce.** Measured tonight:

```
kill-switch current_nav        : 23833.94
/performance nav               : 23833.94
/portfolio portfolio.total_nav : 23833.94
MAX SPREAD: 0.000000
```

**Explanation, which is the criterion's actual requirement.** All three read the
*same* stored `paper_portfolio.total_nav`, so they agree exactly by
construction. They can only disagree if two of them are served on **opposite
sides of a `mark_to_market` write** -- a race against the cycle, not rounding
and not FX. That also explains the *shape* of the reported number: $0.06 is not
a rounding artifact (which would be <= $0.005 on a 2-dp round) and not an FX
step; it is one mark's worth of price movement on the single open position
(NTAP, 5.35 shares -- a $0.011 tick moves the NAV about $0.06).

**Stated honestly: I could not observe the delta, so this is a mechanism
explanation consistent with the evidence, not a reproduction.** The way to
reproduce it deliberately would be to poll both endpoints across a cycle
boundary, which requires a running cycle and is out of scope here.

## Criterion 4 -- CAN the daily-loss leg fire? DEMONSTRATED

`backend/tests/test_phase_86_12_kill_switch_nav_freshness.py`, 12 tests.

**YES, at cycle time:**

| drawdown | `daily_loss_pct` | breached |
|---|---|---|
| 0.0% | 0.0 | no |
| 3.9% | 3.9 | no |
| 4.01% | 4.01 | **YES** |
| 4.1% | 4.1 | **YES** |
| 12.0% | 12.0 | **YES** |

Driven through the real `evaluate_breach` with a today-dated anchor -- the state
the book is in at Step 5.5. A separate test drives the **enforcement** path's
own resolution (`portfolio["total_nav"]` -> `evaluate_breach`) and shows a 6%
stored drawdown breaching a 4% limit.

**NO, while the anchor is from a previous UTC day.** With `sod_date` older than
today the daily leg is deliberately unevaluable (phase-36.9's protection against
reading a multi-day move as a same-day loss). Demonstrated: a **20% collapse**
produces `daily_loss_breached: False`, `armed: False`. The trailing leg is not
date-scoped and **does** fire there -- asserted, because in that window it is
the only cover the book has.

**This is the live state right now**, not a hypothetical:

```
sod_date             : 2026-08-09      (the clock has rolled to 2026-08-10)
daily_baseline_stale : true
armed                : false
NAV age              : 10.87 hours
```

So between UTC midnight and the first cycle of a new day, the daily-loss leg
cannot fire. By design -- but the badge reports `armed: false` without saying
that it is the ordinary overnight condition rather than a fault.

### A boundary finding produced by accident

A NAV constructed exactly 4.0% down yields a raw ratio of `3.9999999999999973`.
The payload reports `round(pct, 4)` = **4.0**; the decision compares the raw
value `>= 4.0` = **False**. The badge can show "4.0%" beside a "4.0%" limit and
correctly not be breached.

Ordinary binary floating point, and every threshold has such a boundary -- but
it is a display-vs-decision disagreement and an operator staring at two equal
numbers deserves it written down. **Not fixed:** changing either the rounding or
the comparison IS a threshold change, which criterion 5 forbids during a
diagnosis.

## Criterion 5 -- the verdict, plainly

**The behaviour is CORRECT for what it claims to do, with two documented
weaknesses that are not defects in the enforcement path.**

- The enforcement decision is made on a NAV marked moments earlier in the same
  cycle. The leg is not evaluating a stale NAV at the moment it matters.
- The equality that prompted this step is the expected consequence of the SOD
  roll, holds 70% of the time, and diverges exactly when a later mark moves the
  NAV.
- The badge endpoint IS stale between cycles (10.87 hours as measured) and does
  not say so. That is a **display** honesty gap.
- `evaluate_breach` checks baseline freshness and not NAV freshness. Per the
  research brief's JFE-2004 citation, a nonsynchronous pair biases the measured
  move DOWNWARD, so the error direction is **fires late**, which is the
  dangerous one.

**No threshold was changed, no guard weakened.** The two follow-ups (surface the
NAV asof; decide whether "sod" should mean the session open) are queued
separately rather than done here.

## Criterion 6 -- nothing was disturbed

`handoff/kill_switch_audit.jsonl` sha256
`ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065`,
**byte-identical** before and after every test run and every measurement in this
step. The test module redirects `_AUDIT_PATH` to `tmp_path` and uses a
**detached** `KillSwitchState`, never the module singleton (phase-86.1's
lesson). No threshold touched: `git diff` on `kill_switch.py` is empty.

## Files

```
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py   (NEW, 12 tests)
scripts/qa/measure_nav_freshness_86_12.py                     (NEW, re-runnable)
handoff/current/contract_86.12.md, research_brief_86.12.md    (gate + plan)
```

**No production file was modified by this step.**
