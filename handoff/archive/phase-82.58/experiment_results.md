# Experiment Results -- phase-82.58

**Step:** 82.58 (P1) -- the cost-budget hard-block's only alarm has never fired.
**Date:** 2026-08-06. **Cycle:** 2 (cycle-1 Q/A returned CONDITIONAL; both WARNs cured -- see §3 and §1).
**Contract:** `handoff/current/contract_82.58.md`
**Research brief:** `handoff/current/research_brief_82.58.md` (`gate_passed: true`,
audit-class, dry after 6 rounds / 2 dry, 9 sources read in full, 53 URLs, 20 files)

---

## 1. What changed

| File | Change | Lines |
|------|--------|-------|
| `backend/services/observability/spend.py` | `detail=` -> `details=`, `severity="P2"` -> `"P1"`, with the three-blocker rationale in-code | +16 / -2 |
| `backend/tests/conftest.py` | Slack-egress guard installed at import time (see §4) | +40 / -0 |
| `backend/tests/test_phase_82_58_spend_alert_delivery.py` | new -- 10 tests | 418 (new) |
| `.claude/masterplan.json` | queued 82.59 / 82.60 / 82.61 -- **and see the disclosure below** | +77 / -0 |

Figures from `git diff --numstat` and `wc -l`, run as the last action before
writing this file -- not carried from an earlier draft.

### Disclosure: `.claude/masterplan.json` is shared, and I churned it

The Q/A caught that my table above described this file as only my three steps.
Two things were undisclosed, and one of them was my own damage:

1. **A concurrent session's step `4000.10` is in the same file.** Git cannot
   stage part of a file, so committing my three steps necessarily commits
   theirs. Parsed delta: added `['4000.10','82.59','82.60','82.61']`,
   **removed: none**.
2. **I re-serialized the whole file with `ensure_ascii=True`**, escaping every
   em-dash to `\uXXXX` across all 1141 steps -- 155 lines of pure encoding
   churn that had nothing to do with this step. **Fixed**: re-written with
   `ensure_ascii=False`, which took the diff from `+231/-154` to `+77/-0`.
   The remainder is additions only.

The ~20 untracked `phase-4000.3` artifacts also sitting in the tree are **not**
mine and are NOT committed here -- this step commits with an explicit pathspec,
never `git add -A`.

## 2. Verbatim output of the immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_58_spend_alert_delivery.py -q
..........                                                               [100%]
10 passed, 8 warnings in 85.76s (0:01:25)
```

*(The 8 warnings are `SyntaxWarning: invalid escape sequence` raised while the
call-site sweep `ast.parse`s unrelated repo files. They are properties of the
files being scanned, not of this suite.)*

## 3. Mutation matrix -- 7 mutants, all killed

Every mutant targets the **production** call site or the **production** latch,
never a test helper. Restores write back captured bytes rather than
`git checkout`, so a concurrent session's edits cannot be clobbered.

```
baseline (unmutated):
  rc=0 GREEN

M_A revert details= -> detail=                 DIED  (criterion 1 dies on the original TypeError defect)
M_B revert P1 -> P2                            DIED  (criterion 1+2 die on the deduper/webhook drop)
M_C severity -> P3 (severity-only guard)       DIED  (criterion 2 dies on the severity literal alone,
                                                      independent of delivery)
M_C2 severity -> P2 (severity-only guard)      DIED  (the original P2 defect dies on it too)
M_D break the once-per-process latch           DIED  (the alert-fatigue latch is actually asserted)
M_E alert on the HEALTHY path too              DIED  (the negative control catches an always-firing guard)
M_F blind the sweep                            DIED  (the recall test catches an instrument that sees nothing)

=== 7 mutants died, 0 survived ===
```

This licenses exactly "these 7 mutants died". It does not license "no
survivors" in general.

M_C and M_C2 are run under a `-k` selector that matches **only** the
severity-only guard, so their kills are attributed by construction rather than
by my assertion about which line failed.

### M_D survived the first run, and that was a real defect in my own guard

On the first matrix run **M_D SURVIVED**. `test_the_alert_fires_once_per_process`
drove three degradations and asserted one POST -- but with the `_ALERTED` latch
mutated to `if True:`, the **deduper** still suppressed calls 2 and 3 on its own
(P1 takes the critical branch, which re-fires only after `repeat_hours`). The
test passed either way. It was asserting a property the deduper guarantees while
its docstring credited the latch: a guard that could not fail.

Fixed by resetting the deduper between degradations, which removes the confound
and leaves the latch as the only thing that can suppress. M_D then died. The
reason is recorded in the test's own docstring so the reset is not mistaken for
noise and removed later.

### Cycle 2: I wrote an unfalsifiable guard while auditing for unfalsifiable guards

The cycle-1 Q/A returned **CONDITIONAL** on this (WARN, not BLOCK -- all four
criteria were already MET). It is worth stating plainly because it is the same
shape as M_D, one layer up.

My matrix credited M_C's kill to *"criterion 2 is bound to the live critical
set"*, pointing at `assert severity in live`. **That assertion cannot fail.**
With `slack_webhook_url` empty, `alerting.py:210-224` routes to
`_bot_token_fallback` **only** when the severity is already critical -- so any
POST the fixture captures necessarily carries a deliverable severity. The
severity check was reading a value that had already been filtered by the very
property it claimed to test. M_C really died on `assert captured_posts`, the
delivery guard.

The criterion's literal wording was satisfied either way, which is exactly why
this was easy to miss: the guard was *correct*, *green*, and *load-bearing for
nothing*.

Fixed by adding `test_the_call_sites_severity_literal_is_deliverable_independently_of_delivery`,
which AST-reads the severity literal out of the production call site and checks
it against the live set **with no POST in the picture** -- so it fails on a
severity regression even if delivery broke for an unrelated reason. M_C and
M_C2 now die on that guard alone.

## 4. A hazard the fix creates -- guard landed FIRST

Repairing `spend.py` **arms a live Slack POST inside the existing test suite**.
`test_phase_75_5_1_spend_metric.py:294` and `test_phase_75_llm_rail.py:582`
already drive `_record_degradation` for real; measured:
`SLACK_BOT_TOKEN present: True | starts xoxb: True | len: 59`, webhook length 0.
Post-fix the P1 alert routes to `_bot_token_fallback` and posts to the
operator's real channel from a routine `pytest` run.

So the guard went in **before** the `spend.py` edit, and the suite was not run
against a repaired `spend.py` until it was in place. Proven to work, both
directions, before proceeding:

```
OK blocked: phase-82.58 test guard: refusing a live Slack POST from the test suite (url='htt ...
OK non-slack host reached the real urlopen (network error is fine): URLError
```

Scoped to `slack.com` only -- it is not a general network jail.

**No real page was sent.** Delivery is asserted at the socket seam with a dummy
token. Sending an actual message to the operator's channel would be an
outward-facing side effect that this step does not have authorisation for.

## 5. Regression + lint

```
$ python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py \
    backend/tests/test_phase_75_llm_rail.py \
    backend/tests/test_phase_82_54_cost_budget_columns.py -q
81 passed, 1 warning in 21.22s
```

The 82.54 guard (`:344-368`) watches this exact defect and requires an open step
naming `spend.py` + `detail` while the code is broken. It has
`if not still_broken: return`, so it degrades to a clean no-op now the fix has
landed -- confirmed green above.

**Lint:** `ruff check` reports 3 `BLE001` in `spend.py`. All three are
**pre-existing** and are the deliberate fail-open excepts. Measured rather than
asserted -- same rule set against the HEAD versions of the same files:

```
HEAD versions, BLE/F/E9 errors: 3
CURRENT versions, same rules: 3
```

My two new files contribute zero.

## 6. Criterion 4 -- the sweep, and the count the step got wrong

The step asserts "the ONLY malformed call site of **15** audited repo-wide".
**The denominator is wrong.** Derived twice, independently:

```
by area: {'backend': 28, 'tests/tests': 5}
TOTAL: 33
```

My sweep found **28** under `backend/`; the research gate found **33**
repo-wide; the difference is exactly the **5** sites in repo-root `tests/`.
Both are right for their scope. **The numerator is correct: exactly 1
signature mismatch**, which this step fixes.

The in-test sweep is **import-resolved**, not bare-name. The gate measured that
matching on the bare attribute name yields 65 candidates for 3 real hits
(colliding with `csv.writer`, `yfinance.history`, `json.loads`,
`numpy.percentile`). The derived set is asserted non-empty **and** recall-tested
against three known-present anchors, because "found no mismatches" and "the
sweep is broken" are indistinguishable from the outside.

**Further mismatches -- queued, each verified by me before queueing:**

| Step | Finding | How I verified it |
|------|---------|-------------------|
| **82.59** (P1) | `assistant_lifecycle.py:181` missing required `set_suggested_prompts` + passes non-existent `client`/`set_status`; `:188` unexpected `client`. Production-wired at `app.py:33`. | runtime `inspect.signature().bind()`, verbatim errors captured |
| **82.60** (P2) | 9 red tests, `TypeError: trigger_thursday_batch() got an unexpected keyword argument 'log_fn'` | ran them: `9 failed in 0.07s` |
| **82.61** (P2) | 11 remaining production sites carry a severity undeliverable while the webhook is empty; `drawdown_alarm.py:154` is non-literal | the AST sweep above; classified, not blanket-raised |

## 7. Corrections I owe from earlier in this session

**The budget caps are 25.0 / 300.0, not 5.0 / 50.0.** I read 5.0/50.0 from the
`getattr` fallbacks at `llm_client.py:437-438` and stated them as the live caps.
Those fallbacks are unreachable -- the settings attributes exist, so the real
values come from `settings.py:392-393`. The conclusion is unchanged
(`0.0 >= 25.0` is still False, so the block still cannot trip on the fail-open
value) but a fixture pinning 5.0/50.0 would have exercised a dead branch.

**The research gate's first run researched the wrong step.** I passed `args` as
a JSON string instead of an object; the rail script's `catch (_e) { a = {} }`
swallowed the parse failure and the agent received `step UNSPECIFIED`, so it
derived its own objective and researched phase-85.1 (~168K tokens). That script
now throws instead of defaulting. The brief was not discarded -- 85.1 is a real
pending P1 step, so it is preserved at
`handoff/current/research_brief_85.1.md` for whoever works it. The irony is
recorded because it is the point: **a silently swallowed exception in my own
tooling, which is the exact defect class this step exists to fix.**

## 8. What I did NOT do

- **No live page sent** (§4).
- **No live-system re-curl**, because there is no live surface: `spend.py`'s
  degradation state is exposed only by `spend_guard_status()`, which the gate
  measured has **ZERO non-test callers** repo-wide -- no endpoint, no frontend
  tile, no cron. That absence is precisely why this alert matters.
- **No change to the deduper, `_CRITICAL_SEVERITIES`, or any other call site.**
  P2 -> P1 at this call site only; making P2 globally deliverable would page for
  8 deliberately ticket-class feed sites and re-create the storm recorded at
  `alerting.py:46-53`.
- **The 3 pre-existing `BLE001`s are left alone** -- they are the intended
  fail-open behaviour, not this step's business.
