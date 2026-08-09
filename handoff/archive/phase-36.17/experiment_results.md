# Experiment Results -- phase-82.6

**Step:** 82.6 (P2) -- DESIGN (not build) the registry-to-live selection bridge.
**Date:** 2026-08-06. **Cycle:** 2 (cycle-1 Q/A returned CONDITIONAL; all 3 criteria MET, five findings on my claims and guards -- see §10).
**Contract:** `handoff/current/contract_82.6.md`
**Research brief:** `handoff/current/research_brief_82.6.md` (`gate_passed: true`,
audit-class, dry after 13 rounds / 2 dry, 6 sources read in full, 34 URLs, 20 files)

---

## 1. What changed

| File | Change | Lines |
|------|--------|-------|
| `docs/design/registry-to-live-selection-bridge.md` | new -- the design | 184 (new) |
| `backend/tests/test_phase_82_6_bridge_design.py` | new -- 12 tests | 363 (new) |
| `.claude/masterplan.json` | queued 82.64 / 82.65 / 82.66 | see §7 |

**No production code was touched.** That is the point of the step: it ships a
document and a guard that the live selection path is unchanged.

## 2. Verbatim output of the immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_6_bridge_design.py -q
  warnings.warn(
............                                                             [100%]
12 passed in 2.21s
```

## 3. BOTH halves of the step's premise are refuted

The step says *"the registry's EXIT PARAMS cross over; its SELECTION LOGIC does
not."* Measured by me, then independently by the gate:

**The exit params do not cross over either.** `summary["strategy_params"]`
(`autonomous_loop.py:434-437`) has **zero readers repo-wide**, frontend included.
`tp_pct` / `sl_pct` appear nowhere else in `backend/services/`. Live exits come
from `settings.paper_default_stop_loss_pct` and the Risk Judge.

**So today nothing the optimizer produces changes live trading behaviour** --
one display number and one audit label.

The provenance of the error is worth recording: the source spec
(`incumbent_live_strategy_spec.md:35`) said the params cross over **"as display
fields"**. The masterplan step dropped the qualifier, and the qualifier was the
whole meaning.

**Minor:** `:431` does not read `optimizer_best.json` directly; it calls
`load_promoted_params(bq)`, which prefers BQ `promoted_strategies` and uses the
JSON only as fallback. All five of the step's cited anchors do resolve.

## 4. THE HEADLINE -- and a correction to the gate's framing

The gate reported that `paper_trader.py` "silently disarms a risk control".
**I read the branch before writing that down, and it overstates it.** The code is
more careful than either of us said:

> *Kaminski-Lo Proposition 2: mean-reverting strategies (and cointegrated pairs)
> lose expected return when trailing-stop cumulative-loss thresholds fire; SKIP
> for those. Fail-CLOSED-conservative: when entry_strategy is None/unknown, treat
> as momentum (trail IS applied).*

The skip is deliberate, research-cited, and its default is already fail-safe. It
is not a defect, and I am not repeating that characterisation.

**The real hazard is sequencing, and it survives the correction.**
`paper_positions.entry_strategy` is NULL on every row (measured live; the table
holds **1** row) so the branch is unreachable -- while
`scripts/migrations/phase_32_2_add_entry_strategy.py:16-17` already names that
column as the bridge's intended wire. So the bridge's first natural wire
**activates a dormant live risk-behaviour change as a side effect of a change
whose stated purpose is selection.** Today every position is trailed; the day
that column is populated, `mean_reversion` and `pairs` positions stop being
trailed.

The design's requirement: populating `entry_strategy` must be separately flagged
and separately reviewed from wiring selection -- never the same commit.

## 5. The bridge was already designed

`backend/autoresearch/strategy_selector.py` (phase-47.6) is complete, tested, and
**dark with zero production callers**; its docstring specifies this exact bridge.
Steps 47.6 / 48.1 / 48.2 / 48.3 / 48.4 are all `done`, and 48.3's own name records
the deployment bridge as the deferred piece. So this step **ratifies and
documents** rather than re-designing. **This is a deployment problem, not a design
problem.**

**Hard prerequisites, measured:** `optimizer_best.json` has **no `pbo` key at
all** and `PromotionGate` is fail-closed on a missing pbo -- so **82.23** and
**82.26** are build-time blockers, not advisories.

## 6. Mutation matrix -- 7 mutants, all killed

```
baseline: rc=0 GREEN

M_A doc drops the trailing-stop hazard          DIED  (the §2 hazard is actually required)
M_B a file:line ref rots                        DIED  (criterion 2 catches a rotted anchor)
M_C doc drops the build prerequisites           DIED  (the ungateable-bridge warning is required)
M_D doc stops naming the rollback               DIED  (criterion 1's rollback part)
M_E live cycle references a registry label      DIED  (criterion 3 -- selection wired)
M_F live cycle imports the backtest ENGINE      DIED  (criterion 3 -- engine reachable)
M_G a SECOND strategy read appears              DIED  (label-only baseline is pinned)

=== 7 died, 0 survived ===
```

Licenses exactly "these 7 mutants died", not "no survivors".

### Two mutants survived the first run, and BOTH were my own construction errors

This matters more than the final table, because a survivor I mis-read as a guard
failure would have sent me editing a correct guard.

- **M_E survived** because I injected the string `'_label_mean_reversion'` -- and
  the registry's actual values are `_compute_mean_reversion_label` and friends.
  **The mutant named something that was never in the registry**, so a
  registry-value sweep correctly ignored it. A mutation that cannot succeed
  proves nothing about the guard. Re-run with a real registry value: it dies.
- **M_C survived** because `82.23` appears **3 times** in the design and I
  mutated only the bolded occurrence. The exact string `**82.23**` was unique,
  so my uniqueness assert passed while the mutation was still semantically a
  no-op.

Both are the same lesson from the other direction: a mutation matrix's *negative*
results are only as trustworthy as the mutants, and "SURVIVED" must be diagnosed
before it is believed.

## 7. Criterion 3, and two things my own tests caught

The sweep targets registry **method values** (`_compute_*_label`), never the
**keys** -- `triple_barrier` legitimately appears in the live path as a label
string, while its labeller must not. Traps avoided, each measured by the gate:
`len(set(values)) == len(keys)` **fails on correct code** (6 keys, 5 distinct
values -- `meta_label` shares `triple_barrier`'s); a blanket "no
`backend.backtest` import" assertion is a **false positive** (`universe_lists`
and `markets` are legitimate); `perf_metrics.py:151` names the engine in a
comment.

**My own guard found two real things while I was writing the doc:**

1. **`autonomous_loop.py` is genuinely ambiguous -- there are TWO.**
   `backend/autonomous_loop.py` is the phase-3.3 planner/evaluator harness;
   `backend/services/autonomous_loop.py` is the live cycle. My first draft cited
   the bare basename, which is exactly the kind of anchor that misleads. The
   resolution test refused to resolve it, and the doc now uses full paths and
   states the ambiguity.
2. A prose token was matched case-sensitively ("Kill switch" vs "kill switch").
   Fixed in the test: identifiers exact, prose case-insensitive.

## 8. Discovered defects -- queued, verified by me before queueing

| Step | Finding | How I verified it |
|------|---------|-------------------|
| **82.64** (P2) | `promoter.py:134` writes `float(trial.get("pbo") or 0.0)` -- a missing PBO is recorded as a **perfect 0.0** on the promoted row | read both sides; `optimizer_best.json` confirmed to have no `pbo` key |
| **82.65** (P2) | `strategy_decisions` heartbeat ~6 days stale, write swallowed | live BQ: 51 rows, newest 2026-07-31 |
| **82.66** (P3) | 3 stale registry enumerations + 2 dead configs | registry has 6 keys; two docs say "five" |

**82.64 is stated carefully rather than dramatically.** The *gate* is fail-closed
and does not promote on absent data -- that part is correct. The defect is the
**record written afterwards**: any downstream consumer of
`promoted_strategies.pbo` reads a fabricated pass. Overstating it as "the gate
fails open" would have been wrong, and the queued step says so explicitly.

## 9. What I did NOT do

- **No selection wired**, no production code touched at all. The live book is
  working; this step is a document plus a guard.
- **Did not re-design the selector** (§5) -- that would duplicate settled work.
- **Did not schedule `run_friday_promotion`** or populate `entry_strategy`. Both
  are named in the design as deliberate, separately-gated future decisions.
- **Did not run the full test suite.** This step adds no production code, so the
  regression surface is the new test file itself; the targeted run is in §2. That
  is a narrower check than 82.51/82.59 got, and it is stated rather than implied.

## 10. Cycle 2 -- three claims of mine that did not reproduce, and two recall bounds

The cycle-1 Q/A returned **CONDITIONAL** with all three criteria MET. It ran a
17-mutant matrix, confirmed criterion 3's guard is genuinely live, and **ratified
all three challenges I raised** -- including my correction of its predecessor's
"silently disarms a risk control" framing, and 82.64's careful gate-vs-record
distinction. Every finding below is mine.

### 10.1 Two universal claims in the design were false

- **"zero production callers"** for `strategy_selector`. `select_best_strategy`
  **is** called at `backend/autoresearch/strategy_candidate_producer.py:181`,
  reached via `rotation_runner.py:53`. All production modules.
- **"`run_friday_promotion` has no caller anywhere."** **25** invocations exist across the tracked repo (987 `.py` files via `git ls-files`): 12 in `tests/autoresearch/test_friday_promotion.py`, 7 in `scripts/harness/phase10_friday_promotion_test.py`, 4 in `tests/autoresearch/test_slot_usage_wiring.py`, 2 in `tests/verify_phase_25_A3.py` -- every one a test or harness call, none a scheduler or production caller.
  My cycle-2 correction said "four" (unmeasured) and my cycle-3 correction said
  "7, all in one file" -- which reproduced ONLY inside a scope I chose,
  `grep ... backend scripts`, that structurally cannot see `tests/`. **Three
  consecutive corrections of the same claim, each wrong a different way.** The
  count above is derived over `git ls-files '*.py'`, not a directory list I
  picked.

**Both came from the research gate, and I republished them without re-deriving.**
The Q/A traced the gate's grep failure precisely: an **unquoted
`--include=*.py`, glob-eaten by zsh** -- the same instrument failure I hit twice
myself today. A "zero X" claim is a set claim, and I shipped two of them into a
document a future builder would trust.

Corrected to the claims that are true and that actually carry the argument:
*zero callers on the live trading path*, and *no scheduled caller*. **The
design's material conclusion is unaffected** -- the selector still has no path to
live trading, so this remains a deployment problem, not a design problem.

### 10.2 I edited a block presented as a verbatim source quote

§2's fenced block was anchored to `paper_trader.py:1425-1428` but showed **three**
lines, dropping the real first line `if pos.get("stop_advanced_at_R"):` and
appending a trailing comment (`# skip the HWM trailing stop`) **that does not
exist in the source**.

The consequence is material, not cosmetic: the skip applies only to positions
**past the breakeven ratchet**, so my prose "Today every position is trailed"
overstated the blast radius of the very hazard the section exists to raise. This
is the discipline I have been applying to my own figures all day, failing on a
code quote.

Fixed by **regenerating the block with `sed` and pasting the output**, never
retyping it, plus a prose correction and a new assertion that the design states
the `stop_advanced_at_R` precondition. A small irony worth recording: the
fabricated comment was the only unhyphenated instance of "trailing stop" in the
document, so removing it broke my own token check -- **my guard had been partly
passing on text I invented.**

### 10.3 Two recall bounds, both closed

The Q/A found **five** wiring shapes criterion 3's sweep could not see:
submodule import, `importlib.import_module`, registry **key** dispatch,
f-string-built `getattr`, and a `["strategy"]` subscript read. None is reachable
today, and criterion 3 as written was satisfied -- but the test docstring
over-claimed "None of them may be reachable".

All five are now guarded, and each was mutation-verified to die:

```
submodule import         DIED
importlib string         DIED
registry KEY dispatch    DIED
f-string getattr         DIED
subscript read           DIED
```

The `getattr` shape is the one that mattered most: the design itself says the
label methods are dispatched by `getattr` inside the engine, so a copy-paste
wiring would take exactly that form. The docstring now states what the sweep
proves rather than what I wished it proved.

**Criterion 1's token tests also passed on a comment-stuffed stub.** HTML
comments are now stripped before the scan, so the tokens must appear in prose.

### 10.4 Final matrix

7 original mutants + 5 recall shapes = **12 kills, 0 survivors.** Still licenses
only "these 12 mutants died".

## 11. Cycle 3 -- I reported a fix that never applied

The cycle-2 Q/A cured 4 of 5 findings by execution (all five recall shapes now
die, the stub is rejected, the §2 block matched byte-for-byte). It blocked on one
thing, and it is the sharpest self-inflicted finding of the day.

**I wrote in §10.1 that both universals were "Corrected". Only one was.** My
`str.replace` for the second targeted the single-line form
`` `run_friday_promotion` has no caller anywhere. `` while the document has it
**line-wrapped** as `has no\ncaller anywhere.`. The replace matched nothing,
returned the string unchanged, and I reported success without checking.

**That is a no-op that looks exactly like a success** -- the standing failure
shape I have a note about and have caught in production code twice this week --
committed in the middle of a section whose entire subject is claims that do not
reproduce.

And the replacement I *intended* was itself wrong: I wrote "Four call sites"
without measuring. The real count is **7** invocations.

### Fixed, and this time verified

All three edits assert their anchor exists **and** re-read the file to confirm
the old text is gone:

```
1. design fixed AND verified absent
2. write-up count corrected to 7
3. 82.66 corrected AND verified
```

- **The design** now says "no SCHEDULED caller", with the measured count, the
  file, and why the ledger/plan slots (`cron_budget.yaml`'s
  `friday_promotion_gate`, `sprint_calendar.yaml`'s `fri_promotion`,
  `slot_accounting.py`'s logged name string) are not invocations. The Q/A
  independently confirmed that replacement claim is sound.
- **This write-up** carries the measured 7 and the command that produced it, and
  says plainly that my correction was also untested.
- **Queued step 82.66** no longer ships the refuted universal to an executor who
  would have no way to know it was measured false in the cycle that queued it.

The lesson is not "check your replaces". It is that **a claim of having fixed
something is itself a claim, and gets the same burden of proof as the claim it
replaces.** I applied that burden to the code all day and not to my own edits.

## 12. Cycle 4 -- FAIL, and the verdict is correct

The cycle-3 Q/A returned **FAIL**. All three immutable criteria were MET, the
cycle-2 blocker was confirmed cured, and no production code was touched -- but
my replacement claim failed independent re-derivation for the **third
consecutive cycle**, and the 3rd-cycle rule removed CONDITIONAL as an option.

**The finding:** the design said `run_friday_promotion` has *"7 invocations, all
in `scripts/harness/phase10_friday_promotion_test.py`"*. That reproduces **only
inside the scope I chose** -- my pinned command was
`grep ... --include="*.py" backend scripts`, which **structurally cannot see
`tests/`**. Derived over the whole tracked repo the answer is **25**, with 18
invocations in three tracked test files I never looked at.

**Three corrections of one claim, each wrong a different way:**

| cycle | claim | why it was wrong |
|-------|-------|------------------|
| 1 | "no caller anywhere" | inherited from the gate, never re-derived; the gate's grep had been glob-eaten by zsh |
| 2 | "Four call sites" | typed, not measured -- and the edit never applied at all |
| 3 | "7, all in one file" | measured, but over a scope I picked that excluded `tests/` |

The third is the subtlest and the worst, because it *looks* like the fix: a
command, a number, a file. **A tool that reports success over a scope the author
chose is not evidence** -- which is the rule I have been applying to production
code all day, failing on my own prose three times running.

### Fixed, derived over `git ls-files`

```
tracked .py files scanned: 987
TOTAL invocations: 25
   12  tests/autoresearch/test_friday_promotion.py
    7  scripts/harness/phase10_friday_promotion_test.py
    4  tests/autoresearch/test_slot_usage_wiring.py
    2  tests/verify_phase_25_A3.py
```

The scope is now `git ls-files '*.py'` -- the repo's own list of tracked files,
not a directory list I supply. All three sites corrected and each verified by
re-reading the file: the design, this write-up, and queued step **82.66**, which
now warns explicitly that deleting or rewiring the function **breaks 25 call
sites across four files**, and tells its executor to re-derive over the whole
tracked repo because two earlier drafts of that very step were false.

**What was never in question across all four cycles:** the three immutable
criteria, the guards (12 kills, 0 survivors), that no production code was
touched, and the design's load-bearing conclusion -- the Q/A checked all 25 sites
itself and confirmed none is a scheduler or production caller, so *"no scheduled
caller"* and *"a deployment problem, not a design problem"* are both true.

### Cycle 4 -- PASS, and two residuals accepted rather than fixed

`violated_criteria: []`. The Q/A **refused my scope entirely** -- it scanned the
whole workspace, tracked and untracked, every file type, and classified each hit
by reading the line -- and confirmed 25 with the same four-file breakdown. It
also verified the design's load-bearing conclusion at that maximum scope: none of
the 25 is a scheduler or production caller.

It recorded two NOTE-level residuals and **explicitly instructed me not to open a
cycle 5 for them**, on the grounds that a 5th cycle to change a noun and a
denominator, on a step whose criteria have now been confirmed four times, would
be the "harness is logging, not correcting" pathology. I am taking that
instruction, and recording the residuals rather than quietly editing evidence
that has already been graded:

- **R1:** 24 of the 25 are calls. The 25th
  (`tests/autoresearch/test_slot_usage_wiring.py:5`) is a **module docstring
  line** that a `run_friday_promotion(` substring match catches. So "invocations"
  is loose for exactly 1 of 25 -- in a file already named, which a rewire would
  want to update anyway.
- **R2:** `git ls-files '*.py' | wc -l` returns **986**, not the 987 I stated in
  three places. Most likely 986 tracked plus the 1 new untracked test file, which
  `git ls-files` does not list. The denominator carries no part of the argument.

Both are for 82.66's executor, who is already instructed to re-derive.

**Four cycles, and the substance never moved.** The three criteria were MET in
cycle 1 and in every cycle since; no production code was ever touched; the guards
went from 11 to 12 tests and from 7 to 12 mutation kills. Every blocker was a
claim about a population, and the claim that took four attempts was a *count of
call sites* -- wrong first by inheritance, then by typing, then by scope.
