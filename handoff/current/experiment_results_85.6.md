# Experiment results — phase-85.6 (P0 DEADLOCK: the book cannot be un-paused)

**Step id:** 85.6 · **Cycle:** 184 — 2026-08-09
**Contract:** `handoff/current/contract_85.6.md` · **Gate:** `research_brief_85.6.md` (`gate_passed: true`, 7 sources read in full, 22 URLs)

## HEADLINE: the book is un-paused. `POST /resume` returned **HTTP 200** at 2026-08-08T20:58:43Z.

```
{'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}
```

Before this step, that call had returned 409 every time for six days.

---

## 1. What was built

| File | Change |
|---|---|
| `backend/services/paper_trader.py` | **NEW** `PaperTrader.roll_daily_anchor()` — the start-of-day roll as a standalone, callable operation. Same NAV source, same `sod_anchor_needs_reroll` guard as `:1298`; only the moment differs. Never raises. |
| `backend/services/autonomous_loop.py` | **Step 0**: the roll now runs FIRST in the cycle, before screening, and records its outcome in `summary["sod_anchor_roll"]` + `summary["steps"]`. |
| `backend/api/paper_trading.py` | The stale-anchor 409 body rewritten: both false promises removed, the real unblock condition named, the weekend case stated, and forging a `sod_snapshot` row explicitly warned against. |
| `backend/tests/test_phase_85_6_anchor_deadlock.py` | **NEW**, 16 tests, incl. a cycle that dies mid-analysis and a **spend guard**. |
| `scripts/qa/mutation_matrix_85_6.py` | **NEW**, 9 mutations, all killed. |

## 2. Criterion 1 — the deadlock broken at the CODE level

### The mechanism, re-derived

`update_sod_nav` had exactly **one** production call site (`paper_trader.py:1298`),
inside a method with exactly **one** production caller
(`backend/services/autonomous_loop.py:1375`, Step 5.5 of 10) — **behind** the
analysis phase at `:1148`. A cycle that dies in `analyzing` never rolls.

The gate found the canonical prior art: resilience4j ships
`automaticTransitionFromOpenToHalfOpenEnabled=false` by default, documented as
*"the transition to HALF_OPEN only happens if a call is made"* — a breaker whose
recovery path is reachable only through the thing it is blocking. Its documented
remedy is an **out-of-band transition**, not a change to thresholds. This step
changes no threshold and disarms nothing.

### The fix — criterion 1's SECOND branch

Step 0 of the cycle, before anything that can fail. Options rejected, with reasons:

- **Roll on backend startup — REJECTED.** Google SRE lists restart as the standard
  remediation for a deadlocked server, so it fires exactly when it is most
  dangerous; an MQL5 write-up names the result precisely: *"From every anchor's
  point of view the limit was never breached."*
- **Roll inside `/resume` — REJECTED.** The refusal would manufacture its own
  precondition.
- **Separate scheduled roller — viable, not chosen.** A second writer to a safety
  journal and a second schedule to keep in sync, for the same property.

### Proof — a CYCLE that dies mid-analysis, not an inspection

`test_c1_a_cycle_that_dies_mid_analysis_still_leaves_a_fresh_anchor` drives the
real `run_daily_cycle` with the anchor seeded to the true production value
(`2026-08-05`), injects a hang into `_run_single_analysis`, asserts the cycle
died with `status == "timeout"` and `"analyzing"` in its steps, then asserts
`sod_date == today`. It also asserts its **own precondition** (the analysis phase
was actually entered) so it cannot pass vacuously.

`test_c1_the_anchor_survives_a_crash_in_analysis_too` repeats it for the
exception shape. `test_c1_the_roll_runs_before_screening_not_after_analysis`
pins the ORDER from the cycle's own `summary["steps"]`, and asserts the roll is
`steps[0]`.

## 3. Criterion 2 — the 409 no longer lies

### Verbatim PRE-fix (captured live at 2026-08-08T20:5x from backend pid 143, which predates the fix)

> ... the trailing leg is date-independent and still armed. **NO operator action is
> required: the daily start-of-day roll stamps today's anchor at the top of the
> next paper-trading cycle and this refusal clears itself.** Retry the resume
> after that cycle.

Both halves were false. The roll was not at the top of the cycle, and it did not
clear itself: the 08-06 and 08-07 cycles both timed out in `analyzing` and never
reached it. An operator reading this waited for a self-heal that could not happen.

### Verbatim POST-fix (captured live from backend pid 23676, started after the fix)

> ... **UNBLOCK CONDITION: a paper-trading cycle must START and run its
> start-of-day roll (Step 0, backend/services/autonomous_loop.py,
> PaperTrader.roll_daily_anchor -> kill_switch.update_sod_nav). That roll now runs
> FIRST in the cycle, before screening and analysis, so it no longer depends on
> the cycle finishing. If a cycle is scheduled before you need to trade, retry the
> resume after it starts. If none is scheduled -- the cron is weekday-only, so
> this includes all weekend -- no cycle will run and this refusal will NOT clear
> on its own; trigger a cycle, or leave the book paused.** Verify with GET
> /api/paper-trading/kill-switch ... **Do NOT hand-write a sod_snapshot row into
> handoff/kill_switch_audit.jsonl to force this -- that forges the evidence the
> daily leg is measured against.**

It states a **precondition**, never a promise — phase-36.12 bans telling an
operator to wait for an automatic re-anchor, because for LOST HISTORY that silent
anchor WAS the defect.

`test_c2_the_mechanism_the_message_names_actually_exists` is a **claim audit**,
not a string check: it verifies `PaperTrader.roll_daily_anchor` exists, that it
calls `update_sod_nav`, that the cycle calls it, and that the call is genuinely
ahead of screening in source order — so the message cannot go stale silently.

## 4. Criterion 3 — the 85.4 interaction, with evidence

**Answer: NO. Fixing 85.4 does not clear this, and 85.6's own fix was required.**
Three independent pieces of evidence:

1. **85.4's remedies are not applied.** I closed 85.4 earlier this session
   (`8aa3f52e`). Criterion 5 forbade applying its config remedies inline, so the
   cycle budget is still 7200s, the rail timeout still 150s, and the merged
   dispatch flag is still dark. They are operator asks #23/#24/#25. Monday's cycle
   still projects to **8529s against 7200s** and will still die in `analyzing`.
2. **A completing cycle was already proven insufficient.** On 2026-08-05 all six
   tickers finished, the cycle reached the mark/trade region — and it traded
   nothing, logging `kill-switch active (paused) -- skipping decide/execute`. The
   switch was paused, so completion alone changes nothing.
3. **The two gates are separable and both bind.** 85.4 governs whether a cycle
   REACHES decide/execute; 85.6 governs whether it is ALLOWED to trade when it
   gets there. Today's verification cycle isolates them cleanly: the anchor rolled
   at **t+2 seconds**, long before the analysis phase could fail. Under the old
   code that same cycle would have had to survive ~2.4 hours to roll it.

## 5. Criterion 4 — resume proven END-TO-END, live

Full transcript in `handoff/current/live_check_85.6.md`. Summary:

| step | time (UTC) | evidence |
|---|---|---|
| lockfile read before acting | 20:5x | `state: "released"` — no live cycle |
| pre-fix 409 captured | 20:5x | verbatim, both false promises present; state + journal unchanged (52 lines) |
| backend restarted | 20:57:25 | pid 143 → **23676**, started AFTER the fix commit (22:56:34 local) |
| corrected 409 confirmed live | 20:57:xx | old phrases absent, `UNBLOCK CONDITION` present |
| **verification cycle 1 of 2 triggered** | **20:58:27** | `POST /run-now` → 200 |
| **anchor rolled at Step 0** | **20:58:29** | `+2s`. Log: `phase-85.6: start-of-day anchor rolled '2026-08-05' -> 2026-08-08 (nav=23830.46) at cycle start` |
| real `sod_snapshot` audit row | 20:58:29.379594 | `{"event":"sod_snapshot","nav":23830.46,"date":"2026-08-08"}` — **written by the code path, not by hand** |
| **`POST /resume` → HTTP 200** | **20:58:43** | `paused:false, sod_date:2026-08-08, armed:true` |

**One** of the two authorized verification cycles was used. The second is unspent.

## 6. Criterion 5 — no loosening, and it is tested

- **No threshold moved. No leg disarmed. Nothing un-paused by the roller.**
- `test_c5_the_roller_changes_no_threshold_and_disarms_nothing` pins the roller's
  entire **call set** against the kill-switch state (`{snapshot, update_sod_nav}`),
  so any future scope creep fails whether or not it happens to move a number.
- `test_c5_on_a_falling_book_the_early_anchor_is_never_more_forgiving` is the
  load-bearing safety test. The early roll sees the previous close; the old roll
  saw today's fresh mark. The dangerous direction is anchoring LOWER during a
  drawdown, which shrinks the measured loss and fires the switch LATER. On a
  falling book that is unreachable — the previous close is the higher number. The
  test asserts the measured loss percentages directly: **10.00% under the new
  anchor vs 0.00% under the old**, i.e. strictly more protective.
- The phase-36.12 invariant (the BREACH decision reads a POST-roll state) is
  preserved: by Step 5.5 the anchor is already today's and `:1298` no-ops.
- `test_roll_refuses_a_non_positive_nav_and_says_so` keeps the phase-36.9 F3
  refusal intact — a 0.0 anchor is a semipredicate that blocks its own repair.

## 7. Mutation matrix — 9/9 at cycle-1 (SUPERSEDED: see §12 for 12/12 and §13 for the final 14/14)

```
$ source .venv/bin/activate && python scripts/qa/mutation_matrix_85_6.py
[KILLED] M1 THE DEADLOCK RESTORED: Step 0 roll deleted from the cycle          3 failed
[KILLED] M2 the roll drifts back BEHIND screening (ordering guard)             2 failed
[KILLED] M3 the roller ignores the date guard and re-anchors mid-day           1 failed
[KILLED] M4 the roller latches a non-positive anchor (36.9 F3 regression)      2 failed
[KILLED] M5 the roller anchors on the CURRENT mark (loosening)                 1 failed
[KILLED] M6 the roller un-pauses / mutates the peak (scope creep)              1 failed
[KILLED] M7 the false 409 promise comes back                                   1 failed
[KILLED] M8 the 409 stops naming the real roller                               1 failed
[KILLED] M9 STUB MUTATION: the fake state stops mirroring the F3 refusal        1 failed

MUTATION MATRIX PASSED -- 9/9 mutations killed, tree restored byte-for-byte, suite green.
```

### Two mutations were LIVE on the first run. Both were real weaknesses in my guards.

1. **M2 walked through the ordering guard.** My test recorded when
   `screen_universe` was *called*; moving the roll to just after
   `summary["steps"].append("screening")` still precedes that call, so the test
   stayed green against a re-broken cycle. The source-order check failed too, for
   a second reason: it matched the bare name `roll_daily_anchor`, which also
   appears in the explanatory comment above the call — so `.index()` found the
   comment, which does not move. **Fixed** by reading the cycle's own
   `summary["steps"]` order (and asserting `steps[0] == "sod_anchor_roll"`), and
   by anchoring the source check on the full call expression.
2. **M6 walked through the scope guard.** It injected `update_peak(0.01)`, but my
   fake's peak is monotonic so a low value was silently ignored and the assertion
   "peak did not move" held. **Fixed** by pinning the roller's full call set
   instead of its visible effects.

## 8. Lint gate (qa.md §1a) — run BEFORE evaluation this time

```
$ uvx ruff check --select F821,F401,F811 "${FILES[@]}"   # 5-file git-derived scope, non-empty guard
All checks passed!
ruff_exit=0
```

It was **RED on the first run** (`F401 json`, `F401 fastapi.HTTPException`). The
second finding pointed at a dead helper that was never called and contained an
`eval()` — removed entirely rather than silenced. This is the 85.4 cycle-1 lesson
applied without needing the Q/A to teach it twice.

## 9. Test totals

```
backend/tests/test_phase_85_6_anchor_deadlock.py   19 passed in 14.01s   (16 + 3 from cycle-2)
scripts/qa/mutation_matrix_85_6.py                 12/12 mutations killed (9 + 3 from cycle-2)
```

## 10. Disclosed: a test wrote to the live cycle lockfile, and one run reached the real rail

- **Spend accident, caught and fenced.** While writing the tests, one run reached
  `claude_code_client.claude_code_invoke` and spawned real `claude` CLI
  subprocesses. Cause: emptying `screen_universe` does **not** empty the funnel,
  because the fixture also stubs `rank_candidates` to a constant 2-row list. I
  killed it on discovery and verified the damage: **`llm_call_log` had 0 rows in
  the preceding 30 minutes**, so no metered Vertex spend occurred; the escape was
  on the Max rail. The fixture now stubs the analysis path by default AND installs
  a **spend guard** that raises if any test reaches the rail, so a future escape
  fails loudly instead of quietly spending.
- **Live-file pollution (pre-existing class, 36.28).** The earlier full-suite run
  left `handoff/.autonomous_loop.lock` written by pytest pid 11128
  (`state: "released"`, so it blocked nothing). The 85.6 tests redirect
  `_LOCK_PATH`, `_HISTORY_PATH` and `_HEARTBEAT_PATH` into `tmp_path` and inject
  kill-switch state, so they add nothing to this. It is another artifact for the
  36.28 widening, filed rather than fixed mid-step.

## 11. What this step does NOT fix

- **It does not make cycles complete.** A cycle can now roll the anchor and the
  book can now be un-paused, but reaching decide/execute still requires the
  analysis phase to finish inside the budget — 85.4's measurement says it does
  not, and its remedies are operator asks #23/#24/#25.
- It does not address the **36 redundant `trigger:"manual"` pause rows** the gate
  found, which re-stamp `_paused_at` and would keep the phase-38.1 2h auto-resume
  clock permanently reset. This step deliberately does not lean on
  `check_auto_resume`. Queued rather than absorbed.
- It does not fix **85.5.1**'s fixture (`test_book_safety_69.py:79`). The gate
  established the scope boundary: that RED test is a fixture defect, but
  `sod_date=None` IS production-reachable via `_load_from_audit:285-295`, and the
  trailing leg still fires, bounding exposure to `[daily_limit, trailing_limit)`.


---

## 12. Cycle-2 — what changed after the Q/A CONDITIONAL (EVALUATE pass 1)

Verdict verbatim in `handoff/current/evaluator_critique_85.6.md`. The Q/A found
criteria 1, 2, 3, 4 and 6 MET and reproduced them independently. It blocked PASS
on **criterion 5**, and it was right — I had the safety argument half-wrong.

### The finding: my "strictly more protective" claim was true in one direction only

Step 0 anchors from the last stored mark and stamps **today's date** on it. If
cycles have been failing, that value can be several sessions old. The Q/A
measured the consequence with the **production `evaluate_breach`**, not by
reasoning:

- anchor `23830.46` (the 2026-08-05 mark) against a `22600` mark →
  `daily_loss = 5.16%` → `any_breached = True` → **flatten_all + pause**
- the pre-85.6 code, on the same book, measured **0.00%**
- with an *honest* stale date, phase-36.9's designed disarm fires instead
  (`armed=False`, `daily_baseline_stale=True`, no breach)

So I had converted a **designed disarm** into a **spurious flatten** — the exact
hazard `kill_switch.py`'s own phase-36.9 F1 comment records as measured on this
book on 2026-07-26 (*"a TWO-DAY move reported as a same-day loss … biases toward
a spurious flatten"*).

**It was live-reachable tonight.** Cycle `c67b3b15` reached Step 5.5 and
evaluated the daily leg against an anchor whose value was the 2026-08-05 mark.
It did not flatten only because the 3-session move happened to be −0.0146% —
**by margin, not by design.** My contract disclosed the risen-book direction and
then argued it away with no test and no bound. That was the error.

### The fix: the Step-0 anchor is PROVISIONAL and is upgraded before any breach

- `roll_daily_anchor` sets `self._sod_anchor_provisional = True` when it anchors.
- The post-mark path (`check_and_enforce_kill_switch`) upgrades that anchor to
  **today's freshly-marked NAV** before the breach is evaluated, and clears the
  flag.
- Net effect: the value the breach is judged against is **byte-equivalent to
  pre-85.6**, while the anchor still exists from t+2s so `/resume` works. The
  deadlock fix is kept; the loosening is removed.

**Residual, disclosed rather than hidden:** a cycle that dies *before* Step 5.5
leaves the provisional anchor for the rest of that UTC day, and the flag does not
survive the process. The window is one cycle — and a cycle that dies before Step
5.5 never reaches the breach decision either, so nothing is judged against it in
the meantime.

### Also fixed: an illusory guard the Q/A caught

`BANNED[2]` (`"at the top of the next paper-trading cycle"`) **could never
fail** — in the pre-fix source that phrase was split across two adjacent string
literals, so the substring check was already satisfied by the code it was written
to forbid. The check now normalises literal concatenation and whitespace, **and
self-validates**: it asserts every banned phrase IS present in the real pre-fix
blob (`git show 81f81750^:backend/api/paper_trading.py`), so a vacuous guard
fails loudly.

### New mutations

- **M10** deletes the upgrade branch → the hazard test dies.
- **M11** stops Step 0 flagging provisional → two tests die.
- **M12** flags provisional on a same-day no-op → the mid-day-re-anchor guard dies.

**M10 was LIVE on its first run**, and for an instructive reason: my hazard test
performed the upgrade *itself* in test code instead of calling
`check_and_enforce_kill_switch`. A guard that re-implements the thing it guards
proves nothing. It now drives the real production method.

Matrix: **12/12**. Suite: **19 passed**. Lint gate: `ruff_exit=0`.


---

## 13. Cycle-3 — the pass-2 CONDITIONAL, and why my own residual claim was false

Pass 2 confirmed criteria 1, 2, 3, 4 and 6 MET, and confirmed both of pass 1's
counterexamples were closed by the cycle-2 fix. It blocked on criterion 5 again,
for a reason I had asserted away rather than measured.

### The finding

The provisional marker lived on the `PaperTrader` **instance**, and
`autonomous_loop.py` constructs a **new** `PaperTrader` inside every
`run_daily_cycle`. So the flag did not survive the **cycle** — not merely "the
process", which is what I wrote. Measured through the production path:

1. cycle 1 rolls a provisional anchor at Step 0 and dies in `analyzing`
2. the next same-day cycle's Step 0 returns `anchor_already_current` and does
   **not** re-flag (fresh trader, flag clear)
3. the upgrade branch is therefore unreachable for the rest of the UTC day
4. that cycle's Step 5.5 judges the breach against the 3-session-stale anchor →
   `daily_loss_pct 5.1634` → `any_breached True` → **flatten_all + pause**

The mirror direction was live too: with a provisional anchor of 100 and today's
open at 110, a real **9.09% same-day drawdown reports 0.00% and does not fire**
(trailing 9.09% < 10% does not cover it), where pre-85.6 fires. That is a literal
loosening of the daily-loss leg.

### My disclosure was materially false, in both places

`paper_trader.py` said *"The next cycle re-rolls at Step 0 and upgrades here, so
the window is one cycle"* — it does not re-roll and cannot upgrade. §12 said
*"nothing is judged against it in the meantime"* — the next completing cycle
**does** judge against it, and in the Q/A's measurement it flattens the book.
Both texts are now replaced with the measured behaviour. This is the second time
this session that a residual I *reasoned* about turned out to be wrong when
*measured*; the lesson is banked.

### The fix: the marker is durable, not in-memory

`sod_provisional` is now state on `KillSwitchState`, written into the
`sod_snapshot` audit row and replayed on boot. It survives a dead cycle, a fresh
`PaperTrader`, and a process restart. The upgrade condition reads
`snap["sod_provisional"]`, never an instance attribute.

Declared **class-level**, because the file already documents the trap: fixtures
build detached states via `object.__new__`, so an instance-only attribute
`AttributeError`s inside `_snapshot_locked`. That was not theoretical — it broke
**34 tests**, measured, before the class default was added.

Legacy `sod_snapshot` rows have no `provisional` field → replays as `False` → the
upgrade path stays inert and behaviour is exactly pre-85.6.

### New tests and mutations

- 4 tests drive the **real** `KillSwitchState` with its journal redirected to
  tmp: persistence, replay-after-restart, clearing, legacy-row compatibility.
- 1 test drives the cross-cycle path with a **new** trader, which is the exact
  residual pass 2 measured.
- **M13** (marker not persisted) and **M14** (marker not replayed) both **LIVE on
  their first run** — because my existing tests used `FakeKillSwitchState`, which
  mirrors the marker itself and therefore cannot see a production persistence
  failure. That is the "mutate the stub too" lesson arriving for the third time
  this session.

Matrix **14/14**. Suite **25 passed**. All kill-switch + book-safety suites — **command recorded so the number
reproduces** (the pass-3 Q/A could not reproduce a bare "202" and was right to
say so; scope shape changes the count, the finding does not):

```
$ FILES=(); while IFS= read -r f; do FILES+=("backend/tests/$f"); done \
      < <(ls backend/tests/ | grep -iE "kill_switch|book_safety")
$ FILES+=(backend/tests/test_phase_85_6_anchor_deadlock.py)   # 10 files
$ python -m pytest "${FILES[@]}" -q --timeout=60
1 failed, 202 passed, 1 skipped
```

**202 passed, 1 failed** — the single failure is
`test_book_safety_69::test_valid_nav_still_breaches`, **pre-existing**, one of the
known 26, and the subject of step 85.5.1.

### One pre-existing test was deliberately changed — disclosed, not buried

`test_phase_36_9_kill_switch_armed_liveness.py::test_phase_36_9_the_resume_409_names_staleness_not_absence`
asserted the 409 message **contains** `"NO operator action is required"` — the
exact phrase criterion 2 requires **removed**, because it was measured false and
the book sat unresumable for six days behind it. The assertion is inverted with
the supersession explained inline, and the original intent (the message must
describe the daily roll that actually clears the refusal, not the lost-history
block) is preserved and still asserted. **No other pre-existing test was touched.**

### NOTE (1) from the pass-3 Q/A — a LIVE-state residual, recorded because it is real

The `sod_snapshot` row written by tonight's verification cycle
(`ts 2026-08-08T20:58:29.379594Z`) has **no `provisional` key** — it was written
by backend pid 23676, which predates the durable marker. So the current live
anchor (`23830.46`, stamped `2026-08-08`) is provisional **in fact** but replays
as **confirmed**, and a cycle running before the UTC rollover would judge the
breach against it without upgrading.

Measured exposure, rather than asserted:

- the stored mark at roll time **was** `23830.46` and current NAV is `23833.94`,
  so the move it would measure is **≈0%** — nowhere near the 4% daily limit;
- the cron is **weekday-only** and it is the weekend, so no scheduled cycle runs
  before the rollover;
- the failure direction is **over-protective** (a spurious flatten), never
  permissive;
- it **self-clears at the UTC rollover**, when Step 0 re-rolls and stamps the
  marker.

**Action taken:** the backend was restarted so the durable marker is in force
for Monday — a committed marker in a process that predates it is not a marker
(`feedback_committed_is_not_in_force`). Verified after the restart that the
kill-switch read is unchanged and the book is still resumed.

**Restart verified.** Backend pid **36970**, started **2026-08-09 00:03:29**,
after the cycle-3 commit `0f9e92c0` (2026-08-08 23:48:48). `launchctl` owns the
pid, health 200, no `EADDRINUSE`. The immutable verification command still reads
`{'paused': False, 'sod_date': '2026-08-08'} {'armed': True,
'daily_baseline_stale': False}` — the book is still resumed.

**Stated precisely, so it is not over-claimed:** `GET /api/paper-trading/kill-switch`
does **not** expose `sod_provisional` (it builds its payload from selected
fields, and phase-85.6 did not change that shape). So what is verified live is
that the process now runs post-marker code and the endpoint contract is
unbroken. The marker's persistence and replay are proven by the four tests that
drive the real `KillSwitchState`, not by the HTTP surface.
