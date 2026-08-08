# Experiment Results — phase-85.5 (P0 SAFETY)

**Cycle 181 — 2026-08-08.** Contract: `handoff/current/contract_85.5.md`
(written BEFORE any code change). Research: `research_brief_85.5.md`
(`gate_passed: true`, 10 sources read in full, 22 URLs).

Commits: `1911499b` (fix) · `da98af6b` (guards the mutation matrix caught).

---

## 1. What changed

`backend/services/cycle_lock.py` — the authority model was rewritten.

| # | Change | Why |
|---|---|---|
| 1 | `is_stale = (state == "released") or (not pid_alive)` | Age can no longer condemn a live cycle. Liveness is the sole authority; `released` is a **fact the releasing process recorded**, not a heuristic. |
| 2 | Stale-reacquire branch **deleted** (`acquire`) | Its premise was empty: a live flock holder is live by kernel guarantee, and a dead holder cannot hold a flock. Taking it required unlink+reopen → new inode → split brain. |
| 3 | Verify-after-lock: `(st_dev, st_ino)` of `fstat(fd)` vs `stat(path)`, bounded 5 attempts | portalocker 4.0.0 / issue #115. A lock on an orphaned inode excludes nobody. |
| 4 | Release **never unlinks**; rewrites `{"state": "released"}` in place | The old `finally` unlinked at `:153` *before* `LOCK_UN` at `:157` — split brain on the **normal release path**, no TTL involved. |
| 5 | `clean_stale_lock` is the one unlink site, gated on **winning the flock** | Race-free by construction; previously it unlinked on the predicate alone. |
| 6 | TTL derived at call time from `settings.paper_cycle_max_seconds` | The old constant froze at 5400s citing an 1800s budget while the real budget was 7200s → TTL was 0.75×. Budget is operator-mutable, so an import-time snapshot would re-rot. |

`backend/slack_bot/scheduler.py` — `_cycle_state_line()` gains a `released`
branch and reports `[OVER BUDGET]` without condemning the lock.

**Why item 3 was in scope** (contract §5a): criterion 2's words are "a second
acquirer cannot obtain the lock **while the first process is alive and holds
the flock**." During the old `finally` the holder was alive *and* still held
the flock. Fixing only `is_stale` would have left criterion 2 true in
appearance and false in fact.

## 2. Files

```
backend/services/cycle_lock.py                            (rewritten authority model)
backend/slack_bot/scheduler.py                            (_cycle_state_line only)
backend/tests/test_phase_85_5_cycle_lock_split_brain.py   (NEW, 11 tests at HEAD)
backend/tests/test_phase_38_6_restart_survivable.py       (2 tests updated, 1 replaced)
backend/tests/test_book_safety_69.py                      (1 trailing assertion updated)
```

## 3. Verification command — verbatim

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests -k "cycle_lock or 85_5" -q --timeout=120'
14 passed, 3015 deselected, 1 warning in 6.14s
```

Baseline before the change was `3 passed`. 14 = 11 new tests in the 85.5
module + 3 pre-existing hits. (Regenerated after Q/A cycle 1: the earlier
capture said 13 and was measured before the flock-gate guard was added —
a carried-forward number rather than a regenerated one.) The new module is named so
`-k "cycle_lock or 85_5"` actually collects it — the phase-38.6 module matches
neither keyword, which is why the criterion-3 replacement lives in the new file.

## 4. Mutation matrix — one mutant per change in the §1 table

Reverts each fix **in the production file**, runs the test meant to catch it,
asserts that test FAILS, restores via `git checkout --` and asserts byte-equality.
Each replacement asserts its target exists first (a no-match `replace` looks
identical to success).

```
CAUGHT   M1 is_stale ORs age back in (the original defect)
CAUGHT   M2 acquire re-adds the stale unlink+recreate branch
CAUGHT   M3 TTL frozen back to a constant below the budget
CAUGHT   M4 release unlinks again (the third, unnamed defect)
CAUGHT   M5 verify-after-lock removed
CAUGHT   M6 clean_stale_lock unlinks on the predicate alone (flock gate removed)
All 6 mutants caught.
```

Six mutants for the six changes in the §1 table. **M6 exists because Q/A
cycle 1 caught me shipping an untested guard:** change #5
(`clean_stale_lock` gated on winning the flock) had no mutant, and the Q/A's
own equivalent mutant **survived my entire suite**. It is not an equivalent
mutant — ungated, `clean_stale_lock` deletes a live holder's lockfile, after
which a second acquirer `O_CREAT`s a new inode and its flock contends with
nobody. It is production-reachable from `backend/main.py:265`
(`reason="startup_recovery"`) — a backend restarting while a cycle runs, which
is exactly this step's threat model. Both of my existing tests exited at the
`not is_stale` early return and never reached the gate.

The guard added for it drives `is_stale` True while the flock is genuinely
held, by **both** routes: a forged `released` payload, and the
`_write_payload` truncate window in which a live holder momentarily reads as
malformed.

**The first run had 2 of 5 SURVIVE, and that is the most useful thing in this
document:**

- **M2 survived** because with the fixed predicate a live holder is never
  `is_stale`, so the re-added branch was *unreachable in the scenario my test
  drove*. The test proved "a contender is refused" but not "the bypass is
  gone" — two independent fixes masking each other. Fixed by forcing
  `is_stale` True **while the flock is genuinely held**, which is a real
  window: release writes `released` *before* `LOCK_UN`.
- **M5 survived** because the inode test was near-vacuous — it only asserted
  `inspect_lock()` returned non-None. Replaced with a test that swaps the file
  from inside `flock()` and asserts the payload lands at the **path**, not on
  the orphan.

## 5. Regression — full suite, exact command, environment held constant

**This section was rewritten after Q/A cycle 1.** The original version said
"the broad sweep shows 6 failures … zero regressions introduced" over a
`-k`-filtered slice of ~336 of ~3029 tests (~11%), disclosed no command, and
so could not be reproduced. Calling an 11% slice "broad" was overclaiming.
The corrected evidence follows.

**Command (both runs, verbatim):**

```
python -m pytest backend/tests -q --timeout=120 --tb=no
```

**Full suite at HEAD (`990e409d`):**

```
26 failed, 2985 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 288.54s
```

### The worktree baseline is INVALID — and here is why, rather than a quiet drop

Running the same command in a detached worktree at `cf4d22d8` gives
`30 failed, 2958 passed, 22 skipped, 4 errors`, with **1 failure unique to
HEAD** and **5 unique to the baseline**. None of those six are code
differences. A worktree does not contain the repo's **gitignored live-system
files**, and several tests read them directly:

- `test_phase_23_2_6_backend_log_has_skipping_buy_evidence` reads the
  repo-root `backend.log` (**32,565,172 bytes** in the real repo, **absent**
  in the worktree). It therefore *skipped* in the baseline and *ran and
  failed* at HEAD — appearing as a regression that is purely environmental.
  Its own docstring says so: *"Fails locally because it IS
  live-system-dependent … genuine live-system state, not a code defect."*
- The 5 baseline-only failures read `.claude/masterplan.json` and `handoff/`
  state, which differ in a detached checkout.
- Skip counts corroborate exactly: **12 skipped at HEAD vs 22 in the
  worktree** — ten extra skips for absent live files.

### The valid method: same environment, only the code varies

Injected the **pre-change** `backend/services/cycle_lock.py`
(`git show 1911499b~1:…`, old predicate confirmed present at its line 79) into
the **real repo** and replayed exactly the 26 HEAD failures:

```
OLD CODE: 26 failed, 1 warning in 5.19s
=== all 26 reproduce under OLD code, same environment? ===
  IDENTICAL -- all 26 failures are independent of my change
```

File restored byte-exact afterwards (`cmp` verified; explicit backup/restore,
not `git checkout --`, which this repo's PreToolUse danger hook blocks —
correctly, since that command silently discarded uncommitted work earlier in
this very cycle).

**Conclusion: zero regressions introduced — demonstrated, not asserted.**

One of the 26, `test_book_safety_69.py::test_valid_nav_still_breaches`, is
**not** left as prose: it is queued as its own research-gated masterplan step
(see §7).

**A failed measurement that looked like success:** the first corrected re-run
printed "0 failures" for both sweeps. zsh does not word-split unquoted
parameter expansions, so `-m $SWEEP` passed the whole string as a module name
and pytest never ran. Those numbers were discarded, not published, and the
harness now refuses to report unless a real pytest summary line is present.

Lint gate (ruff F821,F401,F811, git-derived scope, all 5 changed files):
`All checks passed!` — this also cleared two **pre-existing** F401s in the 38.6
module (verified pre-existing by running ruff on the `1911499b~1` version).

## 6. Live evidence

Full transcript in `handoff/current/live_check_85.5.md`. Headlines:

- A **real cycle** (`cycle-1786174058`) exercised the new release path during
  this step and wrote `{"state": "released"}` — a field only the new code
  emits. It released cleanly.
- The exact measured condition from the audit basis now inverts:
  `age_sec=20800 > ttl_sec=10800`, `pid_alive=True` → **`is_stale=False`**
  (was `True` pre-fix).
- Two real processes: contender **REFUSED** against a live holder whose lock
  is older than the TTL; after `SIGKILL`, cleaned and **ACQUIRED**. No deadlock.
- Reconciled values: budget `7200.0`, TTL `10800.0`, `TTL >= budget = True`,
  `_LOCK_TTL_SEC` gone.

**A consequence I did not anticipate and then tested:** because release no
longer unlinks, every cycle now leaves a `released` pidfile behind. One is
sitting in `handoff/` right now. Verified it does **not** block the next cycle
— acquire reuses the **same inode** rather than unlink+recreate — and added a
permanent guard, `test_85_5_c2_acquire_works_over_a_leftover_released_lockfile`.
The file is gitignored (`.gitignore:93`), so it is never committed.

## 7. Out-of-scope defect found — queued, not narrated

`test_book_safety_69.py::test_valid_nav_still_breaches` fails at pristine HEAD:
the kill switch DISARMS because the daily anchor is stale (`sod_date=None`)
when replaying the real `handoff/audit/kill_switch_audit*.jsonl` sources. A
**book-safety** test asserting that a real 20% breach still fires is red. Queued
as its own research-gated masterplan step rather than mentioned in prose.

## 8. Honest limits

- The live evidence uses a **real second process**, not a production trading
  cycle mid-analysis. No production cycle is running (Saturday; next scheduled
  Monday), so the criteria — which say "a test drives the exact measured
  condition" — are met, while the live_check's literal "on a live cycle"
  wording is satisfied by a genuine flock holder rather than a trading cycle.
  Production confirmation lands with Monday's cycle.
- `backend/slack_bot/scheduler.py` is changed in code but the **running bot
  process will not pick it up until its next restart**. No service was
  restarted this cycle (goal rule: sequence restarts clear of other steps'
  evidence windows). Until then the watchdog degrades to the old wording,
  which is cosmetically wrong but safe — the line is appended to alerts and
  never suppresses one.
- I edited `cycle_lock.py` while a backend capable of importing it was live.
  Nothing was harmed (the real cycle released cleanly), but the safer sequence
  would have been to confirm no scheduled cycle could fire first.
