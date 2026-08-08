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
backend/tests/test_phase_85_5_cycle_lock_split_brain.py   (NEW, 9 tests)
backend/tests/test_phase_38_6_restart_survivable.py       (2 tests updated, 1 replaced)
backend/tests/test_book_safety_69.py                      (1 trailing assertion updated)
```

## 3. Verification command — verbatim

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests -k "cycle_lock or 85_5" -q --timeout=120'
13 passed, 3015 deselected, 1 warning in 5.86s
```

Baseline before the change was `3 passed`. The new module is named so
`-k "cycle_lock or 85_5"` actually collects it — the phase-38.6 module matches
neither keyword, which is why the criterion-3 replacement lives in the new file.

## 4. Mutation matrix — every guard proven to bite

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
All 5 mutants caught.
```

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

## 5. Regression — measured against a pristine baseline, not asserted

The broad sweep shows 6 failures. **All 6 fail identically at `cf4d22d8`, the
commit before mine**, measured in a detached worktree with `backend/.env`
symlinked (the first baseline attempt was invalid — 4 pydantic collection
errors because the worktree lacked the gitignored env):

```
                                          baseline(cf4d22d8)   with my change
passed                                           330                338
failed                                             6                  6
```

`338 − 330 = +8` = 9 new tests minus the 1 TTL-literal test replaced. **Zero
regressions introduced.** The 6 pre-existing failures:

```
test_book_safety_69.py::test_valid_nav_still_breaches
test_phase_23_2_4_pause_resume_no_deadlock_live.py::...pause_cycle_under_5s
test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
test_phase_70_4_gate_observability.py::test_price_tolerance_rejection_is_accumulated
test_phase_70_4_gate_observability.py::test_price_tolerance_accumulator_empty_when_within_tolerance
```

These are **not** disclosed as prose and left there — `test_valid_nav_still_breaches`
is queued as its own research-gated masterplan step (see §7).

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
