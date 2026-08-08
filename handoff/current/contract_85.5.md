# Contract — phase-85.5 (P0 SAFETY)

**Step id:** 85.5
**Cycle:** 181 — 2026-08-08
**Title:** A LIVE autonomous cycle's lock is judged stale after 90min, and the stale
path unlink+recreates the lockfile — which defeats the flock the live cycle holds.
Two cycles can then run concurrently on the money path.

**Written BEFORE any code change.** Order this cycle: research → contract → generate
→ Q/A → log → flip. (68.1 was capped at CONDITIONAL for writing the contract after
the code; this cycle does not repeat that.)

---

## 1. Research-gate summary

Researcher spawned before this contract. Brief:
`handoff/current/research_brief_85.5.md` (463 lines).

Envelope: `gate_passed: true` — **10** external sources read in full (floor 5),
**22** URLs collected, **12** snippet-only, recency scan performed, **11** internal
files inspected, tier `moderate`, not audit-class.

Findings that drive this contract:

- **The defect class has a name and a canonical remedy.** "Lock split-brain via
  lockfile inode swap." `portalocker` 4.0.0 (issue #115) ships exactly the remedy
  hypothesised: after `flock` succeeds, re-verify `(st_dev, st_ino)` of `fstat(fd)`
  against `stat(path)` and retry in a bounded loop.
- **Never unlink a lockfile that others coordinate on by path.** `filelock` states
  this as an in-source invariant; Flohr's *"Never Delete Your PID File"* is the
  canonical treatment.
- **Liveness is strictly stronger than age — sourced, not assumed.** The kernel
  releases an advisory lock the instant the holder dies, file present or not. So
  `BlockingIOError` *already* means a live holder, and no age test improves on it.
  `filelock`, `portalocker` and `trbs/pid` carry **no TTL term at all**;
  `flufl.lock`'s TTL is safe only as a *refreshed lease*, which this write-once
  pidfile is not.
- **A third defect the step premise does not name** (researcher-proven on this host,
  scratch dir, no project file touched): the `finally` block unlinks at
  `cycle_lock.py:153` **before** `LOCK_UN` at `:157`. That yields a double-acquire on
  the **normal, successful release path** — no stale lock, no TTL, no
  `clean_stale_lock` involved. See §5 for the scope ruling.

## 2. Hypothesis

The lock's authority is misplaced. Today an *age heuristic* can condemn a lock that
the *kernel* says is live, and the remedy path then destroys the very object that
provides mutual exclusion. If the flock becomes the sole authority for destructive
action — age demoted to pure observability — and the acquire path stops recreating
the file, then two cycles cannot hold the lock simultaneously, while a genuinely
dead holder stays recoverable.

## 3. Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

> 1. a live process's lock is NEVER judged stale: the predicate treats pid_alive (and holding the flock) as vetoing staleness, so age alone cannot condemn a running cycle; a test drives the exact measured condition (age > TTL, pid alive) and asserts is_stale is False
>
> 2. the unlink+recreate bypass is closed: a second acquirer cannot obtain the lock while the first process is alive and holds the flock, proven by a test that spawns a real second process (not a mocked one) and asserts it is refused -- the fixture must first demonstrate the OLD code path acquiring, so the test cannot pass vacuously
>
> 3. the TTL and the cycle timeout are reconciled and the relationship is asserted in code (e.g. TTL derived from the cycle budget rather than a hardcoded constant whose comment cites a value no longer in force), with a test that fails if a future edit makes TTL < cycle timeout again
>
> 4. a genuinely dead holder is still recoverable: a lock whose pid is dead is cleaned as before, proven by a test that kills a real child process mid-hold -- the fix must not trade a double-run hazard for a permanent deadlock
>
> 5. no change to order/sizing/risk logic; fresh Q/A PASS

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests -k "cycle_lock or 85_5" -q --timeout=120'`

**live_check:** `live_check_85.5.md` with the re-measured `inspect_lock()` output on a
live cycle showing `is_stale` False, the two-real-process acquisition transcript
(refused while alive, granted after death), and the reconciled TTL-vs-cycle-timeout
values.

## 4. Measured baseline (re-derived this cycle — the audit basis said to trust nothing)

| Fact | Where | Value |
|---|---|---|
| Cycle live right now? | `handoff/.autonomous_loop.lock` | **absent — no live cycle** |
| `_LOCK_TTL_SEC` | `cycle_lock.py:41` | `90*60` = 5400 |
| Real cycle budget | `settings.py:33` `paper_cycle_max_seconds` | `7200.0` |
| Budget consumer | `autonomous_loop.py:439` | reads the same field (fallback literal is a stale `1800.0`) |
| TTL vs budget | derived | **0.75×** — the audit's claim reproduces |
| Verification command | as-written | **3 passed**, 3016 deselected |
| 38.6 lock suite | `test_phase_38_6_restart_survivable.py` | **8 passed** |

Budget is **runtime-mutable** (`settings_api.py:171`, `300 ≤ x ≤ 21600`, env
`PAPER_CYCLE_MAX_SECONDS`), so criterion 3 must bind to it **lazily at call time**.
An import-time snapshot would reintroduce the identical staleness class.

## 5. Scope rulings (stated up front so the evaluator can contest them)

**(a) The release-path unlink (defect 3) is IN scope.** Criterion 2's literal words
are "a second acquirer cannot obtain the lock **while the first process is alive and
holds the flock**." During the `finally` block the first process is alive *and* still
holds the flock (`unlink` at `:153` precedes `LOCK_UN` at `:157`), so a second
acquirer in that window achieves precisely what criterion 2 forbids. Fixing only
`is_stale` would leave criterion 2 satisfied in appearance and false in fact. I am
fixing it and flagging it here rather than widening silently — if the evaluator reads
criterion 2 more narrowly, this is the item to challenge.

**(b) An existing test must be weakened, and I am naming it.**
`test_phase_38_6_restart_survivable.py:151` hard-asserts `_LOCK_TTL_SEC == 90*60`.
Criterion 3 cannot be met while that assertion stands. It is a test, not immutable
criteria, so changing it is legitimate — but "the fix required weakening an existing
guard" is exactly the shape a rubber-stamp hides, so it is disclosed here, will be
disclosed in `experiment_results_85.5.md`, and is replaced by a **stronger,
mutation-resistant** assertion (TTL *tracks* the setting — monkeypatch the budget and
assert the derived TTL moves) rather than deleted.

**(c) The Slack watchdog wording is touched, minimally, and needs no restart today.**
`scheduler.py:133` branches on `is_stale` to print "STALE lock — backend looks DOWN".
Once release stops unlinking, a completed cycle leaves a pidfile with a dead pid, so
that branch would fire routinely and read as a fault. The line is **appended to
alerts and never suppresses one** (`scheduler.py:125`), so this is cosmetic, not a
safety regression. `cycle_lock` will write a `{"state": "released"}` payload and
`_cycle_state_line()` will read it. The running bot process picks this up only on its
next restart; until then it degrades to the cosmetically-wrong-but-safe wording. **No
service restart is performed this cycle** (goal rule: sequence restarts clear of other
steps' evidence windows).

**(d) Out of scope, no prose-only mentions.** Nothing else is changed. Criterion 5
forbids touching order/sizing/risk logic and I touch none of it.

## 6. Plan

1. **Predicate** — `is_stale = (not pid_alive)`. Age never condemns. `inspect_lock()`
   keeps reporting `age_sec` and gains a separate `age_exceeds_budget` field for
   observability, which gates nothing.
2. **Acquire path** — delete the stale-reacquire branch (`:119-132`) outright. On
   `BlockingIOError`: close fd, raise `CycleLockError`. A live flock holder means a
   live cycle by kernel guarantee; a dead holder cannot hold a flock; the branch's
   premise is empty.
3. **Verify-after-lock** — bounded `(st_dev, st_ino)` re-verification loop
   (max 5 attempts) per portalocker #115.
4. **Release** — stop unlinking. Write `{"state": "released", ...}`, then `LOCK_UN`,
   then close. Keep exactly one unlink site (`clean_stale_lock` at startup), gated on
   the strong test: only a process that has just *won the flock* may remove the file.
5. **TTL** — derive lazily from `settings.paper_cycle_max_seconds` × a multiple,
   for observability only. No fallback to the stale `1800.0`.
6. **Tests** — new `backend/tests/test_phase_85_5_cycle_lock_split_brain.py`, named so
   `-k "cycle_lock or 85_5"` collects it. Real second process for criteria 2 and 4
   (`multiprocessing`/`subprocess`, not mocks), non-vacuous fixture that first
   demonstrates the old path acquiring, `_LOCK_PATH`/`_HANDOFF` monkeypatched to
   `tmp_path` so nothing touches the real lock.
7. **Mutation-test every guard** — revert each fix in turn and confirm the
   corresponding test fails. A guard that cannot fail does not count.

## 7. References

- `handoff/current/research_brief_85.5.md` — the gate output (10 sources in full)
- portalocker 4.0.0 / issue #115 — verify-after-lock remedy
- `filelock` in-source invariant; Flohr, *Never Delete Your PID File*
- `backend/services/cycle_lock.py` — subject
- `backend/services/autonomous_loop.py:302-330, 439, 1710-1719` — sole acquire + release
- `backend/main.py:265-276` — startup recovery
- `backend/slack_bot/scheduler.py:120-141` — read-only observability consumer
- `backend/config/settings.py:33`, `backend/api/settings_api.py:171` — budget source
