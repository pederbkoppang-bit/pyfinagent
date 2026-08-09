# Kill-switch cluster reconciliation — 2026-08-09

Deliverable of §2 of `goal_full_day_2026-08-09.md`. Every status below is
**measured at source today**, not relayed from the step text. Where a step's own
audit_basis is now stale, that is stated explicitly — several are.

Session: cycle 186 prep. Repo at `6edd0c79` + working tree. Measured ~06:00 CEST.

---

## 0. Live state the reconciliation is measured against

```
GET /api/paper-trading/kill-switch   (localhost:8000, backend pid 36970 under launchd)
  paused: false          sod_nav: 23830.46     sod_date: "2026-08-08"
  peak_nav: 24666.57     current_nav: 23833.94
  breach.armed: false    breach.daily_baseline_stale: true
  breach.daily_baseline_missing: false          breach.trailing_baseline_missing: false
  breach.trailing_dd_pct: 3.3755  (limit 10.0)  breach.daily_loss_pct: 0.0 (limit 4.0)
```

`handoff/kill_switch_audit.jsonl`: **62 lines**,
`sha256 90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653`.

`armed: false` is **case C** — the UTC date rolled past `sod_date`, the daily leg
is correctly unevaluable, the trailing leg is date-independent and still firing.
Not a defect. Not touched.

**Verification NOT performed and why:** phase-85.6's Step-0 roll re-anchors the
daily anchor at the *start of the next cycle*. No cycle has run since
2026-08-08T20:58Z, and the cron is weekday-only, so **no cycle will run today**.
Confirming the Step-0 roll live therefore requires spending the one remaining
authorized verification cycle. Deferred — with the rail dead (ask #26 unchanged,
`sha256[:12]=9f8c63a185d8`), that cycle would also produce 6/6 degraded analyses
and 0 trades, which is already measured. **The Step-0 roll remains proven only by
test, not by a live cycle.** Stated so nobody reads it as confirmed.

---

## 1. The census

35 steps carry a `36.x` or `86.x` id. 14 are `done`/`deferred` and out of scope.
The **19 pending** members of the kill-switch cluster:

| Step | P | Status after inspection | Disposition |
|---|---|---|---|
| **36.21** | P1 | **DUPLICATE of 86.3**, and its stated mechanism is **refuted** | → `superseded` by 86.3 |
| **86.3** | P1 | **OPEN** — root cause identified, survivor of the merge | execute #1 |
| **36.26** | P1 | **OPEN — NOT closed by 85.6.** Live-reachable right now | keep; execute after the 86.x block |
| **36.28** | P2 | **OPEN** — genuinely distinct from 86.3 (read vs write channel) | keep; cross-ref added |
| **36.15** | P1 | **CORE MECHANISM ALREADY FIXED** by phase-36.8; narrow residual remains | → re-scope, do NOT close |
| **86.1** | P1 | **OPEN** — confirmed at source | execute #2 |
| **86.2** | P1 | **OPEN** — confirmed at source | execute #3 |
| **36.17** | P1 | **OPEN** — untouched by 85.4/85.6 | execute #4 |
| **36.10** | P1 | **OPEN** — confirmed by grep, and live-relevant today | keep |
| **36.20** | P1 | **OPEN** — and rendering the wrong badge *right now* | keep; paired with 36.26 |
| **86.5** | P2 | OPEN — triage, blocked on 86.3 | execute #5 |
| **86.4** | P3 | OPEN | tail |
| 36.11, 36.14, 36.16, 36.18, 36.19, 36.22, 36.23 | P1/P2 | OPEN, not inspected in depth this pass | tail |
| 36.24, 36.25, 36.27, 36.29, 36.30 | P1/P2 | OPEN, **not kill-switch** (memory / harness / archive) | out of cluster |

---

## 2. `36.21` ≡ `86.3` — merged, and 36.21's hypothesis is refuted

### They are the same defect

36.21 (filed 2026-07-26) records: running
`pytest backend/tests/ -q -k "paper_trading or resume"` appends **four real rows —
two pause/resume pairs with `trigger=manual`** — to the live journal.

86.3 (filed 2026-08-09) records **8 rows across two full-suite arms** — 4 per arm,
`4 pause/resume pairs` at `22:29:41-43Z` and `22:36:59-22:37:01Z`.

Four rows per run, pause/resume pairs, `trigger=manual`. **Same signature.**

### The writer, and why 36.21 could not find it

`backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py`:

- `BACKEND_URL = "http://localhost:8000"` (line 42)
- `@pytest.mark.skipif(not _backend_is_up(), ...)` where `_backend_is_up()` probes
  `GET /api/health` — **measured 200 right now**
- `test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s` POSTs
  `pause → resume → pause`, then, **if the book was unpaused pre-cycle**, a fourth
  POST `resume` to restore state.

**Unpaused book → 4 rows: pause, resume, pause, resume.** That is exactly the
22:29 and 22:36 clusters, timestamps to the second.

36.21 hypothesised "an ORDERING or STATE-POLLUTION effect … most likely one module
leaving `kill_switch._state` or `_AUDIT_PATH` attached to the real path". **That
hypothesis is wrong.** There is no ordering effect and no singleton leak. The
mechanism is a plain HTTP POST to the running backend.

36.21's census missed it for a precise, reproducible reason. Its file selection
grepped for `resume_trading` / `.resume()` / `.pause()`:

```
$ grep -nE 'resume_trading|\.resume\(\)|\.pause\(\)' \
      backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py
(no match)
```

The file never calls those Python methods — it calls
`_post_state_transition("/api/paper-trading/pause", "PAUSE")`. **It was invisible
to the census that was supposed to find it**, which is why running "each file
alone" never reproduced the write.

(Note: 36.21 says 12 files matched that pattern; today **4** do. The test tree has
changed in the 14 days since. The 4-vs-12 gap is repo drift, not a contradiction,
and no claim here rests on it.)

### The pause-only rows are the same writer, in the deadlock era

Manual-row census of the live journal:

| date | pause | resume |
|---|---|---|
| 2026-07-25 | 4 | 4 |
| 2026-07-27 | 1 | 1 |
| 2026-08-03 | 3 | 0 |
| 2026-08-04 | 12 | 0 |
| 2026-08-06 | 6 | 0 |
| 2026-08-07 | 1 | 0 |
| 2026-08-08 | 17 | 5 |

The pause-only days are 2026-08-03 → 08-07 — **exactly the window in which the
stale-anchor deadlock made `POST /resume` return 409**. `_post_state_transition`
re-raises any non-2xx except a 503 on resume, so the test errored after its first
POST and left **one orphan pause row per run**. Once 85.6 unblocked resume at
20:58Z on 08-08, the runs at 22:29 and 22:36 produced complete 4-row pairs again.

One writer explains both signatures. This also confirms ask #21's finding (12
pause rows on 08-08) as the same mechanism, not a second one.

**2026-07-27's single pause + single resume is the real operator action** — the
36.26 live incident. It is the only manual pair in the file that a human made.

### Disposition

- `36.21` → **`superseded`**, pointing at 86.3.
- 86.3 **absorbs 36.21's two criteria that 86.3 lacked**:
  - a *session-scoped* protection so a new module cannot regress this by omission
    (36.21's criterion 3), and
  - `test_phase_23_2_4_audit_log_clean_transitions` still passes with its trigger
    allowlist **byte-unchanged** (36.21's criterion 4).
- 36.21's ban carries over verbatim: **do not** fix this by gitignoring the
  journal or relaxing the trigger allowlist.

---

## 3. `36.26` is **NOT** closed by 85.6 — and it is live-reachable today

The goal file's hypothesis was that 85.6 closed this. **Measured: it did not.**

85.6 moved the daily-anchor roll to Step 0 of the cycle and rewrote the refusal
message. It did **not** implement 36.26's fix. The staleness gate at
`backend/api/paper_trading.py:621-627` still refuses on the anchor date alone:

```python
if (breach.get("daily_baseline_stale")
        and not breach.get("daily_baseline_missing")
        and not breach.get("trailing_baseline_missing")):
    raise HTTPException(409, "Cannot resume: the daily-loss anchor is STALE ...")
```

`pause_reason` is **never consulted** — grep over `backend/api/paper_trading.py`
finds it at line 526 only, inside the snapshot payload. 36.26's fix direction (a)
— allow resume when `pause_reason == 'manual'` and `any_breached` is false — is
unimplemented.

**Live reachability, today:** `daily_baseline_stale` is `true` right now
(`sod_date=2026-08-08`, UTC today is `2026-08-09`). If the operator pauses
manually this weekend they cannot un-pause. **85.6's own new message says so:**

> "If none is scheduled — the cron is weekday-only, so this includes all weekend —
> no cycle will run and this refusal will **NOT** clear on its own; trigger a
> cycle, or leave the book paused."

85.6 converted a false promise into an honest one. That is a real improvement and
it is what 85.6 was scoped to do. It is not a fix for 36.26: the operator still
cannot reverse their own action.

**Disposition: `36.26` stays `pending`, P1.** Its criteria 1 and 2 are unmet, and
the condition it describes is reachable on the live book at this moment. I did
**not** flip it, and I did not verify it by driving a real pause — doing so would
repeat exactly the harm 86.3 documents.

---

## 4. `36.15` — the core mechanism is already fixed; re-scope the residual

36.15 states the mechanism as: the `peak_reset` replay branch does
"a bare `self._peak_nav = _coerce_nav(row.get('new_peak'))` with NO None-check".

**That is stale.** `backend/services/kill_switch.py:382` now reads:

```python
self._apply_authoritative_peak(row.get("new_peak"), "peak_reset")
```

and `_apply_authoritative_peak` (`:397-430`) coerces, and on `None`:

```python
logger.error("kill_switch: IGNORING an authoritative peak assignment from %s -- "
             "value %r does not coerce to a positive finite NAV. The prior peak "
             "(%s) is retained; ...", source, raw, self._peak_nav)
return
```

Introduced by **phase-36.8, commit `09125a81`** ("an in-stream authority boundary").
That satisfies 36.15's criterion 2 — a malformed `peak_reset` is ignored, logged
loudly, and the true prior peak survives — and removes both harm directions (a)
and (b) that 36.15 describes.

**Residual, genuinely still open:**

1. **No test names this guard.** `grep -rl "_apply_authoritative_peak\|IGNORING an
   authoritative peak" backend/tests/ tests/` returns **nothing**. 36.8's own test
   file exists (`test_phase_36_8_kill_switch_archive_merge_authority.py`) but
   whether it covers the malformed-`peak_reset` case must be measured, not assumed
   — that belongs in 36.15's own research gate. Criterion 6 (mutation-test) is
   therefore unproven either way.
2. **36.15's criterion 5 is undecided:** whether `_append_audit` should also
   *refuse to write* a non-positive `peak_reset`. The read side is guarded; the
   write side is not, and no decision is recorded.

**Disposition: `36.15` stays `pending`, but must be RE-SCOPED before execution** —
an executor following its current text will hunt a defect that no longer exists
and may "fix" an already-guarded branch. The re-scope is a masterplan text edit
and is queued, not applied in this pass (editing a step's `name` is safe; its
`verification` block is immutable and is **not** touched).

---

## 5. `36.28` vs `86.3` — related class, different channel. Both stay open.

- **36.28** is the **read** coupling: `PaperTrader` constructed without the
  `kill_switch_state` injection seam falls back to the module singleton, which
  replays the real on-disk journal — so test *greenness* depends on the operator's
  live pause state. Fix = inject at construction sites.
- **86.3** is the **write** channel: a test POSTs to `localhost:8000` and mutates
  the switch. Fix = contain the HTTP channel.

A worktree relocates `Path(__file__).parents[N]` and therefore addresses part of
36.28. **It does not relocate a TCP connection**, so it cannot address 86.3. These
are not duplicates and must not be merged. 86.3's audit_basis already carries the
instruction to coordinate; that stands.

---

## 6. `36.10` and `36.20` — both confirmed open, both live-relevant today

**36.10** — measured, `armed` is read by nothing outside the UI/API:

```
scripts/away_ops/healthcheck.sh          armed_hits=0
scripts/away_ops/sentinel.sh             armed_hits=0
scripts/away_ops/run_away_session.sh     armed_hits=0
scripts/ops/send_confirmation_digest.py  armed_hits=0
backend/slack_bot/scheduler.py           armed_hits=0
```

The book is `armed: false` **right now** and no away-ops surface can see it. The
current instance is benign (case C, self-clearing at the next cycle) — but the
blindness is total, so a genuine disarm would be equally silent.

**36.20** — `daily_baseline_stale: true` and `armed: false` today means the cockpit
renders the DISARMED alarm badge and disables Resume on a healthy book, which is
exactly the state 36.20 describes. Not re-verified in the browser this pass (36.20
itself requires the isolated `:3100` skip-auth rig, never `:3000`).

**36.20 and 36.26 are two layers of one operator-visible symptom** — 36.20
disables the button, 36.26 is the server-side 409 behind it. Fixing either alone
leaves the operator stuck. They should be sequenced together, and 36.20's criterion
"the Resume button is enabled for exactly the states the server will accept"
**cannot be satisfied until 36.26 decides what the server accepts.** Recorded as a
real ordering dependency: **36.26 before 36.20.**

---

## 7. Execution order this establishes

1. **86.3** (absorbing 36.21) — until the suite stops POSTing to the live book,
   every other baseline measurement corrupts safety state.
2. **86.1** — landmine armed by the owed KS-PEAK-RESET token (79.6, APPROVED).
3. **86.2** — the only measured path to a *total* disarm.
4. **36.17** — halted cycle stops enforcing stop-losses.
5. **86.5** — 26-failure triage (needs #1 first).
6. **36.26** → then **36.20** (ordering established in §6).
7. **36.15** re-scope, **36.10**, **86.4**, remaining 36.x P1s.

## 8. Masterplan edits made by this reconciliation

| Edit | Evidence |
|---|---|
| `36.21` → `superseded` | §2 |
| `86.3` name/audit_basis absorbs 36.21's two extra criteria + the ban | §2 |
| `36.28` audit_basis gains the read-vs-write cross-ref to 86.3 | §5 |

**Nothing was flipped to `done`.** No step's `verification` block was altered —
success criteria are immutable. Every other cluster member keeps its status.
