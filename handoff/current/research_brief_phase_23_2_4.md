# phase-23.2.4 Research Brief -- pause/resume deadlock non-regression

**Tier:** simple (operator override: per `feedback_never_skip_researcher`,
researcher spawns per step even on small fix verification)
**Date:** 2026-05-23
**Step:** Verify pause/resume deadlock did not regress (P0)
**Origin step:** phase-23.1.22 (commit `0ed72940`, 2026-04-30)

---

## Section A -- Internal audit (file:line)

### A.1 The fix commit

- **SHA `0ed72940`** -- "phase-23.1.22: kill_switch reentrant-lock
  deadlock + 23.1.20+21 hardening" (2026-04-30 18:57 +0200).
- Symptom: pause/resume causing 30s timeouts; the entire asyncio
  event loop frozen.
- Diagnosis: `faulthandler.register(SIGUSR1)` (shipped in
  phase-23.1.21) dumped a live hung backend at 18:42:54 showing
  `snapshot()` blocked while `resume()` already held `self._lock`.
  Confirms `threading.Lock` non-reentrancy was the root cause.
- Pre-fix live timing: infinite deadlock on resume.
- Post-fix live timing (from commit message):
  - pause:   0.00s
  - resume:  1.49s
  - pause-2: 0.00s

### A.2 The locking site -- `backend/services/kill_switch.py`

The deadlock surface is confined to ONE module. `paper_trader.py`
has zero `_lock`/`threading.` references (verified via grep -- no
output). The fix shape:

- L46: `self._lock = threading.Lock()` -- still `Lock`, NOT `RLock`.
  The fix kept the non-reentrant primitive and restructured the
  call graph to avoid re-entry. This is the documented preferred
  pattern (see Section B refs).
- L109-123: `_snapshot_locked()` -- private helper, "Caller MUST
  already hold `self._lock`". Pure read; no lock acquisition.
- L125-127: `snapshot()` -- public path that DOES take the lock,
  then delegates to `_snapshot_locked()`.
- L130-157: `pause(trigger, details)` -- acquires lock, mutates,
  appends audit row, calls `_snapshot_locked()` (NOT `snapshot()`),
  exits lock; post-lock alert dispatch outside critical section.
- L159-166: `resume(trigger, details)` -- mirrors pause; calls
  `_snapshot_locked()` inside the `with` block.
- L168-189: `update_sod_nav`, `update_peak` -- mutate state inside
  lock but DO NOT call `snapshot()`, so no re-entry. Clean.

Lock-call inventory (six `with self._lock:` sites, all tight):

| Line | Method | Re-entry risk? |
|------|--------|----------------|
| 106 | `is_paused` | No (single read) |
| 126 | `snapshot` | No (calls `_snapshot_locked`) |
| 131 | `pause` | No (calls `_snapshot_locked`, NOT `snapshot`) |
| 160 | `resume` | No (calls `_snapshot_locked`, NOT `snapshot`) |
| 179 | `update_sod_nav` | No (no snapshot call) |
| 186 | `update_peak` | No (no snapshot call) |

### A.3 The API surface -- `backend/api/paper_trading.py`

- L451 `GET /api/paper-trading/kill-switch` -- read-only state
  + breach calc; BQ portfolio fetch wrapped in
  `async with asyncio.timeout(5)` (L463); degrades gracefully to
  `nav=0.0` on timeout.
- L492 `POST /api/paper-trading/pause` -- requires
  `confirmation == "PAUSE"` (L494); calls `pause(trigger="manual")`;
  no BQ I/O; near-instant.
- L501 `POST /api/paper-trading/resume` -- requires
  `confirmation == "RESUME"` (L503); 5s `asyncio.timeout` on BQ
  breach-check (L514); returns 503 + `Retry-After: 5` if BQ hangs;
  returns 409 if any limit is still breached (L532).
- L544 `POST /api/paper-trading/flatten-all` -- requires
  `confirmation == "FLATTEN_ALL"`; closes positions THEN pauses
  with `trigger="manual_flatten"` and `details=result`.
- L46 `KillSwitchActionRequest(BaseModel)` -- pydantic model
  carrying the `confirmation` string.

### A.4 The audit-log shape -- `handoff/kill_switch_audit.jsonl`

Append-only JSONL written by `KillSwitchState._append_audit`
(L92-102). Live distribution as of 2026-05-23 (226 rows total):

```
   1 "event": "cleanup"
 155 "event": "pause"
  10 "event": "peak_update"
  44 "event": "resume"
  16 "event": "sod_snapshot"
```

Row schema (verified from tail-20 sample):

```json
{"ts": "2026-05-22T05:07:21.273041+00:00",
 "event": "resume",
 "trigger": "manual",
 "details": {}}
```

A "clean transition" is the pair `pause` -> `resume` rows landed
within milliseconds of the API call returning. A regression
signature would be: HTTP request returned 30s timeout AND no row
landed in the JSONL (process hung between `with self._lock:` entry
and `_append_audit` line append), OR the row landed but the API
call hung at the snapshot step (row exists, response never
returned).

### A.5 Existing regression tests (already in repo)

- `tests/services/test_kill_switch_no_deadlock.py` -- 4 tests:
  1. `test_pause_does_not_deadlock_on_self_lock` -- spawns a
     daemon thread doing `state.pause(trigger="test")`, asserts
     `threading.Event` set within 2s.
  2. `test_resume_does_not_deadlock_on_self_lock` -- mirror for
     resume.
  3. `test_pause_resume_cycle_is_fast` -- full pause -> resume
     -> pause cycle asserted `<1s` (the same shape the masterplan
     step verifies, but with a tighter budget than the operator's
     `<5s`).
  4. `test_snapshot_locked_helper_present` -- source-level guard
     via regex: `pause()` body must contain `self._snapshot_locked()`
     and the file must carry the `phase-23.1.22` marker. This is
     the **mutation-resistance test**: it fails the next time a
     well-meaning refactor reverts `_snapshot_locked` -> `snapshot`.
- `tests/api/test_pause_resume_timeout.py` -- 3 tests:
  1. `test_resume_returns_503_when_bq_hangs` -- `mock` slow BQ
     `time.sleep(6)`; resume must return 503 + `Retry-After: 5`
     in <9s (5s asyncio.timeout + ~1s overhead + threadpool
     shutdown wait).
  2. `test_kill_switch_status_degrades_gracefully_when_bq_hangs`
     -- GET endpoint returns paused + thresholds even with
     `nav=0.0` on BQ timeout.
  3. `test_pause_unaffected_no_bq_call` -- pause has no BQ I/O;
     `<1s`.
- Both files use a `tmp_path` `_isolated_kill_switch_audit`
  fixture (phase-23.2.22) that monkeypatches `_AUDIT_PATH` so
  tests cannot pollute the production `handoff/kill_switch_audit.jsonl`.
  Critical for regression hygiene -- earlier cycles polluted prod
  log with `bench-1/2/3` rows.

---

## Section B -- External sources (>=5 in full)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.python.org/3/library/threading.html | 2026-05-23 | Official doc | WebFetch in full | "A primitive lock is a synchronization primitive that is not owned by a particular thread when locked." Lock acquire blocks until release from ANOTHER thread -> same-thread reacquire = indefinite block. RLock: "may be acquired multiple times by the same thread." |
| https://realpython.com/python-thread-lock/ | 2026-05-23 | Authoritative blog | WebFetch in full | Names the exact anti-pattern: `deposit()` holds lock, calls `_update_balance()` which tries to re-acquire same lock -> "The thread hangs indefinitely because `._update_balance()` tries to acquire the lock that's already held by `.deposit()`." Direct analogue of the pre-23.1.22 `pause()` -> `snapshot()` bug. |
| https://fastapi.tiangolo.com/async/ | 2026-05-23 | Official doc | WebFetch in full | When a sync library must be called from an async route, declare the route `async def` and isolate the blocking call (FastAPI handles via threadpool when route declared `def`; explicit `asyncio.to_thread`/`run_in_threadpool` when route declared `async def`). |
| https://www.techbuddies.io/2026/01/10/case-study-fixing-fastapi-event-loop-blocking-in-a-high-traffic-api/ | 2026-05-23 | Industry blog (2026) | WebFetch in full | Event-loop block symptoms: "throughput plateaued far earlier than it should have, while latency ballooned ... moderate CPU, healthy dependencies, but rising tail latency and timeouts." Detection: background coroutine that schedules tiny tasks and measures lateness -- exactly what `faulthandler` on SIGUSR1 produces on demand for Python. Fix: `await asyncio.to_thread(func, ...)`. |
| https://runebook.dev/en/docs/python/library/threading/threading.RLock | 2026-05-23 | Official-derived doc | WebFetch in full | "If your code is not recursive ... a standard `threading.Lock` is sufficient and slightly simpler." Recommends Lock as default; RLock only for "recursive functions or complex nested locking scenarios." This validates the 23.1.22 design choice (keep Lock; refactor to avoid re-entry; do NOT swap to RLock). |
| https://www.digitalapplied.com/blog/agent-audit-trail-design-7-best-practices-2026 | 2026-05-23 | Industry guidance (2026) | WebFetch in full | Append-only JSONL: "no schema migrations against historical records; new fields are added forward-only, with the old records preserved exactly as written." Required fields: identity, action, provenance, outcome. Validates the existing schema `{ts, event, trigger, details}` and the prod 226-row chain-of-custody. |

### Snippet-only (snippet evidence, did NOT count toward gate floor)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://superfastpython.com/thread-anti-patterns/ | Blog | Page exists but does NOT cover the nested-lock anti-pattern (confirmed via WebFetch; 5 other anti-patterns covered) |
| https://realpython.com/lessons/avoiding-deadlocks-rlock/ | Video tutorial | Title relevant but video transcript not fetched -- relpython.com blog (above) covers the same material in text |
| https://www.geeksforgeeks.org/python/python-difference-between-lock-and-rlock-objects/ | Tutorial | Redundant with realpython + Python docs |
| https://www.bomberbot.com/python/mastering-concurrency-a-deep-dive-into-pythons-lock-and-rlock-objects/ | Blog | Redundant with realpython + Python docs |
| https://medium.com/@codingguy/understanding-thread-lock-and-thread-release-and-rlock-b95e1ceb4a17 | Blog | Redundant |
| https://oneuptime.com/blog/post/2026-01-24-fix-race-condition-test-failures/view | Blog (2026) | Race-condition test failures, tangential to deadlock verification |
| https://pypi.org/project/pytest-timeout/ | Tool | Already covered via the existing test fixtures using `threading.Event.wait(timeout=2.0)` -- the project pattern is event-based, not pytest-timeout based |
| https://py-free-threading.github.io/testing/ | Official guide | Python free-threading not relevant to GIL-on production |
| https://docs.pytest.org/en/stable/explanation/flaky.html | Official doc | Tangential to deadlock-specific tests |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | Vendor doc | Re-fetched; does NOT contain explicit regression-verification-as-step guidance. Snippet-only because the answer to "why verify a fix is still in place" is the harness contract itself, not a quotable passage. |
| https://www.3forge.com/audit-trail.html | Vendor doc | Audit-trail product page; the project's existing schema citations already cover the ESMA / 3forge angle from earlier phases |
| https://github.com/fastapi/fastapi/discussions/14893 | Discussion | "Footgun -- using Request directly in a sync route can cause thread to get stuck forever" -- relevant but tangential to the kill_switch class |

---

## Section C -- Recommended verification protocol

### C.1 The required evidence (from masterplan 23.2.4)

> "Run live pause-resume-pause cycle through the API; each must
> complete in <5s; tail handoff/kill_switch_audit.jsonl for clean
> transitions"

### C.2 Step-by-step verification

Order: pytest first (cheap, deterministic), then live curl
sanity-check, then JSONL tail. Pytest already encodes the
mutation-resistance check that protects against silent
re-introduction of the bug -- run it first.

1. **Backend up**. From repo root:
   ```
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000
   ```
   Wait for "Uvicorn running on http://127.0.0.1:8000".

2. **Run the regression suite (with tmp_path isolation)**:
   ```
   source .venv/bin/activate
   python -m pytest -q \
     tests/services/test_kill_switch_no_deadlock.py \
     tests/api/test_pause_resume_timeout.py
   ```
   Expected: 7 passed in <15s. These run in tmp_path; they will
   NOT pollute `handoff/kill_switch_audit.jsonl`. Quote the full
   exit-0 line + count in `experiment_results.md`.

3. **Current state check** (DO NOT mutate yet):
   ```
   curl -s http://127.0.0.1:8000/api/paper-trading/kill-switch \
     | python -m json.tool
   ```
   Note the `"paused"` field. If `true`, the live cycle below
   needs to START with resume; if `false`, START with pause.
   Capture this in `experiment_results.md`.

4. **Capture pre-cycle JSONL tail position** (for diff later):
   ```
   wc -l handoff/kill_switch_audit.jsonl
   ```
   Record the count.

5. **Live cycle -- pause** (timed via `time`):
   ```
   time curl -s -X POST http://127.0.0.1:8000/api/paper-trading/pause \
     -H "Content-Type: application/json" \
     -d '{"confirmation": "PAUSE"}' \
     | python -m json.tool
   ```
   Expected: HTTP 200, `{"status": "paused", "state": {...}}`,
   wall-clock `real` time well under 5s (per A.1 live proof, ~0.00s).

6. **Live cycle -- resume**:
   ```
   time curl -s -X POST http://127.0.0.1:8000/api/paper-trading/resume \
     -H "Content-Type: application/json" \
     -d '{"confirmation": "RESUME"}' \
     | python -m json.tool
   ```
   Expected: HTTP 200 (or 409 if limits breached -- in which case
   the test must first force healthy NAV; for verification
   purposes we expect a HEALTHY resume), `{"status": "resumed",
   "state": {...}}`, wall-clock under 5s.

7. **Live cycle -- pause-2**:
   ```
   time curl -s -X POST http://127.0.0.1:8000/api/paper-trading/pause \
     -H "Content-Type: application/json" \
     -d '{"confirmation": "PAUSE"}' \
     | python -m json.tool
   ```
   Expected: HTTP 200, under 5s.

8. **Tail JSONL for clean transitions**:
   ```
   tail -3 handoff/kill_switch_audit.jsonl
   ```
   Expected output: exactly THREE new rows with matching `ts`
   values from the curl calls in steps 5-7 -- one `event=pause`,
   one `event=resume`, one `event=pause`, all with
   `trigger=manual`, `details={}`. Quote verbatim in
   `experiment_results.md`.

9. **(Cleanup, optional)** If the system started healthy
   (unpaused) but we end paused after step 7, leave a paired
   `resume` to restore the original state -- mention it in
   `experiment_results.md` for audit chain-of-custody:
   ```
   curl -s -X POST http://127.0.0.1:8000/api/paper-trading/resume \
     -H "Content-Type: application/json" \
     -d '{"confirmation": "RESUME"}'
   ```

### C.3 Expected timings

| Step | Pre-fix (regression signature) | Post-fix (expected) | Step threshold |
|------|--------------------------------|---------------------|----------------|
| Pytest (7 tests) | timeouts on test_*does_not_deadlock | <15s total | <30s |
| Live pause | 30s hang (frontend AbortController) | ~0.00-0.05s | <5s |
| Live resume | 30s hang | 1-2s (5s BQ timeout floor) | <5s |
| Live pause-2 | 30s hang | ~0.00-0.05s | <5s |
| JSONL tail | row missing or duplicated | exactly 3 chronologically-ordered rows | one per call |

### C.4 Failure-mode catalog

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `test_pause_does_not_deadlock_on_self_lock` FAILED with "deadlocked" assertion message | someone reverted `_snapshot_locked()` -> `snapshot()` in pause/resume | grep `_snapshot_locked` in `kill_switch.py`; restore the helper-call pattern |
| `test_snapshot_locked_helper_present` FAILED | helper removed entirely OR phase-23.1.22 marker removed | restore the helper + marker; the marker is a load-bearing fingerprint per the test |
| Live curl pause hangs >5s | event loop block somewhere else (likely NOT in kill_switch but in a middleware) | send `kill -SIGUSR1 <backend-pid>`; read stderr for `faulthandler` stack dump |
| `resume` returns 503 | BQ portfolio fetch hung past 5s asyncio.timeout | inspect BQ status -- this is the documented degraded path, NOT a regression |
| `resume` returns 409 | system has live limit breach | this is expected behavior, not a regression; choose a different timing window or document the 409 as "couldn't run resume due to live breach" |
| JSONL tail shows fewer rows than expected | a write between `with self._lock:` and `_append_audit` failed silently | check `backend.log` for `kill_switch: audit write failed` warning |

---

## Section D -- Recency scan (last 2 years, 2024-2026)

Performed against the topic vectors "Python Lock reentrancy", "FastAPI event-loop block", "pause/resume audit-trail JSONL". Two 2026-current sources read in full:

1. `techbuddies.io 2026-01-10` -- FastAPI event-loop blocking
   case study. Confirms diagnostic technique used by phase-23.1.21
   (background coroutine measuring scheduling latency) is the
   current best-practice 2026 approach. The faulthandler
   SIGUSR1 dump used to find the deadlock is the on-demand
   equivalent.
2. `digitalapplied.com 2026` -- Agent audit-trail design.
   Validates the existing JSONL schema; reinforces append-only,
   immutable-record requirement, which the project already meets.

**No new findings supersede the 23.1.22 fix design.** Python's
`threading.Lock`/`RLock` semantics are stable across the 3.9-3.14
window and Python 3.14 docs (read in full) re-confirm the
non-reentrancy. The fix chose to refactor (extract
`_snapshot_locked`) rather than swap to RLock -- the
`runebook.dev` Python 3.14 derived doc explicitly recommends this
direction: keep Lock as the simpler default, refactor to avoid
re-entry.

---

## Section E -- 3-variant queries

Per `.claude/rules/research-gate.md` "Search-query composition":

1. **Current-year frontier (2026)**:
   - "Python threading.Lock RLock reentrant deadlock anti-pattern 2026"
   - "FastAPI asyncio.to_thread sync code blocking event loop pause resume 2026"
   - "ESMA 3forge kill-switch audit trail JSONL pause resume operator 2026"

2. **Last-2-year window (2025)**:
   - "pytest test pause resume kill switch state machine timing assertion 2025"
   - "regression test re-entrant lock fix verification pytest threading.Event timeout 2026"
     (this query returned 2025-flagged sources too)

3. **Year-less canonical**:
   - "threading.Lock deadlock snapshot helper extract private method"
     Surfaced: Microsoft `lock` doc, Python 3.14 threading doc,
     Java tutorial deadlock + livelock, Oracle Java tutorial deadlock,
     bogotobogo Python multithreading 2020. The Python 3.14 doc
     was already in our read-in-full set; Java/C# refs confirm
     the lock-ordering / minimize-critical-section pattern is
     cross-language canon.

---

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

Hard-blocker checklist (`.claude/rules/research-gate.md`):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (18)
- [x] Recency scan (last 2 years) performed + reported (Section D)
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Section A)

---

## Section G -- Application notes for the planner

Five bullets the contract should reflect:

1. **Test order: pytest -> live curl -> JSONL tail.** The pytest
   suite already encodes the mutation-resistance check via
   `test_snapshot_locked_helper_present` (regex grep on
   `kill_switch.py`). Run it FIRST so the static guard is
   asserted before any live state is mutated. If pytest fails, do
   NOT proceed to live curl -- there is a regression in the
   source; live timing can mask it.

2. **Assert <5s with a 3s budget in practice.** The masterplan
   sets the threshold at <5s, but live proof from the
   phase-23.1.22 fix commit shows pause/resume at ~0.00s
   (pause) / ~1.5s (resume includes the 5s asyncio.timeout
   floor on BQ breach-check, returning fast on healthy BQ).
   Any single API call >3s in the post-fix codebase is a yellow
   flag worth investigating, even if <5s.

3. **JSONL tail must show exactly THREE new rows with timestamps
   matching the curl wall-clock.** Each row carries `trigger="manual"`
   and `details={}` for operator pause/resume; the system event
   alphabet is `pause | resume | sod_snapshot | peak_update`.
   A regression signature would be: HTTP 200 returned but no
   matching JSONL row (write failed silently), OR more than 3
   rows (some other thread auto-paused during the cycle).
   Quote the verbatim tail in `experiment_results.md`.

4. **Cleanup: restore the original paused/unpaused state.** Step 3
   captures the starting state. If the cycle ends in `paused=true`
   and the system started `paused=false`, append a paired manual
   `resume` to leave the audit log balanced. Document this
   explicitly in the experiment results so the auditor sees the
   chain of custody. Earlier phases polluted the prod log with
   `bench-1/2/3` triggers; the test fixtures in phase-23.2.22
   added `tmp_path` isolation specifically to prevent this -- the
   live curl path does NOT have that protection, so the operator
   IS the isolation layer.

5. **Failure mode: if `resume` returns 409 ("limit still
   breached"), that is NOT a deadlock regression.** It is the
   documented `evaluate_breach()` precondition (`paper_trading.py`
   L532). The test should pivot to forcing a healthy NAV first OR
   document the 409 + the breach numbers in the experiment
   results and re-run when the breach window has cleared. Per the
   commit message, the 5s `asyncio.timeout` on the BQ call is the
   defense-in-depth layer; the actual deadlock fix is the
   `_snapshot_locked` refactor. Verify both.

---

## Sources

- [Python 3.14 `threading` documentation -- Lock + RLock](https://docs.python.org/3/library/threading.html)
- [Real Python -- "Python Thread Safety: Using a Lock and Other Techniques"](https://realpython.com/python-thread-lock/)
- [FastAPI official -- "Concurrency and async / await"](https://fastapi.tiangolo.com/async/)
- [Techbuddies -- "Case Study: Fixing FastAPI Event Loop Blocking in a High-Traffic API" (2026-01-10)](https://www.techbuddies.io/2026/01/10/case-study-fixing-fastapi-event-loop-blocking-in-a-high-traffic-api/)
- [Runebook (Python 3.14 derived) -- "RLock vs. Lock"](https://runebook.dev/en/docs/python/library/threading/threading.RLock)
- [Digital Applied -- "Agent Audit Trail Design: 7 Best Practices for 2026"](https://www.digitalapplied.com/blog/agent-audit-trail-design-7-best-practices-2026)
- [Anthropic -- "Harness Design for Long-Running Apps"](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic -- "How We Built Our Multi-Agent Research System"](https://www.anthropic.com/engineering/built-multi-agent-research-system)
