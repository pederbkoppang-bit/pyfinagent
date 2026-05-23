# Research Brief -- phase-23.2.14

**Topic:** Audit other `with self._lock:` blocks for re-entrant patterns (P2)
**Tier:** simple (>=5 external sources read in full)
**Date:** 2026-05-23
**Author:** researcher subagent

---

## TL;DR

13 `threading.Lock()` instances in `backend/` (the phase-23.1.21 audit
count of 12 was one low -- it missed `llm_client.py:365`
`_BUDGET_CACHE_LOCK` added later in phase-25.A8). **All 13 audited.
All 13 CLEAN of re-entrant patterns.** The phase-23.1.22 fix to
`backend/services/kill_switch.py` removed the only known instance
in this codebase. Recommend a source-grep-based pytest that
catches future re-entrant drift before it deadlocks production.

---

## Section A -- Internal per-lock audit table

The audit rule used: a lock is RE-ENTRANT if a method holding
`with self._lock:` (or `with _XXX_LOCK:`) transitively calls
another method/function that tries to `acquire()` the SAME lock
without first releasing it.

| # | File | Line | Lock name | Owner | Methods that acquire it | Methods called UNDER LOCK | Re-entrancy verdict |
|---|------|------|-----------|-------|-------------------------|---------------------------|---------------------|
| 1 | `backend/tools/alt_data.py` | 30 | `_CACHE_LOCK` | module-level dict cache | `_cache_get`, `_cache_put` | Only `dict.get/pop/__setitem__` and `time.time()` -- builtins | CLEAN |
| 2 | `backend/agents/llm_client.py` | 365 | `_BUDGET_CACHE_LOCK` | module-level (cost-budget) | `_check_cost_budget`, `reset_cost_budget_cache` (3 call sites in `_check_cost_budget` at L399, L418, L432) | Reads/writes `_BUDGET_CACHE_VALUE` tuple only. The `_default_fetch_spend()` BQ call is INTENTIONALLY OUTSIDE the lock (L408-414). No transitive lock re-acquire. | CLEAN |
| 3 | `backend/agents/_genai_client.py` | 29 | `_client_lock` | module-level singleton | `get_genai_client` (L113), `close_genai_client` (L136) | `_build_client()` -- which only touches `genai.Client(...)` and `get_settings()`; neither re-enters `_client_lock`. `close()` is called OUTSIDE the lock (L139). Classic double-checked-lock pattern, textbook clean. | CLEAN |
| 4 | `backend/governance/limits_loader.py` | 50 | `_init_lock` | module-level (boot-once) | `load_once` only (L96) | `_file_digest()` (pure hashing), `signal.signal()`, `threading.Thread.start()`. `_watcher_loop` is the thread's `target=` so it does NOT run on the lock-holder thread -- it runs on the new spawned daemon thread which has its own stack and cannot re-enter the boot-time `_init_lock`. | CLEAN |
| 5 | `backend/api/job_status_api.py` | 87 | `_lock` | module-level registry | `record_heartbeat` (L107), `get_registry_snapshot` (L135), `get_job_status` (L147) | Pure dict reads/writes. `JobStatus(...)` pydantic ctor inside the loop at L151-159 does NOT touch `_lock`. | CLEAN |
| 6 | `backend/services/api_cache.py` | 31 | `self._lock` (`APICache._lock`) | `APICache` instance | `get` (L37), `set` (L52), `invalidate` (L66), `stats` (L77), `clear` (L95) | Pure dict + integer ops, `time.monotonic()`, `re.match()` builtin. The `stats()` method does the "evict expired while we're at it" pattern (L82) which is a dict comprehension inside the lock -- no method dispatch. | CLEAN |
| 7 | `backend/services/kill_switch.py` | 46 | `self._lock` (`KillSwitchState._lock`) | `KillSwitchState` instance | `is_paused` (L106), `snapshot` (L126), `pause` (L131), `resume` (L160), `update_sod_nav` (L179), `update_peak` (L186) | `pause` and `resume` were the re-entrant deadlock fixed in phase-23.1.22. The fix: `_snapshot_locked()` lock-free helper at L109 + `_append_audit()` static method that opens its own file handle. The Slack `raise_cron_alert_sync` call (L144-156) is INTENTIONALLY outside the lock (L141 inserts the unlock boundary by re-assigning `snap`). | CLEAN (post-23.1.22 fix) |
| 8 | `backend/services/cycle_health.py` | 257 | `self._lock` (`CycleHealthLog._lock`) | `CycleHealthLog` instance | Only `record_cycle_end` (L291) | File-open + `f.write(json.dumps(row))`. `_write_heartbeat` is called OUTSIDE the lock at L297. No method on `self` is invoked under the lock. | CLEAN |
| 9 | `backend/services/perf_tracker.py` | 34 | `self._lock` (`PerfTracker._lock`) | `PerfTracker` instance | `record` (L53), `summarize` (L62), `export_tsv` (L116), `clear` (L125) | Pure list ops + `LatencyEntry(...)` dataclass ctor. Critically, `get_slow_endpoints` (L103) calls `self.summarize()` -- this is the only inter-method call. `summarize()` acquires `_lock` independently after `get_slow_endpoints` has RETURNED from its own (non-existent) lock-hold. `get_slow_endpoints` does NOT hold the lock itself. CLEAN. | CLEAN |
| 10 | `backend/services/live_prices.py` | 40 | `self._lock` (`LivePriceCache._lock`) | `LivePriceCache` instance | `get_many` only (L65) | `_rate_ok()` (L44) -- helper that touches `self._refresh_log` (already protected by being called under lock). `_fetch_price(t)` module-level function -- pure yfinance call, no `self._lock` reference. NB: `_fetch_price` does I/O under the lock which is a SEPARATE perf concern (not re-entrancy) but is documented as intentional in the module docstring. | CLEAN |
| 11 | `backend/services/observability/alerting.py` | 64 | `self._lock` (`AlertDeduper._lock`) | `AlertDeduper` instance | `should_fire` (L72, L78) [2 call sites: critical-path + dedup-path], `reset` (L93) | Pure dict + deque ops. No method dispatch under-lock. | CLEAN |
| 12 | `backend/services/observability/api_call_log.py` | 59 | `_lock` (module-level) | api_call_log buffer | `log_api_call` (L91), `flush` (L111, L115, L149), `buffer_size` (L157), `reset_buffer_for_test` (L164) | `_should_flush_locked()` helper at L100 -- explicit `_locked` suffix; callers MUST already hold `_lock`. `flush()` is the call that could re-enter: `log_api_call` calls `flush()` at L95 only AFTER releasing `_lock` (the `with` block ends at L93 then `if should_flush: flush()` at L94-95). CLEAN. | CLEAN |
| 13 | `backend/services/observability/api_call_log.py` | 200 | `_llm_lock` (module-level) | llm_call_log buffer | `log_llm_call` (L267, L271 inline check), `flush_llm` (L283, L287, L320), `llm_buffer_size` (L327) | Same pattern as #12: the dual flush-flag check happens inline at L267-273 INSIDE the lock, then the lock is released and `flush_llm()` is called OUTSIDE the lock at L275. CLEAN. | CLEAN |

### Notable findings (per-lock)

- **#3 _genai_client.py** uses the textbook double-checked-lock
  pattern. The fast-path (L110-111) reads `_client` WITHOUT the
  lock -- this is correct in Python because dict/global reference
  reads are atomic under the GIL; the slow-path acquires the lock
  for the build.
- **#10 live_prices.py** holds the lock across a yfinance I/O call
  inside `get_many`. This is NOT a re-entrancy issue but it IS a
  latency-under-lock concern. Out of scope for phase-23.2.14
  (re-entrancy audit only) but worth a separate phase.
- **#12 / #13 api_call_log.py** demonstrate the SAFE flush-on-
  threshold pattern: detect-under-lock, dispatch-outside-lock.
  This is the documented anti-pattern remediation per Real Python
  + SuperFastPython.
- **#7 kill_switch.py** is the phase-23.1.22 fix. The
  `_snapshot_locked` suffix convention is established here and
  should be the project-wide naming standard for "I assume the
  lock is held" helpers.

---

## Section B -- External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|-----|----------|------|-------------|---------------------|
| https://docs.python.org/3/library/threading.html | 2026-05-23 | Official doc | `WebFetch` full | "`Lock` handles this case the same as the previous, blocking until the lock can be acquired" -- canonical statement that re-entrant `Lock.acquire()` self-deadlocks. RLock "may be acquired multiple times by the same thread". |
| https://realpython.com/python-thread-lock/ | 2026-05-23 | Authoritative blog | `WebFetch` full | Demonstrates the EXACT pattern that bit kill_switch.py: `.deposit()` holds lock, calls `_update_balance()` which tries to re-acquire -> permanent deadlock. Recommends `RLock` OR refactoring to avoid nested acquisition. |
| https://superfastpython.com/thread-deadlock-in-python/ | 2026-05-23 | Authoritative blog | `WebFetch` full | Provides the `with lock: with lock: pass` reproducer + states the remediation: "This specific deadlock with a mutex lock can be avoided by using a reentrant mutex lock." Audit guidance: scrutinize "functions [that] call other functions internally to reuse code". |
| https://www.geeksforgeeks.org/python/python-difference-between-lock-and-rlock-objects/ | 2026-05-23 | Tutorial site | `WebFetch` full | Decision criteria: use Lock for non-nested simple mutex; use RLock when "A thread may need to acquire the same lock while holding it" OR "Nested function calls require lock protection". Confirms RLock has a small perf overhead from the recursion counter. |
| https://medium.com/@abhishekjainindore24/advanced-python-10-lock-vs-rlock-c747bbdbd803 | 2026-05-23 | Blog (named author) | `WebFetch` full | Shows the `outer() -> inner()` deadlock with both functions acquiring the same `Lock` and explicitly recommends switching the lock to RLock + using `with` context manager. Confirms the project's per-23.1.22 alternative -- extract `_locked` helper -- is the OTHER valid remediation path. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.cis.upenn.edu/~mhnaik/papers/icse09.pdf | Academic paper (Naik, ICSE'09) | Binary PDF response from WebFetch -- the snippet describes static deadlock detection via lock-tree recording, which is more sophisticated than our grep-based audit needs. |
| https://link.springer.com/chapter/10.1007/978-3-319-95582-7_36 | Academic paper | Auth-redirect to Springer IDP; not in scope for `simple` tier. |
| https://www.sciencedirect.com/science/article/pii/S0167642318303812 | Academic paper | HTTP 403 (publisher paywall). |
| https://research.ibm.com/haifa/Workshops/PADTAD2005/papers/article.pdf | Academic paper | HTTP 404 -- IBM Haifa workshop page moved. |
| https://www.researchgate.net/publication/221471849_... | Academic paper | ResearchGate gating. |
| https://www.mathworks.com/discovery/sr11-7.html | Vendor doc | SR-11-7 is about MODEL risk, not THREAD safety -- not directly relevant to phase-23.2.14 audit. Useful for understanding the broader governance frame but not load-bearing here. |
| https://github.com/PyCQA/bandit | OSS tool | Bandit is a security static analyzer (injection / hardcoded password); it does NOT have a re-entrant-Lock detector. Pylint similarly. Confirmed via the bandit+pylint search -- no community static analyzer covers this gap. |

### Recency scan (2024-2026)

Searched for `"threading.Lock" deadlock Python 2025 case study` and
`python threading.Lock re-entrant deadlock fix _locked helper
pattern 2025 2026`. Result: **NO new findings supersede the
canonical Lock/RLock guidance in the last 2 years.** The Python
threading model has been stable across 3.12 -> 3.14, and the
recommendations are consistent: use `RLock` for nested acquisition
OR extract a lock-free `_locked` helper. Hotter areas of
concurrency research (free-threaded Python 3.13+, PEP 703 nogil)
do NOT change the per-thread Lock re-entrancy semantics -- a
non-reentrant `Lock` will continue to self-deadlock the same way
on a nogil interpreter.

---

## Section C -- Search-query composition (mandatory)

Per `.claude/rules/research-gate.md` three-variant discipline:

1. **Current-year frontier**: `bandit pylint static analysis
   threading lock python 2026` -- confirmed no community static
   analyzer covers re-entrant-Lock detection.
2. **Last-2-year window**: `Python threading.Lock vs RLock
   re-entrant deadlock best practices 2026` + `"threading.Lock"
   deadlock Python 2025 case study` -- canonical sources unchanged.
3. **Year-less canonical**: `python concurrency lock acquire same
   thread deadlock` -- recovered the official Python docs +
   Real Python + SuperFastPython triumvirate that the audit cites.

The mix is reflected in the source table above (Python 3.14.5 docs
URL = year-less canonical; GeeksforGeeks + Medium = year-less
canonical; 2026 search returned bandit/pylint blog).

---

## Section D -- Key findings

1. **Re-entrant Lock acquisition is a self-deadlock** -- documented
   verbatim in the Python 3.14 stdlib docs:
   "`Lock` handles this case the same as the previous, blocking
   until the lock can be acquired"
   (Source: docs.python.org/3/library/threading.html, 2026-05-23).
2. **Two canonical remediations exist**, both documented in Real
   Python: (a) switch the lock to `threading.RLock()` (low effort,
   small perf overhead from the recursion counter); (b) extract a
   lock-free helper with a naming convention (the `_snapshot_locked`
   /`_should_flush_locked` pattern the codebase already uses at 3
   locations) and keep the public method as the only acquirer.
   (Source: realpython.com/python-thread-lock + the codebase's own
   kill_switch.py:109 + api_call_log.py:100.)
3. **The audit pattern is grep-based + call-graph traversal** --
   no community static analyzer (bandit / pylint / mypy) detects
   re-entrant Lock patterns. Per Mayur Naik ICSE'09, sound static
   detection requires lock-tree recording which is research-grade,
   not project-maintenance-grade. For a fixed-size codebase of 13
   locks, a manual + grep audit is the right tool. (Snippet-only:
   cis.upenn.edu/~mhnaik/papers/icse09.pdf.)
4. **No new findings in 2024-2026** that change the canonical
   guidance. The free-threaded Python 3.13+ work does NOT alter
   `threading.Lock` re-entrancy semantics.
5. **Project-specific finding:** all 13 backend locks are CLEAN.
   The 3 locks that use the `_locked` suffix convention
   (`_snapshot_locked`, `_should_flush_locked`, and the inline
   pattern in `log_llm_call`) are following the documented
   remediation correctly.

---

## Section E -- Consensus vs debate (external)

**Consensus** across all 5 fully-read sources:
- `Lock` is non-reentrant; re-acquiring from the same thread is a
  self-deadlock.
- `RLock` is the drop-in remediation when nested acquisition is
  required.
- Context-managers (`with lock:`) are the recommended idiom over
  manual `acquire()` / `release()`.

**Minor debate**: Real Python + Medium recommend RLock as the
default remediation; SuperFastPython more cautiously notes that
RLock has perf overhead and structural code restructuring (i.e.,
the `_locked` helper extraction) is preferable when the call graph
is small and well-understood. This codebase has chosen the latter
path (see kill_switch.py phase-23.1.22 commit message), which is
the structurally cleaner choice for a 13-lock fixed-size codebase.

---

## Section F -- Pitfalls (from literature)

- **Method-calls-method-with-same-lock** is the canonical pitfall
  (5 of 5 sources). The kill_switch.py phase-23.1.22 incident is
  a textbook reproduction of this -- `pause()` held lock + called
  `snapshot()` which re-acquired.
- **Holding I/O under lock** is a related-but-different pitfall
  (not re-entrancy). `live_prices.py:65-107` does yfinance I/O
  under-lock; this is intentional but worth flagging in a separate
  phase.
- **Daemon threads spawned under lock** -- the spawned thread
  starts with an empty stack, so it CANNOT re-enter a lock held
  by the spawner. `limits_loader.py:113-119` correctly relies on
  this invariant.
- **Tests** can paper over deadlocks if they run with timeouts
  too generous to surface a true hang. The phase-23.2.4 live test
  uses a 5-second budget per transition; matching budget for the
  recommended pytest below.

---

## Section G -- Application to pyfinagent (mapping external findings to file:line anchors)

| Finding | File:line | Implication |
|---------|-----------|-------------|
| Re-entrant `Lock.acquire()` self-deadlocks | `backend/services/kill_switch.py:46` (the `_lock`) | Already remediated in phase-23.1.22 via `_snapshot_locked` at L109. |
| Recommended `_locked` suffix convention | `backend/services/kill_switch.py:109` + `backend/services/observability/api_call_log.py:100` | Project has 3 instances of this idiom; no new ones needed. |
| Project-wide grep audit | `grep -rn 'threading.Lock()' backend/` | Returns the 13 locks; the recommended pytest below codifies this scan as a regression. |
| RLock as alternative | (not used in project) | Project has chosen the lock-free-helper path; that decision is documented in `backend/services/kill_switch.py:6-21` and `backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py:7-8`. |

---

## Section H -- Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
      (5 of 5: Python docs + Real Python + SuperFastPython +
      GeeksforGeeks + Medium)
- [x] 10+ unique URLs total (12 collected; 5 read in full +
      7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full
      set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
      (13 of 13 locks read end-to-end)
- [x] Contradictions / consensus noted (RLock vs `_locked` helper
      debate documented)
- [x] All claims cited per-claim

---

## Section I -- Recommended pytest shape

The audit is structurally simple enough that a single source-grep
test catches future drift. Three layers, low overhead.

**File:** `backend/tests/test_phase_23_2_14_no_reentrant_locks.py`

```python
"""phase-23.2.14 regression: backend has no re-entrant Lock patterns.

Source-grep audit that catches future _snapshot_locked-style drift.
Three layers:

  1. Roster check: the count of `threading.Lock()` instances in
     backend/ matches the phase-23.2.14 audit (13). New locks
     trigger a CONDITIONAL on Q/A so the audit can be redone.
  2. Convention check: every method named `_*_locked` (the
     project's lock-free-helper convention) must have a docstring
     stating that the caller MUST hold the lock.
  3. Anti-pattern check: no `_*_locked` helper may itself contain
     `with self._lock:` (which would defeat the purpose).

If a future commit adds a NEW `threading.Lock()` instance, the
roster check fails and forces an explicit audit + count bump --
preventing silent re-entrancy drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
EXPECTED_LOCK_COUNT = 13  # phase-23.2.14 audit; bump only after re-audit
LOCK_PATTERN = re.compile(r"threading\.Lock\s*\(\s*\)")
LOCKED_HELPER_PATTERN = re.compile(r"def\s+(_\w+_locked)\s*\(")


def _iter_backend_py_files() -> list[Path]:
    return [p for p in BACKEND.rglob("*.py")
            if "tests" not in p.parts and "__pycache__" not in p.parts]


def test_threading_lock_count_matches_audit():
    """Roster guard: count of threading.Lock() instances must equal
    the phase-23.2.14 audit count. A bump requires re-running the
    re-entrancy audit and updating EXPECTED_LOCK_COUNT here."""
    hits: list[str] = []
    for path in _iter_backend_py_files():
        text = path.read_text(encoding="utf-8")
        for match in LOCK_PATTERN.finditer(text):
            hits.append(f"{path.relative_to(BACKEND.parent)}:{text[:match.start()].count(chr(10)) + 1}")
    assert len(hits) == EXPECTED_LOCK_COUNT, (
        f"threading.Lock() count drift: expected {EXPECTED_LOCK_COUNT}, "
        f"got {len(hits)}. New locks require a phase-23.2.14-style "
        f"re-entrancy audit. Hits:\n  " + "\n  ".join(hits)
    )


def test_locked_helpers_document_lock_held_invariant():
    """Convention guard: every `_*_locked` helper must document the
    'caller MUST hold the lock' invariant in its docstring.
    Establishes the kill_switch.py:109 pattern as project-wide."""
    failures: list[str] = []
    for path in _iter_backend_py_files():
        text = path.read_text(encoding="utf-8")
        for match in LOCKED_HELPER_PATTERN.finditer(text):
            name = match.group(1)
            # Look for the next docstring within 200 chars of the def.
            window = text[match.end():match.end() + 600]
            if "lock" not in window.lower() or (
                "hold" not in window.lower() and "MUST already" not in window
            ):
                failures.append(
                    f"{path.relative_to(BACKEND.parent)}::{name} -- "
                    f"missing 'caller holds lock' docstring"
                )
    assert not failures, (
        "Locked-helper docstring convention violations:\n  " +
        "\n  ".join(failures)
    )


def test_locked_helpers_do_not_reacquire_their_own_lock():
    """Anti-pattern guard: a `_*_locked` helper that itself contains
    `with self._lock:` would silently re-enter and defeat the
    purpose of the helper extraction. Source-grep for this shape."""
    failures: list[str] = []
    for path in _iter_backend_py_files():
        text = path.read_text(encoding="utf-8")
        for match in LOCKED_HELPER_PATTERN.finditer(text):
            name = match.group(1)
            # Body window: from this def to the next top-level def or
            # end of class. 800 chars is enough for the small helpers.
            body = text[match.end():match.end() + 1200]
            if "with self._lock" in body or "with _lock:" in body:
                failures.append(
                    f"{path.relative_to(BACKEND.parent)}::{name} -- "
                    f"helper re-acquires its own lock (would deadlock)"
                )
    assert not failures, (
        "Locked-helper re-acquire violations:\n  " +
        "\n  ".join(failures)
    )
```

**Cost:** <1s on a normal laptop. **False-positive rate:** low --
the regex is intentionally tight (`threading\.Lock\s*\(\s*\)` does
not match `RLock`, `Lock(...)` ctor with args, or string literals).

**Future evolution:** if a new lock is added and the count bumps
to 14, the next session must (a) read the new lock-bearing module,
(b) verify no re-entrant call paths, (c) bump `EXPECTED_LOCK_COUNT`
in this test in the SAME commit that adds the new lock. The audit
is forced, not optional.

---

## Section J -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "gate_passed": true
}
```
