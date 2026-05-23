# phase-23.2.14 — Re-entrant threading.Lock Audit

**Date:** 2026-04-29
**Scope:** 9 files, mechanical scan for the kill_switch.py re-entrant-lock bug class.

## Reference: the kill_switch.py pattern (fixed in phase-23.1.22)

`pause()` and `resume()` acquired `self._lock`, then called `self.snapshot()` which
tried to re-acquire the same `threading.Lock` (non-reentrant) — instant deadlock.
Fix: extract `_snapshot_locked()` that assumes the lock is already held; callers
inside the locked block call the `_locked()` variant; callers outside call the
public variant which acquires the lock then delegates.

---

## A. Per-file inventory

### 1. `backend/services/api_cache.py`

```
=== APICache::get (line 37) ===
LOCK: self._lock
CALLS_INSIDE: self._store.get(), time.monotonic(), del self._store[key] — all stdlib, no self.* method calls
RE_ENTRANCE_RISK: no

=== APICache::set (line 52) ===
LOCK: self._lock
CALLS_INSIDE: CacheEntry(...) constructor, dict assignment — no self.* method calls
RE_ENTRANCE_RISK: no

=== APICache::invalidate (line 66) ===
LOCK: self._lock
CALLS_INSIDE: regex.match(), del self._store[k] — no self.* method calls
RE_ENTRANCE_RISK: no

=== APICache::stats (line 77) ===
LOCK: self._lock
CALLS_INSIDE: time.monotonic(), dict comprehension, self._store reassignment — no self.* method calls
RE_ENTRANCE_RISK: no

=== APICache::clear (line 95) ===
LOCK: self._lock
CALLS_INSIDE: len(), self._store.clear(), counter resets — no self.* method calls
RE_ENTRANCE_RISK: no
```

**Verdict: CLEAN.** No method inside a locked block calls another public or private method of `self`.

---

### 2. `backend/services/perf_tracker.py`

```
=== PerfTracker::record (line 53) ===
LOCK: self._lock
CALLS_INSIDE: self._entries.append(), len(), list slice — no self.* method calls
RE_ENTRANCE_RISK: no

=== PerfTracker::summarize (line 62) ===
LOCK: self._lock  (short acquisition; releases before per-endpoint computation)
CALLS_INSIDE: list comprehension — no self.* method calls
NOTE: lock released at line 64 before the heavy computation. Clean pattern.
RE_ENTRANCE_RISK: no

=== PerfTracker::export_tsv (line 116) ===
LOCK: self._lock
CALLS_INSIDE: list(self._entries) — no self.* method calls
RE_ENTRANCE_RISK: no

=== PerfTracker::clear (line 125) ===
LOCK: self._lock
CALLS_INSIDE: self._entries.clear() — no self.* method calls
RE_ENTRANCE_RISK: no
```

**Special note — `get_slow_endpoints` (line 103):**
This method is NOT inside a lock but it calls `self.summarize()` at line 105.
`summarize()` itself acquires `self._lock`. So `get_slow_endpoints` → `summarize()` is
a lock-acquiring call chain, but since neither `get_slow_endpoints` nor its caller
holds the lock first, there is NO re-entrance. Clean.

**Verdict: CLEAN.**

---

### 3. `backend/services/cycle_health.py`

```
=== CycleHealthLog::record_cycle_end (line 106) ===
LOCK: self._lock
CALLS_INSIDE: _HISTORY_PATH.open(), f.write(), json.dumps()
  -- after the with self._lock block exits (line 112): self._write_heartbeat()
RE_ENTRANCE_RISK: no — _write_heartbeat() is called AFTER the lock block exits,
  not inside it. Lines 106-111 are the locked block; line 112 is outside it.
```

Reading lines 106-112 carefully:
```python
with self._lock:          # line 106
    try:
        with _HISTORY_PATH.open(...) as f:
            f.write(...)
    except Exception as e:
        logger.warning(...)
self._write_heartbeat(...)  # line 112 — outside the with block
```

`_write_heartbeat()` does not acquire `self._lock` (it uses file I/O only, no lock).

**Verdict: CLEAN.**

---

### 4. `backend/services/live_prices.py`

```
=== LivePriceCache::get_many (line 65) ===
LOCK: self._lock
CALLS_INSIDE:
  - self._cache.get() — dict method, no self lock
  - self._rate_ok(now) — line 80, called INSIDE the locked block
  - self._cache[t] = ... — dict assignment
  - self._refresh_log.append() — list method
  - _fetch_price(t) — module-level function (not a method); calls yfinance, no lock
RE_ENTRANCE_RISK: self._rate_ok() — INSPECT
```

`_rate_ok()` (line 44-47):
```python
def _rate_ok(self, now: float) -> bool:
    cutoff = now - 60.0
    self._refresh_log = [t for t in self._refresh_log if t >= cutoff]
    return len(self._refresh_log) < self._max_refresh
```

`_rate_ok` accesses `self._refresh_log` and `self._max_refresh` — both plain attributes, no lock acquisition. It does NOT call any other `self.*` method.

**Verdict: CLEAN.** `_rate_ok` is a pure helper that mutates a list attribute; no lock re-acquisition.

---

### 5. `backend/services/observability/alerting.py`

```
=== AlertDeduper::should_fire — critical path (line 62) ===
LOCK: self._lock
CALLS_INSIDE: self._state.setdefault(), deque.append(), datetime.now()
RE_ENTRANCE_RISK: no

=== AlertDeduper::should_fire — normal path (line 68) ===
LOCK: self._lock
CALLS_INSIDE: self._state.setdefault(), deque.append()/popleft(), len(), timedelta comparisons
RE_ENTRANCE_RISK: no

=== AlertDeduper::reset (line 85) ===
LOCK: self._lock
CALLS_INSIDE: self._state.clear()
RE_ENTRANCE_RISK: no
```

**Verdict: CLEAN.** All operations inside the lock are stdlib dict/deque methods. No self-method calls.

---

### 6. `backend/services/observability/api_call_log.py`

This file uses TWO separate module-level locks: `_lock` (api log) and `_llm_lock` (llm log).
They guard separate global buffers and never nest.

```
=== log_api_call (line 91) ===
LOCK: _lock
CALLS_INSIDE: _buffer.append(), _should_flush_locked()
  -- AFTER lock release: flush() is called if should_flush is True
RE_ENTRANCE_RISK: _should_flush_locked() — INSPECT
```

`_should_flush_locked()` (lines 100-105): accesses `_buffer` and `_last_flush_ts` (globals),
calls `len()` and `datetime.now()`. Does NOT acquire `_lock`. Safe to call inside the lock.

The pattern on lines 91-95:
```python
with _lock:
    _buffer.append(row)
    should_flush = _should_flush_locked()   # no lock inside
if should_flush:
    flush()                                  # called OUTSIDE lock
```

`flush()` itself acquires `_lock` (line 111) and `_llm_lock` is separate. Since `flush()`
is called AFTER the `with _lock` block exits, no re-entrance.

```
=== flush (lines 111, 115-116) ===
LOCK: _lock (acquired twice in sequence, never nested)
  First acquisition: line 111 — drain buffer
  Second acquisition: line 115 — update _last_flush_ts (only when rows is empty)
  Or line 149 — update _last_flush_ts after BQ insert
RE_ENTRANCE_RISK: no — sequential acquisitions, not nested
```

```
=== log_llm_call (line 232) ===
LOCK: _llm_lock
CALLS_INSIDE: _llm_buffer.append(), inline threshold check (len + datetime), no self.* method calls
  -- AFTER lock release: flush_llm() called if should_flush
RE_ENTRANCE_RISK: no — flush_llm() called outside the lock block
```

```
=== flush_llm (lines 248, 252, 285) ===
LOCK: _llm_lock (acquired twice in sequence, never nested)
  Line 248: drain buffer
  Line 252: update timestamp when empty
  Line 285: update timestamp after BQ insert
RE_ENTRANCE_RISK: no — sequential, not nested
```

```
=== buffer_size / llm_buffer_size / reset_buffer_for_test (lines 155, 161, 291) ===
LOCK: _lock or _llm_lock respectively
CALLS_INSIDE: len(), list.clear() — pure stdlib
RE_ENTRANCE_RISK: no
```

**Verdict: CLEAN.** The two locks are independent. `flush()` is always called outside
the `_lock` block in `log_api_call`. `_should_flush_locked()` is a pure read helper
with no lock acquisition.

---

### 7. `backend/agents/cost_tracker.py`

```
=== CostTracker::record (line 167) ===
LOCK: self._lock
CALLS_INSIDE: self.entries.append(entry)
RE_ENTRANCE_RISK: no — pure list append

=== CostTracker::total_cost (property, line 175) ===
LOCK: self._lock
CALLS_INSIDE: sum(), generator over self.entries
RE_ENTRANCE_RISK: no

=== CostTracker::check_budget (line 178) ===
NOT directly locked. Calls self.total_cost (line 180) which acquires self._lock.
RE_ENTRANCE_RISK: no — check_budget does not hold self._lock before calling total_cost

=== CostTracker::summarize (line 196) ===
LOCK: self._lock (short acquisition at line 197 to snapshot entries)
CALLS_INSIDE: list(self.entries)
Lock is released before the computation at lines 199+.
RE_ENTRANCE_RISK: no
```

**Verdict: CLEAN.** `check_budget` calls `self.total_cost` but does not hold the lock first.
Lock re-entrance cannot occur here.

---

### 8. `backend/agents/_genai_client.py`

```
=== get_genai_client (line 113) ===
LOCK: _client_lock
CALLS_INSIDE: _build_client() — module-level function
RE_ENTRANCE_RISK: _build_client() — INSPECT
```

`_build_client()` (lines 33-95): imports `genai`, calls `get_settings()`, constructs
`genai.Client(**kwargs)`. Does NOT acquire `_client_lock` anywhere. Clean.

```
=== close_genai_client (line 136) ===
LOCK: _client_lock
CALLS_INSIDE: _client = None (simple assignment)
  -- AFTER lock release: existing.close() (if exists) — outside the block
RE_ENTRANCE_RISK: no — close() called after lock exits
```

**Verdict: CLEAN.**

---

### 9. `backend/tools/alt_data.py`

```
=== _cache_get (line 34) ===
LOCK: _CACHE_LOCK
CALLS_INSIDE: _CACHE.get(), time.time(), _CACHE.pop() — all stdlib, no function calls
RE_ENTRANCE_RISK: no

=== _cache_put (line 45) ===
LOCK: _CACHE_LOCK
CALLS_INSIDE: _CACHE[key] = ... — dict assignment only
RE_ENTRANCE_RISK: no
```

`get_google_trends()` calls `_cache_get()` (line 61) and `_cache_put()` (lines 105, 109, 116, 162)
from outside any lock. Neither `_cache_get` nor `_cache_put` calls the other while holding the lock.

**Verdict: CLEAN.**

---

## B. Summary — which files have the kill_switch.py bug pattern

**None of the 9 audited files exhibit the re-entrant lock bug.**

The kill_switch.py pattern was:
  `locked_method_A()` → calls `locked_method_B()` → tries to re-acquire same non-reentrant lock → deadlock.

Closest candidates examined and cleared:
- `live_prices.py::get_many` calls `self._rate_ok()` inside the lock — but `_rate_ok` is a pure
  helper with no lock acquisition. CLEAR.
- `perf_tracker.py::get_slow_endpoints` calls `self.summarize()` — but `get_slow_endpoints` itself
  does not hold the lock. CLEAR.
- `cost_tracker.py::check_budget` calls `self.total_cost` — but `check_budget` holds no lock. CLEAR.
- `api_call_log.py::log_api_call` calls `_should_flush_locked()` inside the lock — but that helper
  has no lock acquisition. `flush()` is called after the lock releases. CLEAR.
- `cycle_health.py::record_cycle_end` calls `self._write_heartbeat()` — but after the `with` block
  exits, not inside it. CLEAR.

---

## C. Proposed fixes for found bugs

**None required.** No bug found.

---

## D. Recommendation

**No fixes needed.** The 9 audited files are clean with respect to the re-entrant lock
bug class identified in phase-23.1.22.

The existing patterns are safe because:
1. All files use `threading.Lock` (non-reentrant by design).
2. Every locked block either (a) only calls stdlib methods on `self._store`/`self._entries`/etc.,
   or (b) calls a private helper that was correctly written to not re-acquire the same lock.
3. Where a method calls another lock-acquiring method of `self` (e.g., `check_budget` → `total_cost`,
   `get_slow_endpoints` → `summarize`), the outer method does NOT hold the lock — so no re-entrance.
4. Where `flush()` / `_write_heartbeat()` are called from inside another method,
   they are always invoked AFTER the `with lock:` block exits.

**Action:** Close phase-23.2.14 as PASS with no code changes required.
