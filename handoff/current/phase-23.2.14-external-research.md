# phase-23.2.14 — External Research Brief
## Re-entrant threading.Lock Deadlocks in Python

**Tier:** simple (mechanical scan)
**Date:** 2026-04-29

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.python.org/3/library/threading.html | 2026-04-29 | official docs | WebFetch | "If the same thread owns the lock, [Lock's] acquire blocks until the lock can be acquired" — confirms Lock is non-reentrant by spec; RLock's nested acquire/release pairs may be nested |
| https://realpython.com/python-thread-lock/ | 2026-04-29 | authoritative blog | WebFetch | "The thread hangs indefinitely because `._update_balance()` tries to acquire the lock that's already held by `.deposit()`" — canonical example of indirect self-deadlock via method calls |
| https://superfastpython.com/thread-deadlock-in-python/ | 2026-04-29 | authoritative blog | WebFetch | "you may have a custom class that has a lock as a member variable...some functions call other functions internally" — directly names the kill_switch.py pattern |
| https://www.geeksforgeeks.org/python/python-difference-between-lock-and-rlock-objects/ | 2026-04-29 | educational reference | WebFetch | Comparison table: Lock cannot be reacquired without release (deadlock); RLock allows same thread to acquire multiple times via acquisition counter |
| https://www.tech-reader.blog/2026/04/the-secret-life-of-python-deadlock.html | 2026-04-29 | blog (2026) | WebFetch | Covers circular-wait deadlocks; confirms same fix (lock ordering, context managers) — does not add new self-deadlock content beyond the above |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://realpython.com/lessons/avoiding-deadlocks-rlock/ | tutorial | Video lesson, no machine-readable text body |
| https://bugs.python.org/issue27422 | bug tracker | Covers multiprocessing+threading interaction, not the self-deadlock pattern |
| https://dev.to/hiteshchawla/r-lock-vs-lock-in-python-3po9 | blog | Redundant with GeeksForGeeks; snippet sufficient |
| https://medium.com/@abhishekjainindore24/advanced-python-10-lock-vs-rlock-c747bbdbd803 | blog | Redundant |
| https://archambaultv-prof.github.io/programmation-python/docs/advanced/concurrence.mdx/threading/race_conditions_rlock | educational | Redundant |
| https://docs.python.org/3/library/threadsafety.html | official docs | Covers thread-safety guarantees at module level, not lock semantics |
| https://www.bogotobogo.com/python/Multithread/python_multithreading_Synchronization_RLock_Objects_ReEntrant_Locks.php | tutorial | 2020, redundant with docs.python.org fetch |

## Recency scan (2024-2026)

Searched: "Python threading.Lock reentrant deadlock detection 2026", "Python thread safety lock re-entry 2025".

Result: the 2026-04-dated tech-reader.blog post was found and fetched in full. It covers circular-wait deadlocks but does not introduce new findings about the self-deadlock (same-thread re-acquisition) pattern. The Python threading.Lock / RLock behavior has been stable since Python 3.2 (2011) and has not changed in any 2024-2026 release. Official CPython docs (Python 3.14, the stack version for pyfinagent) confirm the same semantics. No new findings in the 2024-2026 window that supersede the canonical sources.

---

## Key findings

1. **`threading.Lock` is non-reentrant by spec.** A thread that holds a `Lock` and attempts to call `acquire()` again will block indefinitely — self-deadlock. (Source: Python 3.14 docs, https://docs.python.org/3/library/threading.html)

2. **Indirect re-acquisition is the dangerous form.** Direct `with lock: with lock:` is obvious; the kill_switch.py bug class is subtler: method A holds the lock and calls method B, which also tries to acquire the same lock. The superfastpython source names this pattern explicitly: "some functions call other functions internally." (Source: https://superfastpython.com/thread-deadlock-in-python/)

3. **Two canonical fixes exist:**
   - **`_locked()` helper pattern (pyfinagent choice):** Extract the logic that needs the lock into a private `_foo_locked()` method that assumes the lock is already held. Public callers acquire then delegate. Zero overhead. Preferred when the class design is clear.
   - **Upgrade to `RLock`:** Drop-in replacement for `threading.Lock`; allows the same thread to re-acquire. Slight overhead (tracks owning thread + acquisition count). Preferred when the call graph is complex or recursive and the `_locked()` pattern would require many helpers. (Source: https://realpython.com/python-thread-lock/, https://www.geeksforgeeks.org/...)

4. **`RLock` has a release-ownership constraint `Lock` does not.** A `Lock` can be released by any thread; an `RLock` can only be released by the thread that acquired it. This is generally the correct behavior for the lock-per-instance pattern used in pyfinagent services, but callers must not share `RLock` across thread boundaries with the expectation of cross-thread release. (Source: GeeksForGeeks comparison table)

5. **Python 3.14 (pyfinagent's stack) has not changed Lock/RLock semantics.** The behavior is stable since Python 3.2. The `threadsafety` module docs (Python 3.14) address thread safety of stdlib objects but do not change Lock primitives. (Source: https://docs.python.org/3/library/threadsafety.html, snippet)

---

## Internal code inventory

(See `phase-23.2.14-internal-codebase-audit.md` for the full per-method inventory.)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/api_cache.py` | 140 | In-memory TTL cache singleton | CLEAN |
| `backend/services/perf_tracker.py` | 148 | Per-endpoint latency recorder | CLEAN |
| `backend/services/cycle_health.py` | 228 | Cycle JSONL writer + heartbeat | CLEAN |
| `backend/services/live_prices.py` | 123 | yfinance price cache with rate gate | CLEAN |
| `backend/services/observability/alerting.py` | 152 | Alert dedup tracker | CLEAN |
| `backend/services/observability/api_call_log.py` | 294 | BQ-buffered API + LLM call log | CLEAN |
| `backend/agents/cost_tracker.py` | 255 | Per-agent token cost accumulator | CLEAN |
| `backend/agents/_genai_client.py` | 152 | genai.Client singleton factory | CLEAN |
| `backend/tools/alt_data.py` | 164 | Google Trends with 24h TTL cache | CLEAN |

---

## Consensus vs debate

**Consensus:** `threading.Lock` is non-reentrant; same-thread re-acquisition deadlocks. `RLock` is the standard fix when the call graph requires re-entry. All sources agree. No debate.

**Debate (minor):** Whether to prefer `RLock` universally vs. the `_locked()` helper pattern:
- Performance purists prefer `Lock` + `_locked()` helpers (no overhead, explicit contract).
- Defensive engineers prefer `RLock` everywhere (eliminates entire bug class at small cost).
- pyfinagent's phase-23.1.22 fix chose the `_locked()` helper, which is auditable and zero-overhead.

## Pitfalls (from literature)

1. **Indirect calls hide the re-entrance.** `with self._lock: ... self.method()` where `method()` also acquires `self._lock` is a deadlock, but grep-level audits miss it unless they trace call chains.
2. **`RLock` release ownership.** If a helper is called from a thread that did NOT acquire the RLock, `release()` raises `RuntimeError`. Cross-thread patterns must use `Lock`.
3. **Double sequential acquisitions are fine.** `with _lock: ...; with _lock: ...` (separate, non-nested) is safe. The bug is only nested acquisition by the same thread.

## Application to pyfinagent

- The `_locked()` helper fix from phase-23.1.22 is the right pattern for pyfinagent's use case: simple per-instance locks with clear ownership.
- None of the 9 audited files require remediation. All locked methods call only stdlib collection methods or private helpers that do not re-acquire the lock.
- If any future method is added that (a) holds `self._lock` AND (b) calls another public method of the same class, the developer MUST either use `_locked()` variants or upgrade to `RLock`.
- The `_rate_ok()` pattern in `live_prices.py` (called inside `get_many()`'s locked block) is a correct reference implementation: private helper, no lock acquisition, accesses only attributes.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total incl. snippet-only (12 URLs collected)
- [x] Recency scan (last 2 years) performed + reported (2026 dated source found and evaluated; no new findings supersede canonical sources)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see internal codebase audit file)

Soft checks:
- [x] Internal exploration covered every relevant module (all 9 files read in full)
- [x] No contradictions; consensus clear
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
