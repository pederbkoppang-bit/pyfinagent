# Research Brief — phase-80.11 (session-probe stampede + provider consolidation)

**Tier:** T2 (moderate/complex depth, Opus 5 effort high). NOT audit-class.
**Started:** 2026-07-25
**Status:** IN PROGRESS (write-first; appended incrementally)

## Questions

- A. Single-flight / promise-dedup in TypeScript (store promise, clear in `.finally()`, rejection fan-out)
- B. Cache-invalidation interaction with the 401 path
- C. React 19 / Next 15 duplicate-provider consolidation
- D. Polling-loop discipline (5-consecutive-failure budget) under consolidation
- E. INTERNAL census of every fetch/poll site firing on `/paper-trading/positions`

---

_(sections appended below as sources are read)_
