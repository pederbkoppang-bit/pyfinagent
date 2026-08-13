# Live check — step 86.9

**Date:** 2026-08-14 (~06:45 CEST) — **REFRESHED**. The prior version was written
2026-08-11 and had gone stale in two criterion-relevant ways (see §0).
**Backend:** pid **93024**.

> **SCOPE, STATED UP FRONT.** This artifact supplies **criteria 1, 2 and 4** with fresh
> measurements. **Criteria 3 and 5 are NOT done** — see §5. It is a refresh so that the
> step's remaining attempt is not spent on staleness; **it is not a claim the step is
> ready to PASS**, and I have **not** spawned that attempt.

---

## 0. What had gone stale, and why it mattered

86.9 has **3 prior Q/A spawns**, so its next attempt is **PASS-or-FAIL** under the
counter repointed in phase-86.75. Two criteria depended on facts that changed after
2026-08-11:

| | prior artifact (08-11) | actual now |
|---|---|---|
| **Criterion 1** — running pid | `pid 66306` | **pid 93024** |
| **Criterion 2** — completed cycles under the new budget | one | **four** |

The recorded pid was **two generations stale** (66306 → 99231 → 93024; the last restart
was a peer session at 2026-08-13T20:30:59Z on the operator's session-end batching
instruction). A criterion that explicitly asks for the pid would not have survived that
— and the attempt would have been spent on bookkeeping while the evidence that answers
the step sat uncollected.

---

## 1. The effective value in the RUNNING process — **10800.0**

The criterion is explicit that a fresh interpreter is **not** sufficient. Both are
recorded so the distinction is visible:

```
RUNNING PROCESS (endpoint served by pid 93024):
  GET /api/settings/  ->  paper_cycle_max_seconds = 10800.0

fresh interpreter (NOT sufficient for this criterion, shown for contrast):
  python -c "from backend.config.settings import get_settings as g; print(g().paper_cycle_max_seconds)"
  -> 10800.0
```

**pid 93024**, started **2026-08-13T20:30:59Z**. The two agree, but the criterion is
satisfied by the **endpoint** reading, not the interpreter one.

---

## 2. Cycles completed end-to-end under the new budget — **four, verbatim**

```
2026-08-10 21:15:34,706  Paper trading cycle complete: NAV=$23897.88, P&L=19.49%, trades=0, cost=$0.3300
2026-08-11 21:21:29,452  Paper trading cycle complete: NAV=$23881.12, P&L=19.41%, trades=0, cost=$0.5100
2026-08-12 20:23:25,003  Paper trading cycle complete: NAV=$23900.18, P&L=19.50%, trades=0, cost=$0.6000
2026-08-13 21:31:52,301  Paper trading cycle complete: NAV=$23920.63, P&L=19.60%, trades=1, cost=$0.6000
```

Wall-clocks from `handoff/cycle_history.jsonl` (`duration_ms`, `status=completed`):

| Cycle | Duration | of 10,800 s | Trades |
|---|---:|---:|---:|
| 2026-08-11 `86667da7` | 4,889 s | 45.3% | 0 |
| 2026-08-12 `2eab42d6` | 1,405 s | 13.0% | 0 |
| 2026-08-13 `c7ac27f2` | **5,512 s** | **51.0%** | **1** |

**The raise was SUFFICIENT on the evidence available.** Maximum observed wall-clock is
**5,512 s against a 10,800 s budget — 51%**, with **no timeout** in any of these cycles,
and the longest one both **completed and traded**. The criterion's alternative outcome
("report that the raise was INSUFFICIENT rather than closing") does **not** apply.

**Caveat that belongs with that claim:** four cycles is a small sample, and the 08-12
cycle at 1,405 s ran with `degraded: True, degraded_analyses: 6/6` — a fast cycle
because work was skipped, not because it was efficient. The honest reading is that the
budget is not currently binding, **not** that headroom is proven under full load.

---

## 4. `_run_single_analysis` still has NO inner per-ticker timeout — confirmed

```
backend/services/autonomous_loop.py:2088-2305   (218-line body, ENTIRE body scanned)
timeout-shaped lines (wait_for | asyncio.timeout | timeout=) in that body:  0
```

**Positive control — the probe is not blind:** the identical regex finds **3** matches
elsewhere in the same file, including the **outer** ceiling it is being contrasted with:

```
:426  # wrapped in `async with asyncio.timeout(...)` -- a timeout mid-cycle
:509  # phase-23.2.18: outer asyncio.timeout ceiling so a stuck
:514  async with asyncio.timeout(_cycle_timeout)
```

So the outer cycle has a ceiling; **the per-ticker call does not.**

**Answering the criterion's actual question — does a longer outer budget increase the
window?** **Yes, and by exactly the raise.** With no inner timeout, a single hung ticker
is bounded only by `_cycle_timeout`. Raising 7,200 → 10,800 s raises the maximum time
one stuck analysis can consume before anything reclaims the cycle **by 3,600 s**. The
budget raise is therefore *sufficient for throughput* (§2) **and simultaneously widens
the hang-exposure window** — both are true, and the step should not report the first
without the second.

---

## 5. NOT DONE — criteria 3 and 5. Do these BEFORE spawning.

**Criterion 3 (per-ticker mean / projected total, re-derived with
`scripts/diagnostics/measure_analysis_phase.py` against cycles run AFTER the 2026-08-09
rail repair).** The tool **exists** (`13,820 b`, 2026-08-08) and there are **now more
post-repair cycles than on 08-11** — so this is re-derivable and its inputs have
improved. **I did not run it.**

**Criterion 5 (asks #24 rail timeout 150→210 and #25 merged dispatch, each re-evaluated
against post-fix data and explicitly recommended or withdrawn).** **Not evaluated.** The
criterion's own wording — *"a budget raise that leaves 26% of rail calls timing out…"* —
implies a rail-timeout rate must be re-measured post-fix. **I did not measure it.**

**Criterion 6 (no other setting changed; `paper_analyze_top_n` NOT lowered; `.env`
backup retained).** Not re-verified here. Note independently: `paper_analyze_top_n = 5`
in the running process, unchanged.

---

## What this artifact licenses

- **Does:** replace stale criterion-1/2 evidence with current measurements, and answer
  criterion 4 with a positive-controlled source read.
- **Does NOT:** satisfy criteria 3, 5 or 6, or license spawning the final attempt.
  **Spawn only after 3 and 5 are genuinely settled** — otherwise the one remaining
  PASS-or-FAIL is spent on a known gap, which is the exact trap this refresh exists to
  remove.
- **Nothing was changed:** no file under `backend/` or `scripts/` modified; no `.env`
  write; no flag promotion; no restart; no manual cycle.
