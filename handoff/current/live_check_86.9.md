# Live check — step 86.9

**Date:** 2026-08-14 (~06:45 CEST) — **REFRESHED**. The prior version was written
2026-08-11 and had gone stale in two criterion-relevant ways (see §0).
**Backend:** pid **93024**.

> **SCOPE, STATED UP FRONT.** This artifact supplies **all six criteria** with fresh
> measurements. Criterion 6 carries a scope limit (§6) and criterion 5 carries a
> **named gap**: I measured the rail timeout *rate*, not the "26% of rail *time*
> discarded" figure the criterion cites — different quantities, and §5 says so rather
> than treating one as the other. **I have NOT spawned the remaining attempt.**

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

## 3. Per-ticker mean and projected total — RE-DERIVED with the named tool

Command (**read-only**: the tool's only write is `Path(args.json).write_text(...)` at
`:316`, gated on `--json`, which defaults to `None` and was **not** passed):

```
python scripts/diagnostics/measure_analysis_phase.py --log backend.log --budget-sec 10800
```

All cycles below post-date the 2026-08-09 rail repair, as the criterion requires:

| Cycle | per-ticker mean | median | eff. parallelism | projected cycle | vs 10,800 s |
|---|---:|---:|---:|---:|---:|
| 2026-08-11 | **1,609.6 s** | 1,699.2 s | 2.17 | 4,850 s | −5,950 s |
| 2026-08-12 | **336.3 s** | 360.0 s | 2.56 | 1,366 s | −9,434 s |
| 2026-08-13 | **1,707.5 s** | 1,789.5 s | 2.02 | 5,454 s | −5,346 s |

Every cycle: `planned=6 dispatched=6 finished=6 unfinished=[]`, concurrency cap 3,
end reason `reached_mark_to_market`. Tool verdict on each: **within budget**.

**The criterion's own hypothesis is CONFIRMED: "the 2310-2320s figure predates that fix
and may no longer hold."** It no longer holds — the post-fix per-ticker mean on full
cycles is **~1,610–1,708 s**, roughly **26–30% below** the 2,310–2,320 s figure. The
08-12 outlier at 336 s is the degraded cycle (6/6 degraded) and should not be averaged in
as though it were a healthy fast cycle.

**Projected total, worst observed: 5,454 s against a 10,800 s budget (50.5%).**

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

## 5. Asks #24 and #25 — re-evaluated, with one quantity I could NOT measure

### What I measured: the rail TIMEOUT RATE, post-fix

| Cycle | rail calls started | timed out | **rate** | `subprocess_timeout_s` |
|---|---:|---:|---:|---:|
| (earlier window) | 152 | 1 | **0.66%** | 150 |
| 2026-08-11 | 170 | 8 | **4.71%** | **120** |
| 2026-08-12 | 75 | 1 | **1.33%** | 150 |
| 2026-08-13 | 173 | 7 | **4.05%** | 150 |

Note the 08-11 cycle ran at `subprocess_timeout_s=120`, not 150, and carries the highest
rate — coherent, and worth keeping because it is the one cycle whose cap differed.

### What I did NOT measure, and it is the criterion's own number

Criterion 5's wording is **"26% of rail TIME being discarded"** — a fraction of *time*,
not of *calls*. **I measured calls. These are different quantities and I am not treating
one as a refutation of the other.**

The tool reports `rail_calls_started` / `rail_calls_timed_out` only (`:220`, printed at
`:311`); it emits **no rail-time total**, and `agent latency` returns **`None`** in every
window, so the Trace-based latency source is unavailable here. **The 26% figure is
therefore neither confirmed nor refuted by this measurement.**

### Recommendations, bounded by that gap

- **Ask #24 (rail timeout 150 → 210): RECOMMEND WITHDRAWAL — provisionally.** On the
  quantity I *can* measure, timeouts are **0.66%–4.71%** of calls, not a dominant failure
  mode, and no cycle came near the budget. Raising the cap would extend the tail of the
  slowest calls rather than fix them. **Provisional because** if 26% of rail *time* is
  genuinely being discarded, the time-side argument is untouched by my call-side numbers.
  **To settle it properly, derive rail time discarded** = Σ(timed-out call durations) ÷
  Σ(all rail call durations); that needs a latency source this tool did not produce.
- **Ask #25 (merged dispatch): RECOMMEND DEFERRAL.** Effective parallelism is
  **2.02–2.56 against a cap of 3** (67–85% utilisation), so there is real headroom — but
  the worst cycle used **50.5%** of budget, so dispatch efficiency is not currently the
  binding constraint. Revisit if projected cycle time approaches the budget, or after
  86.69 (the analysis-emptiness regression) changes the load profile.

---

## 6. No other setting changed; `paper_analyze_top_n` NOT lowered; `.env` backup retained

- **`paper_analyze_top_n = 5`** in the RUNNING process (`GET /api/settings/`, pid 93024)
  — **not lowered**, which is what the criterion forbids.
- **`.env` backup retained:** `backend/.env.bak.20260809T155016` — dated **2026-08-09**,
  the day of the budget raise itself, so it is the correct pre-change snapshot rather
  than an unrelated older copy. (Others exist and are older:
  `.env.bak.phase23.3.7`, `.env.env.bak-20260417-224659`.)
  Filenames were **derived by listing** `backend/.env*` rather than assumed.
- **No setting was changed by this step**, and none by this session:
  `git status --porcelain -- backend/ scripts/` is **empty**, no `.env` write, no flag
  promotion, no restart, no manual cycle.

**Scope limit, stated:** I verified the two settings the criterion names and the backup's
existence. I did **not** diff the whole `.env` against its backup to prove *nothing else*
changed — that would require reading `backend/.env`, which is denied.

---

## What this artifact licenses

- **Does:** replace stale criterion-1/2 evidence with current measurements, and answer
  criterion 4 with a positive-controlled source read.
- **Does NOT:** close the step. Two limits are named rather than hidden — criterion 5's
  "26% of rail **time**" figure is not derived (I measured calls), and criterion 6 is
  verified only for the two settings the criterion names, because reading `backend/.env`
  is denied. A Q/A may reasonably weigh either.
- **Nothing was changed:** no file under `backend/` or `scripts/` modified; no `.env`
  write; no flag promotion; no restart; no manual cycle.
