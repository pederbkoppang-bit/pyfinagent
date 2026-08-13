# Live check — step 86.62

**Date:** 2026-08-14
**Backend:** pid **93024**, started 2026-08-13T20:30:59Z (restarted by a peer session on
the operator's session-end batching instruction; I did not restart it).
**Requirement:** *"live_check_86.62.md quoting each degradation verbatim with its
timestamp, plus the measured recurrence of each across the available cycles."*

**Population for every count:** all `^{"timestamp"` lines in `backend.log` + rotated
`handoff/logs/backend.log.*.gz`, rebuilt after the 2026-08-13 cycle completed —
**912,459 lines, 2026-07-24..2026-08-13, 21 days, 19 cycles**
(rule: lines containing `Paper trading: Step 1`).

---

## Degradation 1 — promoted-strategy 404

**Verbatim, with timestamp:**

```
2026-07-24 20:00:03,983  Promoted strategy BQ unavailable, falling back to optimizer_best:
  404 Not found: Table sunny-might-477607-p8:pyfinagent_data.promoted_strategies
  was not found in location US; reason: notFound

2026-07-27 20:00:01,866  Promoted strategy BQ unavailable, falling back to optimizer_best:
  404 Not found: Table sunny-might-477607-p8:pyfinagent_data.promoted_strategies
  was not found in location US; reason: notFound
```

**Measured recurrence: 19 occurrences across 17 distinct days = 19 of 19 cycles (100%).**

**Not transient. Not a permission** — `reason: notFound` (403 is the permission code),
dataset **is** `US`. **Specific missing object:**
`sunny-might-477607-p8:pyfinagent_data.promoted_strategies`.

---

## Degradation 2 — p95 latency breach

**Verbatim, with timestamp:**

```
2026-07-28 18:46:00,230  MetaCoordinator decision: perf_opt (reason=p95 latency 6602ms > 500ms threshold)
2026-07-30 20:42:25,142  MetaCoordinator decision: perf_opt (reason=p95 latency 2750ms > 500ms threshold)
```

**Measured recurrence: 10 breaches of 14 `MetaCoordinator decision` lines = 71.4%**,
across 9 distinct days. Parsed values: **n=10, min 2,750ms, max 13,341ms, 10 of 10 over
threshold (100%)**.

**Positive control for the zero below:** `MetaCoordinator decision` appears **14**
times, so the channel works.
**`quant_opt` appears 0 times in 21 days** — the remedial action the breach is meant to
trigger has never fired.

**The p95 is over HTTP request latencies in a rolling 300-second window**
(`perf_tracker.py:59 summarize(window_seconds=300)` → `meta_coordinator.py:267`), not
per cycle and not per analysis.

---

## Degradation 3 — Alpha Vantage social-sentiment rate limit

**Verbatim, with timestamp:**

```
2026-07-24 20:01:39,275  Alpha Vantage rate limit for FTNT: Thank you for using Alpha Vantage!
  Please consider spreading out your free API requests more sparingly (1 request per second)...
2026-07-24 20:01:39,522  Alpha Vantage rate limit for DELL: Thank you for using Alpha Vantage!
  Please consider spreading out your free API requests more sparingly (1 request per second)...
```

Social-specific shape, which is the one this step is about:

```
Alpha Vantage rate limit in social_sentiment for AAPL   (7)
Alpha Vantage rate limit in social_sentiment for DELL   (5)
Alpha Vantage rate limit in social_sentiment for HPE    (4)
```

**Measured recurrence: 27 social-sentiment events across 14 distinct days.**

**Membership rule stated, because two defensible populations exist here:**

| Rule | Count | Days |
|---|---:|---:|
| any `rate limit` | 68 | 19 |
| `rate limit in social_sentiment` | **27** | **14** |

**27/14 is the correct figure for this step.** My first probe used the broad rule and
returned 68/19, disagreeing with the research gate's 27/14 — the gate was right and my
probe was measuring a different population. Recorded rather than silently reconciled.

---

## The consequence that makes this a defect and not just noise

`backend/tools/social_sentiment.py::_keyword_score`:

```python
    total = pos + neg
    if total == 0:
        return 0.0
```

A rate-limited fetch falls back to keyword-scoring headlines; a no-match returns
**exactly `0.0`**, inside the **NEUTRAL** band. **"No sentiment data" and "sentiment is
genuinely neutral" become the same number**, and the `yfinance_fallback` provenance that
would distinguish them is dropped — `save_report` has no column for it.

---

## Why none of the three was ever triaged

`handoff/cycle_history.jsonl`, verified directly:

```
cycle_id: 86667da7 | status: started   | degradation: None | error_count: 0
cycle_id: 86667da7 | status: completed | degradation: None | error_count: 0
```

**The cycle self-reports CLEAN while all three degradations are firing.** No criterion
in this step owns that finding, so it is recorded here and belongs in a queued step —
not claimed as a deliverable.

---

## Nothing was changed

- **No threshold moved** (criterion 6). The 500ms figure is untouched.
- `git diff --stat HEAD -- backend/` : **empty**.
- No `.env` write, no flag promotion, no restart, no manual cycle.
- Immutable command `test -f backend.log && grep -c "Paper trading cycle complete"`
  → `4`, exit 0. It proves the log exists and is countable; it is **not** evidence for
  any criterion, each of which carries its own command above.
