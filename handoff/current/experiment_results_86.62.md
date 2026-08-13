# Experiment results — step 86.62

**Step:** 86.62 — three degradations logged, none triaged
**Date:** 2026-08-14
**Contract:** `handoff/current/contract_86.62.md`
**Research gate:** PASSED (`wf_07a0d6c8-b7c`) — `handoff/current/research_brief_86.62.md`

**Outcome: a triage report. NO production code was changed** — criterion 6 forbids the
only change that would be tempting (moving the 500ms threshold), and nothing else here
requires one.

---

## Population rule (every count below is over this, and only this)

Every line matching `^{"timestamp"` in `backend.log` plus all rotated
`handoff/logs/backend.log.*.gz`, concatenated once and **rebuilt after the 2026-08-13
cycle completed** (the earlier working snapshot predated it):

```
912,459 JSON lines · 2026-07-24 .. 2026-08-13 · 21 distinct days
CYCLES (rule: lines containing "Paper trading: Step 1") = 19
```

---

## Criterion 1 — each degradation traced, with a MEASURED recurrence

**"Transient" is not available for any of the three.** Each rate is derived from the
population above; none is asserted.

| Degradation | Count | Distinct days | Rate | Transient? |
|---|---:|---:|---|---|
| promoted-strategy 404 | **19** | 17 | **19 of 19 cycles = 100%** | **No** |
| p95 breach | **10** | 9 | **10 of 14 MetaCoordinator decisions = 71.4%** | **No** |
| AV social rate limit | **27** | 14 | 27 events over 21 days | **No** |

**`quant_opt` fired 0 times in the whole 21-day window** — positive-controlled by
`MetaCoordinator decision` appearing **14** times, so the log channel demonstrably
works and the zero is about `quant_opt` specifically.

### A disagreement I hit, pinned rather than papered over

My first probe for the AV rate limit matched the bare string `rate limit` and returned
**68 events on 19 days**, against the research gate's **27 on 14**. Neither number was
wrong — **they are different populations**:

```
ALL Alpha Vantage rate limits:                       68 events, 19 days
SOCIAL-SENTIMENT only ("rate limit in social_sentiment"): 27 events, 14 days
```

The step is about the **social-sentiment** overlay, so **27 on 14 days is the correct
figure** and the gate's number stands. Recorded because a count without its
membership rule is exactly the error this project keeps paying for.

---

## Criterion 2 — the p95 RE-DERIVED, and the population it is a p95 OF

Verbatim, two of the ten:

```
2026-07-28 18:46:00,230  MetaCoordinator decision: perf_opt (reason=p95 latency 6602ms > 500ms threshold)
2026-07-30 20:42:25,142  MetaCoordinator decision: perf_opt (reason=p95 latency 2750ms > 500ms threshold)
```

Parsed from the log: **n=10, min 2,750ms, max 13,341ms, 10 of 10 over 500ms (100%)**.

**The population, traced through source rather than guessed:**

```
meta_coordinator.py:157   if health.p95_latency_ms > self.latency_threshold_ms
meta_coordinator.py:267   health.p95_latency_ms = perf_tracker.summarize()["p95_ms"]
perf_tracker.py:59        def summarize(self, window_seconds: float = 300)
perf_tracker.py:~63       recent = [e for e in self._entries if e.timestamp >= cutoff]
                          latencies = [e.latency_ms for e in recent]
```

**It is the 95th percentile of HTTP request latencies in a rolling 300-second window.**
Not per cycle, not per analysis, not per LLM call — **the last five minutes of API
traffic.**

**INFERENCE, labelled as such and NOT measured:** a 500ms threshold is an
*interactive-API* figure, while a 5-minute window sampled during a paper-trading cycle
contains long analysis requests. That would make a breach the expected state rather
than a signal, which is consistent with 10 of 14 breaching while the remedial action
never fires. **I did not verify which endpoints populate those entries**, so this is a
hypothesis for whoever changes the threshold — and per criterion 6 that change is not
made here.

---

## Criterion 3 — the 404 is a MISSING TABLE, and the cycle SHOULD have proceeded

Verbatim:

```
2026-07-24 20:00:03,983  Promoted strategy BQ unavailable, falling back to optimizer_best:
  404 Not found: Table sunny-might-477607-p8:pyfinagent_data.promoted_strategies
  was not found in location US; reason: notFound
```

**Specific missing object: `sunny-might-477607-p8:pyfinagent_data.promoted_strategies`.**

**Not a permission.** The discrimination is clean: `reason: notFound` (a permission
failure returns **403**), the dataset **is** `US` so it is not a location mismatch, and
the request reached job creation at all.

**Should the cycle have run on fallback parameters? YES.**
**Consequence for that cycle's decisions: NIL.** `best_params` populates two summary
fields and the heartbeat; `decide_trades` (`portfolio_manager.py:164-172`) does not
take it as an argument. The fallback path is the *correct* behaviour here, and the
404's real cost is 19 cycles of noise in the log that trained everyone to ignore it.

---

## Criterion 4 — BOTH branches exist, and the production path ZEROES

Verbatim, `backend/tools/social_sentiment.py::_keyword_score`:

```python
def _keyword_score(text: str) -> float:
    """Return a sentiment score in [-1, 1] using keyword matching."""
    words = set(text.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
```

When the AV fetch is rate-limited, the fallback keyword-scores yfinance headlines. A
no-match returns **exactly `0.0`** — which sits **inside the NEUTRAL band**.

**A zeroed signal and an absent signal are different inputs to a score, and this path
produces the zero.** "We have no sentiment data" and "sentiment is genuinely neutral"
are represented by the identical number. The provenance that would distinguish them
(`yfinance_fallback`) is produced and then **dropped** — `save_report` has no
social-provenance column.

**This is the same defect class as 86.69** (a failed analysis persisted as `HOLD`) and
86.58 (an order reason persisted as a recommendation): **an absence recorded as a
value.** Third instance in this codebase, found independently.

---

## Criterion 5 — causal links, demonstrated or ruled out

**86.60 (blind overlays) — LINK IS REAL, mechanism shown.** The social overlay is one
of the eight; when rate-limited it contributes a `0.0` in the neutral band rather than
abstaining, so it perturbs the score with a non-signal. **However**, 86.60's finding is
that the overlays slice an *unsorted* head-of-universe, so they were already not
entry paths. **The two compound; neither causes the other.**

**86.47 (trade drought) — NOT DEMONSTRATED. Recorded as UNTESTED.** A neutral-band
`0.0` is directionally weak, and I have **not** measured whether removing it changes
any candidate's rank or any BUY decision. That measurement requires replaying the
scorer with the overlay abstaining versus zeroing, which this step did not do.
**I am not claiming a link and I am not ruling one out** — criterion 5 permits an
untested link recorded as untested, and that is what this is.

**The stronger candidate for the drought remains 86.69** (81.2% of analyses persisted
as empty placeholders), which is separately measured and P0.

---

## Criterion 6 — NO threshold changed

The 500ms threshold is **untouched**. `git diff --stat` for `backend/` across this
step: **empty**. The p95-population inference in criterion 2 is offered *as the
argument someone would need* if they wanted to change it — with its own evidence, as a
separate step. **Making that change here would be the exact breach criterion 6 names.**

---

## Adjacent finding — NOT owned by any criterion, recorded so it is not lost

**The cycle self-reports CLEAN while all three degradations fire.** Verified by me in
`handoff/cycle_history.jsonl`:

```
cycle_id: 86667da7 | status: started   | degradation: None | error_count: 0
cycle_id: 86667da7 | status: completed | degradation: None | error_count: 0
```

**That is why none of the three was ever triaged** — the reporting channel says there is
nothing to triage. No criterion here owns this, so it is **not** claimed as a
deliverable; it belongs in a queued step.

---

## Scope honesty

- **No file under `backend/` was modified.** Verified: empty diff.
- **The 19/19 figure spans the rotated archives.** The live `backend.log` alone holds
  **4** Step-1 cycles, so it cannot be reproduced from the live log in isolation. The
  population is stated above; I am not claiming more than it supports.
- **The p95-population interpretation is an INFERENCE, not a measurement** — I did not
  verify which endpoints feed `perf_tracker`.
- The immutable command (`test -f backend.log && grep -c "Paper trading cycle complete"`)
  proves only that the log exists and is countable; it is not evidence for any criterion
  above, which is why each carries its own command.
