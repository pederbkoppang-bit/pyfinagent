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

> **CORRECTED (cycle 2, Q/A `wf_52e33912-843` FAIL).** This paragraph previously read
> *"`quant_opt` fired 0 times in the whole 21-day window"* and elsewhere called it *"the
> remedial action the breach is meant to trigger"*. **Both halves were wrong, and the
> evidence refuting me was quoted in my own artifact two lines above.**

**What `meta_coordinator.py:156-172` actually does — an EARLY-RETURN LADDER:**

```python
# Priority 1: Latency issues (cheap to fix, user-visible)
if health.p95_latency_ms > self.latency_threshold_ms:
    return CoordinatorDecision(action="perf_opt", ...)     # <-- RETURNS HERE

# Priority 2: Low Sharpe (quant params need tuning)
if (health.sharpe_ratio < self.sharpe_target and ...):
    return CoordinatorDecision(action="quant_opt", ...)
```

- The p95 branch returns **`perf_opt`**, and it fired **10 of 10** breaches
  (`MetaCoordinator decision: perf_opt` = **10** in the population). **The remedial
  action for p95 fired every single time.**
- **`quant_opt` is Priority 2 — the LOW-SHARPE action — and has nothing to do with
  p95.** My claim inverted the code.
- **My "0 times" count was literally false:** `quant_opt` occurs **17** times in the
  population (module `quant_optimizer`). Only the narrower
  `MetaCoordinator decision: quant_opt` is 0 — and even under that charitable reading
  the gloss was wrong.

**THE FINDING I MISSED, and it is the better one:** because Priority 1 **returns**, a
chronic p95 breach **STARVES** Priority 2 and Priority 3. On **10 of 14** decisions,
`quant_opt` and `skill_opt` were **never evaluated at all**. The degradation is not
"the remedy never fires" — it is "one remedy fires so reliably that the other two are
unreachable."

Verified counts: `MetaCoordinator decision: perf_opt` = **10**;
`MetaCoordinator decision: quant_opt` = **0**; bare `quant_opt` = **17**.

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

> **WITHDRAWN AND REFUTED (cycle 2).** I offered, labelled as an unmeasured inference,
> the hypothesis that the 300s window "contains long analysis requests" so a 500ms
> threshold would make breach the expected state — and nominated it as *"the argument
> someone would need"* to change the threshold. **The Q/A refuted it with two checks I
> could have run myself:**
>
> **(a) Live, backend idle, no cycle running:** `GET /api/observability/latency?window=300`
> returned **p50 5.2ms, p95 2680.2ms, p99 4594.1ms over 37 requests** — the threshold is
> breached **5.4x with ZERO batch traffic present**.
>
> **(b) Historical endpoint mix** in the 300s before each of the 10 breaches, from
> 147,416 uvicorn access lines: **all ten windows are dominated by frontend dashboard
> polling**, and no analysis or agent endpoint appears in any top-6. The
> 2026-08-11 21:21:28 / 6267ms window — the cycle this step is named for — is 111
> requests: live-prices 17, portfolio 16, snapshots 16, kill-switch 16, freshness 15,
> gate 15.
>
> **The interactive endpoints ARE the slow ones.** `/api/paper-trading/portfolio` alone
> shows p95 **4,724.7ms** against `/api/health` at **5.4ms**. So the 500ms threshold is a
> **TRUE POSITIVE about user-visible latency**, not batch contamination.
>
> **The lesson:** labelling a claim an inference does not make it safe when it points at
> loosening a gate. This one would have seeded a future threshold change with a refuted
> argument. It is withdrawn, not softened.

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

## Criterion 4 — CORRECTED: the codebase does BOTH, and I never read the consumer

> **CORRECTED (cycle 2).** The criterion prescribes a METHOD — *"determined by reading
> the consumer"* — and **no deliverable cited a single consumer file:line**. A census of
> `avg_sentiment` / `analysis.py` / `NO_DATA` returned **0/0/0** across all three
> artifacts. I then asserted flatly *"the production path ZEROES"* **on the exact
> dichotomy the criterion exists to resolve.**

**THE CONSUMER** — `backend/tasks/analysis.py:251`:

```python
social_sentiment_score=social_data_dict.get("avg_sentiment") if isinstance(social_data_dict, dict) else None,
```

**THE PRODUCER** — `backend/tools/social_sentiment.py:73-81`, **two** rate-limit branches:

```python
if not feed:
    if fallback_articles:
        return _score_fallback_articles(ticker, fallback_articles)   # -> avg_sentiment 0.0
    return {
        "ticker": ticker,
        "signal": "NO_DATA",
        "summary": "No social sentiment data available.",           # <-- NO avg_sentiment key
    }
```

**CORRECTED AGAIN (cycle 3) — MEASURED BY EXECUTION, and it is WORSE than "zeroes".**
I called this a *zeroing* branch twice. I executed `_score_fallback_articles` rather
than reading it:

```
neutral-words article  -> avg_sentiment= 0.0  signal='NEUTRAL'  data_source='yfinance_fallback'
positive-words article -> avg_sentiment= 1.0  signal='BULLISH'  data_source='yfinance_fallback'
negative-words article -> avg_sentiment=-1.0  signal='BEARISH'  data_source='yfinance_fallback'
```

**It is a SUBSTITUTION branch, not a zeroing branch.** `_keyword_score` returns
`(pos-neg)/total` over a 20-word positive / 22-word negative list and yields `0.0`
**only when no keyword matches**; `_score_fallback_articles` returns the mean of those.
So an Alpha Vantage rate limit can **fabricate a full-strength directional social
signal (±1.0)** from crude keyword matching over yfinance headlines — and the
`yfinance_fallback` provenance that would reveal it is dropped at `save_report`.

**Structurally reachable whenever the primary feed is empty:** `orchestrator.py:2041`
passes `articles or fallback_articles or None`. ~~And it is the COMMON case, not an
equal-odds branch.~~ **STRUCK (cycle 5): NO FREQUENCY CLAIM IS MADE.** That line governs
whether the fallback ARG is **supplied**, not whether the branch is **taken** — the AV
feed's emptiness decides that — and I never counted `yfinance_fallback` vs `NO_DATA` in
production.

| Branch | Producer returns | Consumer `.get("avg_sentiment")` | Effect |
|---|---|---|---|
| `fallback_articles` present (structurally reachable; **frequency NOT measured**) | `avg_sentiment` anywhere in **[-1.0, +1.0]** | that value | **SUBSTITUTES** a fabricated directional signal |
| `fallback_articles` absent | `NO_DATA` dict, no such key | `None` | **OMITS** — correct behaviour |

My earlier "zeroes into the neutral band" **understated the defect**, and understated
criterion 5's 86.60 mechanism with it: the perturbation is bounded by **[-1.0, +1.0]** (a RANGE BOUND, not a point magnitude:
`_score_fallback_articles` returns the MEAN of per-article scores, so ±1.0 requires
unanimity; the module's own signal thresholds are ±0.15 / ±0.25), not a
neutral non-signal.

**The defect class is now sharper than "an absence recorded as a value":** it is an
absence recorded as a **fabricated directional value**, which is a third and worse kind
of input than either "zero" or "missing". The omitting branch behaves correctly.

## Criterion 5 — causal links, demonstrated or ruled out

**86.60 (blind overlays) — LINK IS REAL, and STRONGER than I first scoped it
(corrected cycle 3).** It applies to the **substitution** branch, which
`orchestrator.py:2041` (`articles or fallback_articles or None`) makes structurally reachable whenever the primary feed is empty — note this governs whether the fallback ARG is SUPPLIED, not whether the branch is TAKEN, and I did **not** count `yfinance_fallback` vs `NO_DATA` in production, so no frequency claim is made —
and the perturbation ranges over **[-1.0, +1.0]** (a range bound, not a point magnitude: `_score_fallback_articles` returns the MEAN of per-article scores, so ±1.0 needs unanimity). On the `NO_DATA` branch the
signal is omitted and the link does not apply. My cycle-2 scoping under-claimed in both
membership and magnitude. The social overlay is one
of the eight; when rate-limited **on the fallback branch** it contributes a
**fabricated directional value in `[-1.0, +1.0]`** rather than abstaining, so it
perturbs the score with a **non-signal that carries a direction**.
(~~contributes a `0.0` in the neutral band ... perturbs the score with a non-signal~~ —
**STRUCK (cycle 4)**: refuted by the execution measurement in criterion 4, and it
contradicted the sentence four lines above it. This is the third cycle in which a
correction was declared complete while its superseded text survived beside it.) **However**, 86.60's finding is
that the overlays slice an *unsorted* head-of-universe, so they were already not
entry paths. **The two compound; neither causes the other.**

**86.47 (trade drought) — NOT DEMONSTRATED. Recorded as UNTESTED.** I have **not**
measured whether removing this input changes any candidate's rank or any BUY decision.
(~~"A neutral-band `0.0` is directionally weak"~~ — **STRUCK (cycle 4)**. That was a
**speculative downgrade** of the link's strength, refuted by my own execution
measurement: the branch can emit a full-signed value. The untested record stands, but it
must carry **no** characterisation of magnitude in either direction — which is exactly
what criterion 5's clause is for.) That measurement requires replaying the
scorer with the overlay abstaining versus substituting, which this step did not do.
**I am not claiming a link and I am not ruling one out** — criterion 5 permits an
untested link recorded as untested, and that is what this is.

**The stronger candidate for the drought remains 86.69** (81.2% of analyses persisted
as empty placeholders), which is separately measured and P0.

---

## Criterion 6 — NO threshold changed

The 500ms threshold is **untouched**. `git diff --stat` for `backend/` across this
step: **empty**; `meta_coordinator.py:120 DEFAULT_LATENCY_THRESHOLD_MS = 500.0` intact.

> **CORRECTED (cycle 3).** This paragraph previously ended: *"The p95-population
> inference in criterion 2 is offered as the argument someone would need if they wanted
> to change it."* **That sentence survived the withdrawal it contradicted** — declared
> "withdrawn, not softened" ~100 lines above, still standing here, **inside the section
> that certifies criterion-6 compliance**. It is the same sentence attempt 1 flagged,
> and I left it while asserting the withdrawal was complete.
>
> **There is no argument for changing this threshold in this artifact.** The measured
> evidence says the opposite: the 500ms threshold is a **TRUE POSITIVE** about
> user-visible latency. Anyone wanting to change it must start from scratch, against
> that evidence.

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
- ~~**The p95-population interpretation is an INFERENCE** — I did not verify which
  endpoints feed `perf_tracker`.~~ **CORRECTED (cycle 3): FALSIFIED, struck through so
  the error stays visible.** `backend/main.py:574` is `@app.middleware("http")` and
  `:617` calls `get_perf_tracker().record(...)` after `await call_next(request)` — so
  **every successfully-dispatched HTTP request feeds it** — precision corrected cycle 4:
  `main.py:605` returns a `JSONResponse` on auth failure **before** `:611`/`:617`, so
  401-rejected requests are never recorded, and "EVERY" was overstated. The question was
  answerable in one read, and the
  artifact's own criterion-2 endpoint mix already answered it empirically.
- The immutable command (`test -f backend.log && grep -c "Paper trading cycle complete"`)
  proves only that the log exists and is countable; it is not evidence for any criterion
  above, which is why each carries its own command.
