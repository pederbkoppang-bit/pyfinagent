# Research Brief — step 86.62

**Topic:** Triage of the three 2026-08-11 cycle degradations (MetaCoordinator p95
latency ~6267ms vs 500ms threshold; promoted-strategy 404 → FALLBACK params;
Alpha Vantage rate limit on the social-sentiment overlay).
**Tier:** simple (floors unchanged: >=5 WebFetch full reads, >=10 URLs, recency scan).
**Audit-class:** NO (coverage reported for information only).
**Role:** DIAGNOSIS. Report only — no production code changes, no threshold loosening.

---

## ENVELOPE (born inert — phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 13,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "sources_read_in_full": [
    "https://sre.google/sre-book/monitoring-distributed-systems/",
    "https://prometheus.io/docs/practices/histograms/",
    "https://arxiv.org/html/2206.14254",
    "https://stefvanbuuren.name/fimd/sec-simplesolutions.html",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11924964/",
    "https://arxiv.org/html/2511.01196"
  ],
  "brief_path": "handoff/current/research_brief_86.62.md",
  "gate_passed": true
}
```

---

## Status log

- [t0] Brief created. Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- Work in progress below; sections are appended as evidence lands.


---

## INTERNAL — Degradation 2: promoted-strategy 404

**Verbatim log line** (`backend.log:85009`, the 2026-08-11 cycle):

```
{"timestamp": "2026-08-11 20:00:02,044", "level": "WARNING", "module": "autonomous_loop", "message": "Promoted strategy BQ unavailable, falling back to optimizer_best: 404 Not found: Table sunny-might-477607-p8:pyfinagent_data.promoted_strategies was not found in location US; reason: notFound, message: Not found: Table sunny-might-477607-p8:pyfinagent_data.promoted_strategies was not found in location US\n\nLocation: US\nJob ID: 64121585-4bb9-4338-9a9b-6cdcfbe13b51\n"}
```

**Emitter:** `backend/services/autonomous_loop.py:70-74` — the `except Exception` arm of
`load_promoted_params()` (defined `:47`, docstring `:48-58`). Called once per cycle at
`backend/services/autonomous_loop.py:499` (`best_params = load_promoted_params(bq)`), which
falls through to `load_best_params()` at `:75` (reads `optimizer_best.json`).

**MISSING OBJECT, NOT A PERMISSION.** The error is `404 notFound: Table
sunny-might-477607-p8:pyfinagent_data.promoted_strategies was not found in location US`.
A permission failure on BigQuery is `403 accessDenied`, not `404 notFound`; and the job
was *accepted and executed* (it carries a Job ID), which itself proves the caller had
`bigquery.jobs.create`. So the resolution is: **the table `pyfinagent_data.promoted_strategies`
does not exist** (in location US).

**Measured recurrence (NOT transient):** 4 of 4 available daily cycles in the live
`backend.log` — 2026-08-10 20:00:01, 2026-08-11 20:00:02, 2026-08-12 20:00:01,
2026-08-13 20:00:02. That is **100% of cycles in the log window**, at the same
clock position each day. Recurrence rate 4/4.


---

## INTERNAL — Degradation 1: MetaCoordinator p95 latency

**Verbatim log line** (`backend.log:90485`, the 2026-08-11 cycle):

```
{"timestamp": "2026-08-11 21:21:28,984", "level": "INFO", "module": "autonomous_loop", "message": "MetaCoordinator decision: perf_opt (reason=p95 latency 6267ms > 500ms threshold)"}
```

All four in the live log (`/usr/bin/grep -n "p95" backend.log`, 4 hits, 4 of 4 are `perf_opt`):

```
33060: 2026-08-10 21:15:34,265  p95 latency 6812ms > 500ms threshold
90485: 2026-08-11 21:21:28,984  p95 latency 6267ms > 500ms threshold
148384: 2026-08-12 20:23:24,425 p95 latency 6500ms > 500ms threshold
209590: 2026-08-13 21:31:51,861 p95 latency 3973ms > 500ms threshold
```

**Emitter chain (file:line):**
- `backend/services/autonomous_loop.py:1831-1834` — the `logger.info(f"MetaCoordinator decision: ...")` in Step 10, inside a `try:` whose `except` at `:1835-1836` marks the whole step non-fatal.
- The reason string is built at `backend/agents/meta_coordinator.py:157-162`:
  `if health.p95_latency_ms > self.latency_threshold_ms:` → `reason=f"p95 latency {health.p95_latency_ms:.0f}ms > {self.latency_threshold_ms:.0f}ms threshold"`.
- Threshold constant: `backend/agents/meta_coordinator.py:120` `DEFAULT_LATENCY_THRESHOLD_MS = 500.0`.
- Value source: `backend/agents/meta_coordinator.py:263-269` `gather_health()` → `summary = perf_tracker.summarize(); health.p95_latency_ms = summary.get("p95_ms", 0.0)`.
- Computation: `backend/services/perf_tracker.py:59-117` `summarize(window_seconds=300)` +
  `:144-154` `_percentile()` (linear interpolation, no minimum-N guard, returns `0` on empty).

**THE POPULATION IS UNSTATED AND IS NOT THE TRADING CYCLE.** Three measured facts:

1. **Window = the last 300 seconds only** (`perf_tracker.py:59` default `window_seconds=300`,
   called with no argument at `meta_coordinator.py:266`). The decision fires ~80 min AFTER the
   cycle starts (20:00:02 → 21:21:28 on 2026-08-11), so the 5-minute window it summarises
   contains **none of the cycle's own work**.
2. **The population is every HTTP endpoint mixed together**, not a trading operation.
   Sampling `backend.log` for the window ending at the 2026-08-11 decision, the recorded
   traffic is exclusively the frontend dashboard poll set:
   `/api/paper-trading/live-prices` (11), `/kill-switch` (11), `/cycles/history?limit=1` (11),
   `/snapshots?limit=365` (10), `/portfolio` (10), `/gate` (10), `/freshness` (10).
   These are **cache/BQ-backed dashboard reads**, and `snapshots?limit=365` is by construction
   a heavy one. A p95 over that mixture is a statement about the operator's open browser tab,
   not about the trading engine.
3. **The denominator is computed and then thrown away.** `perf_tracker.summarize()` returns
   `total_requests` (`perf_tracker.py:109`) and a full `per_endpoint` breakdown (`:116`), but
   `gather_health` reads **only** `summary.get("p95_ms")` (`meta_coordinator.py:267`) and the
   cycle summary records only sharpe/accuracy/p95 (`autonomous_loop.py:1825-1829`). So the
   emitted reason string is a percentile with **no N, no window, and no endpoint** attached —
   it is unfalsifiable as logged. With `_percentile` interpolating on a sorted list and no
   minimum-N guard (`perf_tracker.py:144-154`), at small N the "p95" is arithmetically pinned
   to the largest sample.

**Consequence:** `perf_opt` is Priority 1 in `MetaCoordinator.decide` (`meta_coordinator.py:156-162`),
so a breach of this metric **pre-empts** the Sharpe check (Priority 2, `:165-173`) and the
agent-accuracy check (Priority 3, `:176-186`) for that cycle. The dashboard's latency is
therefore able to suppress the quant/skill optimisation signal. **And nothing consumes the
decision**: `/usr/bin/grep -rn "perf_opt" backend --include='*.py'` returns only the enum
comment (`meta_coordinator.py:110`) and the construction site (`:159`). `summary["coordinator"]`
(`autonomous_loop.py:1820`) has **no reader anywhere in `backend/`**. The decision is recorded,
never acted on.


**Measured recurrence (NOT transient):** across the 21-day population
(2026-07-24..08-13; four log segments — `handoff/logs/backend.log.20260729T171222Z.gz`,
`...20260804T182713Z.gz`, `...20260810T064130Z.gz`, live `backend.log`):

| Segment | cycles started | MetaCoordinator decisions | of which `perf_opt` |
|---|---|---|---|
| 20260729 gz | 4 | 4 | (see below) |
| 20260804 gz | 5 | 3 | |
| 20260810 gz | 6 | 3 | |
| live backend.log | 4 | 4 | 4 |
| **TOTAL** | **19** | **14** | **10 of 14** |

`perf_opt` = 10, `skill_opt` = 4, `quant_opt` = 0, `idle` = 0 across the window.
**Positive control on the population:** my cycle count (4+5+6+4 = **19**) reproduces the
caller's independently-stated "19 cycles started", so the segment set is the right one.
`quant_opt` fired **zero** times in 21 days — consistent with Priority-1 `perf_opt`
pre-empting it whenever the dashboard p95 is above 500ms, which was 10 of the 14 decisions.

---

## INTERNAL — Degradation 3: Alpha Vantage rate limit (social-sentiment overlay)

**Verbatim log line** (`backend.log:29249`; the same signature recurs on the 2026-08-11 cycle):

```
{"timestamp": "2026-08-10 20:06:12,465", "level": "WARNING", "module": "social_sentiment", "message": "Alpha Vantage rate limit in social_sentiment for PANW"}
```

A second, distinct AV limiter fires in the news tool (different module, do not conflate):

```
{"timestamp": "2026-08-11 20:06:57,564", "level": "WARNING", "module": "alphavantage", "message": "Alpha Vantage rate limit for DELL: ... (25 requests per day) ..."}   backend.log:85461
{"timestamp": "2026-08-11 20:06:57,878", "level": "INFO", "module": "alphavantage", "message": "AV unavailable for DELL -- using 10 yfinance articles"}                  backend.log:85462
```

**Emitters:** `backend/tools/social_sentiment.py:68` (`logger.warning("Alpha Vantage rate
limit in social_sentiment for %s", ticker)`) and `backend/tools/alphavantage.py:84`
(`"AV unavailable for %s -- using %d yfinance articles"`).

**Measured recurrence (NOT transient):** `rate limit in social_sentiment` occurs on
**14 distinct days of the 21-day window** — 07-25, 07-26, 07-27, 07-28, 07-29, 07-30,
08-03, 08-04, 08-08, 08-09 (from the gz segments) plus 08-10, 08-11, 08-12, 08-13 (live log).
Event counts per segment: 14 / 4 / 5 / 4 = **27 events** (per-ticker, so > 1 per cycle).
The AV free tier is **25 requests/day** (quoted verbatim in the vendor's own limiter message
above) and the cycle analyses several tickers with **two** AV endpoints each
(`alphavantage.py` news + `social_sentiment.py`), so this is a **structural budget breach,
not a burst**.

### CRITERION 4 — ZEROED or OMITTED? Answer: **BOTH, and which one fires is decided by an argument the caller passes, not by the data.**

Consumer chain, every hop anchored:

1. `backend/tools/social_sentiment.py:67-69` — rate limit detected (`"Information" in data or
   "Note" in data`) → **`feed = []`**. The limiter response is HTTP **200**, so
   `raise_for_status()` (`:63`) does not fire and the `except` at `:145` is never reached.
2. `:73` `if not feed:` → two mutually exclusive branches.

**BRANCH A — `fallback_articles` present → a NUMBER IS SCORED (zero-imputation).**
`:75-76` calls `_score_fallback_articles(ticker, fallback_articles)` (`:150-201`), which
scores **yfinance headlines with a 20-word/20-word keyword lexicon** (`_POSITIVE` `:26-30`,
`_NEGATIVE` `:31-35`, `_keyword_score` `:38-46`). This returns a full dict including
`avg_sentiment` — so the consumer at `backend/tasks/analysis.py:251`
(`social_sentiment_score=social_data_dict.get("avg_sentiment")`) receives **a float, not None**.
Two independent zero-substitutions live inside it:
  - `_keyword_score` `:44-45` — `if total == 0: return 0.0`. An article whose title matches
    **no** lexicon word is scored **exactly 0.0**, identical to an article with balanced
    positive and negative words.
  - `:162` — `avg_sentiment = sum(all_scores) / len(all_scores) if all_scores else 0`.
    Empty evidence → **0**.
  And `0.0` falls inside the NEUTRAL band (`:177-185`: BULLISH needs `>0.15`, BEARISH
  `<-0.15`), so **"we could not measure it" is emitted as "the market is neutral."**

**BRANCH B — no `fallback_articles` → the signal is OMITTED.**
`:77-81` returns `{"ticker", "signal": "NO_DATA", "summary"}` — the `avg_sentiment` key is
**absent**, so `.get("avg_sentiment")` at `analysis.py:251` yields `None` and the BQ column is
written **NULL**. Same for `sentiment_velocity` at `analysis.py:299`.

**Which branch runs in production: A.** `backend/agents/orchestrator.py:2041` calls
`_safe(self.fetch_social_sentiment, "Social", ticker, articles or fallback_articles or None)`
and `:2085` `self.fetch_social_sentiment(ticker, articles)`; `fetch_social_sentiment`
(`:1241-1243`) forwards them. So whenever any news article list exists — which is the normal
case, since `alphavantage.py:82-84` itself substitutes 10 yfinance articles on ITS rate
limit — **Branch A fires and a keyword-derived number is scored as if it were Alpha
Vantage social sentiment.**

**THE PROVENANCE IS PRODUCED AND THEN DROPPED.** `social_sentiment.py:196` sets
`"data_source": "yfinance_fallback"` in the returned dict. But
`backend/db/bigquery_client.py` `save_report` accepts only `social_sentiment_score` (`:97`,
`:208`) and `social_sentiment_velocity` (`:145`, `:256`) — there is **no social
data-source parameter**, and `analysis.py:251,299` pass only those two. So a degraded
keyword score and a genuine AV score are written to the **same column with no distinguishing
mark**, and the pipeline's own `report["social_sentiment"]` wrapper
(`orchestrator.py:1306`, `{"text": ..., "data": sentiment_data}`) is where
`analysis.py:128-129` reads it from.

**POSITIVE CONTROL (the mechanism exists; it is simply not wired here).** The repo already
persists exactly this provenance for a different tool: `orchestrator.py:2007`
`source="yfinance_fallback"` into the `data_source_events` stream, aggregated by
`bigquery_client.py:951-952` as `pct_yfinance_fallback_dominance = COUNTIF(source='yfinance_fallback') / COUNT(*)`.
So a zero-count for social-sentiment provenance is a real absence, not a grep artefact.


---

## EXTERNAL — read in full (6; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-13 | Official docs (Google SRE Book, ch.6) — *year-less canonical* | WebFetch, full page | "If you run a web service with an average latency of 100 ms at 1,000 requests per second, 1% of requests might easily take 5 seconds." Recommends collecting **request counts bucketed by latency** rather than a scalar. Alert test: *"Does this rule detect an otherwise undetected condition that is urgent, actionable, and actively or imminently user-visible?"* |
| 2 | https://prometheus.io/docs/practices/histograms/ | 2026-08-13 | Official docs (Prometheus) — *year-less canonical* | WebFetch, full page | "Aggregating the precomputed quantiles from a summary rarely makes sense. In this particular case, averaging the quantiles yields statistically nonsensical values." Quantile error depends on distribution shape: "With a broad distribution, small changes in φ result in large deviations in the observed value." |
| 3 | https://arxiv.org/html/2206.14254 | 2026-08-13 | Peer-reviewed/preprint — Lenz et al., *No Imputation without Representation* | WebFetch, arXiv native HTML (per the html→ar5iv→pdfplumber chain) | "missing values may in principle contribute useful information that is lost through imputation." Mode imputation "uses one of the non-missing values", making missing values **indistinguishable from actual observations**. Recommends mean/mode imputation **plus a binary missing-indicator** as the safe default. |
| 4 | https://stefvanbuuren.name/fimd/sec-simplesolutions.html | 2026-08-13 | Authoritative book (van Buuren, *Flexible Imputation of Missing Data*, §1.3.3-1.3.7) — *year-less canonical* | WebFetch, full section | "Mean imputation will underestimate the variance, disturb the relations between variables, bias almost any estimate other than the mean." Imputing at a constant "actually creates a bimodal distribution". The indicator method "can yield severely biased regression estimates, even under MCAR and for low amounts of missing data". |
| 5 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11924964/ | 2026-08-13 | Peer-reviewed — Ehrig, Bullock, Leng, Pajewski, Speiser (2025), *JMIR Medical Informatics* — **recency (2025)** | WebFetch, full text | Simulation, 250 patients x 5 visits, 20%/50% missingness under MAR and MNAR: "the missing indicator method may not improve imputation quality or model performance, even when data are MNAR. However, it does not seem that including missing indicators harms imputation quality or model performance either." And: "it is not possible to empirically test whether missingness is informative." |
| 6 | https://arxiv.org/html/2511.01196 | 2026-08-13 | Preprint review — Jicong Fan, *An Interdisciplinary and Cross-Task Review on Missing Data Imputation*, arXiv:2511.01196v3 — **recency (2025/26)** | WebFetch, arXiv native HTML | "Filling missing values with zeros often yields statistical bias because the means of the variables are not necessarily zero." Notes the one legitimate case is count-like data "where a zero could be either an observed value or a missing value" — i.e. exactly the ambiguity to avoid elsewhere. Taxonomy MCAR / MAR / MNAR. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://openreview.net/forum?id=BylsKkHYvH — *Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training NNs* | Peer-reviewed (ICLR) | **Attempted WebFetch, BLOCKED** by an OpenReview browser-verification interstitial. Snippet establishes the "variable sparsity problem": model output "largely varies with respect to the rate of missingness in the given input". |
| https://medinform.jmir.org/2025/1/e64354/ | Peer-reviewed (2025) | **Attempted WebFetch, returned empty body**; read in full via the PMC mirror instead (row 5 above) — same paper, so counted once. |
| https://dl.acm.org/doi/10.1145/3580305.3599911 — *The Missing Indicator Method: From Low to High Dimensions* (KDD '23) | Peer-reviewed | ACM paywall/bot wall; superseded by rows 3+5. |
| https://www.nature.com/articles/s43856-023-00356-z — *Impact of imputation quality on ML classifiers* | Peer-reviewed (Nature Comms Medicine) | Redundant with rows 4-6 for this question. |
| https://arxiv.org/abs/2111.00138 — *The Missing Covariate Indicator Method is Nearly Valid Almost Always* | Preprint | Snippet used only for the recency scan's counterpoint. |
| https://arxiv.org/pdf/2406.00549 — *Zero Inflation as a Missing Data Problem* | Preprint | Directly on "is this zero a zero or a gap" but PDF-only; rows 3+6 cover the claim. |
| https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-01068-x | Peer-reviewed | Adjacent (indicators as proxies for unmeasured vars); not load-bearing. |
| https://www.mdpi.com/2571-905X/5/2/29 | Peer-reviewed | Limits-of-detection variant; adjacent. |
| https://redis.io/blog/p95-latency/ | Industry blog | Vendor-tier; superseded by rows 1-2. |
| https://web-alert.io/blog/latency-percentiles-p50-p95-p99-explained-monitoring-guide | Community blog | Lowest tier in the hierarchy; snippet only ("low sample counts produce unreliable p95", "windowing matters: sliding vs discrete windows change values"). |
| https://nirvanalabs.io/blog/understanding-latency-metrics-p90-p95-p99-explained | Community blog | Same. |
| https://www.blog.trainindata.com/your-guide-to-missing-values-imputation/ | Community blog | Same. |
| https://www.sciencedirect.com/science/article/pii/S0023643825002695 | Peer-reviewed (2025) | Paywalled; recency-scan snippet only. |

**URLs collected: 19** (6 read in full + 13 snippet-only).

### Search-query composition (three-variant discipline, both topics)

- *Year-less canonical:* `zero imputation versus missing data indicator bias machine learning`;
  `how not to measure latency percentiles p95 sample size monitoring distributed systems`
  → surfaced Lenz et al., the ICLR zero-imputation paper, van Buuren, the SRE book, Prometheus docs.
- *Last-2-year window:* `missing data indicator method zero substitution bias 2025 2026 study`
  → surfaced the JMIR 2025 simulation and arXiv:2511.01196.
- *Current-year frontier (2026):* covered by the same 2025/2026 query; arXiv:2511.01196**v3 is dated 2026-04-24**, inside the current year.

### Recency scan (2024-2026) — REQUIRED SECTION

Searched the 2024-2026 window explicitly. Result: **2 new findings, both COMPLEMENTING rather
than superseding the canonical sources, plus 1 genuine qualification.**

1. **arXiv:2511.01196v3 (Fan, v3 dated 2026-04-24)** restates the canonical result in current
   terms — "Filling missing values with zeros often yields statistical bias because the means of
   the variables are not necessarily zero" — and identifies the *only* domain where a zero is
   defensible: count data where "a zero could be either an observed value or a missing value."
   pyfinagent's `avg_sentiment` is a **bounded continuous mean in [-1,1] whose neutral point IS
   zero**, which is the worst possible case: the imputed value collides exactly with the most
   meaningful observed value.
2. **Ehrig et al. 2025 (JMIR Med Inform)** is a genuine QUALIFICATION of the "always add an
   indicator" advice: in longitudinal EHR simulation the indicator "may not improve imputation
   quality or model performance, even when data are MNAR", though it "does not harm" either.
   **This does not weaken the finding here**, because the pyfinagent defect is not "we omitted an
   indicator" — it is "we substituted a value and *destroyed* the distinction". Ehrig et al.
   also state the epistemically important limit: "it is not possible to empirically test whether
   missingness is informative", which is exactly why the distinction must be **preserved in the
   data**, not resolved by assumption.
3. No 2024-2026 source found that defends constant-zero substitution for a continuous
   sentiment-like score. The 2025-26 literature is consistent with the 2018-2022 canon here.


---

## KEY FINDINGS (each cited per-claim)

**F1 — A percentile with no stated population is not actionable, and this one has none.**
Google SRE's alert test is *"Does this rule detect an otherwise undetected condition that is
urgent, actionable, and actively or imminently user-visible?"*
(https://sre.google/sre-book/monitoring-distributed-systems/, accessed 2026-08-13). The
MetaCoordinator reason string fails all three clauses: it is not user-visible (it summarises
the operator's own dashboard polling), not actionable (no consumer of `perf_opt` exists —
`/usr/bin/grep -rn "perf_opt" backend --include='*.py'` returns only
`meta_coordinator.py:110` and `:159`), and its condition is *not otherwise undetected* — the
same numbers are already in `/api/observability/latency`. The SRE book's prescription is to
collect "request counts bucketed by latencies (suitable for rendering a histogram), rather
than actual latencies"; `perf_tracker.summarize()` **already computes** exactly that
(`total_requests` `perf_tracker.py:109`, `per_endpoint` `:116`) and `gather_health` then
**discards all of it** (`meta_coordinator.py:267` reads only `p95_ms`).

**F2 — The scalar is additionally fragile in ways the literature names.** Prometheus:
"With a broad distribution, small changes in φ result in large deviations in the observed
value" (https://prometheus.io/docs/practices/histograms/, accessed 2026-08-13). The measured
population here is maximally broad — seven heterogeneous endpoints from a cached
`kill-switch` read to a 365-row `snapshots` scan — which is the textbook bimodal/multi-modal
case where a single percentile is least informative. `perf_tracker.py:144-154` also has **no
minimum-N guard**, so at small N the "p95" degenerates toward the maximum.

**F3 — Substituting 0 for a missing continuous score is a named statistical error, and it is
worst when 0 is also the neutral point.** van Buuren: "Mean imputation will underestimate the
variance, disturb the relations between variables, bias almost any estimate other than the
mean" and imputing at a constant "actually creates a bimodal distribution"
(https://stefvanbuuren.name/fimd/sec-simplesolutions.html §1.3.3, accessed 2026-08-13).
Fan 2026: "Filling missing values with zeros often yields statistical bias because the means
of the variables are not necessarily zero", the sole exception being count data "where a zero
could be either an observed value or a missing value"
(https://arxiv.org/html/2511.01196, accessed 2026-08-13). `avg_sentiment` is a bounded mean
in [-1,1] whose **NEUTRAL reading is literally 0.0** and whose NEUTRAL band is |x| <= 0.15
(`social_sentiment.py:177-185`) — so the imputed value is not merely biased, it is
**perfectly camouflaged** as the most common legitimate answer.

**F4 — The fix class the literature endorses is to preserve the distinction, not to pick a
better constant.** Lenz et al.: "missing values may in principle contribute useful
information that is lost through imputation"; imputation that "uses one of the non-missing
values" makes missing values indistinguishable from actual observations; the recommended
default is mean/mode imputation **paired with a binary missing-indicator**
(https://arxiv.org/html/2206.14254, accessed 2026-08-13). The 2025 qualification: Ehrig et al.
found the indicator "may not improve imputation quality or model performance, even when data
are MNAR ... [but] does not seem [to] harm" it, and — the decisive epistemic point — "it is
not possible to empirically test whether missingness is informative"
(https://pmc.ncbi.nlm.nih.gov/articles/PMC11924964/, accessed 2026-08-13). Since it cannot be
tested after the fact, the distinction must be **retained in the record**.

---

## INTERNAL CODE INVENTORY

| File | Lines cited | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 3752 total; `:47-75`, `:499-505`, `:1662-1669`, `:1810-1836`, `:1849-1859` | The live daily cycle (the `backend/services/` one, per the trap note — `backend/autonomous_loop.py` is the OTHER file and is not this path) | Live |
| `backend/agents/meta_coordinator.py` | 328; `:58`, `:101`, `:110`, `:120`, `:149-192`, `:238-271` | Builds the p95 reason string; `gather_health` drops the denominator | Live; **decision has no consumer** |
| `backend/services/perf_tracker.py` | 163; `:29-57`, `:59-117`, `:144-154` | Computes p95 over a 300s all-endpoint window; no min-N guard | Live |
| `backend/tools/social_sentiment.py` | 201; `:38-46`, `:49-81`, `:150-201` | The overlay. Rate limit → `feed=[]` → keyword-scored yfinance proxy **or** `NO_DATA` | Live; **two zero-substitutions** |
| `backend/tools/alphavantage.py` | `:82-84` | The *other* AV limiter (news, 25 req/day); substitutes 10 yfinance articles | Live |
| `backend/agents/orchestrator.py` | `:1241-1243`, `:1301-1306`, `:2041`, `:2085`, `:2007` | Passes `fallback_articles` (so Branch A is the production path); wraps result as `{"text", "data"}`; **and already persists `source="yfinance_fallback"` for a different tool** | Live (positive control) |
| `backend/tasks/analysis.py` | `:128-129`, `:251`, `:299` | THE CONSUMER. `social_sentiment_score = social_data_dict.get("avg_sentiment")` | Live |
| `backend/db/bigquery_client.py` | `:97`, `:145`, `:208`, `:256`, `:741-760`, `:800-820`, `:951-952` | Persists score+velocity, **no social provenance column**; `get_latest_promoted_strategy` | Live |
| `backend/services/portfolio_manager.py` | `:164-172` | `decide_trades(...)` — signature confirmed to take **no `best_params`** | Live |
| `scripts/migrations/create_promoted_strategies_table.py` | `:1`, `:26-29` | Creates the missing table; has an explicit `--apply` flag | **Apparently never applied** |
| `handoff/cycle_history.jsonl` | 170 rows / 81 completed | The cycle's durable record | See F7 |
| `.claude/masterplan.json` | steps 86.47, 86.60, 86.69 | Related pending steps | Pending |

---

## THE CALLER'S FOUR DIRECT QUESTIONS

**Q1 — Is any of the three "transient"? NO. All three are chronic, with measured rates.**

| Degradation | Measured recurrence over 2026-07-24..08-13 | Verdict |
|---|---|---|
| p95 breach → `perf_opt` | **10 of 14** MetaCoordinator decisions; `quant_opt` fired **0** times | Chronic |
| promoted-strategy 404 | **19 of 19 cycles = 100%**, same clock slot daily | Chronic / permanent |
| AV rate limit (social) | **27 events on 14 distinct days of 21** | Chronic (structural budget breach) |

"Transient" is not available as a conclusion for any of them.

**Q2 — Does the 404 resolve to a missing OBJECT or a missing PERMISSION? A MISSING OBJECT.**
The BigQuery error is `404 ... reason: notFound` for
`sunny-might-477607-p8:pyfinagent_data.promoted_strategies` **in location US**, and
`pyfinagent_data` **is** a US dataset (CLAUDE.md dataset table), so this is not a
location mismatch either. A permission failure would be `403 accessDenied`; moreover the
request was accepted as a **job with an ID** (`64121585-4bb9-4338-9a9b-6cdcfbe13b51`), which
proves the caller held `bigquery.jobs.create`. The table simply does not exist — and
`scripts/migrations/create_promoted_strategies_table.py` exists with an explicit `--apply`
flag, so the most economical reading is that the phase-25.A3 migration was **never applied**.
*(Not independently confirmed against live BQ in this session — flagged as the one open
verification for PLAN.)*

**Q3 — Should the cycle have proceeded on fallback parameters? YES — and the consequence for
that cycle's decisions is NIL.** `best_params` is used at exactly three places
(`/usr/bin/grep -n "best_params" backend/services/autonomous_loop.py`): `:499` assignment,
`:500-505` writing `summary["best_params_sharpe"]` and `summary["strategy_params"]`, and
`:1850-1851` filling `decided_strategy` on the heartbeat row. The trade decision is
`decide_trades(...)` at `:1662-1669`, whose signature
(`backend/services/portfolio_manager.py:164-172`) is
`(current_positions, candidate_analyses, holding_analyses, portfolio_state, settings,
candidates_by_ticker, blocked_out)` — **`best_params` is not among them**. Live
risk/sizing/turnover is driven by `settings.paper_*`. Two independent in-repo comments state
the same thing outright: *"best_params is NOT threaded into decide_trades -- flipping a
promoted_strategies row alone changes only the heartbeat, not live orders"*
(`backend/autoresearch/strategy_backtest_adapter.py:43`) and the near-identical text at
`backend/autoresearch/strategy_registry.py:40`. So halting the cycle on the 404 would have
sacrificed a full trading day to protect a value that reaches only two log fields.
**The real defect is the inverse of the one the log implies:** the fallback is harmless
*because the promotion pipeline is not connected to live orders at all* — which is a strictly
worse finding than "we ran on stale params", and it belongs to the promotion bridge, not to
this cycle's error handling.

**Q4 — Zeroed or omitted? BOTH — see the Degradation-3 section. In the production path
(Branch A, `orchestrator.py:2041`), a rate-limited social signal is ZEROED-BY-PROXY:** a
keyword score over yfinance headlines is written into the same `social_sentiment_score`
column as a genuine Alpha Vantage reading, with `_keyword_score` returning **exactly 0.0**
for any headline matching no lexicon word (`social_sentiment.py:44-45`) and
`avg_sentiment` defaulting to **0** on empty evidence (`:162`). Only when no fallback
articles exist at all (Branch B) is the field OMITTED as NULL (`:77-81` → `analysis.py:251`
`.get()` → `None`). **The difference is the finding**, and the sharpest part of it is that
the returned dict *does* carry `"data_source": "yfinance_fallback"` (`:196`) which the
persistence layer then **drops** — `bigquery_client.save_report` has no social provenance
parameter (`:97`, `:145`).

---

## CRITERION 5 — CAUSAL LINK TO 86.47 (trade drought) AND 86.60 (blind overlays)

Answers are given as DEMONSTRATED / RULED OUT / **UNTESTED**, per the caller's instruction
that speculation in either direction must be recorded as untested.

| Link | Verdict | Basis |
|---|---|---|
| p95 breach → 86.47 drought | **RULED OUT (demonstrated)** | The decision has no consumer anywhere in `backend/` (grep result above) and the metric's population is post-cycle dashboard traffic in a 300s window ~80 min after the cycle started. It cannot reach a trade decision. |
| p95 breach → suppression of `quant_opt` | **DEMONSTRATED, in the coordinator's own logic; effect on trading UNTESTED** | `meta_coordinator.py:156-173`: `perf_opt` is Priority 1 and returns early, so Priority-2 `quant_opt` is unreachable on those cycles. Measured: `quant_opt` = 0 of 14 decisions in 21 days. Whether that suppression would have changed any trade is **UNTESTED** — and note it is partly moot given Q3, since no coordinator action is executed at all. |
| promoted-404 → 86.47 drought | **RULED OUT (demonstrated)** | Q3: `best_params` never reaches `decide_trades`. The parameters in force were `settings.paper_*` either way; the 404 changed nothing about the orders. |
| AV rate limit → **86.60 (blind overlays)** | **DEMONSTRATED (same defect family)** | 86.60's claim is that alternative signals cannot promote a ticker. This adds a second, independent blinding mechanism on the same overlay: on 14 of 21 days the social overlay was not measured, and the substitute is a 40-word keyword lexicon over yfinance headlines carrying no provenance. An overlay whose value is silently a proxy cannot promote anything on its own evidence. |
| AV rate limit → 86.47 drought | **UNTESTED** | Plausible: a 0.0/NEUTRAL social input is HOLD-leaning. But I did NOT measure the counterfactual (what the AV score would have been) and did NOT measure whether `social_sentiment_score` is load-bearing in the final recommendation. Recording as UNTESTED, per instruction. A cheap first test: join `social_sentiment_score` against `recommendation` and check the mass at exactly 0.0. |
| AV rate limit → 86.69 (81.2% empty rows scored 0.0 / HOLD) | **SAME FAILURE CLASS, causal link UNTESTED** | Both are "an absence recorded as 0.0 and then read as neutral". But 86.69 concerns a whole analysis persisting as an empty placeholder, while this is one field inside a populated analysis. **Do not conflate them** — sharing a class is not sharing a cause. Untested. |
| Any of the three → the caller's cited "Degraded-scoring guard fired" (11 on 10 days), "Meta-scorer no-X fallback" (8 on 7 days), "QuantAgent NoneType" (10 on 5 days) | **UNTESTED — and the cycle record cannot answer it** | See F7. |

**F7 — THE TRIAGE GAP, MEASURED.** The 2026-08-11 cycle's own durable record
(`handoff/cycle_history.jsonl`, cycle `86667da7`, `completed_at 2026-08-11T19:21:29Z`) reads:
`"degradation": null`, `"meta_scorer_degraded": false`, `"rail_skipped": false`,
`"breaker_tripped": false`, `"error_count": 0`, `"n_trades": 0`. **The cycle self-reports as
CLEAN on the exact day three degradations fired.** Across all 81 completed rows the
`degradation` key is present on only **2**, and both of those carry a different family
(`fallback_rate` / `degraded_analyses`), never these three. `meta_scorer_degraded` appears on
40 of 81, `funnel`/`rail_skipped`/`breaker_tripped` on 26 of 81 — i.e. the schema itself grew
over time and is sparsely populated. **This is why none of the three was triaged: they exist
only as free-text log lines, and the one durable per-cycle artefact says nothing happened.**
That is a stronger and more general finding than any of the three individually.

---

## PITFALLS FROM THE LITERATURE (carry into PLAN)

1. **Do not "fix" F1 by moving the threshold.** Criterion 6 forbids it, and the literature
   agrees for an independent reason: the SRE book's objection is to a metric that is not
   *actionable* and not *user-visible*, which no threshold value repairs. The defect is the
   missing population, not the number 500.
2. **Do not replace 0.0 with another constant.** van Buuren §1.3.3-1.3.7 shows every simple
   substitution distorts variance and relations; the indicator method itself "can yield
   severely biased regression estimates, even under MCAR."
3. **Do not assume the indicator will improve accuracy.** Ehrig et al. 2025 measured "minimal
   effect on AUROC". The argument for preserving missingness here is **auditability**, not
   predicted accuracy — say so plainly or the change will be mis-sold.
4. **Do not average percentiles if this is ever aggregated across cycles.** Prometheus:
   averaging precomputed quantiles "yields statistically nonsensical values."
5. **Do not conflate the two Alpha Vantage limiters.** `alphavantage.py:84` (news, module
   `alphavantage`) and `social_sentiment.py:68` (module `social_sentiment`) are separate
   emitters with separate budgets against the same 25/day key.

---

## APPLICATION TO PYFINAGENT (external → file:line)

- SRE-book "bucketed counts, not a scalar" → the buckets already exist at
  `perf_tracker.py:94-104` (`per_endpoint`) and are dropped at `meta_coordinator.py:267`.
  A diagnosis-only change is to *log what is already computed* (N, window, top endpoint)
  alongside the p95 in `autonomous_loop.py:1825-1829` — no threshold moves.
- Lenz et al. missing-indicator → the provenance value already exists at
  `social_sentiment.py:196` and dies at `analysis.py:251` / `bigquery_client.py:97`. The
  repo's own precedent for carrying it is `orchestrator.py:2007` +
  `bigquery_client.py:951-952`.
- Fan 2026 "zeros bias because the means are not zero" → `social_sentiment.py:44-45` and
  `:162` are the two exact lines that manufacture the zero.
- Promotion bridge → the 404 is a symptom; the substantive gap is
  `strategy_backtest_adapter.py:43` (`best_params` never reaches `decide_trades`). That is a
  separate step, not this one.

---

## RESEARCH GATE CHECKLIST

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **6** (2 official docs,
      1 authoritative book, 3 peer-reviewed/preprint). Two further attempts failed and are
      disclosed in the snippet-only table with the reason.
- [x] 10+ unique URLs total — **19**.
- [x] Recency scan (last 2 years) performed + reported — dedicated section above, 2 findings
      + 1 qualification, non-empty.
- [x] Full pages/papers read, not abstracts — arXiv native HTML used for both arXiv papers
      per the html→ar5iv→pdfplumber chain; no `arxiv.org/pdf/` URL was WebFetched.
- [x] file:line anchors for every internal claim.

Soft checks:
- [x] Internal exploration covered every module in the stated scope (`backend.log` + 3 gz
      rotations, `backend/services/autonomous_loop.py`, the overlay + its consumer,
      `handoff/cycle_history.jsonl`, masterplan 86.47/86.60/86.69).
- [x] Contradictions noted — Ehrig 2025 qualifies Lenz 2024 on indicator *usefulness*; both
      recorded, and the brief does not let the qualification masquerade as a refutation.
- [x] Per-claim citation with URL + access date.
- [ ] **Disclosed gaps:** (a) the `promoted_strategies` non-existence is inferred from the BQ
      error text + the un-run migration, not from a live `list-tables` call; (b) I did not
      measure whether `social_sentiment_score` is load-bearing in the final recommendation, so
      the AV→drought link is UNTESTED, not ruled out; (c) `error_count: 0` on the 08-11 row is
      reported as-is — I did not audit what increments it.
- [ ] **Brief length exceeds the `simple` tier's <=300-word guidance.** Disclosed, not hidden:
      the external analysis is held at simple-tier depth (6 sources, one page of findings);
      the overrun is entirely the internal half, which the caller scoped explicitly
      (verbatim line + emitter + measured recurrence for each of three degradations, plus a
      consumer trace and a six-row causal matrix). Flagging rather than truncating evidence.

