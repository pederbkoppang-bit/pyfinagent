# Research Brief -- phase-86.38

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Objective:** Vertex AI Gemini 429 `RESOURCE_EXHAUSTED` quota semantics -- how to distinguish a
per-minute rate limit from a per-day quota from a billing/capacity state when the error body carries
none of that -- plus the design of degraded-mode observability: threshold semantics for fallback-rate
alarms (strict `>` vs `>=`, boundary behaviour), and detecting write-only "dead field" telemetry that
is set but never consumed.

**Internal scope:** `backend/services/autonomous_loop.py` (the full->lite fallback site
`_run_single_analysis`, the `_fallback_reason` / `_intended_path` tags, `_fallback_rate_check` + its
single call site), whether `_intended_path` is consumed anywhere, how the 28-agent orchestrator's
Gemini calls are accounted in `pyfinagent_data.llm_call_log`, and where a degraded cycle surfaces to
the operator.

---

## ENVELOPE (born inert -- phase-86.37; flipped to COMPLETE only as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 21,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "Vertex generative AI has NO per-day quota -- every enforced dimension is per-minute (TPM/RPM), so the step's per-minute-vs-per-day framing is a category error imported from the AI Studio free tier. Under Dynamic Shared Quota there is no project-level number at all, which is why 429s appear 'within quota'. Google's own docs say classify out-of-band from the quota metric, never from the error string. Strict > vs >= has NO external best practice (the SRE Workbook uses both); it is already pinned by test_phase_60_1_deep_pipeline.py:125-133, and at n=5 tickers the choice is a provable no-op. Internally: _intended_path is confirmed write-only (1 write, 0 reads) -- but FOUR MORE dead fields matter more (summary fallback_rate/fallback_reasons/degraded/degraded_analyses), dropped at three proven boundaries. The P1 page is the ONLY operator channel; cycle_history.jsonl never receives them.",
  "brief_path": "handoff/current/research_brief_86.38.md",
  "gate_passed": true
}
```

---

## Search queries run (three-variant discipline)

| Variant | Query | Purpose |
|---|---|---|
| year-less canonical | `Vertex AI Gemini 429 RESOURCE_EXHAUSTED quota error per minute vs per day` | prior art / canonical docs |
| year-less canonical | `Google API error model QuotaFailure ErrorInfo quota_metric quota_limit RESOURCE_EXHAUSTED details` | the structured-detail question |
| year-less canonical | `detecting write-only fields dead code unused field static analysis telemetry never read` | dead-field prior art |
| current-year 2026 | `Vertex AI generative AI quota rate limit RESOURCE_EXHAUSTED 2026` | frontier |
| last-2-year 2025 | `Vertex AI Gemini 429 quota debugging Cloud Monitoring serviceruntime quota exceeded metric 2025` | recency scan |
| last-2-year 2025/2026 | `observability dead telemetry unused metrics never queried instrumentation waste 2025 2026` | recency scan |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| S1 | https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/error-code-429 | 2026-08-11 | official doc (Google Cloud) | curl + tag-strip (WebFetch nav-only) | The 429 message is determined by the **quota framework, not the window**: "Quota framework / Message -- Pay-as-you-go / `Resource exhausted, please try again later.` ; Provisioned Throughput / `Too many requests. Exceeded the Provisioned Throughput.`" The doc lists FOUR distinct causes that all surface as the SAME PayGo string: shared-pool unavailability, per-model quota exhaustion, **acceleration limits** ("You may encounter 429 errors because of acceleration limits if your project has a sharp increase in usage"), and regional-vs-global endpoint capacity. It never mentions a per-minute vs per-day distinction. Also: for Provisioned Throughput under the purchased amount, "errors that might otherwise be 429 are returned as 5XX" -- i.e. the SAME underlying capacity condition changes HTTP class based on billing state. Page footer "Last updated 2026-08-07 UTC". |
| S2 | https://cloud.google.com/vertex-ai/generative-ai/docs/quotas | 2026-08-11 | official doc (Google Cloud) | curl + tag-strip, 39,721 chars extracted | **Every generative-AI quota in the table is a PER-MINUTE rate quota.** Grep across the whole extracted body returns 20+ "per minute" rows (e.g. `aiplatform.googleapis.com/generate_content_image_input_per_base_model_id_and_resolution`, "Generate content requests with image input per minute per base model and resolution -- 34,000,000") and **zero** "per day" rows. Batch inference explicitly has **no quota at all**: "There are no predefined quota limits on batch inference for Gemini models. Instead, the batch service provides access to a large, shared pool of resources, dynamically allocated based on the model's real-time availability and demand across all customers." So on Vertex generative AI there is no per-day quota to distinguish FROM -- the "per-day" hypothesis is a category error imported from the *Gemini Developer API* (AI Studio) free tier, which is a different product. |
| S3 | https://cloud.google.com/docs/quotas/troubleshoot | 2026-08-11 | official doc (Cloud Quotas) | curl + tag-strip, 9,026 chars (whole article) | Defines the only first-class taxonomy Google exposes: **rate quotas vs quota values**. "Rate quotas reset after a predefined time interval that is specific to each service." The HTTP class is decided by *transport*, not by cause: "If you exceeded a quota value with an HTTP/REST request, Google Cloud returns an HTTP `429 TOO MANY REQUESTS`... If you exceeded a quota value using gRPC, Google Cloud returns a `ResourceExhausted` error. **How this error appears to you depends on the service.**" Compute Engine returns `403 QUOTA_EXCEEDED` for the same condition, and `403 RATE_LIMIT_EXCEEDED` if it is a rate quota -- proof that the HTTP code alone carries no window semantics. Recommends out-of-band observability, not error parsing: "If you want to be alerted when errors happen, you can create custom alerts for specific quota errors" + the IAM & Admin > Quotas & System Limits Monitoring charts for "current and peak usage". Also documents a *third* state that mimics quota exhaustion: a **service rollout** in progress, where the console value and the enforced value disagree and the error carries `quotaExceeded.futureLimit`. |
| S4 | https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/unusedwrite | 2026-08-11 | official doc (Go x/tools) | WebFetch (full) | The canonical shipped analyzer for exactly the "write-only dead field" class: "Package unusedwrite checks for unused writes to the elements of a struct or array object... The analyzer reports instances of writes to struct fields and arrays that are never read." Its detection rule is narrow and **value-copy-specific** (the write lands on a copy: `for i, v := range input { v.x = i }`, and non-pointer receivers `func (t T) f() { t.x = i }`). It does NOT do whole-program reachability, so it would NOT catch pyfinagent's `_intended_path` case (a dict key written on a live object that no consumer ever reads). Note the doc states no limitations/false-positive conditions -- absence of a stated caveat, not a claim of completeness. |
| S5 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-11 | official doc / book (Google SRE Workbook, ch.5) | WebFetch (full) | Directly settles both halves of the threshold question. (a) **Google's own examples use BOTH operators**: the trivial approach is `job:slo_errors_per_request:ratio_rate10m{job="myjob"} >= 0.001`, the burn-rate approaches are `... ratio_rate36h{job="myjob"} > 0.001`. The chapter never argues for one over the other -- threshold selection should prioritise detecting "significant events". So strict `>` vs `>=` is a **local spec decision, not a best practice** -- it must be pinned by a test, because the two differ only on the exact-equality boundary. (b) **The small-denominator problem is the dominant failure mode at pyfinagent's volume**: "If a system receives 10 requests per hour, then a single failed request results in an hourly error rate of 10%. For a 99.9% SLO, this request constitutes a 1,000x burn rate and would page immediately." Mitigations listed: synthetic traffic, combining services into monitoring groups, client-side retries with backoff, **lowering the SLO if a single failure doesn't materially impact users**, and increasing the alerting window. Precision = "the proportion of events detected that were significant"; recall = "the proportion of significant events detected". Multi-window table: page at burn rate 14.4 (1h/5m), 6 (6h/30m); ticket at 1 (3d/6h); short window should be "1/12 the duration of the long window". |
| S6 | https://raw.githubusercontent.com/OneUptime/blog/master/posts/2026-02-17-how-to-manage-quotas-and-rate-limits-for-gemini-api-requests-in-vertex-ai/README.md | 2026-08-11 | industry blog (OneUptime, 2026-02-17) | WebFetch (full, raw markdown) | Supplies the **actionable out-of-band signal** the official docs only gesture at: the Cloud Monitoring filter `metric.type="serviceruntime.googleapis.com/quota/rate/net_usage" resource.type="consumer_quota" resource.labels.service="aiplatform.googleapis.com"`. Confirms the enforcement dimensions are **TPM (tokens per minute) and RPM (requests per minute)** -- again, no daily dimension. Recommends alerting at **70-80% utilisation** (i.e. alarm on the *approach*, not on the 429), and client-side rate limiting rather than relying on server-side enforcement, with exponential backoff `2^attempt` (1,2,4,8,16s) + up to 50% jitter, max 5 attempts. Notably it "does not differentiate between multiple causes of 429 errors" and says nothing about the error body -- corroborating that the body is not the discriminator. |
| S7 | https://www.cncf.io/blog/2026/06/22/telemetry-that-matters-designing-sustainable-high-impact-observability-pipelines/ | 2026-08-11 | foundation blog (CNCF; Todea/VictoriaMetrics, Luttmer/Dynatrace, Jimenez Martinez/Cisco) | WebFetch (full) | The quantitative anchor for the dead-telemetry half: "**around 50% of collected metrics are never queried or acted upon**". Frames the cost as "steep engineering overhead, increases alert noise, and heightens cognitive load", not just storage. Its only recommended *method* is a judgement question, not a tool: "If this specific data stream stopped flowing tomorrow, what would we actually lose?" **[Partly ADVERSARIAL to the framing]** -- it provides NO concrete detection procedure (no usage auditing, query-log analysis, or ownership framework), which is itself the finding: the industry has no standard automated detector for emitted-but-unconsumed signals, so pyfinagent must build a bespoke reachability check rather than adopt one. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-handle-429-resource-exhaustion-errors-in-your-llms | official blog | Superseded by S1+S6 on the same content; budget |
| https://discuss.ai.google.dev/t/intermittent-429-resource-exhausted-despite-low-quota-usage-billing-enabled/115831 | forum | Community tier; corroborates "429 despite low quota usage" but anecdotal |
| https://github.com/googleapis/python-genai/issues/2001 | issue tracker | "Repeated 429 Despite Being Within Quota (Paid Tier 1)" -- symptom match, not authority |
| https://discuss.ai.google.dev/t/api-returns-quota-exceeded-limit-0-while-dashboard-shows-2-usage-gemini-3-pro-preview-gemini-3-1-pro-preview-description/127541 | forum | "limit: 0 while dashboard shows 2% usage" -- billing-state confound evidence |
| https://discuss.ai.google.dev/t/429-quota-exceeded-for-quota-metric-generate-content-api-requests-per-minute/83133 | forum | Shows the per-minute metric name IS sometimes in the message |
| https://support.hashicorp.com/hc/en-us/articles/1500009347021-Google-API-Error-429-Quota-exceeded-for-Queries-and-limit-reached-for-Queries-per-minute-per-user | vendor KB | Non-Gemini Google API; same error shape |
| https://discuss.google.dev/t/resource-exhausted-quota-exceeded-for-resource-model-googleapis-com-error-in-vertex-ai/168337 | forum | `model.googleapis.com` variant |
| https://firebase.google.com/docs/ai-logic/quotas | official doc | Firebase AI Logic surface, not the Vertex SDK path pyfinagent uses |
| https://cloud.google.com/vertex-ai/generative-ai/docs/resources/provisioned-throughput | official doc | PT not purchased on this project |
| https://docs.cloud.google.com/bigquery/docs/troubleshoot-quotas | official doc | BQ quotas, adjacent not central |
| https://github.com/google-gemini/gemini-cli/issues/10925 | issue tracker | "When user quota is used up, queries should fail quickly" -- CLI surface |
| https://github.com/google-gemini/gemini-cli/issues/3430 | issue tracker | "Quota already exceeded early in the morning" -- daily-reset folklore |
| https://oneuptime.com/blog/post/2026-01-24-gcp-quota-exceeded-errors/view | industry blog | Generic GCP, superseded by S6 |
| https://arxiv.org/abs/2506.11076 | preprint (DCE-LLM, submitted 2025-06-04) | **Attempted read-in-full and FAILED**: `arxiv.org/html/2506.11076`, `...v1`, `...v2` all returned HTTP 404, so no native HTML exists; per the rules I did not WebFetch the `/pdf/` URL. Only the abstract page was retrieved -> does NOT count. |
| https://cloudnativenow.com/contributed-content/the-telemetry-debt-crisis-why-cloud-native-teams-are-optimizing-the-wrong-metric/ | industry blog | "Telemetry debt" taxonomy; superseded by S7 |
| https://thenewstack.io/can-opentelemetry-save-observability-in-2026/ | industry blog | Frontier context only |
| https://www.aivosto.com/articles/deadcode.html | industry article | Names the exact class ("readers alive but writers dead, or vice versa... variable is in partial use only") but VB6-era tooling |
| https://phpstan.org/blog/detecting-unused-private-properties-methods-constants | tool blog | PHP analogue of S4; same narrow scope |
| https://vfunction.com/blog/dead-code/ | vendor blog | Vendor marketing tier |
| https://learn.adacore.com/courses/SPARK_for_the_MISRA_C_Developer/chapters/08_unreachable_and_dead_code.html | official course | MISRA-C unreachable-vs-dead distinction; adjacent |
| https://kccnceu2026.sched.com/2026-03-26/list/descriptions/type/Observability | conference index | Recency probe only |

## Recency scan (2024-2026) -- MANDATORY, performed

Searched three windows (see the query table). **Result: 4 new findings in the 2024-2026 window, and one of them SUPERSEDES the older mental model.**

1. **[SUPERSEDES]** Vertex AI has moved to **Dynamic Shared Quota (DSQ)** for Gemini 2.0+ models -- capacity is "dynamically distributed among all customers for a specific model and region, removing the need to set quotas and submit quota increase requests." Under DSQ **there is no project-level numeric quota to be "at" or "under"**, which is exactly why the ai.google.dev threads report 429s "despite low quota usage" and "limit: 0 while dashboard shows 2% usage". Any pyfinagent logic that tries to infer "we hit our quota" from a 429 is inferring a number that no longer exists for the models it uses.
2. The official 429 doc was **last updated 2026-08-07** -- four days before this brief. It is current, not stale, and it still declines to distinguish causes in the error body.
3. CNCF (2026-06-22, S7): ~50% of collected metrics are never queried -- and no standard automated detector for unconsumed signals exists.
4. OneUptime (2026-02-17, S6) supplies the current `serviceruntime.googleapis.com/quota/rate/net_usage` monitoring filter.

No 2024-2026 source contradicts the older SRE Workbook guidance in S5; the workbook remains the canonical threshold reference.

## Key findings (external)

1. **The error body genuinely cannot discriminate, and that is by design -- but the *transport* can.** The HTTP status is a function of transport and product, not of cause: REST gives 429, gRPC gives `ResourceExhausted`, Compute Engine gives 403, and "how this error appears to you depends on the service" (S3, https://cloud.google.com/docs/quotas/troubleshoot). The *only* structured discriminator Google defines is the `google.rpc.QuotaFailure` / `google.rpc.ErrorInfo` detail block carrying `quota_metric` + `quota_limit` -- and its presence is per-service and not guaranteed. So a robust implementation must treat "no detail block" as a first-class case, not an error.
2. **On Vertex generative AI, "per-day quota" does not exist.** Every enforced dimension is per-minute (S2, https://cloud.google.com/vertex-ai/generative-ai/docs/quotas): TPM and RPM (S6). The per-day model belongs to the Gemini Developer API free tier, a different product. **A design that branches on "per-minute vs per-day" is branching on a distinction the platform does not make** -- the real trichotomy is *rate limit (per-minute) / DSQ capacity shortfall / billing-or-model-state* (404 retired pin, disabled billing, unavailable region).
3. **The correct discriminator is out-of-band, not in-band.** Both Google (S3: "create custom alerts for specific quota errors" + the Quotas Monitoring charts) and industry (S6: `serviceruntime.googleapis.com/quota/rate/net_usage`, alert at 70-80% utilisation) tell you to read the *quota metric*, never the error string. Corollary for pyfinagent: alarm on the **approach** to the limit and on the **local fallback rate**, not on classifying the 429.
4. **Strict `>` vs `>=` has no external best practice.** Google's own SRE Workbook uses `>=` in one example and `>` in others (S5). It is therefore a spec decision that must be pinned by a boundary test, because the two differ on exactly one input.
5. **Small denominators dominate at pyfinagent's cycle volume.** S5's 10-requests-per-hour example is directly analogous to a 5-ticker cycle: one fallback is 20%. The workbook's own listed mitigation for this case is *not* "add more alarm logic" but longer windows / lower sensitivity / grouping.
6. **There is no off-the-shelf detector for the pyfinagent dead-field shape.** Go's `unusedwrite` (S4) only catches writes to *value copies*; PHPStan's analogue only covers private properties; CNCF (S7) offers a judgement question and no procedure. A dict key written onto a live object that every downstream consumer whitelists away is invisible to all of them -- it needs a bespoke reachability grep at the serialization boundary.

## Internal code inventory

| File | Lines (anchors) | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | :2168-2190 | Full->lite fallback site inside `_run_single_analysis`. `except Exception` builds `_fb_reason = f"{type(e).__name__}: {e}"` (:2176), then stamps `_lite["_fallback_reason"] = _fb_reason[:500]` (:2188) and `_lite["_intended_path"] = "full"` (:2189) | LIVE (`_fallback_reason`) / **DEAD (`_intended_path`)** |
| same | :2605-2626 | `_fallback_rate_check(analyses, threshold)` pure predicate. Fires on `n_total > 0 and (n_fallback / n_total) > threshold` (:2625) -- **strict `>`**, matching its own docstring "strictly EXCEEDS `threshold`... 3/5 fires, 2/4 does not" | LIVE, single definition |
| same | :1326-1359 | The ONLY call site. Reads `settings.fallback_alarm_threshold` (:1327), calls the predicate on `candidate_analyses + holding_analyses` (:1328-1330), and on fire sets `summary["fallback_rate"]` (:1332) + `summary["fallback_reasons"]` (:1333), logs a warning, and `await raise_cron_alert(source="autonomous_loop", error_type="fallback_rate", severity="P1", ...)` (:1339-1357). Whole block wrapped in `try/except Exception` that logs "non-fatal" (:1358-1359) | LIVE alert; **summary fields DEAD** |
| same | :1293-1317 | Sibling `_degraded_scoring_check` guard. Same shape: sets `summary["degraded"]` (:1298) + `summary["degraded_analyses"]` (:1299), raises a P1 `degraded_scoring` alert (:1305-1315) | LIVE alert; **summary fields DEAD** |
| same | :1930-1951 | `record_cycle_end` invocation. `_funnel` is built from an **explicit whitelist tuple** (:1932-1939: `universe_source, universe_size, screened, candidates, new_to_analyze, reeval_tickers`) and the call passes a **hand-picked argument list** -- `meta_scorer_degraded` IS forwarded (:1947), `degraded` / `degraded_analyses` / `fallback_rate` / `fallback_reasons` are NOT | **This is the drop point** |
| same | :3273-3307 | `_persist_analysis`. Copies ONLY `_path` (:3294), `_fallback_reason` (:3295-3296) and `_degraded`/`_degraded_reason` (:3301-3307) into `full_report_json`. `_intended_path` is **not** copied -> never reaches BQ `analysis_results` | **Second drop point** |
| `backend/config/settings.py` | :50 | `fallback_alarm_threshold: float = Field(0.5, ...)` with a docstring that itself states the strict semantics ("strictly exceeds this value") and the origin incident ("The away week ran 9 days at 100% fallback ... with zero alerts") | LIVE |
| `backend/tests/test_phase_60_1_deep_pipeline.py` | :107-145 | Four tests pin the predicate, including `test_fallback_alarm_threshold_is_strictly_greater_than` (:125-133): "2/4 = 0.5 is NOT > 0.5 -> quiet", "3/5 = 0.6 IS > 0.5 -> fires"; plus deliberate-lite-mode quiet (:136-140) and empty-cycle quiet (:143-145) | LIVE -- **the boundary is already pinned** |
| `backend/services/cycle_health.py` | :429-478 | `record_cycle_end` builds `row` from a **closed literal dict** (:449-471). Its own comment at :467-469 records that phase-66.2 added `funnel` precisely because those counts were "previously summary-only (log-parse to recover)" | **Internal precedent for the fix shape** |
| `backend/api/paper_trading.py` | :1409-1418, :1485-1493 | The only two production callers of `run_daily_cycle`. Both are fire-and-forget: `result = await run_daily_cycle(settings)` followed by `logger.info(... result.get('status'))`. **Neither returns `result` in an HTTP response; nothing else reads the dict.** | **Third drop point -- closes the dead-field proof** |
| `backend/agents/llm_client.py` / `orchestrator.py` / `cost_tracker.py` | `llm_client.py:1115-1139` (Gemini `llm_call_log` retrofit, phase-35.2), `orchestrator.py:887-921` (separate row for Gemini `code_execution` calls, phase-26.3) | The 28-agent orchestrator's Gemini calls ARE metered into `llm_call_log` -- but **every writer is wrapped in a swallowing `except` that only `logger.debug`s** ("llm_call_log write skipped: %r") | LIVE but **silently lossy** |

### The dead-field finding, stated precisely

`_intended_path` is written at `autonomous_loop.py:2189` and read **nowhere in the repository**. A
repo-wide grep for `intended_path` (all file types, `.venv` excluded) returns exactly three hits: the
write at `:2189`, a prose mention in `handoff/archive/phase-60.1/experiment_results.md:32`, and this
brief. It is also structurally unreachable: `_persist_analysis` whitelists which keys reach
`full_report_json` (:3293-3307) and `_intended_path` is not among them.

**The same defect class is present three more times and is strictly worse**, because these fields are
the *payload* of the alarms rather than a redundant tag: `summary["fallback_rate"]`,
`summary["fallback_reasons"]`, `summary["degraded"]`, `summary["degraded_analyses"]`. They are written
at :1298-1299 and :1332-1333, are excluded from the `record_cycle_end` whitelist at :1932-1951, and
their only two production consumers (`paper_trading.py:1409-1418` and `:1485-1493`) read
`result.get('status')` and discard the rest. There is no frontend consumer (grep of `frontend/src` for
`fallback_rate` / `meta_scorer_degraded` / `rail_skipped` / `breaker_tripped` returns zero hits).

**Consequence:** the ONLY channel by which a degraded or high-fallback cycle reaches the operator is
the live `raise_cron_alert` P1 page (:1305, :1339) plus `backend.log` lines. If that page is
suppressed, throttled, or -- as in the phase-66.1 incident recorded in the inline comments at :1304
and :1338, where the import target `backend.services.alerting` did not exist and the
`ModuleNotFoundError` was swallowed by the fail-open `except` -- the degradation leaves **no durable
trace at all**, because `cycle_history.jsonl` never received the fields. That is precisely the
"9 days at 100% fallback with zero alerts" failure the `settings.py:50` docstring describes.

## Consensus vs debate (external)

**Consensus:** classify quota state out-of-band from the quota metric, never from the error string
(S1, S3, S6); use truncated exponential backoff with jitter (S1, S6); alarm before the limit, not on
the error (S6). **Debate / genuine gap:** (a) strict `>` vs `>=` -- Google's own workbook is internally
inconsistent (S5), so there is no external answer, only a local one; (b) whether unused telemetry
should be deleted or retained -- S7 declines to give a rule and offers only "what would we actually
lose?"; (c) S7 is adversarial to the premise that a detector exists at all: the industry admits ~50%
waste while shipping no standard detection procedure, and S4 shows the shipped analyzers cover only a
narrow value-copy subset.

## Pitfalls (from literature + measured here)

1. **Do not branch on per-minute vs per-day for Vertex** -- the per-day dimension does not exist (S2);
   under DSQ, even the per-project *number* does not exist for Gemini 2.0+ models.
2. **Do not treat 429 as proof of "our" over-use.** Under DSQ a 429 can mean another tenant's demand,
   an acceleration limit from a traffic spike (S1), or a rollout-window mismatch (S3).
3. **Do not infer the state from the HTTP code.** The same condition surfaces as 429, 5XX (Provisioned
   Throughput under purchase, S1), or 403 (Compute Engine, S3).
4. **A boundary is only pinned if a test asserts the equality case.** `>` and `>=` differ on exactly
   one input; `test_fallback_alarm_threshold_is_strictly_greater_than` (:125-133) already does this,
   so *any* change to the operator must update that test -- it cannot be changed silently.
5. **Small denominators**: at 5 tickers, the reachable fallback fractions are 0/.2/.4/.6/.8/1.0. A
   0.5 threshold is *never* hit exactly at n=5, so the `>` vs `>=` choice is a no-op at n=5 and only
   becomes observable at n=2,4,6,8... This must be stated before anyone "fixes" the operator.
6. **A swallowing `except` around telemetry makes absence unfalsifiable** -- `llm_call_log` writers
   only `logger.debug` on failure (`llm_client.py:1139`, `:1909`, `:2442`; `orchestrator.py:921`), so a
   zero row-count is not evidence that no call was made (this repeats the memory note "llm_call_log
   BUFFERED so a zero is not evidence").

## Application to pyfinagent

- The step's premise "distinguish per-minute from per-day" should be **re-scoped before the contract
  is written**: the platform's real trichotomy is *rate (per-minute) / DSQ-capacity / model-or-billing
  state*, and only the third is cheaply distinguishable locally (a 404 retired-pin `ClientError` is
  already captured verbatim in `_fallback_reason` at `autonomous_loop.py:2176` -- the away-week test
  fixture at `test_phase_60_1_deep_pipeline.py:111` literally carries
  `"ClientError: 404 Publisher Model gemini-2.0-flash was not found"`). Classification by *exception
  string prefix* on the already-captured `_fallback_reason` is the cheap, local, honest win; parsing
  a 429 body for a window is not achievable.
- The highest-value repair is **not** the `_intended_path` tag (which is redundant with
  `_fallback_reason` being non-empty) but the **durability gap**: forward `fallback_rate` /
  `fallback_reasons` / `degraded` / `degraded_analyses` through `record_cycle_end`
  (`autonomous_loop.py:1940-1951` + `cycle_health.py:429-471`) exactly as phase-66.2 already did for
  `funnel`. That precedent is in-tree and its rationale comment (`cycle_health.py:467-469`) states the
  same motive.
- Any threshold change must be argued against S5's small-denominator guidance and must update
  `test_phase_60_1_deep_pipeline.py:125-133`; note the n=5 no-op above so the change is not sold as
  behavioural when it is not.
- `_intended_path` is safe to delete OR to make live, but the two are different steps -- deleting it
  is a pure cleanup; persisting it is a behaviour change to the BQ row shape.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7: S1-S7; S1-S3 via curl+tag-strip because
      `feedback_gcloud_docs_fetch.md` records WebFetch returns nav-only on cloud.google.com, S4-S7 via
      WebFetch)
- [x] 10+ unique URLs total (28: 7 read-in-full + 21 snippet-only)
- [x] Recency scan (2024-2026) performed + reported, with 4 findings and 1 marked SUPERSEDES
- [x] Full pages read, not abstracts. **One honest failure recorded**: arXiv 2506.11076 (DCE-LLM) has
      no `/html/` rendering (404 on the bare id, v1 and v2); per `.claude/rules/research-gate.md` I did
      not WebFetch the `/pdf/` URL, so it is listed snippet-only and does NOT count toward the floor.
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the internal scope, plus `cycle_health.py`
      and `paper_trading.py` which the scope implied but did not name
- [x] Contradictions noted (S5 internally inconsistent on `>` vs `>=`; S7 adversarial on detector
      availability)
- [x] Claims cited per-claim with URL or file:line
- [ ] **Gap:** I did not query BigQuery to count actual `llm_call_log` rows per cycle, so the claim
      that orchestrator Gemini calls ARE metered rests on the code path (`llm_client.py:1115-1139`,
      `orchestrator.py:887-921`), not on measured rows. Given the swallowing `except`es, Main should
      treat per-cycle row counts as unverified until measured.

---

## Status log (append-only, write-first discipline)

- [start] Brief created; envelope born inert. Read `.claude/agents/researcher.md` +
  `.claude/rules/research-gate.md` in full as operating instructions.
- [S1] Google Cloud "Error code 429" read in full via curl + tag-strip (31,616 chars).
- [S2] Vertex AI generative-AI quotas doc read in full (39,721 chars); zero "per day" rows.
- [S3] Cloud Quotas troubleshoot read in full (9,026 chars, whole article).
- [S4/S5] `unusedwrite` analyzer + SRE Workbook ch.5 read in full via WebFetch.
- [S6/S7] OneUptime quota post + CNCF telemetry post read in full via WebFetch.
- [internal] Confirmed `_intended_path` write-only (repo-wide grep: 1 write, 0 reads); found 4 MORE
  dead summary fields and proved the drop at all three boundaries (`_persist_analysis:3293-3307`,
  `record_cycle_end:1932-1951`, `paper_trading.py:1409-1418/:1485-1493`).
- [end] Envelope flipped to COMPLETE.
</content>
