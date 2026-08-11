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
  "brief_status": "INCOMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "coverage": {
    "audit_class": false,
    "rounds": 0,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "IN PROGRESS -- do not read as a pass.",
  "brief_path": "handoff/current/research_brief_86.38.md",
  "gate_passed": false
}
```

---

## Status log (append-only, write-first discipline)

- [start] Brief created; envelope born inert. Read `.claude/agents/researcher.md` +
  `.claude/rules/research-gate.md` in full as operating instructions.
- [S1] Read in full: Google Cloud "Error code 429" doc (curl + tag-strip; WebFetch is nav-only on
  cloud.google.com per `feedback_gcloud_docs_fetch.md`). 31,616 chars extracted, article body at
  lines 1458-1540 of the stripped text. Page footer says "Last updated 2026-08-07 UTC".

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| S1 | https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/error-code-429 | 2026-08-11 | official doc (Google Cloud) | curl + tag-strip (WebFetch nav-only) | The 429 message is determined by the **quota framework, not the window**: "Quota framework / Message -- Pay-as-you-go / `Resource exhausted, please try again later.` ; Provisioned Throughput / `Too many requests. Exceeded the Provisioned Throughput.`" The doc lists FOUR distinct causes that all surface as the SAME PayGo string: shared-pool unavailability, per-model quota exhaustion, **acceleration limits** ("You may encounter 429 errors because of acceleration limits if your project has a sharp increase in usage"), and regional-vs-global endpoint capacity. It never mentions a per-minute vs per-day distinction. Also: for Provisioned Throughput under the purchased amount, "errors that might otherwise be 429 are returned as 5XX" -- i.e. the SAME underlying capacity condition changes HTTP class based on billing state. Page footer "Last updated 2026-08-07 UTC". |
| S2 | https://cloud.google.com/vertex-ai/generative-ai/docs/quotas | 2026-08-11 | official doc (Google Cloud) | curl + tag-strip, 39,721 chars extracted | **Every generative-AI quota in the table is a PER-MINUTE rate quota.** Grep across the whole extracted body returns 20+ "per minute" rows (e.g. `aiplatform.googleapis.com/generate_content_image_input_per_base_model_id_and_resolution`, "Generate content requests with image input per minute per base model and resolution -- 34,000,000") and **zero** "per day" rows. Batch inference explicitly has **no quota at all**: "There are no predefined quota limits on batch inference for Gemini models. Instead, the batch service provides access to a large, shared pool of resources, dynamically allocated based on the model's real-time availability and demand across all customers." So on Vertex generative AI there is no per-day quota to distinguish FROM -- the "per-day" hypothesis is a category error imported from the *Gemini Developer API* (AI Studio) free tier, which is a different product. |
| S3 | https://cloud.google.com/docs/quotas/troubleshoot | 2026-08-11 | official doc (Cloud Quotas) | curl + tag-strip, 9,026 chars (whole article) | Defines the only first-class taxonomy Google exposes: **rate quotas vs quota values**. "Rate quotas reset after a predefined time interval that is specific to each service." The HTTP class is decided by *transport*, not by cause: "If you exceeded a quota value with an HTTP/REST request, Google Cloud returns an HTTP `429 TOO MANY REQUESTS`... If you exceeded a quota value using gRPC, Google Cloud returns a `ResourceExhausted` error. **How this error appears to you depends on the service.**" Compute Engine returns `403 QUOTA_EXCEEDED` for the same condition, and `403 RATE_LIMIT_EXCEEDED` if it is a rate quota -- proof that the HTTP code alone carries no window semantics. Recommends out-of-band observability, not error parsing: "If you want to be alerted when errors happen, you can create custom alerts for specific quota errors" + the IAM & Admin > Quotas & System Limits Monitoring charts for "current and peak usage". Also documents a *third* state that mimics quota exhaustion: a **service rollout** in progress, where the console value and the enforced value disagree and the error carries `quotaExceeded.futureLimit`. |
| S4 | https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/unusedwrite | 2026-08-11 | official doc (Go x/tools) | WebFetch (full) | The canonical shipped analyzer for exactly the "write-only dead field" class: "Package unusedwrite checks for unused writes to the elements of a struct or array object... The analyzer reports instances of writes to struct fields and arrays that are never read." Its detection rule is narrow and **value-copy-specific** (the write lands on a copy: `for i, v := range input { v.x = i }`, and non-pointer receivers `func (t T) f() { t.x = i }`). It does NOT do whole-program reachability, so it would NOT catch pyfinagent's `_intended_path` case (a dict key written on a live object that no consumer ever reads). Note the doc states no limitations/false-positive conditions -- absence of a stated caveat, not a claim of completeness. |
| S5 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-11 | official doc / book (Google SRE Workbook, ch.5) | WebFetch (full) | Directly settles both halves of the threshold question. (a) **Google's own examples use BOTH operators**: the trivial approach is `job:slo_errors_per_request:ratio_rate10m{job="myjob"} >= 0.001`, the burn-rate approaches are `... ratio_rate36h{job="myjob"} > 0.001`. The chapter never argues for one over the other -- threshold selection should prioritise detecting "significant events". So strict `>` vs `>=` is a **local spec decision, not a best practice** -- it must be pinned by a test, because the two differ only on the exact-equality boundary. (b) **The small-denominator problem is the dominant failure mode at pyfinagent's volume**: "If a system receives 10 requests per hour, then a single failed request results in an hourly error rate of 10%. For a 99.9% SLO, this request constitutes a 1,000x burn rate and would page immediately." Mitigations listed: synthetic traffic, combining services into monitoring groups, client-side retries with backoff, **lowering the SLO if a single failure doesn't materially impact users**, and increasing the alerting window. Precision = "the proportion of events detected that were significant"; recall = "the proportion of significant events detected". |


</content>
</invoke>
