# Research Brief -- step 86.9

**Topic:** Is raising a global timeout ever the right fix for a batch job that overruns?
Per-item timeout + bounded queue vs a larger per-batch deadline; timeout-raise as
hung-dependency masking; empirically right-sizing a budget; reading a RUNNING process's
effective config.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Session:** 2026-08-11.

> **Length disclosure:** this brief exceeds the moderate tier's ~700-word guide. The caller
> set five specific internal establishments plus four external sub-questions, and the
> measured tables are load-bearing. Prose is kept tight; no padding.

---

## ENVELOPE

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 13,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "Raising the GLOBAL per-batch deadline is the one remedy the literature rejects; raising a PER-ITEM cap against a measured censored distribution is the one it endorses -- and pyfinagent applied exactly the former (ask #23, 7200->10800) while the latter (ask #24, rail 150->210) stays open. Criterion 1's premise is REFUTED: GET /api/settings/ exposes the value and returns 10800.0 live from pid 66306.",
  "brief_path": "handoff/current/research_brief_86.9.md",
  "gate_passed": true
}
```

---

## Method disclosure (read this before the source tables)

**`WebSearch` was unavailable for this entire session.** The first search call returned:
*"Web search was not performed: this session has used its web search budget (200 of 200
WebSearch calls)."* The budget is session-shared and was exhausted **before this agent was
spawned**. Consequences, stated plainly:

- The mandatory **three-variant search discipline** (current-year / last-2-year / year-less)
  in `.claude/rules/research-gate.md` **could not be executed.** I did not run it and I am
  not claiming I did.
- All 8 read-in-full sources were reached by **direct `WebFetch` of canonical URLs** (from
  domain knowledge) plus **citation-chaining** from inside sources I had already read.
- The recency scan was performed by a **different mechanism** -- fetching each doc's current
  revision and reading its embedded publication/revision metadata -- and it did return
  substantive in-window findings. See "Recency scan".

This is a genuine methodological shortfall on the search-composition rule. It does not
touch the hard blockers (>=5 read in full, >=10 URLs, recency reported), so I return
`gate_passed: true`, but Main should weigh it.

---

## Read in full (>=5 required; counts toward the gate) -- 8 sources

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-11 | Official book (Google SRE ch.22) | WebFetch, full | "Having deadlines several orders of magnitude longer than the mean request latency is usually bad."; deadline propagation; bounded queues |
| 2 | https://sre.google/sre-book/handling-overload/ | 2026-08-11 | Official book (Google SRE ch.21) | WebFetch, full | "Different queries can have vastly different resource requirements"; retry budgets (3/request, 10%/client) |
| 3 | https://grpc.io/blog/deadlines/ | 2026-08-11 | Official vendor doc | WebFetch, full | "when you don't set a deadline, resources will be held for all in-flight requests"; choosing one needs to know "which RPCs are serial, and which can be made in parallel" |
| 4 | https://prometheus.io/docs/practices/histograms/ | 2026-08-11 | Official doc | WebFetch, full | "averaging the quantiles yields statistically nonsensical values"; quantile estimates carry bucket-width error |
| 5 | https://sre.google/workbook/configuration-design/ | 2026-08-11 | Official book (SRE Workbook ch.14) | WebFetch, full | **NEGATIVE RESULT** -- covers versioning + code/data separation, does NOT address verifying a change is in force in a running process |
| 6 | https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead | 2026-08-11 | Official vendor architecture doc | WebFetch, full | "the resources that the client's request uses might remain unavailable for an extended period... those resources might be exhausted" |
| 7 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-08-11 | Official vendor architecture doc | WebFetch, full | **"To resolve this problem, set a shorter time-out."** + "Inappropriate time-outs on external services" |
| 8 | https://docs.spring.io/spring-boot/reference/actuator/endpoints.html | 2026-08-11 | Official framework doc | WebFetch, full | `configprops` / `env` / `startup` -- the concrete instrumentation exemplar for reading a RUNNING process's effective config |

## Identified but NOT read in full (context; does NOT count toward the gate) -- 13

No search snippets were available (see Method disclosure). Provenance is stated per row:
**[chain]** = hyperlinked from inside a source I read in full; **[domain]** = canonical URL
from prior knowledge; **[fail]** = fetch attempted and failed.

| URL | Kind | Provenance / why not read in full |
|-----|------|-----------------------------------|
| https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/ | Industry | [fail] 301 to builder.aws.com |
| https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter | Industry | [fail] refetched; returned only the "AWS Builder Center" header -- JS-rendered body, no text. Honest failure; NOT counted |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/retry | Official doc | [chain] from #7; budget |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/health-endpoint-monitoring | Official doc | [chain] from #7; budget |
| https://learn.microsoft.com/en-us/azure/architecture/best-practices/transient-faults | Official doc | [domain]; budget |
| https://resilience4j.readme.io/docs/getting-started | Library doc | [chain] from #6 |
| https://www.pollydocs.org/ | Library doc | [chain] from #6 |
| https://sre.google/workbook/managing-load/ | Official book | [domain]; overlaps #2 |
| https://sre.google/sre-book/monitoring-distributed-systems/ | Official book | [domain]; overlaps #4 |
| https://sre.google/workbook/configuration-specifics/ | Official book | [domain]; #5 already showed the gap |
| https://12factor.net/config | Methodology | [domain]; budget |
| https://queue.acm.org/detail.cfm?id=2903468 | Peer-reviewed (ACM Queue) | [domain] Hartmann, "Statistics for Engineers" -- wanted for percentile sample-size; not reached |
| https://research.google/pubs/the-tail-at-scale/ | Peer-reviewed (CACM) | [domain] Dean & Barroso; not reached |

**URLs collected: 21** (8 read in full + 13 identified).

---

## Recency scan (2024-2026) -- performed, with 2 in-window findings

**Method (deviation disclosed above):** year-scoped search queries were impossible. Instead
I fetched each source's current revision and read the publication/revision metadata embedded
in the returned content. Two of the eight carry explicit in-window dates **in the fetched
bytes**, not by inference:

- Azure **bulkhead**: `ms.date: 2026-03-19`, `updated_at: 2026-06-24T05:04:00Z`
- Azure **circuit-breaker**: `ms.date: 2025-02-05`, `updated_at: 2026-07-02T17:35:00Z`
- Spring Boot Actuator docs are the current 4.1.0 line; Prometheus native histograms are a
  recent-generation addition.

**Result: 2 new findings that COMPLEMENT (do not supersede) the canonical Google SRE material.**

1. **AI/inference workloads are now called out as needing strict bulkheads.** The 2026
   bulkhead revision adds: *"AI and inference workloads often require strict bulkheads
   because of deployment-level quotas and concurrency limits. For example, isolate model
   deployments or Foundry resources per workload or per tenant."* This is directly on point
   for pyfinagent, whose batch IS an LLM fan-out under a concurrency cap.
2. **Adaptive/ML-tuned thresholds replace static ones.** The 2025-2026 circuit-breaker
   revision adds: *"Traditionally, circuit breakers relied on preconfigured thresholds, such
   as failure count and time-out duration. This approach resulted in a deterministic but
   sometimes suboptimal behavior. Adaptive techniques that use AI and machine learning can
   dynamically adjust thresholds based on real-time traffic patterns, anomalies, and
   historical failure rates."* Same revision adds an "Adaptability to compute
   diversification" consideration (cold starts, serverless vs containerized).

Nothing in the window contradicts the 2016-era Google SRE guidance; the deadline-propagation
and bounded-queue advice is unchanged and still canonical.

---

## Key findings

### (a) Per-item timeout + bounded queue beats a bigger per-batch deadline

1. **A deadline orders of magnitude above typical work is explicitly an anti-pattern.**
   *"Having deadlines several orders of magnitude longer than the mean request latency is
   usually bad."* and *"Setting either no deadline or an extremely high deadline may cause
   short-term problems that have long since passed to continue to consume server resources
   until the server restarts."* (Google SRE ch.22, https://sre.google/sre-book/addressing-cascading-failures/, 2026-08-11)

2. **Deadline propagation is the named mechanism: one absolute deadline set high in the
   stack, checked at every stage.** *"With deadline propagation, a deadline is set high in
   the stack (e.g., in the frontend). The tree of RPCs emanating from an initial request will
   all have the same absolute deadline."* and *"The server should check the deadline left at
   each stage before attempting to perform any more work on the request."* The failure mode
   is named exactly: *"If server B uses deadline propagation, it should set a 2-second
   deadline, but suppose it instead uses a hardcoded 20-second deadline for the RPC to server
   C... server C processes the request thinking it has 15 seconds to spare, but is not doing
   useful work."* (ibid.) **A batch with one outer deadline and no inner checks is the
   "hardcoded, non-propagated" side of that example.**

3. **Bounded queues, sized small relative to the worker pool.** *"it is usually better to
   have small queue lengths relative to the thread pool size (e.g., 50% or less), which
   results in the server rejecting requests early when it can't sustain the rate of incoming
   requests"*; long queues cause *"latency increases (the requests are queued for longer
   amounts of time) and the queue uses more memory."* (ibid.)

4. **Bulkheading is the structural answer to "one hung item eats everyone's budget."**
   *"When the consumer sends a request to a misconfigured or unresponsive service, the
   resources that the client's request uses might remain unavailable for an extended period.
   As requests to the service continue, those resources might be exhausted... At that point,
   the consumer's requests to other services are affected."*
   (https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead, 2026-08-11)

5. **Batch size is the wrong unit of capacity; per-item cost varies wildly.** *"Different
   queries can have vastly different resource requirements. A query's cost can vary based on
   arbitrary factors..."* -- which is why Google rejects QPS as a capacity metric.
   (https://sre.google/sre-book/handling-overload/, 2026-08-11) The same logic says a
   per-BATCH second-count is a poor budget unit: it conflates "how many items" with "how
   expensive each item is."

### (b) A timeout raise MASKING a hung dependency -- and how to tell

6. **The canonical prescription for a hanging dependency is a SHORTER timeout plus fail-fast,
   not a longer one.** *"you can configure an operation that invokes a service to implement a
   time-out. If the service fails to respond within this period, the operation replies with a
   failure message. However, this strategy can block concurrent requests to the same
   operation until the time-out period expires. These blocked requests might hold critical
   system resources... In these situations, an operation should fail immediately... **To
   resolve this problem, set a shorter time-out. But ensure that the time-out is long enough
   for the operation to succeed most of the time.**"*
   (https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker, 2026-08-11)

7. **Long timeouts actively defeat the detection machinery.** *"Inappropriate time-outs on
   external services: A circuit breaker might not fully protect applications from failures in
   external services that have long time-out periods. If the time-out is too long, a thread
   that runs a circuit breaker might be blocked for an extended period before the circuit
   breaker indicates that the operation failed. During this time, many other application
   instances might also try to invoke the service through the circuit breaker and tie up
   numerous threads before they all fail."* (ibid.) The same doc separates the two cases the
   caller asks about: transient faults *"typically correct themselves after a short period of
   time"* (retry/wait is right) vs *"faults that take longer to fix"* where *"an application
   shouldn't continually retry an operation that's unlikely to succeed."*

8. **THE DISCRIMINATOR -- and wall-clock data alone is NOT sufficient.** You cannot separate
   "genuinely needs more time" from "a hung dependency" from batch wall-clock, because a
   batch that hits its deadline is **right-censored**: the recorded duration is the budget,
   not the work. The discriminating test is on the **per-item** distribution:
   - **Censoring test.** Count successes landing within epsilon of the per-item cap. A mass of
     successes piled just under the cap means the cap is **truncating** the distribution --
     the cap is censoring, not protecting, and the "timeouts" are mostly slow successes.
   - **Bimodality test.** A hung dependency produces a bimodal shape: a normal-latency mode
     plus a spike exactly AT the cap. Genuinely-slow work shifts the whole distribution.
   - **Yield test.** Wasted-time = (timeouts x cap). If that exceeds the overrun, the
     overrun is an artifact of the failing dependency, not of an undersized budget.
   pyfinagent already implements the first of these -- see Internal #4.

### (c) Right-sizing a budget empirically

9. **A mean is the wrong statistic for a deadline** -- deadlines are tail decisions. Google
   uses the mean only as a *lower* sanity bound ("orders of magnitude longer than the mean
   ... is usually bad"), never as the setting. Prometheus is explicit that per-quantile
   aggregation is invalid: *"aggregating the precomputed quantiles from a summary rarely
   makes sense. In this particular case, averaging the quantiles yields statistically
   nonsensical values"*, flagging `avg(...{quantile="0.95"})` as **"BAD!"**.
   (https://prometheus.io/docs/practices/histograms/, 2026-08-11)
10. **Quantile estimates carry explicit error bars.** Prometheus documents that a classic
    histogram with 200-300ms buckets yields an estimate of 295ms with only *"the guarantee
    that the true value is between 200ms and 300ms"* -- so a percentile is a range, and a
    deadline set at a measured p95 needs headroom for that error. (ibid.)
11. **Sample size.** *Not sourced -- I could not reach the ACM Queue / Tail-at-Scale sources
    (see snippet table), and neither Prometheus nor the SRE chapters state a minimum n
    (Prometheus explicitly contains no sample-size discussion).* Recorded as a **gap**. What
    I can state without a citation is the arithmetic fact that with n samples the largest
    observation sits near the n/(n+1) plotting position, so **n = 7 cycles cannot express a
    p95 at all** (7/8 = 0.875) -- any "p95" from pyfinagent's current sample is extrapolation.
12. **Composition matters more than a single number.** Choosing a deadline requires knowing
    *"the end to end latency of the whole system, which RPCs are serial, and which can be
    made in parallel."* (https://grpc.io/blog/deadlines/, 2026-08-11) A batch deadline is a
    function of per-item cost AND effective parallelism -- so a scheduling change moves the
    required budget without any per-item speedup.

### (d) Verifying a config is IN FORCE, not merely on disk

13. **NEGATIVE RESULT, and it is a real finding.** The canonical config chapter of the SRE
    Workbook does **not** address this. Fetched in full, it covers code/data separation
    (*"having both code and data, but separating the two, is optimal"*) and versioning
    (*"Versioning configuration... allows you to go back in time to see what the
    configuration looked like at any given point in time"*), and the closest it comes is
    *"When consuming the final configuration data, you will find it useful to also store
    metadata about how the configuration was ingested."* It stops short of runtime exposure.
    (https://sre.google/workbook/configuration-design/, 2026-08-11) **So "read the effective
    config from the running process" is a framework-level practice, not an SRE-book one.**
14. **The framework-level answer is a config-introspection endpoint.** Spring Boot Actuator
    ships exactly the three instruments the caller hypothesised:
    - `configprops` -- *"Displays a collated list of all `@ConfigurationProperties`. Subject
      to sanitization."* (the effective, bound, typed config)
    - `env` -- *"Exposes properties from Spring's `ConfigurableEnvironment`. Subject to
      sanitization."* (raw resolved sources, so you can see WHICH source won)
    - `startup` -- *"Shows the startup steps data collected by the `ApplicationStartup`."*
    - `health` -- *"Shows application health information."*
    Crucially it also ships the safety rail: values from `/env` and `/configprops` are
    *"always fully sanitized by default (replaced by `******`)"*, unsealed only via
    `show-values` = `never` / `always` / `when-authorized`.
    (https://docs.spring.io/spring-boot/reference/actuator/endpoints.html, 2026-08-11)
    **The design lesson for pyfinagent: a config endpoint is standard practice AND must
    default to sanitized, since `Settings` holds API keys.**

---

## Internal code inventory

| File | What I looked at | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | :305-318, :406, :420-432, :500-520, :1204-1229, :1888-1930, :2088, :2136-2138 (3752 lines) | The batch job + its deadline | **Single global deadline, no per-item timeout** |
| `backend/api/settings_api.py` | :123, :171, :308, :330-400, :451, :497 | Config read/write API | **Exposes the budget -- refutes the premise** |
| `backend/config/settings.py` | :33, :655-656 | Settings model + `lru_cache` | Default 7200.0; cache is runtime-clearable |
| `backend/agents/claude_code_client.py` | :302, :396, :417, :428, :591, :593, :600, :751 | The per-item (rail) timeout | `timeout_s=150` cap; ask #24 targets it |
| `scripts/diagnostics/measure_analysis_phase.py` | head + **executed twice** | The measurement instrument | Exists, works, **re-run by me since the raise** |
| `backend.log` (15,367,244 B) | via diagnostic, 72,671 lines | Live cycle records | 1 cycle w/ analysis phase |
| `handoff/logs/backend.log.20260810T064130Z.gz` | via diagnostic, 234,836 lines | Rotated cycle records | 6 more cycles |
| `handoff/current/research_brief_85.4.md` | :321 | Prior research | Contains ask #24's measured rationale |
| `handoff/current/day_report_2026-08-09.md` | :174-186 | Ask ledger | #23 applied; #24/#25 open |
| `handoff/harness_log.md` | :14495, :21598-21662, :24575-24864, :31925 | Raise history | 1800 -> 3600 -> 7200 -> 10800 |
| `handoff/current/evaluator_critique_85.4*.{md,json}` | grep | Prior Q/A | Confirms 8554s/8529s projections |
| `handoff/current/experiment_results_85.4.md` | :20, :246, :281 | Prior GENERATE | Merged-dispatch flag is DARK |
| `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | StandardOut/ErrorPath | Log location | Log is repo-root `backend.log` |
| `handoff/archive/misc/live_check_27.6.md` | :93 | Prior live evidence | Endpoint used this way before |

### (1) CRITERION 1 -- the premise is REFUTED. A means DOES exist.

**`GET /api/settings/` exposes `paper_cycle_max_seconds`.** Measured by me on 2026-08-11
against the RUNNING backend (pid 66306, started 2026-08-10 21:33:01, confirmed via
`ps -eo pid,lstart,command`):

```
HTTP=200 ; 45 keys
paper_cycle_max_seconds = 10800.0
paper_screen_top_n = 10 ; paper_analyze_top_n = 5
```

Anchors: `backend/api/settings_api.py:123` (`FullSettings.paper_cycle_max_seconds: float =
7200.0`), `:171` (`SettingsUpdate`, `ge=300.0 le=21600.0`), `:308` (`_FIELD_TO_ENV` ->
`PAPER_CYCLE_MAX_SECONDS`), `:383` (`_settings_to_full` copies it). Added phase-cycle-7 /
step 38.12; the same endpoint was used as live evidence before
(`handoff/archive/misc/live_check_27.6.md:93` records `paper_cycle_max_seconds=7200.0`).

**This is direct runtime evidence, not the mtime inference the criterion rejects.** But it
comes with two caveats Main must carry into the contract:

- **(i) The process does NOT hold a frozen startup snapshot.** `get_settings()` is
  `lru_cache`d (`backend/config/settings.py:655-656`) but the cache is cleared at runtime in
  three places: `settings_api.py:451` and `:497` (on PUT), and -- decisively --
  `backend/services/autonomous_loop.py:2136-2138`, which calls
  `_get_settings_fresh.cache_clear()` then `_get_settings_fresh()` **on every ticker** inside
  `_run_single_analysis` (comment: *"cure the get_settings() lru_cache desync across uvicorn
  workers"*). So the process re-reads `backend/.env` from disk many times per cycle.
  **Corollary: a `.env` edit reaches the NEXT cycle without a restart** -- the raise was
  probably in force before the 21:33 restart, though I did not observe a pre-restart cycle
  that proves it.
- **(ii) The GET is served through an API cache.** `get_all_settings` does
  `cache.get("settings:full")` / `cache.set(..., ENDPOINT_TTLS["settings:full"])`, so a
  response can be up to one TTL stale.

**What the endpoint proves:** the value the process's `Settings()` currently carries.
**What it does NOT prove:** the deadline an *in-flight* cycle is running under, because
`_cycle_timeout` is captured **once** at `:507` and never re-read.

**Instrumentation that would close the remaining gap (cheap, three options):**
1. **A startup log line** -- one `logger.info` of the effective budget at boot. Zero risk.
   Currently the backend logs it **nowhere**; the only budget string in the logs is the
   failure path at `:1896` (`"Paper trading cycle TIMED OUT after %.0fs"`), i.e. you learn
   the budget only by blowing it.
2. **Log the captured value at cycle start**, next to `:507`. This is the one that answers
   "what is THIS cycle's deadline" -- strictly more informative than the endpoint.
3. **A `/healthz` detail or `/config` block** in the Spring-Actuator `configprops` shape
   (finding #14), sanitized by default because `Settings` holds API keys.

### (2) Where the budget is consumed, and what happens on overrun

- `autonomous_loop.py:507` --
  `_cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))`
  Captured once, from the settings bound at `:406` (`settings = settings or get_settings()`).
- `autonomous_loop.py:514` -- `async with asyncio.timeout(_cycle_timeout):` wraps the
  **entire** cycle body (Steps 0-9). The textbook "one big per-batch deadline".
- **NO inner per-ticker timeout.** `_run_single_analysis` (`:2088`) is awaited from
  `_run_and_persist_one` (`:1204`, call at `:1229`) with no `asyncio.wait_for`/`timeout`.
  Concurrency is bounded by a semaphore (cap 3) only. A hung ticker consumes the whole
  remaining batch budget -- exactly finding #2/#4.
  - There IS a per-item timeout one layer down, at the subprocess: `claude_code_client.py:593`
    `def __init__(self, model_name: str, timeout_s: int = 150)`, `:591`
    `recommended_step_timeout = 150`, `:600` `self.recommended_step_timeout = timeout_s + 30`,
    enforced at `:417` `timeout=timeout_s`. So the shape is: **per-CALL cap (150s) ->
    [no per-TICKER cap] -> per-BATCH cap (10800s)**. The missing middle is where a slow
    ticker becomes unbounded, because one ticker makes many rail calls.
- On overrun (`:1895-1899`):
  ```python
  except asyncio.TimeoutError:
      logger.error("Paper trading cycle TIMED OUT after %.0fs", _cycle_timeout)
      summary.update({"status": "timeout", "error": f"cycle exceeded {_cycle_timeout:.0f}s"})
  ```
  Completed work is kept; remaining steps (execute trades / snapshot / outcome) are skipped.
  `finally` (`:1917-1927`) releases the cycle lock. **No rollback, no partial-batch resume.**

**Config drift -- THREE different in-code defaults for one knob:**

| Site | Default | Anchor |
|---|---|---|
| `Settings` field | `7200.0` | `backend/config/settings.py:33` |
| API response model | `7200.0` | `backend/api/settings_api.py:123` |
| **The actual consumer** | **`1800.0`** | `backend/services/autonomous_loop.py:507` |

The consumer's fallback is stale by three raises. Currently unreachable (the field always
exists), but it is a live landmine: any refactor handing the loop a settings-like object
without the attribute silently reverts the budget to 30 minutes.

### (3) The MEASURED per-cycle distribution (n = 7)

Instrument: `scripts/diagnostics/measure_analysis_phase.py`, run by me on 2026-08-11 against
`backend.log` (72,671 lines) and the rotated `backend.log.20260810T064130Z.gz` (234,836
lines, decompressed to scratch). Read-only; its docstring says it exists *"so the 7200s
budget can be judged against a number instead of an opinion."*

| # | Cycle start | Wall / projected | Terminal | Finished | rail calls | rail timeouts | rate |
|---|---|---|---|---|---|---|---|
| 1 | 2026-08-05 20:00:01 | 5670s (projected) | (none) | 6/6 | 124 | 29 | **0.2339** |
| 2 | 2026-08-06 20:00:01 | **7200.117s** | **timeout** | 5/6 (NTAP) | 175 | 26 | **0.1486** |
| 3 | 2026-08-07 20:00:01 | **7200.077s** | **timeout** | 5/6 (NTAP) | 177 | 32 | **0.1808** |
| 4 | 2026-08-08 22:58:29 | 340.4s | completed | 6/6 | 20 | 0 | 0.0 |
| 5 | 2026-08-09 15:03:44 | 322.1s | completed | 6/6 | 33 | 0 | 0.0 |
| 6 | 2026-08-09 15:25:29 | 5942.7s | completed | 6/6 | 172 | 17 | 0.0988 |
| 7 | 2026-08-10 20:00:02 | **4532.1s** | completed | 6/6 | 152 | 1 | 0.0066 |

**Answers to the caller's question 3:**
- **How many cycles: 7.**
- **Order statistics** (322, 340, 4532, 5670*, 5943, 7200, 7200): **min 322s, median 5670s,
  max 7200s.** *= projected.
- **p50 ~ 5670s. p95: NOT COMPUTABLE at n=7** (see finding #11) -- reporting one would be
  fabrication.
- **How many ever exceeded 7200s: 2 of 7** (08-06, 08-07), both by ~1330s
  (projected 8554s and 8529s).
- **The premise "cycles were timing out" is CONFIRMED but STALE.** Both overruns predate the
  raise. **Since the raise (2026-08-09 13:50Z) there have been zero overruns -- and also zero
  cycles that would have needed more than 7200s.** Cycle #7 finished at 4532s, **-2708s
  against the OLD budget.** The extra 3600s has never once been used.
- **The max is CENSORED.** Two of seven "durations" are the budget itself, not the work. Any
  percentile from this column is biased low; the `PROJECTED` column is the uncensored estimate.

**The correlation that decides this step.** Per-cycle wall-clock tracks the **rail
subprocess-timeout rate**, not the ticker count (constant at 6 throughout):

- The 2 over-budget cycles ran at 14.9% and 18.1% rail timeouts.
- The fastest 2 cycles (340s, 322s) ran at **0%**.
- The post-raise cycle ran at **0.66%** and finished ~2708s early.

**Wasted-time arithmetic (the yield test, finding #8).** Cycle 3: 32 timeouts x 150s cap =
**4800s of subprocess time that produced nothing.** The overrun was 1329s. At that cycle's
observed effective parallelism (~1.85 on the comparable cycle 7), that is roughly
**~2600s of wall-clock waste vs a 1329s overrun.** *Caveat, stated against interest: I did
not capture cycle 3's own parallelism figure, so the wall-clock conversion is an estimate;
the subprocess-seconds figure is exact.* Even halving it clears the overrun.

**=> The overrun was caused by a failing dependency burning the budget, not by the batch
being genuinely too big for 7200s.** This is finding #8's discriminator, satisfied on real
data.

### (4) `measure_analysis_phase.py` -- what it measures, and yes it has been re-run

`scripts/diagnostics/measure_analysis_phase.py` (phase-85.4 C1/C2, 13,820 B, mode 0755).
Per its docstring it reports, per cycle: analysis-phase start/end, tickers dispatched vs
finished, per-ticker wall-clock, serial ticker-seconds, observed effective parallelism, the
projected uncensored wall-clock, and *"the cc_rail subprocess-call latency distribution in
the window, including how many 'successes' landed within 5s of the 150s subprocess cap **(a
truncated distribution means the cap is censoring, not protecting)**."*

**That last clause is the censoring test from finding #8, already implemented in this repo.**
It is the single most important existing asset for this step.

**Has it been run since the raise?** It had not been -- until now. Its outputs are cited in
phase-85.4 artifacts dated 2026-08-08/09 (pre-raise). **I re-ran it on 2026-08-11, post-raise,
and the table in (3) is that output.** Note `agent latency: None` on the current log: the
censoring statistic did not populate for these runs, so the 5s-of-cap check needs a look
before it can be relied on.

### (5) Operator asks #23 / #24 / #25

Ledger at `handoff/current/day_report_2026-08-09.md:180`:

> `| 23/24/25 | cycle budget / rail timeout / merged dispatch | #23 applied; **#24 and #25
> still open and both reduce cost rather than raise a ceiling** |`

Origin, `handoff/harness_log.md:31925`: *"The two config remedies the gate identified -- rail
`timeout_s` 150->210 and `paper_cycle_max_seconds` 7200->10800 -- were not applied inline;
filed as asks #23/#24, with #25 for the dark flag."*

- **#23 -- `paper_cycle_max_seconds` 7200 -> 10800. APPLIED 2026-08-09.** The global
  per-batch deadline raise. This is step 86.9's subject.
- **#24 -- cc_rail `timeout_s` 150 -> 210. STILL OPEN.** Rationale verbatim from
  `handoff/current/research_brief_85.4.md:321`: *"Rationale is measured, not guessed: **p90 =
  134 s and max success = 145 s against a 150 s cap -- the distribution is truncated at the
  cap, so the 17.7% "timeouts" are mostly slow successes.** Recovers up to 4,650 s of serial
  time (~1,845 s wall) and removes the retry multiplier behind it. Highest value, lowest
  risk."* Target `claude_code_client.py:593`; `:600` auto-tracks `recommended_step_timeout`
  to 240, so it is a one-line edit.
- **#25 -- promote `paper_merged_analysis_dispatch_enabled`. STILL OPEN, ships DARK**
  (`experiment_results_85.4.md:20`; live-verified False by the 85.4 Q/A). Fixes the two
  sequential `asyncio.gather`s sharing one `Semaphore(3)`; measured to have idled a free slot
  **1923s** on 2026-08-07 (`autonomous_loop.py:305-318`).

---

## Consensus vs debate (external)

**Consensus** (unanimous across Google SRE, Azure, gRPC): deadlines should be short,
propagated, and checked at each stage; queues bounded; failing dependencies fail fast; a
timeout far above typical work is an anti-pattern. **No source I read recommends raising a
global deadline to fix an overrun.**

**Where the sources genuinely qualify each other** -- and this is the debate that matters
here: Azure's circuit-breaker doc pairs *"set a shorter time-out"* with *"But ensure that the
time-out is long enough for the operation to succeed most of the time"*, and Google concedes
*"Short deadlines can cause some more expensive requests to fail consistently. Balancing
these constraints to pick a good deadline can be something of an art."* So the literature is
**not** "always shorter". It is: **the cap must sit above the success distribution's tail and
below the point where waiting is pointless.** Ask #24 is precisely that case -- a cap at 150s
sitting BELOW the observed max success of 145s + variance, i.e. cutting into real successes.

**This yields the answer to the step's headline question.** "Is raising a timeout ever
right?" -- **Yes, but only a PER-ITEM cap, and only against a measured censored
distribution.** Raising a **global per-batch** deadline is not the same act: it buys time for
whatever is wasting it, and it is the one the sources reject. pyfinagent applied the
rejected one (#23) and has left the endorsed one (#24) open.

## Pitfalls (from literature)

1. **The budget-for-waste trap** -- a raised deadline lets *"short-term problems that have
   long since passed to continue to consume server resources"* (SRE ch.22).
2. **Detection blindness** -- a long timeout blocks the thread before the breaker can trip
   (Azure circuit-breaker, "Inappropriate time-outs on external services").
3. **Censored measurement** -- deadline-terminated runs record the budget, not the work; every
   percentile from that column is biased low.
4. **Mean-based sizing** -- means hide the tail; per-quantile averaging is *"statistically
   nonsensical"* (Prometheus).
5. **Non-propagated inner deadlines** -- an inner cap larger than the caller's remaining
   budget means the sub-call "is not doing useful work" (SRE ch.22).
6. **Retry multiplication** -- retries on top of a per-item cap multiply the wasted time;
   Google bounds them (3/request, 10%/client).
7. **Unsanitized config exposure** -- a `/config` endpoint must default to sanitized
   (Spring Actuator), because `Settings` holds API keys.

## Application to pyfinagent

| Finding | Anchor | Implication for 86.9 |
|---|---|---|
| Global deadline is the only cap on the batch; no per-ticker cap | `autonomous_loop.py:507`, `:514`, `:1229`, `:2088` | The "one hung item eats the batch" shape is present. A per-ticker `asyncio.wait_for` is the missing middle layer. |
| Per-call cap exists but is censoring | `claude_code_client.py:593` (`timeout_s=150`), `:600` | Ask #24 is the *endorsed* kind of raise (p90=134s / max-success=145s vs a 150s cap). |
| Overruns correlate with rail-timeout rate, not batch size | Table in (3); 32x150s = 4800s wasted vs a 1329s overrun | The 7200s budget was never the binding constraint. #23 masked #24. |
| The raise has never been exercised | Cycle #7: 4532s vs old 7200s budget (-2708s) | There is no post-raise evidence the extra 3600s is needed. Reverting to 7200 is defensible; leaving it is *low-risk but non-load-bearing*. |
| Deadline captured once, never re-checked | `:507` capture; no per-stage remaining-budget check | Deadline **propagation** (SRE ch.22) is absent -- Step 6-9 begin with no check of the remaining budget. |
| Three conflicting defaults | `settings.py:33` / `settings_api.py:123` = 7200 vs `autonomous_loop.py:507` = **1800** | Config-drift landmine; align or remove the fallback. |
| Config IS readable at runtime | `GET /api/settings/` -> 10800.0 on pid 66306 | **Criterion 1's premise is refuted.** Use this, not mtime ordering. |
| But nothing logs the budget on the success path | only `:1896` on failure | Cheapest fix: one `logger.info` at `:507` naming THIS cycle's captured deadline. |
| Instrument already implements the censoring test | `measure_analysis_phase.py` docstring | Re-run post-raise (done, 2026-08-11); `agent latency: None` needs a look. |

**Recommended framing for the contract (Main owns PLAN; this is input, not a decision):**
the defensible position is that #23 was a *stopgap*, that the *fixes* are #24 (per-item cap
against a censored distribution) and #25 (the scheduling barrier), and that criterion 1 is
satisfiable **today** via `GET /api/settings/` with the two caveats in (1), with a one-line
startup/cycle-start log as the durable instrumentation.

---

## Research Gate Checklist

**Hard blockers:**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total (incl. identified-only) -- **21**
- [x] Recency scan (last 2 years) performed + reported -- **yes, 2 in-window findings**, by
      revision-metadata method; **the year-scoped search variants could not be run** (WebSearch
      budget exhausted session-wide before spawn) -- disclosed in Method disclosure
- [x] Full pages read (not abstracts) for the read-in-full set -- all 8 returned body text
- [x] file:line anchors for every internal claim

**Soft checks:**
- [x] Internal exploration covered every module the caller named (5/5 establishments answered)
- [x] Contradictions / consensus noted (the "shorter vs long enough" tension is the crux)
- [x] All claims cited per-claim with URL + access date
- [ ] **GAP:** no source obtained for percentile sample-size (finding #11); ACM Queue
      "Statistics for Engineers" and Dean & Barroso not reached
- [ ] **GAP:** AWS Builders' Library timeouts article is JS-rendered and returned no body on
      two attempts; not counted as read
- [ ] **GAP:** three-variant search discipline not executed (mechanism unavailable, not skipped)

**Verdict: `gate_passed: true`** -- all hard blockers met. Three soft gaps disclosed above;
none is padded over.
