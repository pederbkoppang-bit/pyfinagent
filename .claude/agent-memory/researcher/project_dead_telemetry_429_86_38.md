---
name: dead-telemetry-429-86-38
description: Vertex has NO per-day quota (per-minute only, DSQ removes the number entirely); _intended_path is write-only but FOUR alarm-payload fields matter more; the > vs >= choice is a provable no-op at n=5
metadata:
  type: project
---

Step 86.38 research (2026-08-11). Three findings that a future step must not
re-derive, and one that would have sent the contract in the wrong direction.

**1. The premise "per-minute vs per-day quota" is a CATEGORY ERROR on Vertex.**
Every enforced generative-AI quota is per-minute (TPM/RPM) -- 20+ "per minute"
rows and ZERO "per day" rows in the whole quotas doc
(https://cloud.google.com/vertex-ai/generative-ai/docs/quotas). The per-day
model belongs to the *Gemini Developer API* (AI Studio) free tier, a different
product. Worse: Gemini 2.0+ runs on **Dynamic Shared Quota**, where there is no
project-level number at all -- which is why forum reports say "429 despite 2%
usage" and "limit: 0". The real trichotomy is *rate / DSQ-capacity /
model-or-billing state*.
**Why:** a design that branches on per-minute-vs-per-day branches on a
distinction the platform does not make.
**How to apply:** classification must come from the already-captured exception
string (a 404 retired-pin `ClientError` is distinguishable; a window is not),
or out-of-band from `serviceruntime.googleapis.com/quota/rate/net_usage`.
Never parse a 429 body for a window. Related: [[project_gemini_lifecycle_pipeline_restoration]].

**2. The HTTP code carries no cause.** Same condition surfaces as 429 (REST),
`ResourceExhausted` (gRPC), 403 QUOTA_EXCEEDED (Compute Engine), or **5XX**
(Provisioned Throughput under the purchased amount). Google's own words: "How
this error appears to you depends on the service."

**3. `> ` vs `>=` has NO external best practice -- and at n=5 it is a NO-OP.**
Google's SRE Workbook uses `>=` in its trivial example and `>` in every
burn-rate example. So it is a local spec decision, already pinned by
`test_phase_60_1_deep_pipeline.py::test_fallback_alarm_threshold_is_strictly_greater_than`.
**The trap:** at 5 tickers the reachable fractions are 0/.2/.4/.6/.8/1.0, so
0.5 is never hit exactly and the operator choice changes nothing. Only n=2,4,6,8
makes it observable. State this before anyone "fixes" the operator and sells it
as behavioural. Related: [[feedback_measure_dont_assert_claims]].

**4. The dead field everyone names is the LEAST important one.** `_intended_path`
(autonomous_loop.py:2189) is genuinely write-only -- but it is redundant with
`_fallback_reason` being non-empty. The load-bearing dead fields are the alarm
PAYLOADS: `summary["fallback_rate"]`, `["fallback_reasons"]`, `["degraded"]`,
`["degraded_analyses"]`, dropped at THREE proven boundaries -- `_persist_analysis`
key whitelist, the `record_cycle_end` hand-picked arg list + `_funnel` whitelist
tuple, and both `run_daily_cycle` callers (`paper_trading.py`) which read only
`result.get('status')` and discard the dict.
**Why:** the P1 `raise_cron_alert` page is therefore the ONLY operator channel;
`cycle_history.jsonl` never receives them, so a suppressed page (exactly the
phase-66.1 `backend.services.alerting` ModuleNotFoundError swallowed by a
fail-open except) leaves ZERO durable trace.
**How to apply:** the fix precedent is in-tree -- phase-66.2 added `funnel` to
`cycle_health.record_cycle_end` for this exact reason, and its comment says so.
Forward the four fields the same way. Related: [[feedback_fail_open_guards_hide_their_own_breakage]].

**5. Method note:** no off-the-shelf detector finds this shape. Go's
`unusedwrite` only catches writes to *value copies*; PHPStan only private
properties; CNCF admits ~50% of metrics are never queried and offers no
procedure. A dict key on a live object that every consumer whitelists away needs
a bespoke reachability grep **at the serialization boundary** -- grep the
whitelist, not the writer.
</content>
