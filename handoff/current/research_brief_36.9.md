# Research Brief -- masterplan step 36.9 (P0 kill-switch correctness)

Tier: **moderate** (caller-specified). Audit-class: **false**.
Written: 2026-07-26. Researcher agent (Layer-3 harness MAS).
Status: **COMPLETE**. `gate_passed: true` (11 sources read in full, 42 URLs,
recency scan performed, 2 disclosed gaps recorded in the checklist).

Question: `evaluate_breach` reports `armed: true` while a leg cannot actually
fire, three ways (stale `sod_date`, `nav_invalid` returning armed:true,
`sod_nav=0.0` wedging resume). Is "fail loud + conservative" the right fix,
or does the literature argue against disarming on stale input?

---

## Queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| Q1 | `IEC 61508 dangerous undetected fault diagnostic coverage proof test interval safety instrumented function` | year-less canonical |
| Q2 | `health check endpoint degraded state "unknown is not healthy" monitoring three-state 2026` | current-year 2026 |
| Q3 | `fail-safe versus fail-danger design principle monitor silent failure safety interlock defeated` | year-less canonical |
| Q4 | `limit up limit down reference price stale previous close circuit breaker reset start of day SEC rule` | year-less canonical (domain: trading) |
| Q5 | `IEC 61511 bypass override safety instrumented function compensating measures spurious trip rate tradeoff` | year-less canonical (ADVERSARIAL -- looking for the case AGAINST disarming) |
| Q6 | `stale data detection silent failure observability monitoring 2025 freshness SLO heartbeat` | last-2-year window |
| Q7 | `sentinel value zero versus null anti-pattern defensive programming magic number absent value` | year-less canonical |
| Q8 | `2025 2026 automated trading kill switch risk control liveness "fail closed" stale reference data agent safety` | current-year + last-2-year (recency scan) |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://www.gt-engineering.it/en/insights/process-safety-processi-gt-engineering/proof-test-diagnostic-coverage/ | 2026-07-26 | industry / functional-safety (IEC 61508 normative quotes) | WebFetch, full | Three failure classes. Dangerous UNDETECTABLE failures "cannot be identified by the automatic diagnostics." Diagnostic Coverage per IEC 61508-4:2010 = "Fraction of dangerous failures detected by automatic on-line diagnostic tests", `DC = λdd / λd`. "The safety system remains 'dormant' most of the time, so latent defects cannot reveal themselves during normal operation -- only through deliberate proof testing." |
| 2 | https://instrumentationtools.com/iec-61511-standard-requirements-for-safety-bypass-and-override/ | 2026-07-26 | industry / standards summary (IEC 61511 clause text) | WebFetch, full | **`[ADVERSARIAL-adjacent]`** Cl. 16.2.4: "Continued process operation with a SIS device in bypass shall only be permitted if a hazards analysis has determined that **compensating measures are in place** and that they provide adequate risk reduction." Cl. 16.2.3: compensating measures "shall be applied with the associated operation limits (duration, process parameters, etc.)". Cl. 16.2.6/16.2.7: "All bypasses need authorization and indication"; "The status of all bypasses shall be recorded in a bypass log." Cl. 11.7.3.2: bypasses "should be installed such that alarms and manual shutdown facilities are not disabled." |
| 3 | https://silsafe.net/spurious-trip-rate-explained/ | 2026-07-26 | industry / functional safety | WebFetch, full | **`[ADVERSARIAL]`** "The art of SIF design is balancing safety integrity (low PFDavg) with availability (low STR)." "A SIS with perfect safety but terrible availability -- or vice versa -- fails its mission." "Unnecessary shutdowns don't just reduce production -- they can create safety hazards of their own... frequent nuisance trips erode operators' trust in SIS performance." NOTE: I probed for the stronger claim ("high STR causes operators to bypass the SIF") and the source does NOT make it -- recording the weaker, actually-supported version. |
| 4 | https://runtimeai.io/blog/2026-05-12-kill-switch.html | 2026-07-26 | authoritative vendor engineering blog (dated 2026-05-12) | WebFetch, full | Fail-closed is dimension 2 of 5 for a real kill switch: **"If the policy plane is unreachable, the answer is no. Not 'best-effort.' Not 'log and pass.' Closed."** Auditors "will reject systems that merely 'revoke its OAuth token' in favor of architectures ensuring denial-by-default when the authorization system cannot verify state." |
| 5 | https://www.investor.gov/introduction-investing/investing-basics/glossary/stock-market-circuit-breakers | 2026-07-26 | official (SEC investor education) | WebFetch, full | "These triggers are set by the markets at point levels that are **calculated daily** based on the **prior day's closing price** of the S&P 500 Index." Levels 7% / 13% / 20%. |
| 6 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-26 | authoritative blog | WebFetch, full | "Circuit breakers are a valuable place for monitoring. **Any change in breaker state should be logged and breakers should reveal details of their state for deeper monitoring.**" Half-open = "a trial call, which will either reset the breaker if successful or restart the timeout if not." Clients "need to react to breaker failures". |
| 7 | https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/ | 2026-07-26 | official docs | WebFetch, full | Probe outcome is derived per-probe, never inferred: HTTP "Any code >= 200 and < 400 indicates success. Any other code indicates failure." Readiness failure -> "the pod will be marked unready and will not receive traffic". **Gap found:** the page does NOT state what the pre-first-probe unknown state is -- i.e. even the canonical liveness/readiness doc leaves "unknown" undefined, which is precisely our bug class. |
| 8 | https://web-alert.io/blog/health-check-endpoint-design-livez-readyz-guide | 2026-07-26 | blog | WebFetch, full | "If you only have ok/fail, you'll either over-alert (failing the whole service) or under-alert (**claiming healthy while features are broken**)." Three-state `status: "degraded"` with per-dependency states. **Gap found:** no guidance on an UNMEASURABLE dependency -- "the three-state model assumes all critical dependencies are measurable during each health check cycle." |
| 9 | https://python-patterns.guide/python/sentinel-object/ | 2026-07-26 | authoritative practitioner reference | WebFetch, full | The `str.find()` -> `-1` case: the sentinel "is indistinguishable from legitimate data -- a programmer might accidentally use `-1` as an index." "If it had been invented today, it would instead have used the Sentinel Object pattern... by simply returning `None`... That would have left no possibility of the return value being used accidentally as an index." Recognition must be by **identity (`is`), not value**. `lru_cache` uses `sentinel = object()` to "distinguish a function call whose result is already cached and happened to be `None` from a function call that has not yet been cached." |
| 10 | https://tacnode.io/post/what-is-stale-data | 2026-07-26 | industry blog (dated **2026-02-12**) | WebFetch, full | "Stale data doesn't announce itself. A fraud model scoring transactions against hour-old behavioral signals still returns a confident score. **It's just the wrong score.**" "Stale data passes every check in your data quality monitoring. Engineers see syntactically correct records with all required fields." Freshness SLA table: Fraud/Risk Scoring `< 1 second`. |
| 11 | https://risknowlogy.com/articles/detail/17305/ | 2026-07-26 | industry / functional safety | WebFetch, full (thin page) | "Fault detection reduces the likelihood that systematic failures -- such as software bugs, logic defects, integration mistakes, or incorrect assumptions -- will **silently propagate to the safety function**." Weakest of the 11; recorded honestly -- it did NOT contain the formal DD/DU definitions I fetched it for (source 1 supplied those). |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.cmegroup.com/tools-information/webhelp/globex-credit-controls/Content/Kill-Switch.html | official exchange docs | **Attempted WebFetch, `ETIMEDOUT`.** Would have been the strongest trading-domain analogue (exchange-level kill switch + explicit re-enable). Not re-tried inside budget. |
| https://www.sciencedirect.com/science/article/abs/pii/S0306454915004806 | peer-reviewed (Reliability Eng. & System Safety) | paywalled abstract only; the proof-test-coverage argument is fully covered by source 1 |
| https://www.exida.com/images/uploads/exida_Position_on_IEC_61508_2010_definitions_minimum_HFT_v4.pdf | industry position paper (PDF) | binary PDF; content duplicated by source 1 |
| https://www.cechina.cn/eletter/standard/safety/iec61508-2.pdf | standard text (PDF) | binary PDF of IEC 61508-2 itself |
| https://www.icheme.org/media/25680/hazards-30-paper-11-ye.pdf | conference paper (Hazards 30) | PDF; STR formulae, covered by source 3 |
| https://automationforum.co/iec-61511-safety-bypass-override-sis-maintenance/ | industry | duplicate of source 2's clause coverage |
| https://www.wolterskluwer.com/en/expert-insights/functional-safety-the-next-edition-of-iec-61511 | industry | forward-looking edition notes, not clause text |
| https://infinum.com/handbook/dotnet/api/health-checks | practitioner handbook | `HealthStatus.Degraded` semantics; covered by source 8 |
| https://cloud.google.com/load-balancing/docs/health-check-logging | official docs | "Logs are not generated when the endpoint's health state is UNKNOWN" -- a real datapoint on UNKNOWN being under-served; JS-nav page (see `feedback_gcloud_docs_fetch`) |
| https://developers.cloudflare.com/load-balancing/understand-basics/health-details/ | official docs | degraded-pool semantics; covered |
| https://oneuptime.com/blog/post/2026-02-26-argocd-built-in-health-checks/view | blog (2026-02-26) | ArgoCD's `Unknown` health state propagating to the app -- recency datapoint |
| https://repost.aws/knowledge-center/route-53-fix-unhealthy-health-checks | official docs | operational, not design guidance |
| https://cronalert.com/blog/http-health-check-endpoints | blog | duplicate of source 8 |
| https://www.tradingsim.com/blog/what-is-limit-up-limit-down-in-day-trading-tradingsim | industry | LULD band = "average price over the immediately preceding five-minute trading period" -- key snippet, quoted below |
| https://www.fidelity.com/learning-center/trading-investing/trading-halts | industry | LULD/halt mechanics; covered |
| https://www.sec.gov/news/press/2011/2011-84.htm | official (SEC) | 2011 LULD proposal press release; superseded by source 5 |
| https://en.wikipedia.org/wiki/Sentinel_value | community | semipredicate problem; covered by source 9 |
| https://peps.python.org/pep-0661/ | official (Python PEP) | sentinel-value standardisation; covered by source 9 |
| https://dev.to/kalio/cache-aside-and-the-null-sentinel-pattern-5gjc | community | null-sentinel in caching |
| https://medium.com/@jooramos_37651/catching-silent-failures-in-data-pipelines-with-forecasting-metadata-and-an-llm-d316e1666bb6 | blog (June 2026) | silent-failure detection; recency datapoint |
| https://risingwave.com/blog/feature-pipeline-observability-freshness-monitoring/ | vendor blog | freshness/completeness/drift triad |
| https://www.conduktor.io/glossary/data-freshness-monitoring-sla-management | vendor glossary | freshness SLA definitions |
| https://thesynthesisai.substack.com/p/the-kill-switch | newsletter | BoE Breeden market-wide AI kill-switch proposal (June 2026) |
| https://2moon.ai/risk | vendor | strategy-level vs account-level kill switches; stale-data execution risk |
| https://www.oanda.com/us-en/trade-tap-blog/trading-knowledge/the-autonomous-trader-forex-systems/ | industry | automated-strategy failure modes incl. stale data |
| https://arxiv.org/pdf/2506.13821 | preprint | formal verification of infrastructure software; tangential |
| https://arxiv.org/pdf/2603.15963 | preprint | risk-based auto-deleveraging; tangential |
| https://www.ituonline.com/tech-definitions/what-is-fail-safe/ | community | de-energize-to-trip definition |
| https://www.sciencedirect.com/topics/materials-science/fail-safe-design | reference | fail-safe overview |
| https://www.analog.com/en/resources/analog-dialogue/raqs/raq-issue-226.html | vendor engineering | supervisory circuits / watchdog |
| https://www.ovaledge.com/blog/data-quality-monitoring-tools | vendor (2026) | tool survey |

**Unique URLs collected: 42** (11 read in full + 31 snippet-only).

## Recency scan (2024-2026)

Performed. Queries Q2, Q6 and Q8 were scoped to the 2025-2026 window; Q8 was
run explicitly as the recency pass. **Result: 3 new findings that COMPLEMENT
(and one that SHARPENS) the canonical prior art. None supersede it.**

1. **`runtimeai.io`, 2026-05-12 (read in full).** The strongest new statement
   of the exact principle this step needs, and it is newer than any of the
   canonical safety literature: **"If the policy plane is unreachable, the
   answer is no. Not 'best-effort.' Not 'log and pass.' Closed."** This is a
   2026 restatement of fail-closed specifically for *kill switches over
   autonomous agents* -- our exact object. It also asserts that auditors reject
   controls that cannot deny "when the authorization system cannot verify
   state", which is defect 2 verbatim (`nav_invalid` + `armed:true` = log and pass).
2. **`tacnode.io`, 2026-02-12 (read in full).** The 2026 framing of staleness
   as the archetypal SILENT failure: "stale data passes every check in your
   data quality monitoring... It's just the wrong score." Fraud/risk-scoring
   freshness SLA is `< 1 second`; our SOD anchor is allowed to be **2 days**
   old with no check at all.
3. **Regulatory/industry direction, June 2026 (snippet-only).** Bank of England
   DG Sarah Breeden proposed a market-wide kill switch for AI-agent-driven
   instability; a May-2026 vendor survey found **no** commercial AI kill switch
   ships all five dimensions, with fail-closed named as a gap. Direction of
   travel is toward MORE conservative default-deny, not less.

No 2024-2026 source was found that reverses the canonical IEC 61508/61511
position. The one genuine counterweight (spurious trips, source 3) is itself
canonical, not new. Older canonical sources (IEC 61508-4:2010 definitions,
Fowler's CircuitBreaker, SEC market-wide circuit breakers) remain valid.

## Key findings

1. **"Armed" is a claim about DIAGNOSTIC COVERAGE, and coverage that is
   asserted rather than measured is the definition of a dangerous UNDETECTED
   fault.** IEC 61508-4:2010 defines DC as "Fraction of dangerous failures
   detected by automatic on-line diagnostic tests", `DC = λdd / λd`
   (gt-engineering, URL above). A fault the diagnostics cannot see is DU, and
   "the safety system remains 'dormant' most of the time, so latent defects
   cannot reveal themselves during normal operation." Our `armed` flag is the
   ONLY on-line diagnostic the kill switch has. All three defects move a fault
   from the DD column to the DU column -- they do not add new failure modes,
   they *remove diagnostic coverage*, which is strictly the worse direction.

2. **Fail-closed is the 2026 consensus for kill switches specifically, and
   "log and pass" is named as the anti-pattern.** "If the policy plane is
   unreachable, the answer is no. Not 'best-effort.' Not 'log and pass.'
   Closed." (RuntimeAI 2026-05-12). Defect 2 is literally log-and-pass: NAV
   unmeasurable -> log `nav_invalid`, pass `armed:true`.

3. **A session-anchored risk threshold is re-derived every session in every
   real implementation.** SEC market-wide circuit breakers: triggers "are set
   by the markets at point levels that are calculated daily based on the prior
   day's closing price of the S&P 500 Index" (investor.gov). LULD bands are
   re-derived continuously from "the average price of the stock over the
   immediately preceding five-minute trading period" (TradingSim, snippet).
   **Nobody carries yesterday's anchor into today and calls the result a daily
   move.** This is the direct external precedent for defect 1, and it is
   stronger than a generic staleness argument: the whole point of a *daily*
   limit is that its denominator is today's open.

4. **Staleness is the archetypal silent failure -- it produces a confident
   wrong number, not an error.** "Stale data doesn't announce itself. A fraud
   model scoring transactions against hour-old behavioral signals still returns
   a confident score. It's just the wrong score." (tacnode, 2026-02-12). Our
   stale-`sod_date` case is worse than the fraud example because the wrong
   number is *directionally biased*: a 2-day move is always >= the 1-day move
   in magnitude over the same drift, so a stale anchor systematically
   OVER-reports daily loss and biases toward a spurious flatten.

5. **Sentinels must be recognised by IDENTITY, never by value -- and `0.0` for
   NAV is the `str.find() -> -1` mistake.** "The sentinel value is
   indistinguishable from legitimate data" and the modern fix is "simply
   returning `None`... That would have left no possibility of the return value
   being used accidentally as an index" (python-patterns.guide). Defect 3 is
   exactly this: `0.0` is not distinguishable from a measured baseline at the
   WRITER (`kill_switch.py:525`), so the reader and the re-anchor predicate
   each had to invent their own rule, and they invented DIFFERENT ones
   (`> 0` at `:745` vs `is None` at `paper_trader.py:1142`). This project's own
   phase-80.36 rule ("discriminate on PRESENCE, never on value") is the same
   finding, arrived at independently.

6. **A monitor's state must be observable, and a state CHANGE must be logged.**
   "Any change in breaker state should be logged and breakers should reveal
   details of their state for deeper monitoring" (Fowler). Our
   `_log_disarmed_once` (`kill_switch.py:792-810`) is a process-lifetime
   one-shot that logs only `sod_nav`/`peak_nav` -- it satisfies "logged once"
   but not "reveal details of their state", and it cannot distinguish a new
   disarm REASON from the first one.

7. **`ok`/`fail` is not enough; but the three-state model in the wild does NOT
   define UNKNOWN.** "If you only have ok/fail, you'll either over-alert...
   or under-alert (claiming healthy while features are broken)" (web-alert.io)
   -- and that same source has NO guidance for an unmeasurable dependency; the
   Kubernetes probe doc likewise never states the pre-first-probe state; GCP LB
   "Logs are not generated when the endpoint's health state is UNKNOWN". So the
   literature supports "add a third state", and *simultaneously* shows that
   UNKNOWN is the under-specified corner everywhere. Do not expect to find a
   ready-made answer; expect to have to define it, and define it explicitly.

8. **`[ADVERSARIAL]` The genuine counterweight: disarming IS itself a hazard,
   and the standard treats it as a BYPASS requiring compensating measures.**
   IEC 61511 Cl. 16.2.4: "Continued process operation with a SIS device in
   bypass shall only be permitted if a hazards analysis has determined that
   compensating measures are in place and that they provide adequate risk
   reduction." Cl. 16.2.3 adds "associated operation limits (duration...)";
   Cl. 16.2.6/16.2.7 require authorization, indication, and a bypass log;
   Cl. 11.7.3.2 requires that a bypass not disable "alarms and manual shutdown
   facilities". Separately, "The art of SIF design is balancing safety
   integrity (low PFDavg) with availability (low STR)" and "frequent nuisance
   trips erode operators' trust in SIS performance" (silsafe).
   **This is the closest thing to a case against my instinct, and it does not
   actually oppose it -- it CONSTRAINS it.** See the disposition below.

## Internal code inventory

### A. The three defects, at file:line

**Defect 1 -- STALE `sod_date` never compared to today.**
`backend/services/kill_switch.py:741-747` is the whole armed computation:

```python
s = _state.snapshot()                                  # :741
sod = s.get("sod_nav")                                 # :742
peak = s.get("peak_nav")                               # :743
daily_baseline_missing = not (sod is not None and sod > 0)      # :745
trailing_baseline_missing = not (peak is not None and peak > 0) # :746
armed = not (daily_baseline_missing or trailing_baseline_missing)  # :747
```

`s` DOES carry `sod_date` -- `_snapshot_locked` returns it at
`kill_switch.py:442` -- but `evaluate_breach` never reads the key. So a restored
2026-07-24 anchor is `armed:true` on 2026-07-26 and `daily_loss_pct` at
`:769` computes `(sod_2days_ago - nav_today)/sod * 100` and calls it a DAILY
loss. Confirmed by `paper_trader.py:1148-1153`, which documents that step 36.9
MEASURED exactly this on the live book at 4.0% -- i.e. at the limit.

Restore path that produces the stale date: `_load_from_audit`'s `sod_snapshot`
branch, `kill_switch.py:278-293` -- `self._sod_nav` at `:279`, then `sod_date =
row.get("date")` at `:283` with a `ts`-parse fallback at `:284-292` and the
assignment at `:293`. Replay is last-row-wins for SOD (contrast `peak_update`
at `:294+`, which ratchets), so boot restores whatever date the newest
`sod_snapshot` row carries.

**Defect 2 -- `armed` is computed BEFORE the `nav_invalid` early return.**
`armed` is fixed at `:747`; the invalid-NAV early return is at `:751-764` and
carries `"armed": bool(armed)` at `:763` alongside `"any_breached": False` and
`"nav_invalid": True` at `:757`. So `armed:true` + `any_breached:false` +
`nav_invalid:true` is a reachable, self-contradictory triple: the switch claims
to be measuring while explicitly reporting it could not measure.
Reachability is not hypothetical -- `backend/api/paper_trading.py:510-516`
swallows a 5s BQ timeout into `portfolio = None` (`:515`) and then
`nav = float((portfolio or {}).get(...) or ... or 0.0)` (`:516`) manufactures
`0.0`, which is exactly the `current_nav <= 0` branch at `:751`.

**Defect 3 -- `sod_nav=0.0` latches, and the re-anchor predicate can't clear it.**
Writer: `kill_switch.py:513-527`, `self._sod_nav = float(nav)` at `:525` with
NO positivity guard. A `0.0` therefore persists as a real anchor plus a
`sod_snapshot` audit row (`:527`).
Reader: `daily_baseline_missing = not (sod is not None and sod > 0)` at `:745`
-> `True` -> `armed:false` -> `/resume` 409s at
`backend/api/paper_trading.py:598-616`.
The wedge: `backend/services/paper_trader.py:1142`

```python
if snap.get("sod_nav") is None or snap.get("sod_date") != today:
```

tests `is None`, NOT `<= 0`. With `sod_nav == 0.0` AND `sod_date == today`
both disjuncts are False, so `update_sod_nav` at `:1143` never runs and the
0.0 survives the whole UTC day. Note the asymmetry is inside ONE module pair:
the WRITER has no guard (`:525`), the READER discriminates on `> 0` (`:745`),
the RE-ANCHOR discriminates on `is None` (`paper_trader.py:1142`). Three
different notions of "absent" for one field.
The 409 text at `paper_trading.py:609-615` promises "The next paper-trading
cycle will BLOCK new orders rather than trade on unknown baselines, raise a P1,
and write a `baseline_anchor_on_lost_history` row" -- and 36.12's block at
`paper_trader.py:1195` fires on `not pre_armed and not first_ever_boot`, so the
BLOCK half of that promise IS true. What is FALSE is the re-anchor: nothing on
that path rewrites a `0.0` sod, so the disarmed state does not self-heal.

### B. Consumer enumeration -- every caller of `evaluate_breach` / `snapshot()` / `armed`

| # | Consumer | file:line | Re-anchors first? | What a semantics change breaks |
|---|----------|-----------|-------------------|-------------------------------|
| 1 | `PaperTrader.check_and_enforce_kill_switch` PRE-measure | `backend/services/paper_trader.py:1096-1101` | **No, deliberately** (36.12 measures BEFORE mutating) | `pre_armed` drives the order-block at `:1195` |
| 2 | same, BREACH decision | `backend/services/paper_trader.py:1154-1158` | **Yes** -- `update_peak` `:1133` + SOD roll `:1140-1143` run first | the only path that flattens (`:1162-1167`) |
| 3 | `GET /api/paper-trading/kill-switch` (UI badge) | `backend/api/paper_trading.py:517-521`, state at `:522`, payload `:523-542` | **No** | `armed`, `sod_date` `:529`, `baseline_provenance` `:535` are all already on the wire |
| 4 | `POST /api/paper-trading/resume` | `backend/api/paper_trading.py:580-584`; gates at `:585` and `:598` | **No** | 409 text `:599-616` |
| 5 | MCP `risk_server.kill_switch` tool | `backend/agents/mcp_servers/risk_server.py:73-92` (import `:73`, call `:80-84`, snapshot `:76`) | **No** | returns `state` + `breach` raw to the Layer-2 MAS |
| 6 | `check_auto_resume` (internal) | `backend/services/kill_switch.py:859`; `armed` gate `:873-879` | **No** | already fails safe: `.get("armed", True)` + `breach_still_active` `:860` |
| 7 | Frontend `KillSwitchPanel` | `frontend/src/components/KillSwitchPanel.tsx:25` (type), `:134-137` (`armed === false`), `:189` (`daily_baseline_missing` copy) | n/a | badge + Resume-disabled |
| 8 | Frontend `OpsStatusBar` | `frontend/src/components/OpsStatusBar.tsx:39-40` (type), `:318` (`armed === false`), `:320` (`daily_baseline_missing`) | n/a | ops strip |

So: **1 of the 6 backend consumers re-anchors before evaluating** (#2). Five do
not. `armed` is on THREE operator-facing surfaces (UI badge, resume 409, MCP
tool) plus one control path (the order block).

Both frontend consumers already discriminate with an explicit `=== false`
(`KillSwitchPanel.tsx:137`, `OpsStatusBar.tsx:318`), never `!breach.armed` --
so an ADDED third state must not be encoded as `armed: undefined`, or both UIs
silently render ACTIVE. Documented deliberately at `KillSwitchPanel.tsx:134-136`.

### C. Writers of `sod_nav` (the 0.0 surface)

Exactly ONE production writer: `paper_trader.py:1143`
(`state.update_sod_nav(nav, date=today)`), guarded by `:1142`. Everything else
is tests (`backend/tests/test_38_1_*`, `test_64_3_*`, `test_dod4_*`,
`tests/services/test_sod_daily_roll.py`). A positivity guard therefore has a
single production call site to satisfy.

### D. Existing test pins I must not break

| File | Lines | What it PINS |
|------|-------|--------------|
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | `:214-230` (`daily_baseline_missing is True`), `:246` (`st._sod_date = "2026-07-24"`), `:249-257` (`evaluate_breach(CATASTROPHIC_NAV,...)` -> `daily_baseline_missing is False`), `:270-278`, `:588`, `:622-626`, `:855-866` (key-set incl. `armed`) | **`:246` sets `_sod_date` to 2026-07-24 and then EXPECTS the daily leg to evaluate.** A naive "stale -> disarm" would turn this suite RED. Any fix must either use a fixed clock or the fixture must be updated. This is the single highest-risk pin in the repo for defect 1. |
| `backend/tests/test_phase_36_7_...` | `:858-859` | `normal = evaluate_breach(9900.0,...)` vs `invalid = evaluate_breach(0.0,...)` compared at `:866` on a key set that INCLUDES `armed` -- i.e. the current shape-parity contract between the two return shapes |
| `backend/tests/test_phase_36_12_kill_switch_trading_path_block.py` | `:646-656` | `evaluate_breach` still evaluates the surviving leg; `daily_baseline_missing is False` with peak None |
| `backend/tests/test_phase_23_2_5_kill_switch_no_false_fires.py` | `:130` / `:152` / `:174` / `:248` all set `_sod_date = "2026-05-22"` then expect the daily leg to fire | **Same stale-date hazard as above, x4.** `:242-258` specifically pins "zero sod does not div-zero" |
| `backend/tests/test_64_3_kill_switch_machine.py` | `:62-72` invalid-NAV; `:73-77` clean | the `nav_invalid` return shape |
| `backend/tests/test_book_safety_69.py` | `:67-80` | `evaluate_breach(0.0/-5.0)` -> no phantom breach (the 69.1 guard) |
| `tests/services/test_sod_daily_roll.py` | `:80` and `:100` and `:156` replicate the `snap.get("sod_nav") is None or snap.get("sod_date") != today` predicate **inline in the test** | if the predicate changes in `paper_trader.py:1142`, these three copies drift |
| `tests/verify_phase_23_2_19.py` | `:47-50` asserts the literal string `'state.update_sod_nav(nav, date=today)'` and `'snap.get("sod_date")'` are PRESENT in `paper_trader.py` | **a source-scan pin**: editing the re-anchor predicate's text can break it. Cross-check before editing `:1142-1143`. |
| `frontend/src/components/KillSwitchPanel.disarmed.test.tsx` | `:48-49` (`sod_nav:null, sod_date:null`), `:60` `armed:false`, `:71-77` `sod_date:"2026-07-26"` + `armed:true`, `:85-86` | pins ACTIVE-on-`armed:true`, DISARMED-on-`false`, ACTIVE-on-absent |

### E. The 409 text, verbatim source

`backend/api/paper_trading.py:598-616` -- gate `if not breach.get("armed", True):`
then a 409 whose body is the concatenation at `:601-615`. The
false-promise clause is `:609-615`. There is a SECOND 409 above it at
`:585-591` for a real breach (`any_breached`), which is correct and untouched
by this step.

### F. Related module facts worth knowing before writing the contract

- `kill_switch.py:748-749` calls `_log_disarmed_once(sod, peak)` -- a
  **process-lifetime one-shot** (`_disarmed_logged` global, `:802-805`). Any new
  disarm reason (stale, nav-invalid) shares that one-shot, so the FIRST reason
  encountered is the only one ever logged in a process. If stale-anchor becomes
  a disarm reason, the log line at `:806-810` (which prints only `sod_nav` /
  `peak_nav`) will not say WHY.
- `evaluate_breach`'s docstring at `:716-733` states three explicit
  non-goals -- (1) never set `any_breached=True` on a missing baseline,
  (2) never wholesale early-return, per-leg markers only, (3) never
  discriminate on VALUE, only on presence. A stale-date fix must be written as
  a PRESENCE test on the anchor's validity-for-today, not as a value test, to
  stay inside (3).
- `check_auto_resume:873` already fails OPEN on a missing `armed` key
  (`.get("armed", True)`) -- consistent with `paper_trading.py:598`. Any new
  key must follow the same optional-key discipline or old dicts wedge.

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/kill_switch.py` | 932 | breach eval + state + audit replay | 3 defects, all in `evaluate_breach`/`update_sod_nav` |
| `backend/services/paper_trader.py` | 1433 | the only re-anchoring consumer | `:1142` predicate is the wedge |
| `backend/api/paper_trading.py` | -- | UI badge + resume | 2 non-re-anchoring consumers |
| `backend/agents/mcp_servers/risk_server.py` | -- | MAS-facing tool | non-re-anchoring consumer |
| `frontend/src/components/KillSwitchPanel.tsx` | -- | badge | `=== false` discipline |
| `frontend/src/components/OpsStatusBar.tsx` | -- | ops strip | `=== false` discipline |
| `backend/tests/test_phase_36_7_*` + `test_phase_23_2_5_*` | -- | existing guards | **5 fixtures set a PAST `_sod_date` and expect ARMED** |
| `tests/verify_phase_23_2_19.py` | -- | source-scan verifier | pins the literal `:1142-1143` strings |

## Consensus vs debate (external)

**Consensus (all 11 sources agree):**
- A safety control that cannot verify its input must not report the state it
  would report if the input were verified-good. Fail-closed / deny-by-default
  is unanimous (RuntimeAI, IEC 61508 DU framing, K8s readiness -> unready).
- Absence must be REPRESENTED, not inferred (sentinel-object, three-state
  health, Fowler "reveal details of their state").
- A session-relative threshold is re-derived per session (SEC MWCB daily
  recalculation; LULD rolling 5-min band).

**Genuine debate -- and the ONE place my instinct needed adjusting:**
The functional-safety literature does NOT endorse "disarm and stop measuring"
as a free action. Disarming a safety function = a BYPASS, and IEC 61511 permits
continued operation under bypass **only with compensating measures, operating
limits (incl. duration), authorization, indication, and a bypass log**. The
spurious-trip literature adds that over-conservatism has its own cost (eroded
operator trust; a SIS that trips on nothing is not free).

**Disposition: this does not overturn the plan, it adds three requirements.**
The caller asked specifically whether a monitor that disarms on stale input is
WORSE than one that keeps measuring against a stale anchor. The answer from the
literature is **no, keeping the stale anchor is worse** -- because the stale
anchor does not merely fail to detect, it ACTIVELY MIS-REPORTS in the trip
direction (finding 4: a 2-day move read as a 1-day loss), which is a spurious
trip *and* a coverage loss at the same time. That is the worst quadrant. BUT
the literature is equally clear that "not armed" is a bypass state, so it must
be (a) paired with a compensating measure, (b) time-bounded / self-healing,
(c) visibly indicated and logged. pyfinagent already HAS (a): phase-36.12's
per-cycle order block at `paper_trader.py:1195-1209` is exactly a compensating
measure, and it is `BLOCK, NOT PAUSE` (`:1188-1194`), which preserves the
"manual shutdown facilities are not disabled" requirement of Cl. 11.7.3.2.
**Defect 3 is a violation of (b): a bypass with no exit.** That is the finding
the standard sharpens most.

## Pitfalls (from literature)

1. **Don't let the new disarm reason ride the existing one-shot log.**
   Fowler requires state-CHANGE logging; `_log_disarmed_once`
   (`kill_switch.py:802-810`) fires once per process for ANY reason and prints
   only `sod_nav`/`peak_nav`. Add stale/nav-invalid reasons and the first one
   wins forever, and the message names neither.
2. **Don't convert a bypass into a wedge.** IEC 61511 Cl. 16.2.3 wants
   duration limits on a bypass. Any new disarm condition must have a defined
   exit path, checked against the same predicate that CREATED it. Defect 3 is
   the live proof of what happens otherwise.
3. **Don't over-trip.** silsafe: a control with terrible availability "fails
   its mission". Concretely: do NOT make a stale anchor set `any_breached=True`
   -- `paper_trader.py:1162-1167` would `flatten_all()` a healthy book on a
   restart. `evaluate_breach`'s own docstring already forbids this at `:718-722`
   and it is right.
4. **Don't encode a new state as an absent key.** Both UIs use `=== false`
   (`KillSwitchPanel.tsx:137`, `OpsStatusBar.tsx:318`) and both backend gates
   use `.get("armed", True)` (`paper_trading.py:598`, `kill_switch.py:873`).
   An `armed: undefined` third state renders ACTIVE and resumes freely.
5. **Don't discriminate on value where the rule is presence.** Source 9 +
   `evaluate_breach` docstring `:729-733`. "The anchor is not for today" is a
   PRESENCE test on *today's* anchor, not a value test on the NAV -- frame it
   that way or it collides with the module's own stated non-goal 3.
6. **Beware the semipredicate at the writer.** The fix belongs at
   `update_sod_nav` (`:513-527`) as well as at the readers; otherwise the next
   reader invents a fourth definition of absent.

## Application to pyfinagent

| External principle | Source | pyfinagent anchor |
|---|---|---|
| DU faults dominate; the on-line diagnostic IS the coverage | gt-engineering (IEC 61508-4:2010) | `armed` at `kill_switch.py:747` is the only diagnostic; 5 of 6 consumers trust it without re-anchoring |
| Fail-closed, never "log and pass" | RuntimeAI 2026-05-12 | `kill_switch.py:751-764` -- `nav_invalid:true` + `armed:true` in one dict |
| Session threshold recalculated daily from the prior session's close | investor.gov (SEC MWCB) | `sod_date` is IN the snapshot (`:442`) and IGNORED by `evaluate_breach` (`:741-747`) |
| Staleness is a confident wrong number | tacnode 2026-02-12 | `daily_loss_pct` at `:769` over a 2-day-old `sod`; measured at 4.0% on the live book per `paper_trader.py:1150-1152` |
| Sentinel by identity, not value | python-patterns.guide | `update_sod_nav:525` writes `float(nav)` unguarded; `:745` reads `> 0`; `paper_trader.py:1142` reads `is None` |
| Bypass needs compensating measures + duration limit + log | IEC 61511 Cl. 16.2.3/16.2.4/16.2.6 | compensating measure EXISTS (`paper_trader.py:1195-1209` + `record_lost_history_anchor`); duration limit does NOT (defect 3) |
| Log every state change with details | Fowler | `_log_disarmed_once` `:792-810` -- one-shot, reason-blind |
| Three-state, and UNKNOWN is under-specified everywhere | web-alert.io + k8s + GCP LB | `armed: true|false|undefined` today; `undefined` already means "old backend, treat as ACTIVE" -- so a genuine UNKNOWN needs a NAMED reason field, not a third `armed` value |

## Recommendations per finding

### Finding 1 -- stale `sod_date`: **FIX, and the literature is unambiguous.**
Read `sod_date` from the snapshot that `evaluate_breach` already takes at
`:741` and treat "the daily anchor is not today's" as
`daily_baseline_missing = True` (i.e. the daily leg is unevaluable), NOT as a
breach. Rationale: SEC MWCB recalculates daily from the prior close
(investor.gov); a daily-loss limit whose denominator is 2 days old is not the
control that was specified. Keeping the stale anchor is the WORST option --
it both loses coverage and biases toward a spurious flatten.

Three constraints the research imposes:
- **Per leg, not wholesale.** Only the DAILY leg is session-anchored. `peak_nav`
  is a monotonic high-water mark with no session semantics -- it must NOT be
  disarmed by a stale `sod_date`. This preserves `evaluate_breach`'s stated
  non-goal 2 (`:723-728`) and keeps the trailing leg firing.
- **Do not set `any_breached`.** Non-goal 1 (`:718-722`); a stale anchor on the
  autonomous path would `flatten_all` at `paper_trader.py:1164`.
- **Carry a REASON.** `armed:false` alone is already ambiguous across 3
  operator surfaces. Add a named reason (e.g. `daily_baseline_stale: true` +
  the anchor's date) so the UI badge, the 409 and the MCP tool can say WHICH
  bypass is active -- IEC 61511 Cl. 16.2.6 "authorization and indication".

**Blast-radius warning (measured, not assumed):** 5 existing test fixtures set
`_sod_date` to a PAST date and then expect the daily leg to evaluate --
`test_phase_36_7_...:246` (`2026-07-24`) and `test_phase_23_2_5_...:130/:152/
:174/:248` (`2026-05-22`). A date-aware disarm turns those RED. They are
GENUINE guards (they pin real-breach detection), so the correct move is to
update the fixtures to today's date (or inject a clock), NOT to weaken the
fix. Flag this in the contract: it is the single biggest regression risk here.
Also note `paper_trader.py:1142` already implements the correct predicate
(`sod_date != today`) -- the fix makes `evaluate_breach` agree with the roll
logic that already exists one module over, rather than inventing a new rule.

### Finding 2 -- `nav_invalid` returning `armed:true`: **FIX. Strongest external support of the three.**
"Not 'best-effort.' Not 'log and pass.' Closed." (RuntimeAI 2026-05-12).
An unmeasurable NAV means the diagnostic did not run; reporting `armed:true`
claims coverage that was not exercised (IEC 61508 DU). Make the invalid-NAV
return `armed: false` and keep `nav_invalid: true` as the reason. Note this is
NOT a change to `any_breached` -- that stays `False` (69.1's guard against a
phantom 100% breach on a BQ timeout is correct and must survive).

Two constraints:
- `backend/tests/test_phase_36_7_...:855-866` compares the key SET of the
  normal vs invalid returns and that set includes `armed` -- a VALUE change is
  compatible with that pin; adding/removing keys is not. Verify.
- `test_64_3_kill_switch_machine.py:62-72` and `test_book_safety_69.py:67-80`
  pin the invalid-NAV shape. Read them before editing.
- Consumer effect, enumerated: the UI would render DISARMED during a BQ
  timeout (`paper_trading.py:515-516` is the manufacturer of the 0.0), and
  `/resume` would 409. Both are the CORRECT conservative outcome, but they are
  BEHAVIOUR CHANGES on operator-facing surfaces during a transient -- the
  contract must say so explicitly, and the 409 text must distinguish
  "baselines unrecoverable" from "NAV temporarily unreadable, retry", or the
  operator is told to go do archive restoration for a 5-second BQ blip.
  That distinction is the practical form of the spurious-trip caution
  (silsafe): a transient must not read like a permanent fault.

### Finding 3 -- `sod_nav = 0.0` wedging resume: **FIX, both ends. Highest-confidence item.**
This is the sentinel/semipredicate bug (python-patterns.guide) AND the
"bypass with no duration limit" violation (IEC 61511 Cl. 16.2.3). Two edits,
and both are needed:
1. **Writer guard** at `kill_switch.py:513-527`: refuse to latch a
   non-positive / non-finite `nav` as a baseline. Absent must be `None`, never
   `0.0` -- "no possibility of the return value being used accidentally"
   (source 9). Decide explicitly whether a rejected write is a no-op or writes
   an audit row; prefer an audit row (Cl. 16.2.7 "bypass log") since silence
   here is what made the original bug invisible.
2. **Re-anchor predicate** at `paper_trader.py:1142`: `is None` must become a
   validity test that also covers `<= 0` / non-finite, so a latched `0.0`
   self-heals on the next cycle. Without this the 409 text at
   `paper_trading.py:609-615` stays a FALSE promise.

Guard-rails found in the audit:
- `tests/verify_phase_23_2_19.py:47-50` asserts the LITERAL STRING
  `'state.update_sod_nav(nav, date=today)'` and `'snap.get("sod_date")'` exist
  in `paper_trader.py`. Editing `:1142-1143` can break a source-scan verifier
  that has nothing to do with this step. Check it in the same change.
- `tests/services/test_sod_daily_roll.py:80`, `:100`, `:156` each RE-IMPLEMENT
  the `:1142` predicate inline. If the predicate changes, those three copies
  drift and stop testing the shipped logic. Either update all three or (better)
  export the predicate as a named helper and have the tests import it -- that
  also removes the "sentinel rule defined in three places" root cause.
- `test_phase_23_2_5_...:242-258` pins "zero sod does not div-zero". A writer
  guard makes `sod == 0` unreachable via the public writer but NOT via direct
  `_sod_nav` assignment, which is how that test builds it. Keep the reader's
  `> 0` check at `:745`; belt AND braces.

### Cross-cutting recommendation
All three fixes converge on ONE structural change worth naming in the contract:
**`armed` is currently a boolean that answers three different questions**
(is a baseline present? is it current? was it measurable this call?). The
literature's answer is a reason-carrying state, not a richer boolean --
Fowler ("reveal details of their state"), IEC 61511 Cl. 16.2.6 (indication),
three-state health. Keep `armed` as the boolean the 4 existing gates already
read (`paper_trading.py:598`, `kill_switch.py:873`, `KillSwitchPanel.tsx:137`,
`OpsStatusBar.tsx:318`) so nothing silently regresses, and ADD a
`disarm_reasons` list / named marker keys alongside it. That satisfies the
optional-key discipline already established at `:597` and `:872`, and it fixes
the `_log_disarmed_once` reason-blindness in the same stroke.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **11**
- [x] 10+ unique URLs total (incl. snippet-only) -- **42**
- [x] Recency scan (last 2 years) performed + reported -- Q2/Q6/Q8; 3 findings
- [x] Full pages read (not abstracts) for the read-in-full set -- 11/11 fetched;
      1 attempted fetch (CME) TIMED OUT and is recorded as snippet-only, not
      counted
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module -- `kill_switch.py`,
      `paper_trader.py`, `paper_trading.py`, `risk_server.py`, 2 frontend
      components, 8 test modules; consumer list is an ENUMERATION from
      repo-wide grep of `evaluate_breach` / `armed` / `sod_date` / `sod_nav` /
      `daily_baseline_missing`, not a sample
- [x] Contradictions / consensus noted -- the IEC 61511 bypass/spurious-trip
      counterweight is recorded with its clause numbers and dispositioned
      rather than dismissed; where a source did NOT support the claim I fetched
      it for (silsafe, risknowlogy), I said so
- [x] All claims cited per-claim
- [ ] **Gap, disclosed:** CME Globex Kill Switch (the closest real-world
      trading analogue for the re-enable path) timed out and was not read.
      The SEC MWCB source covers the daily-recalculation point; the
      explicit-re-enable point rests on IEC 61511 Cl. 16.2.6 instead.
- [ ] **Gap, disclosed:** no peer-reviewed source in the read-in-full set. The
      two peer-reviewed candidates (ScienceDirect proof-test-interval,
      IChemE Hazards-30 STR formulae) were paywalled / binary PDF. The
      normative content (IEC 61508-4:2010 and IEC 61511 clause text) is quoted
      via two independent industry sources that agree with each other.
- [ ] **Not verified live:** I did NOT reproduce the three defects against the
      running backend (constraint: GET-only, and reproducing defect 2 requires
      inducing a BQ timeout). All three are read off the source with file:line;
      defect 1's 4.0% figure is quoted from `paper_trader.py:1150-1152`, which
      attributes the measurement to step 36.9 itself -- it is a CODE COMMENT,
      not a measurement I made. Treat it as a lead to re-measure in GENERATE,
      not as evidence.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 11,
  "snippet_only_sources": 31,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 8,
    "dry": false
  },
  "summary": "All three defects are one fault class: `armed` asserts diagnostic coverage that was never exercised -- IEC 61508's dangerous-UNDETECTED quadrant. External consensus backs fail-loud-and-conservative on all three. Strongest supports: RuntimeAI 2026-05-12 fail-closed ('Not log and pass. Closed.') for the nav_invalid case; SEC market-wide circuit breakers recalculated DAILY from the prior close for the stale-sod_date case; the sentinel-object/semipredicate pattern for the 0.0 case. The adversarial check (IEC 61511 Cl. 16.2.3/16.2.4 bypass rules + spurious-trip-rate literature) does NOT oppose the plan but constrains it: disarming is a BYPASS, so it needs compensating measures (36.12's order block already is one), a duration limit (defect 3 is a bypass with no exit -- the strongest item), indication and a log. Keeping a stale anchor is the worst option: it loses coverage AND biases toward a spurious flatten. Internal: 5 of 6 backend consumers of evaluate_breach do NOT re-anchor; the sole re-anchor predicate (paper_trader.py:1142) tests `is None` while the reader tests `> 0` and the writer (kill_switch.py:525) guards nothing -- three definitions of absent for one field. Biggest regression risk: 5 existing fixtures set a PAST _sod_date and expect ARMED.",
  "brief_path": "handoff/current/research_brief_36.9.md",
  "gate_passed": true
}
```
