# Research Brief -- Step 82.58 (P1): cost-budget hard-block alert has never fired

**Tier:** moderate  |  **Audit class:** TRUE (loop-until-dry, K=2)
**Researcher:** Layer-3 Researcher (Workflow rail)  |  **Started:** 2026-08-06
**Status:** IN PROGRESS -- write-first skeleton, appended incrementally.

## Objective

Step 82.58 alleges: `backend/services/observability/spend.py::_record_degradation`
calls `raise_cron_alert_sync(..., detail=...)` while the signature in
`backend/services/observability/alerting.py` uses `details=`. TypeError is
swallowed by a surrounding `except Exception -> logger.debug`. Compounded by
`severity='P2'` + empty `slack_webhook_url` so a kwarg-only fix still yields a
dropped alert. Seven questions Q1-Q7 (see caller prompt).

---

> NOTE ON LENGTH: the `moderate` tier targets <=700 words. This brief exceeds
> that because `coverage.audit_class = true` expanded the internal half into a
> six-round loop-until-dry sweep. Depth of ANALYSIS is moderate-tier; the extra
> length is measurement output (sweep tables, empirical runs), not padding.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.python.org/3/library/unittest.mock.html | 2026-08-06 | official doc (tier 2) | WebFetch | "Methods and functions being mocked will have their arguments checked and will raise a TypeError if they are called with the wrong signature" (autospec). And explicitly: a plain `Mock(spec=f)` does NOT check signatures -- `mock(1,2,c=3)` "Works fine - signature not checked". Without autospec "the test would pass despite the bug". |
| 2 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-06 | authoritative (tier 3) | WebFetch | "Every page should be actionable"; "I can only react with a sense of urgency a few times a day before I become fatigued"; rule test = "an otherwise undetected condition that is urgent, actionable, and actively or imminently user-visible"; "If a page merely merits a robotic response, it shouldn't be a page." Also: signals "not exposed in any prebaked dashboard nor used by any alert, are candidates for removal." |
| 3 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-06 | authoritative (tier 3) | WebFetch | Alert quality = precision / recall / detection time / reset time. "Recall is 100% if every significant event results in an alert." 82.58 is a **recall-0** alert: it can never fire, so recall for the spend-guard-degraded event is exactly zero. |
| 4 | https://mypy.readthedocs.io/en/stable/error_code_list.html | 2026-08-06 | official doc (tier 2) | WebFetch | `call-arg`: "Mypy expects that the number and names of arguments match the called function." This defect class is statically detectable; the repo does not run mypy on it. |
| 5 | https://pylint.readthedocs.io/en/latest/user_guide/messages/error/unexpected-keyword-arg.html | 2026-08-06 | official doc (tier 2) | curl + tag-strip (WebFetch returned nav-only -- JS-rendered; same technique as the gcloud-docs precedent) | E1123 `unexpected-keyword-arg`: "Used when a function call passes a keyword argument that doesn't correspond to one of the function's parameter names." Created by the `typecheck` checker. Exactly this bug, catchable by a linter pylint already ships. |
| 6 | https://docs.astral.sh/ruff/rules/blind-except/ | 2026-08-06 | official doc (tier 2) | curl + tag-strip | BLE001: "Overly broad except clauses can lead to unexpected behavior." Notably, exceptions "logged with exc_info enabled will not be flagged" -- `spend.py:127` uses `logger.debug(... %r, alert_exc)` with NO exc_info, so it would be flagged. |
| 7 | https://docs.astral.sh/ruff/rules/try-except-pass/ | 2026-08-06 | official doc (tier 2) | curl + tag-strip | S110: "Suppressing exceptions may hide errors that could otherwise reveal unexpected behavior"; CWE-703. `spend.py:126-127` is the `logger.debug` variant -- technically not `pass`, but at DEBUG it is functionally equivalent (invisible at the INFO default). |
| 8 | https://engineering.fb.com/2025/05/15/developer-tools/introducing-pyrefly-a-new-type-checker-and-ide-experience-for-python/ | 2026-08-06 | vendor eng blog (tier 2) | WebFetch | "We want users to benefit from types even if they haven't annotated their code -- so automatically infer types"; 1.8M LOC/sec; goal is shifting "checks that used to happen later on CI to happening on every single keystroke." Relevant because pyfinagent's call sites are largely unannotated at the call, which is where mypy defaults are weakest. |
| 9 | https://seifrajhi.github.io/blog/securing-monitoring-stack-dead-man-switch/ | 2026-08-06 | community (tier 5) | WebFetch | The Watchdog / Dead-Man's-Switch pattern: "This is an alert meant to ensure that the entire alerting pipeline is functional. This alert is always firing, therefore it should always be firing in Alertmanager." Frames the whole 82.58 class: an alert that never fires is indistinguishable from a healthy system. |

## Identified but snippet-only (context; does NOT count toward gate)

44 further URLs were surfaced and evaluated but not fetched in full. Grouped:

| Group | URLs | Why not fetched in full |
|---|---|---|
| autospec prior art | github.com/python/cpython/issues/61387; github.com/python/cpython/issues/145754; bugs.python.org/issue17185; runebook.dev/.../unittest.mock.create_autospec; medium.com/@alcaptar/unit-testing-in-python-...; python-testing-debugging.com/.../autospec-strict-mocking/ | superseded by source 1 (the canonical doc) |
| SRE alerting | sre.google/prodcast/transcripts/sre-prodcast-01-03/; sre.google/workbook/monitoring/; sre.google/workbook/on-call/; incident.io/blog/sre-alerting-best-practices; developers.soundcloud.com/blog/alerting-on-slos/; dev.to/meena_nukala_1154d49b984d/-from-400-alertsnight-to-8-... | duplicative of sources 2 + 3 |
| kwarg / typing prior art | rednafi.com/python/annotate-args-and-kwargs/; askpython.com/python/examples/unexpected-keyword-argument-typeerror; typing.python.org/en/latest/spec/directives.html; cs.toronto.edu/~david/pyta/checkers/index.html; pypi.org/project/argument-checks; github.com/jboy/argcheck-python3 | tutorial-tier; sources 4 + 5 are authoritative |
| 2026 type-checker landscape | codegym.cc/.../python-type-checkers-mypy-pyright-pyrefly; danilchenko.dev/posts/pyrefly-vs-mypy-vs-ty/; pydevtools.com/handbook/explanation/how-do-mypy-pyright-and-ty-compare/; pydevtools.com/handbook/reference/pyrefly/; pydevtools.com/blog/pyrefly-1-0-is-the-obvious-mypy-upgrade/; pydevtools.com/handbook/explanation/ty-vs-pyrefly/; dasroot.net/posts/2026/05/pyrefly-v1-0-...; pkgpulse.com/guides/pyrefly-vs-ty-python-type-checkers-2026; github.com/facebook/pyrefly; pyrefly.org/; pypi.org/project/pyrefly/; alternativeto.net/software/pyre-check | recency evidence; source 8 is the primary |
| dead-alert / watchdog | about.gitlab.com/blog/automated-detection-testing-framework/; oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view; streamkap.com/blog/cdc-failures-silent-timeout-detection; deadmancheck.io/airflow-dag-monitoring; medium.com/@manik.ruet08/alert-fatigue-in-dataops-...; team400.ai/blog/2026-04-21-data-factory-monitoring-alerting-best-practices | source 9 covers the pattern |
| pytest network guards | medium.com/@Modexa/7-pytest-fixture-param-tricks-...; blog.jerrycodes.com/no-http-requests/; tomwojcik.com/posts/2021-08-18/globally-block-internet-connection-in-django/; slack-sansio.readthedocs.io/en/stable/testing.html; monkey.work/blog/2025-12-23-block-network-access-in-tests/; blog.pecar.me/disable-network-requets-when-running-pytest/; github.com/miketheman/pytest-socket; pypi.org/project/pytest-socket/0.5.0/ | directly relevant to the HAZARD section; `pytest-socket` (`--disable-socket`) is the named remedy but is NOT a project dep, so an autouse fixture is preferred over adding a dependency |

**Search-query composition (three variants, per `.claude/rules/research-gate.md`):**
- *Year-less canonical*: "unittest.mock autospec create_autospec signature mismatch mocks that pass against broken code"; "Google SRE book alerting philosophy every page must be actionable severity paging vs ticket".
- *Last-2-year*: ""alert that never fires" monitoring dead alert detection testing your alerting pipeline **2025**"; "pytest fixture block network calls in tests prevent accidental live Slack API POST socket guard".
- *Current-year frontier*: "Python type checker catching wrong keyword argument call-arg **2026** ty pyrefly adoption".

## Recency scan (2024-2026)

Performed. **Two new findings in the window; neither supersedes the canonical
sources, both strengthen the prevention recommendation.**

1. **Pyrefly 1.0 shipped 2026-05-12** and is now Meta's default checker on
   Instagram's ~20M-line Python codebase, plus adopted by PyTorch, NumPy and
   JAX; it "checks aggressively, catching errors in unannotated code that mypy
   skips by default" and passes ~96% of the typing conformance suite vs ty's
   ~76% (search 2026-08-06;
   https://engineering.fb.com/2025/05/15/developer-tools/introducing-pyrefly-a-new-type-checker-and-ide-experience-for-python/,
   https://pydevtools.com/blog/pyrefly-1-0-is-the-obvious-mypy-upgrade/).
   Relevance: the class-level prevention for 82.58 (a `call-arg` checker in CI)
   is materially cheaper and more effective in 2026 than it was when the mypy
   guidance was written. Worth a SEPARATE step, not this one.
2. **Dead-Man's-Switch / Watchdog alerting** remains the canonical answer to
   "how do you know an alert still works", with 2026-dated writeups
   (oneuptime 2026-02-06, team400 2026-04-21) and GitLab's 2026 detection-testing
   framework that "simulat[es] real malicious behavior to validate that
   detections fire end-to-end." Relevance: 82.58's real lesson is that the repo
   has no end-to-end alert-delivery test at all; the new test file IS that
   detection test for one alert.

No 2024-2026 source contradicts the older canonical guidance (SRE book 2016,
`unittest.mock` docs). The `_CRITICAL_SEVERITIES` bypass design at
`alerting.py:42-54` is itself consistent with the SRE page/ticket split.

## Internal code inventory

| File | Lines read | Role | Status |
|---|---|---|---|
| `backend/services/observability/spend.py` | 1-259 (full) | the defect site | **BROKEN** at `:115-125` (`detail=`), `:118` (`severity="P2"`) |
| `backend/services/observability/alerting.py` | 1-302 (full) | the emitter + deduper + delivery | correct; `:201-202` deduper gate, `:209` webhook read, `:217-218` bot-token fallback, `:253-259` signature |
| `backend/services/observability/__init__.py` | 1-80 | public surface | OK -- re-exports `fetch_spend`/`spend_guard_status` (`:50-56`, `__all__` `:68`); note it exports `raise_cron_alert` but NOT `raise_cron_alert_sync` |
| `backend/agents/llm_client.py` | 400-479 | the hard-block consumer | correct; `:452` `tripped = daily >= daily_cap or ...`; caps resolve to 25.0/300.0 |
| `backend/config/settings.py` | targeted (`:123,:151-153,:392-394,:593-595,:627-628`) | config source of truth | `get_settings` is `@lru_cache()` -- fixture-relevant |
| `backend/.env` | presence probe only | live config | webhook EMPTY, bot token SET |
| `backend/tests/test_phase_75_5_1_spend_metric.py` | 120-160, 290-343 | existing coverage | **BLIND** -- `:294-301` asserts `alerted is True` (set at `spend.py:102`, before the try) |
| `backend/tests/test_phase_75_llm_rail.py` | 580-600 | existing coverage | BLIND -- same counter-only assertions |
| `backend/tests/test_phase_82_10_freshness_paging.py` | 1-494 (full) | **the pattern to copy** | `:40` patch target, `:341-344` P1 doctrine, `:417-430` negative control w/ precondition, `:449-459` anti-vacuity pin |
| `backend/tests/test_phase_82_54_cost_budget_columns.py` | 325-369 | the guard already watching this defect | passes today; must not be orphaned by an early status flip |
| `backend/tests/conftest.py` | grep | test setup | **no network/slack guard** -> the fix would send live Slack messages |
| `backend/slack_bot/assistant_lifecycle.py` | 24-50, 88-102, 176-192 | 2nd instance of the class | **BROKEN** at `:181` + `:188`; live via `app.py:33` |
| `backend/slack_bot/app.py` | `:18`, `:33` | wires the above | confirms the 2nd instance is reachable |
| `backend/services/drawdown_alarm.py` | 140-162 | non-literal severity site | `:154` passes `severity=severity` from `:145` -- undecidable statically |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 1-97 | Slack watcher | delegates to `fetch_spend` at `:92-94`; has its OWN `alert_fn` path, does not surface degradation |
| `backend/api/cost_budget_api.py` | grep (`:126-197`) | operator endpoint | `/status` + `/today`; `_alert_llm_tokens_failed` at `:126` is P1 and correct |
| `tests/autoresearch/test_slot_usage_wiring.py` | 52-66 + pytest run | 3rd instance of the class | **9 tests failing** (pre-existing, out of scope, queue) |
| `backend/intel/novelty_client.py` | 48-90 | sweep false positive | OK -- `client.embed(...)` is the voyageai SDK, not the module fn |
| `.claude/masterplan.json` | step 82.58 verbatim | the contract input | 4 immutable criteria read |
| repo-wide AST sweep | 950 `.py` files parsed | class derivation | see Q4 |

## Findings (Q1-Q7)

### Q1 -- CONFIRMED. Exact anchors, re-derived 2026-08-06.

- **Malformed call:** `backend/services/observability/spend.py:115-125`.
  Call opens at `:115`, the bad kwarg is `detail=(` at `:120`, closes `:125`.
- **Signature:** `backend/services/observability/alerting.py:253-259` --
  `def raise_cron_alert_sync(source, error_type, severity, title, details) -> bool`.
  The async twin `raise_cron_alert` at `alerting.py:179-185` has the same
  `details` name. Both are positional-or-keyword; there is no `**kwargs`
  catch-all, so `detail=` is a hard `TypeError` at CALL time, before any body
  executes.
- **Swallowing `try`:** opens `spend.py:112` (`try:  # fail-open: alerting must
  never break the money path`), the call is `:115-125`, and the handler is
  `spend.py:126-127`:
  `except Exception as alert_exc:  # pragma: no cover -- fail-open` /
  `logger.debug("observability.spend: alert skipped: %r", alert_exc)`.
  So the call site IS inside the swallowing try. Severity of the swallow is
  DEBUG, i.e. invisible at the default INFO level.
- Structural proof (not eyeballing): `inspect.signature(...).bind()` over the
  AST-extracted kwargs raises
  `TypeError: missing a required argument: 'details'` for this site alone
  (sweep output, Q4 below).
- Note `# pragma: no cover` on `:126` -- the swallow is explicitly excluded from
  coverage, so no coverage report could ever have flagged this line as hot.

### Q4 -- REFUTED (count). The step says "15 audited, 1 malformed". The real
denominator is **33**, not 15. The numerator (1) is correct.

Structural sweep (`ast` walk over 950 repo `.py` files, excluding
`.venv`, `.venv.py313.bak`, `node_modules`, `frontend`, `.next`, `handoff`,
`.claude`, `docs`, `__pycache__`), each call's kwargs bound against
`inspect.signature` of the LIVE function objects:

| Bucket | Count |
|---|---|
| Production call sites | **27** |
| Test call sites (`tests/services/test_cycle_failure_alerts.py`) | 5 |
| Internal delegation (`alerting.py:268`, sync->async) | 1 |
| **TOTAL** | **33** |
| **Signature mismatches** | **1** (`spend.py:115`) |
| Aliased imports (`import ... as X`) that could hide a call site | **0** |

So the "1 malformed" claim survives independent derivation; the "15" does not.
A contract that re-states 15 would be asserting a set it did not measure.

**Second, larger finding the step does not mention -- the non-deliverable
severity class is NOT a single site.** With `slack_webhook_url` empty (verified
below), any severity outside `_CRITICAL_SEVERITIES` (`alerting.py:54` =
`{"P0","P1","critical","CRITICAL"}`) is logged and dropped at
`alerting.py:219-224`. **12 of 27 production call sites** carry such a severity:

| File:line | severity |
|---|---|
| backend/econ_calendar/sources/finnhub_earnings.py:98 | P2 |
| backend/meta_evolution/cron.py:159 | P3 |
| backend/news/sources/alpaca.py:98 | P2 |
| backend/news/sources/alpaca.py:111 | P2 |
| backend/news/sources/benzinga.py:91 | P2 |
| backend/news/sources/benzinga.py:103 | P2 |
| backend/news/sources/finnhub.py:107 | P2 |
| backend/news/sources/finnhub.py:119 | P2 |
| backend/services/autonomous_loop.py:1001 | P2 |
| backend/services/autonomous_loop.py:1820 | P3 |
| backend/services/kill_switch.py:1029 | P2 |
| backend/services/observability/spend.py:115 | P2 (this step) |

These are *deliberate* non-page severities for most of the news/econ sources
(a dead news feed is not a page), so this is NOT 12 more bugs -- but it IS the
denominator criterion (d) forces the contract to look at. Recommend: fix
spend.py, and QUEUE a triage of `kill_switch.py:1029` (a kill-switch alert that
cannot reach the operator is the same class as this defect) rather than
silently widening scope here.

**One production site has a NON-LITERAL severity:**
`backend/services/drawdown_alarm.py:154` passes `severity=severity` from the
loop variable at `:145` (`for tier_name, dd_pct, severity in breached:`), so its
deliverability is data-dependent and cannot be settled by AST alone.

### Q7 (part 1) -- provenance. `detail=` was introduced 2026-07-23 in commit
`3a7942cf` *"fix: phase-75.5 LLM rail + root-cause remediation of the
unmeasured-scope-claim defect"* (`git blame -L 110,128`), and has NEVER been
edited since (`git log -S 'detail=('` returns exactly that one commit). The
whole `_record_degradation` block is from that single commit. The irony is
load-bearing for the contract: the commit that remediated "unmeasured scope
claims" shipped an unmeasured claim.

### Q6 -- CONFIRMED (conclusion) but the STATED CAPS ARE WRONG.

`backend/agents/llm_client.py:452` is verbatim
`tripped = daily >= daily_cap or monthly >= monthly_cap`, with
`daily = float(daily_usd or 0.0)` at `:450` and `monthly` at `:451`.
So `(0.0, 0.0)` gives `0.0 >= cap` -> False for any positive cap: **the
hard-block provably cannot trip while the fetch is degraded. Confirmed.**

But the caller's "caps 5.0/50.0" is **REFUTED**. `llm_client.py:437-438` reads
`float(getattr(settings, "cost_budget_daily_usd", 5.0))` /
`(..., "cost_budget_monthly_usd", 50.0)`. Those literals are only the
`getattr` FALLBACKS, and they are unreachable: the fields exist on Settings at
`backend/config/settings.py:392` (`cost_budget_daily_usd = 25.0`) and `:393`
(`cost_budget_monthly_usd = 300.0`), and `COST_BUDGET_DAILY_USD` /
`COST_BUDGET_MONTHLY_USD` are ABSENT from `backend/.env`. **Live caps are
$25/day and $300/month.** A test that pins 5.0/50.0 would be asserting against
a branch that never runs in production.

Also `cost_budget_use_llm_spend_enabled` (settings.py:394, default False) is
absent from `.env`, so the live breaker reads `fetch_spend()` (BigQuery bytes),
not `fetch_llm_spend()` -- `llm_client.py:439-442`. Both paths funnel through
the SAME `_record_degradation` seam (`spend.py:167` and `spend.py:249`), so the
fix covers both regardless of the flag.

### Environment facts pinned 2026-08-06 (presence-only probe of backend/.env)

| Key | State |
|---|---|
| `SLACK_WEBHOOK_URL` | **ABSENT/EMPTY** -> webhook branch dead, confirmed |
| `SLACK_BOT_TOKEN` | SET (len 59) -> bot-token fallback is VIABLE |
| `SLACK_CHANNEL_ID` | SET (len 11) |
| `ALERT_CONSECUTIVE_FAILURE_THRESHOLD` | absent -> default **3** (settings.py:151) |
| `ALERT_DEBOUNCE_MINUTES` | absent -> default 5 (settings.py:152) |
| `ALERT_REPEAT_HOURS` | absent -> default 1 (settings.py:153) |
| `COST_BUDGET_DAILY_USD` | absent -> default **25.0** (settings.py:392) |
| `COST_BUDGET_USE_LLM_SPEND_ENABLED` | absent -> default False (settings.py:394) |

In-process confirmation (loading real Settings the way pytest does):
`slack_webhook_url == ''`, `slack_bot_token` len 59 prefix `xoxb-`,
`slack_channel_id == 'C0ANTGNNK8D'`, caps 25.0/300.0, flag False.
`get_settings` is `@lru_cache()` (`settings.py:627-628`) -- **a fixture that
sets env vars will NOT take effect unless it calls `get_settings.cache_clear()`
or patches the resolved object.**

### Q2 -- CONFIRMED, and it is the DOMINANT blocker (empirically driven).

The claim is exactly right, with the anchors re-derived:

- `raise_cron_alert` consults the deduper FIRST:
  `alerting.py:200` `deduper = _get_default_deduper()`;
  `alerting.py:201-202` `if not deduper.should_fire(...): return False`.
  The webhook is not even READ until `alerting.py:209`, and the
  `_CRITICAL_SEVERITIES` -> `_bot_token_fallback` branch is `alerting.py:217-218`.
  So a suppressed alert never reaches ANY delivery code, webhook or bot token.
- Non-critical path `alerting.py:95-107`: appends the occurrence, prunes to the
  5-min window, then `if len(st.occurrences) < self.threshold: return False`
  (`:102-103`). Live threshold = **3** (`settings.py:151` default 3, env absent).
- `spend.py:101-103` sets `_ALERTED = True` on the first degradation and never
  again for the life of the process, so `_record_degradation` calls the alert
  **exactly once**. One occurrence < threshold 3 -> `should_fire` returns False.

**Measured, by executing the real code:**

| Scenario | Result |
|---|---|
| `AlertDeduper().should_fire(..., severity="P2")` 1st / 2nd / 3rd | `False` / `False` / `True` |
| `AlertDeduper().should_fire(..., severity="P1")` 1st / 2nd | `True` / `False` |
| live `_get_default_deduper().threshold` | `3` |
| **TODAY**: `_record_degradation()` with `urllib.request.urlopen` patched | `urlopen.called == False` -- nothing delivered |
| **kwarg-only fix** (`details=`, severity kept P2) | `urlopen.called == False` AND `_bot_token_fallback.called == False` |
| **both fixes** (`details=` + `severity="P1"`) | `urlopen.called == True`, POST to `https://slack.com/api/chat.postMessage`, body `[P1] Cost-budget spend fetch degraded -- guard is fail-open -- cost_budget_guard: fetch_spend() fail-open: RuntimeError('bq down'). Callers receive (0.0, 0.0), so the daily/monthly ...` |

The kwarg-only row is the decisive one: `_bot_token_fallback` is **not even
entered**, proving the drop happens at `alerting.py:202` (deduper), one seam
EARLIER than the empty-webhook drop the step describes. So there are **three**
independent blockers stacked, and P2->P1 fixes blockers #2 and #3 at once
(bypasses the consecutive-threshold at `alerting.py:83-93` -- `fire =
st.last_fired_at is None or ...` is True on a virgin key -- AND satisfies the
`severity in _CRITICAL_SEVERITIES` test at `:217`).

### Q3 -- `urllib.request.urlopen` IS the right seam. Critique + fixture spec.

**Why it works, mechanically:** `_bot_token_fallback` does
`import urllib.request as _ur` FUNCTION-LOCALLY at `alerting.py:143` and calls
`_ur.urlopen(req, timeout=10)` at `alerting.py:167`. `_ur` is bound to the real
module object and `urlopen` is resolved as an attribute AT CALL TIME, so
`patch("urllib.request.urlopen")` is seen. Verified live (row 4 above).
`_post` runs inside `asyncio.to_thread` (`alerting.py:170`); `mock.patch`
rebinds a module attribute process-wide, not thread-locally, so the patch holds
across the worker thread. Also verified live.

**Alternatives, and why they are worse:**

| Candidate seam | Verdict |
|---|---|
| `backend.tools.slack.send_notification` | REJECT -- only reachable when `slack_webhook_url` is NON-empty, which directly contradicts criterion (b) ("while slack_webhook_url is empty"). |
| `alerting._bot_token_fallback` | Second-best. Captures `(source, severity, title, details)` and does prove deduper passage, but stops one seam short: it never exercises token/channel resolution (`:149-151`), the dict/str `detail_str` join (`:155-156`), the 1500-char truncation (`:159`), or `ok` parsing (`:168`). Acceptable as a SUPPORTING assertion, not the primary. |
| `asyncio.to_thread` | REJECT -- hides request construction entirely. |
| `raise_cron_alert_sync` return value | REJECT -- `_record_degradation` discards it (`spend.py:115`). |
| **`urllib.request.urlopen`** | **PRIMARY.** Last seam before the network; asserts the actual POST URL, the `Authorization` header, and the JSON body text. |

**THE MOCK TRAP IS REAL AND REPRODUCIBLE** (measured):
- `patch.object(alerting, "raise_cron_alert_sync", MagicMock())` -> `m.called
  is True` against the CURRENT BROKEN CODE. A test asserting `m.called` would
  have shipped green for the defect's whole life.
- `create_autospec(alerting.raise_cron_alert_sync)` -> `m.called is False`
  (the TypeError fires and is swallowed), correctly failing.
  Any mock of this function MUST be `autospec=True` / `create_autospec`.
  Note the repo's own precedent test
  `backend/tests/test_phase_82_10_freshness_paging.py:328-347` uses a PLAIN
  `patch(ALERT_TARGET)` -- it survives only because it asserts
  `kwargs["details"][...]`, which would `KeyError` on `detail=`. That is luck,
  not design; do not copy the plain-mock half.

**Fixture MUST pin (each item is a real failure mode, not boilerplate):**
1. `alerting.reset_default_deduper()` (`alerting.py:290-293`) before AND after.
   `_DEFAULT_DEDUPER` is a process-global; a neighbouring test that already
   fired `("cost_budget_guard","spend_fetch_degraded")` sets `last_fired_at` and
   suppresses this one for `repeat_hours=1`. Without the reset the guard can
   pass or fail for reasons unrelated to the code.
2. `spend.reset_spend_guard_status()` (`spend.py:85-91`) -- resets `_ALERTED`;
   otherwise the second test in the file gets no alert at all.
3. `slack_webhook_url = ""` and a non-empty `slack_bot_token` + `slack_channel_id`,
   pinned in the fixture, NOT inherited from `backend/.env` (else the test is
   machine-dependent). Because `get_settings` is `@lru_cache()`
   (`settings.py:627-628`), pin by patching `get_settings` (or calling
   `cache_clear()`), not by `monkeypatch.setenv` alone.
4. **The test function MUST be SYNC (`def`, not `async def`).**
   `raise_cron_alert_sync` (`alerting.py:268-287`) branches: with NO running
   loop it takes `asyncio.run(coro)` at `:284` and runs to completion, so
   `to_thread` is awaited and `urlopen` really fires. Under `pytest.mark.asyncio`
   a loop IS running, so it takes `loop.create_task(coro)` at `:276` and returns
   True **optimistically without awaiting** -- `urlopen` may never be called
   before the test ends. That would make the delivery assertion flaky/false.
5. Assert severity against the LIVE set:
   `from backend.services.observability.alerting import _CRITICAL_SEVERITIES`
   then `assert sev in _CRITICAL_SEVERITIES` (it is a private name absent from
   `__all__` at `alerting.py:296-301`, but a direct import works).
6. Drive `fetch_spend()` (or `fetch_llm_spend()`) with `google.cloud.bigquery.Client`
   made to raise -- higher fidelity than calling `_record_degradation` directly,
   because it also proves the `except Exception: _record_degradation(exc)` wiring
   at `spend.py:166-167` / `:248-249`. The negative control (criterion c) is the
   same fetch with a healthy fake client.
7. Anti-vacuity pin, mirroring `test_phase_82_10_freshness_paging.py:449-459`:
   `assert not hasattr(spend, "raise_cron_alert_sync")` -- **verified False**,
   the import is function-local at `spend.py:113`, so patching `spend.*` patches
   nothing.

### HAZARD THE FIX CREATES -- NOT IN THE STEP, MUST BE IN THE CONTRACT

`backend/tests/test_phase_75_5_1_spend_metric.py:294-301`
(`test_fail_open_returns_zero_and_fires_degradation_seam`) drives a REAL
`_record_degradation` via `FakeBQClient.raise_exc`, and
`backend/tests/test_phase_75_llm_rail.py:582-600` does the same with a
`wraps=` spy. Today the TypeError blocks delivery, so those tests are inert.
**After the fix they will construct a valid P1 alert and `_bot_token_fallback`
will POST to `https://slack.com/api/chat.postMessage` with the live
`xoxb-` token from `backend/.env` on every suite run.** Confirmed:
`backend/tests/conftest.py` contains no slack/urlopen/socket guard.
The contract must either add an autouse fixture that pins an empty bot token
(or patches `urlopen`) for those tests, or add a session-level network guard.
Shipping the fix without this turns `pytest` into a Slack spammer.

### Q5 -- RECOMMEND P2 -> P1 at this call site. Do NOT make P2 deliverable.

- **Alert-fatigue cost is structurally ZERO here.** `_ALERTED`
  (`spend.py:72`, latched at `:101-103`) makes `_record_degradation` alert at
  most **once per process lifetime**, and it is only reset by
  `reset_spend_guard_status()` (a test seam). Even without that latch, the
  deduper's `repeat` window would cap it at 1/hour (`alerting.py:88-90`,
  `repeat_hours=1`). So the worst case is one page per backend restart while
  BigQuery is down. That is squarely inside Google SRE's bar -- "Every page
  should be actionable" and pages should be for "a novel problem"
  (https://sre.google/sre-book/monitoring-distributed-systems/, accessed
  2026-08-06). A silently disabled $25/day spend ceiling is urgent and
  actionable.
- **Making P2 globally deliverable is the wrong lever.** It would change
  behaviour for the other 11 non-critical production sites (8 of them news/econ
  feed degradations at `news/sources/*.py` and `econ_calendar/sources/*.py`),
  which are deliberately ticket-class, not page-class. SRE's own framing: "You
  might file a ticket to investigate a low rate of errors ... a 100% error rate
  is an emergency." A dead news feed is a ticket; a dead spend ceiling is a page.
  Widening the delivery rule would re-create the phase-66 P1 page storm the
  comment at `alerting.py:46-53` records.
- **Tradeoff stated:** P1 means this alert bypasses the 3-in-5-min consecutive
  threshold, so a single transient BigQuery blip pages the operator. That is
  accepted deliberately: the one-shot `_ALERTED` latch means a blip costs
  exactly one message, and the alternative (staying P2) costs *silence during a
  real outage*, which is the failure mode this step exists to close.
- **Residual gap worth QUEUING, not fixing here:** because `_ALERTED` never
  resets in-process, a degradation that recovers and re-degrades in the same
  process pages only once; and a multi-day outage pages only at restart. That is
  a deliberate anti-storm choice but it is undocumented. Queue a follow-up to
  decide whether `_ALERTED` should clear on a healthy fetch (level->edge
  re-arm), mirroring `freshness_cron`'s transition gate.

### Q7 -- NO test has ever covered `detail=`. Existing coverage is blind by construction.

- Zero repo-wide hits for `spend_fetch_degraded` or `cost_budget_guard` in any
  test.
- `test_phase_75_5_1_spend_metric.py:294-301` asserts only
  `degraded_count == 1`, `alerted is True`, and `last_error` content. **`alerted`
  is set at `spend.py:102`, BEFORE the `try:` at `:112`** -- so it is literally
  the "branch was entered" assertion the step warns about. It passes against the
  bug and will keep passing after the fix; it can never observe delivery.
- `test_phase_75_llm_rail.py:582-600` spies `_record_degradation` with `wraps=`
  and asserts the same counter -- same blindness.
- `# pragma: no cover` at `spend.py:126` excludes the swallow from coverage, so
  no coverage gate could have surfaced it either.
- Provenance: introduced 2026-07-23 in `3a7942cf`, never edited (Q7 part 1).

### Repo precedent to REUSE (the step names it; here are the anchors)

- `backend/tests/test_phase_82_10_freshness_paging.py:40` --
  `ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"`
  (the correct patch target for a function-local import).
- `...:341-344` already encodes this step's severity doctrine verbatim:
  *"severity must stay P1: with slack_webhook_url empty, only critical
  severities reach the bot-token fallback -- a P2 is logged and dropped"*.
- `...:449-459` `test_wrong_patch_target_does_not_exist` -- the anti-vacuity pin.
- `...:417-430` `test_all_healthy_emits_no_alert` with a **precondition
  assertion** so a zero cannot come from a broken fixture -- exactly the shape
  criterion (c) needs.
- `backend/tests/test_phase_82_11_autoresearch_failure_paging.py:16-17` -- same
  function-local-import doctrine, second instance.

### An EXISTING guard already watches this defect -- do not break it

`backend/tests/test_phase_82_54_cost_budget_columns.py:344-368`
(`test_the_spend_py_alert_defect_is_QUEUED_not_silently_inherited`) regex-scans
`spend.py` at `:354` for `raise_cron_alert_sync\([^)]*\bdetail=` and, if the
defect is still present, REQUIRES an open masterplan step whose `name` contains
both `"spend.py"` and `"detail"` and has criteria. Consequences for 82.58:

1. It passes today because 82.58's name satisfies that predicate.
2. When the fix lands, `still_broken` is None and the guard returns early at
   `:356` -- it degrades to a permanent no-op rather than failing. Fine, but
   note that it will no longer protect anything.
3. **Ordering constraint:** do NOT flip 82.58 to `done` before the code fix is
   in the tree. If the step leaves the open set while `detail=` is still
   present, `owner` at `:362-365` becomes empty and that test goes RED.

### Q4 (extended) -- SWEEPING THE WHOLE CLASS, not just the alert function

Criterion (d) says "any further mismatch fixed or queued", so the sweep was run
three ways. The methodology matters, and this is the part that will save the
executor a cycle:

| Sweep | Candidates | Real | Verdict |
|---|---|---|---|
| (1) Alert-function-only, bound via `inspect.signature` | 33 | **1** | the target |
| (2) Naive bare-NAME match against any unique repo def | 65 | **3** | ~95% false positives -- DO NOT USE |
| (3) Precise: callee resolved through the file's `ImportFrom` (Pass A) + same-file method resolution (Pass B) | 15 + 2 | **3** | the methodology to implement |

Sweep (2) is noisy because a bare name like `writer`, `history`, `parse`,
`loads`, `percentile`, `handler`, `_append_audit` collides with `csv.writer`,
`yfinance.Ticker.history`, `json.loads`, `numpy.percentile` etc. **The contract
must specify import-resolved binding, not name matching**, or criterion (d)'s
"derived set" will be 95% garbage and the executor will either drown or
hand-filter it (which destroys the "derived structurally" property).

**Two FURTHER production mismatches of the SAME class, both confirmed at
runtime with `inspect.signature(...).bind()` -- QUEUE, do not fix here:**

1. `backend/slack_bot/assistant_lifecycle.py:181` calls
   `handler.handle_thread_started(body={}, client=app.client, say=say,
   set_status=..., logger=logger)`; the method at `:28-34` is
   `(self, body, say, set_suggested_prompts, logger)`.
   Runtime bind -> `TypeError: missing a required argument:
   'set_suggested_prompts'` (plus unexpected `client`, `set_status`).
2. `backend/slack_bot/assistant_lifecycle.py:188` calls
   `handler.handle_context_changed(body=body, client=..., say=...,
   set_status=..., logger=logger)`; the method at `:90-93` is
   `(self, body, logger)`.
   Runtime bind -> `TypeError: got an unexpected keyword argument 'client'`.

   **These are LIVE**: `backend/slack_bot/app.py:18` imports
   `register_assistant_lifecycle` and `:33` calls it. So the Slack assistant's
   welcome message + suggested prompts + channel-context tracking are all dead.
   Unlike spend.py there is no local swallowing `except` and no `@app.error`
   handler in `backend/slack_bot/app.py`, so Bolt's default handling logs it at
   ERROR -- louder than spend.py, but still never fixed.

**Third finding (pre-existing red, out of scope, QUEUE):**
`tests/autoresearch/test_slot_usage_wiring.py` passes `log_fn=` at 14 call
sites to `trigger_thursday_batch` / `run_friday_promotion` /
`run_monthly_sortino_gate` / `auto_demote_on_dd_breach`, none of which accept
it. Measured: `pytest tests/autoresearch/test_slot_usage_wiring.py -q` ->
**9 failed in 0.08s**. Pre-existing and unrelated to 82.58 (82.58's immutable
command is scoped to its own new test file, so this does not block it), but it
is a mismatch the derived set contains and criterion (d) requires it be queued.

### Q6b -- the degradation reaches NO operator surface today (closes Q6)

`spend_guard_status()` (`spend.py:75-82`) is exported in `__all__`
(`observability/__init__.py:68`) but has **ZERO non-test callers repo-wide**:
no API endpoint, no frontend component, no Slack job, no cron. The only
consumers are `test_phase_75_5_1_spend_metric.py:298` and
`test_phase_75_llm_rail.py:589/600`. `cost_budget_watcher.py` calls
`fetch_spend()` at `:92-94` but never reads the guard status, and
`cost_budget_api.py`'s `/status` + `/today` endpoints (`:186-197`) do not
expose it either.

**Therefore the broken alert is the ONLY path by which an operator could ever
learn that the $25/day ceiling has stopped enforcing.** There is no dashboard
fallback, no log line above DEBUG for the delivery failure, and no metric. This
is the strongest single argument for the step's P1 classification, and it maps
directly to Google SRE's "signals that are collected, but not exposed in any
prebaked dashboard nor used by any alert, are candidates for removal"
(https://sre.google/sre-book/monitoring-distributed-systems/, accessed
2026-08-06) -- here the signal IS collected and is exposed nowhere.

**Blast radius of P2 -> P1: nil.** No test anywhere asserts
`severity == "P2"` for `source="cost_budget_guard"`; the only two hits for that
string are `spend.py:116` itself and an unrelated test *function name*
(`test_phase_75_llm_rail.py:147`).

## Consensus vs debate (external)

**Consensus (no disagreement found across the 9 sources):**
- Mocks must be signature-checked or they mask API drift (source 1; corroborated
  by the pylint/mypy rules in sources 4-5 which exist for the same reason).
- Broad `except` + sub-WARNING logging hides real errors (sources 6-7, CWE-703).
- Pages must be actionable and rare; ticket-class alerts must not page
  (sources 2-3).
- An alert's health cannot be inferred from its silence (source 9).

**Genuine tension, and how it resolves here:** sources 2-3 push toward FEWER
pages (alert fatigue), while source 9 pushes toward proving every alert can
fire. These conflict in general -- raising severity to make an alert deliverable
is exactly the move that causes page storms (and this repo has the scar:
`alerting.py:46-53` records the phase-66 "~120 pages/hour" incident from a
blanket P1 bypass). The tension does NOT bite here because `_ALERTED`
(`spend.py:72`, `:101-103`) caps this alert at one message per process. So we
get source 9's guarantee at zero cost in source 2-3's currency. **State this
explicitly in the contract** -- it is the reason P1 is safe HERE and would not
be safe at, say, `news/sources/finnhub.py:107`.

## Pitfalls (from literature + measured here)

1. **A plain `MagicMock` of the emitter passes against the broken code.**
   Measured: `m.called is True` with `MagicMock`, `False` with
   `create_autospec`. Source 1: without autospec "the test would pass despite
   the bug". Use `autospec=True` for ANY mock of `raise_cron_alert_sync`.
2. **Asserting "the branch was entered" is not asserting delivery.** The
   existing `alerted is True` assertion is set BEFORE the try block.
3. **An async test silently breaks the delivery assertion.**
   `raise_cron_alert_sync:274-281` fire-and-forgets under a running loop.
4. **Process-global state leaks between tests** -- `_DEFAULT_DEDUPER` and
   `_ALERTED` both must be reset, or the test passes/fails for the wrong reason
   (the exact failure mode `test_phase_82_10_freshness_paging.py:91-98`
   documents for its own transition gate).
5. **A bare-name AST sweep is ~95% false positives** (65 candidates, 3 real).
   Resolve the callee through imports.
6. **Fixing the bug arms a live Slack POST inside the existing test suite.**
7. **`# pragma: no cover` on the swallow** means coverage tooling can never
   surface this class -- worth a broader look later.

## Application to pyfinagent (mapping to the 4 immutable criteria)

- **(a) delivery, not branch entry, failing against current code.** Drive
  `spend.fetch_spend()` with `google.cloud.bigquery.Client` raising; patch
  `urllib.request.urlopen`; assert `urlopen.called` and parse the POST body for
  `[P1]` + `"cost_budget_guard"` + the `(0.0, 0.0)` sentence. Verified to fail
  today (`urlopen.called == False`) and pass after both fixes
  (`urlopen.called == True`). Test must be SYNC.
- **(b) severity actually deliverable, asserted against the live set.**
  `from backend.services.observability.alerting import _CRITICAL_SEVERITIES`;
  extract the severity from the captured POST body (or from an
  `autospec`'d spy on `raise_cron_alert_sync`) and assert membership. Do NOT
  hardcode `"P1"`. Additionally assert `get_settings().slack_webhook_url == ""`
  in the fixture as a PRECONDITION, so the test cannot pass via the webhook path.
- **(c) negative control.** Same fetch with a healthy fake client ->
  `urlopen.called is False`, plus a precondition assertion that the fetch
  returned non-degraded (`spend_guard_status()["degraded_count"] == 0`), so a
  zero cannot come from a broken fixture (pattern:
  `test_phase_82_10_freshness_paging.py:417-430`).
- **(d) structural sweep, derived set non-empty, further mismatches queued.**
  Implement the import-resolved sweep (sweep (3) above), assert the derived
  call-site set is non-empty, and bind every site against
  `inspect.signature`. Expected result after the fix: zero mismatches among
  `raise_cron_alert*` sites (33 of them). QUEUE separately:
  `assistant_lifecycle.py:181/:188` (live, P1-ish) and
  `tests/autoresearch/test_slot_usage_wiring.py` (9 red tests).
- **Plus (not in the criteria, but required for a safe merge):** an autouse
  fixture or conftest guard so `test_phase_75_5_1_spend_metric.py:294` and
  `test_phase_75_llm_rail.py:582` cannot POST to Slack once the fix lands.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9; 7 at
      tier 2 official-doc or better)
- [x] 10+ unique URLs total (53: 9 read-in-full + 44 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (2 findings)
- [x] Full pages read (not abstracts) for the read-in-full set; 3 pages that
      WebFetch returned as nav-only were re-fetched via curl + tag-strip
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the spawn prompt, plus
      4 not named (`assistant_lifecycle.py`, `app.py`, `conftest.py`,
      `test_phase_82_54_cost_budget_columns.py`)
- [x] Contradictions / consensus noted (alert-fatigue vs prove-it-fires)
- [x] All claims cited per-claim
- [x] Adaptive coverage: 6 rounds, last 2 dry (K=2 satisfied)

### Coverage loop record (audit-class)

| Round | Activity | New read-in-full / class findings | Dry? |
|---|---|---|---|
| 1 | Alert-specific AST sweep + 4 external fetches + empirical run | 33 sites, 1 mismatch; 3 blockers proven; mock trap reproduced | no |
| 2 | Generalized bare-name sweep + 2025 recency search | 82.54 regex guard found; sweep-noise finding; watchdog concept | no |
| 3 | Precise import-resolved sweep + runtime bind + 5 more fetches | 2 further LIVE mismatches (`assistant_lifecycle.py:181/:188`) | no |
| 4 | Triage of remaining sweep hits + Bolt error path | 9 red tests in `test_slot_usage_wiring.py` | no |
| 5 | Other one-shot-latched emitters; `__init__` surface; spend consumers | **0** | **DRY 1** |
| 6 | Operator-surface exposure of `spend_guard_status`; P1 blast radius | **0** | **DRY 2** |

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 44,
  "urls_collected": 53,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": true,
    "rounds": 6,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_82.58.md",
  "gate_passed": true
}
```

**Status: COMPLETE.**
