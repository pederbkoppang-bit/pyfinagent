# Contract -- phase-82.58

**Step:** 82.58 (P1) -- an alert guarding the cost-budget hard-block has never
fired.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.58.md`,
`gate_passed: true`, **audit_class** with `dry: true` after 6 rounds / 2 dry,
9 sources read in full, 53 URLs, 20 internal files.

---

## 1. Confirmed, refuted, and corrected

**CONFIRMED by me and independently by the gate:**

- The call is `backend/services/observability/spend.py:115-125`, `detail=` at
  `:120`. The signature is `backend/services/observability/alerting.py:253-259`,
  parameter `details`. The call sits inside the `try` opened at `:112`; the
  handler at `:126-127` logs at **DEBUG** and carries `# pragma: no cover`, so
  coverage tooling could never flag it either.
- `detail=` was introduced 2026-07-23 in commit `3a7942cf` and never edited.
  **No test has ever covered this alert** -- zero repo-wide hits for
  `spend_fetch_degraded` or `cost_budget_guard` in any test.

**REFUTED -- the step's call-site count.** The step says "the ONLY malformed
call site of 15 audited repo-wide". The real denominator is **33**, derived
twice independently: my AST sweep found **28** under `backend/`, the gate found
**33** repo-wide; the difference is exactly the **5** sites in repo-root
`tests/`. Both numbers are right for their scope. The **numerator is correct**:
exactly **1** signature mismatch.

**CORRECTED -- a number I published earlier in this session.** I stated the
budget caps are `5.0/50.0`, reading them from the `getattr` fallbacks at
`llm_client.py:437-438`. Those fallbacks are **unreachable**: the settings
attributes exist, so the live caps are **25.0 / 300.0**
(`settings.py:392-393`). The conclusion is unchanged -- `0.0 >= 25.0` is still
False -- but a fixture pinning 5.0/50.0 would exercise a dead branch, so the
guard pins the real values.

## 2. THREE blockers, not the two the step describes

The step describes the kwarg and the severity. There is a **third, and it is
the earliest one in the chain** -- I found it reading `AlertDeduper`, and the
gate confirmed it by measurement.

| # | Blocker | Where | Order |
|---|---------|-------|-------|
| 1 | `detail=` -> `TypeError`, swallowed at DEBUG | `spend.py:120` / `:126` | call never completes |
| 2 | **deduper suppresses before delivery** | `alerting.py:201-202` | **runs BEFORE the webhook read at :209** |
| 3 | P2 + empty webhook -> logged and dropped | `alerting.py:210-224` | last |

Blocker 2 is the one nobody had seen. `should_fire`'s non-critical path
(`:95-105`) returns False until **3 occurrences inside 5 minutes**
(`threshold=3`), but `spend.py::_record_degradation` has a one-shot `_ALERTED`
latch (`:72`, `:101-103`) and calls the alert **exactly once per process**. So
the P2 alert is suppressed *before any delivery code runs* -- **even on a
machine with a working webhook.** The gate verified this empirically: with a
kwarg-only fix applied, `urlopen.called == False` **and**
`_bot_token_fallback.called == False`.

A kwarg-only fix therefore repairs nothing observable. Raising to **P1** repairs
blockers 2 and 3 together (`:83-93` fires when `last_fired_at is None`; `:217`
routes to the bot-token fallback).

## 3. Why this is P1 and not a typo -- verified, not inherited

`llm_client.py:452`: `tripped = daily >= daily_cap or monthly >= monthly_cap`,
and `daily = float(daily_usd or 0.0)`. Nothing coerces the fail-open value, so
`(0.0, 0.0)` cannot trip the block. **And the gate found the fact that settles
it:** `spend_guard_status()` has **ZERO non-test callers** repo-wide -- no API
endpoint, no frontend tile, no cron. This broken alert is the **only** path by
which an operator could ever learn the $25/day ceiling stopped enforcing.

## 4. Immutable success criteria (verbatim)

1. "a fixture in which the spend fetch degrades causes an alert to be DELIVERED
   to the operator channel, asserted by capturing the emitted payload rather
   than by observing that a branch was entered, and failing against the current
   detail=/P2 implementation"
2. "the emitted severity is one that actually reaches the operator while
   slack_webhook_url is empty, asserted against the live _CRITICAL_SEVERITIES
   set rather than a hardcoded string"
3. "a healthy spend fetch emits NO alert, so the guard cannot pass by always
   firing"
4. "every raise_cron_alert_sync call site in the repo is derived structurally
   and its bound keyword arguments checked against inspect.signature, the
   derived set asserted non-empty, and any further mismatch fixed or queued"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_58_spend_alert_delivery.py -q`

## 5. The delivery seam, and the mock trap that would have faked it

Criterion 1 demands **delivery**, not branch entry. The seam is
`urllib.request.urlopen`, reached via
`_record_degradation` -> real `raise_cron_alert_sync` -> `raise_cron_alert` ->
`_bot_token_fallback` -> `_post()` -> `urlopen`. `alerting.py:143` imports
`urllib.request` **function-locally** and `:167` resolves `urlopen` at call
time, so a process-wide `mock.patch` lands even across `asyncio.to_thread`
(`:170`). Asserting on the POST body drives kwarg binding, deduping, severity
routing and payload construction in one shot.

**The trap, reproduced empirically by the gate:** patching
`raise_cron_alert_sync` with a plain `MagicMock` gives `called == True`
**against the broken code** -- a false PASS, because a MagicMock accepts
`detail=` happily. Only `create_autospec` enforces the real signature
(`called == False`). This guard does not mock the emitter at all; it drives the
real one. *(The repo's own `test_phase_82_10_freshness_paging.py:328` uses a
plain patch and survives only because it happens to index `kwargs['details']`.)*

**The test must be synchronous.** Under a running loop
`raise_cron_alert_sync:276` `create_task`s and returns `True` without awaiting,
so an async test could report success while delivery never happened.

Fixture must pin: `slack_webhook_url` empty, a **dummy** bot token, a fresh
deduper, and `_ALERTED` reset -- so the guard is env-independent rather than
passing by accident of this machine's config.

## 6. A HAZARD THE FIX CREATES -- it lands first, not alongside

Found by the gate, absent from the step. **Fixing `spend.py` arms a real Slack
POST inside the existing test suite.**
`test_phase_75_5_1_spend_metric.py:294` and `test_phase_75_llm_rail.py:582`
already drive `_record_degradation` for real; `backend/tests/conftest.py` has
**no network guard**; and `backend/.env` holds a live `xoxb-` token. Post-fix
the chain completes and posts to the operator's channel from a routine
`pytest` run.

So the ordering is **guard first, fix second**: an autouse egress block goes
into `backend/tests/conftest.py` before `spend.py` is touched, and I will not
run the suite against a fixed `spend.py` until it is in place. This is an
outward-facing side effect, so it is not something to discover by observing it.

## 7. Plan

- **D0** -- autouse network guard in `backend/tests/conftest.py` (blocks
  `urllib.request.urlopen`), landing BEFORE the `spend.py` edit.
- **D1** -- `spend.py`: `detail=` -> `details=`, `severity="P2"` -> `"P1"`.
  Both, because either alone delivers nothing (§2).
- **D2** -- the delivery guard per §5, with the negative control for criterion 3.
- **D3** -- criterion 2 asserts membership in the **imported** live
  `_CRITICAL_SEVERITIES`, never a literal `"P1"`.
- **D4** -- criterion 4's sweep: **import-resolved** AST binding, not bare-name.
  The gate measured that a bare-name sweep yields 65 candidates for 3 real hits
  (collisions with `csv.writer`, `yfinance.history`, `json.loads`,
  `numpy.percentile`); import-resolved yields 15+2 and finds all 3. Derived set
  asserted non-empty (a sweep that sees nothing and a sweep that finds nothing
  are indistinguishable).
- **D5** -- queue the discovered defects (§8).

## 8. Discovered defects -- QUEUED, not fixed here

1. **`assistant_lifecycle.py:181` and `:188`** -- two more live signature
   mismatches of the same class, confirmed by runtime
   `inspect.signature().bind()` (`:181` missing `set_suggested_prompts`,
   unexpected `client`/`set_status`; `:188` unexpected `client`). Wired into
   production via `slack_bot/app.py:33`. **Not** `raise_cron_alert` sites, so
   outside criterion 4's literal scope -- but the same defect shape.
2. **9 already-failing tests** in `tests/autoresearch/test_slot_usage_wiring.py`
   (`log_fn=`).
3. **12 of 27 production alert sites carry a severity that cannot deliver**
   while the webhook is empty (P2/P3). Deliberate anti-spam design per
   `alerting.py:211-216`, so NOT a blanket bug -- but it needs a triage of which
   of them guard something that matters.
4. **A CI call-arg checker.** The recency scan surfaced Pyrefly 1.0
   (2026-05-12) which checks unannotated code `mypy` skips -- this entire defect
   class is statically detectable and was not detected for 14 days.

## 9. Non-scope

Do NOT make P2 globally deliverable -- the gate measured that it would page for
8 deliberately ticket-class news/econ feed sites and re-create the alert storm
recorded at `alerting.py:46-53`. P2 -> P1 at **this call site only**. No change
to the deduper, to `_CRITICAL_SEVERITIES`, or to any other call site. No
credential rotation, no live positions.

## 10. References

- `handoff/current/research_brief_82.58.md` (audit-class, dry after 2 rounds)
- Google SRE Workbook ch. 5 -- alerting on symptoms; severity design
- Dead-man's-switch / watchdog pattern -- "an alert that never fires is
  indistinguishable from a healthy system"
- Internal: `backend/services/observability/spend.py:72,101-127`,
  `backend/services/observability/alerting.py:54,83-105,136-176,200-224,253-276`,
  `backend/agents/llm_client.py:437-452`, `backend/config/settings.py:392-393`,
  `backend/tests/test_phase_82_54_cost_budget_columns.py:344-368`
