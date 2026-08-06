---
name: dead-alert-82-58
description: Step 82.58 research -- the cost-budget alert has THREE stacked blockers not two, the deduper being the earliest; the step's "15 call sites" is really 33; and fixing it arms a live Slack POST inside the existing test suite
metadata:
  type: project
---

Research gate for masterplan 82.58 (2026-08-06): `spend.py::_record_degradation`
raises `TypeError` into its own swallowing `except`, so the alert guarding the
LLM cost-budget hard-block has never fired.

**Facts that were NOT in the step description and that I had to measure:**

1. **THREE stacked blockers, not two, and the step names the wrong one as
   second.** `AlertDeduper.should_fire` is consulted at `alerting.py:201-202`,
   BEFORE the webhook is even read at `:209`. On the non-critical path it needs
   3 occurrences in 5 minutes, but `spend.py`'s `_ALERTED` latch calls the alert
   EXACTLY ONCE per process. Measured by executing the real code with a
   kwarg-only fix applied: `urlopen.called == False` AND
   `_bot_token_fallback.called == False` -- so the drop happens one seam EARLIER
   than the "empty webhook" story. P2 -> P1 repairs blockers 2 and 3 at once.
2. **The step's "15 audited" is wrong: the real denominator is 33** (27
   production + 5 test + 1 internal delegation). The numerator (1 malformed) is
   right. 12 of 27 production sites carry a severity that cannot be delivered
   with the webhook empty -- mostly deliberate ticket-class news feeds.
3. **Fixing the bug ARMS A LIVE SLACK POST inside the existing suite.**
   `test_phase_75_5_1_spend_metric.py:294` and `test_phase_75_llm_rail.py:582`
   already drive `_record_degradation` for real; the TypeError is the only thing
   stopping delivery today. `backend/tests/conftest.py` has no network guard and
   `backend/.env` carries a live `xoxb-` token. The fix must ship with a guard.
4. **The stated caps 5.0/50.0 are the unreachable `getattr` FALLBACKS.** Live
   caps are 25.0/300.0 from `settings.py:392-393`. The conclusion ((0,0) cannot
   trip) still holds, but a test pinning 5.0/50.0 tests a dead branch.
5. **Two FURTHER live instances of the same class**, confirmed by runtime
   `inspect.signature().bind()`: `backend/slack_bot/assistant_lifecycle.py:181`
   and `:188`, wired into production via `app.py:33`. Plus 9 already-red tests
   in `tests/autoresearch/test_slot_usage_wiring.py` (`log_fn=`).
6. **`spend_guard_status()` has ZERO non-test callers repo-wide** -- no API, no
   frontend, no cron. The broken alert is the ONLY operator path.

**Why:** the alert guards the $25/day LLM circuit breaker; a fail-open returns
(0.0, 0.0), which is indistinguishable from "you have spent nothing".

**How to apply:** when a step says "N call sites audited", re-derive N -- see
[[measure-dont-assert]] class. When auditing a kwarg-drift class, resolve the
callee through the file's `ImportFrom`, never by bare name: a bare-name sweep
gave 65 candidates for 3 real hits (`csv.writer`, `yfinance.history`,
`json.loads`, `numpy.percentile` all collide). And whenever a fix RE-ENABLES a
dormant outbound path, check what else in the tree was relying on it staying
dormant.
