---
name: project-execution-backend-wiring-68-1
description: phase-68.1 gate — PKLIVE is folklore (Alpaca separates paper/live by DOMAIN not key prefix); paper URL IS offline-assertable; backend restart measured 2.455s; `kickstart -k` does NOT re-read plist edits
metadata:
  type: project
---

Research gate for phase-68.1 (EXECUTION_BACKEND reaching execution_router in the
launchd process), 2026-08-07. Five findings that are not derivable from the code.

**1. The criterion's own discriminator was unfounded.** Criterion 4(b) demanded a
"live-key pattern (PKLIVE-class) rejected" test. Three official Alpaca sources read
in full (auth reference, paper-trading doc, vendor learn page) document **NO prefix
or format difference between paper and live API keys**. Alpaca separates the two by
**DOMAIN**: `https://paper-api.alpaca.markets` vs `https://api.alpaca.markets`.
`execution_router.py:76`'s `key.startswith("PKLIVE")` tests for a string the vendor
does not document as existing. Corroborates the phase-68.0 Q/A's "PKLIVE-folklore"
remark with primary sources.
**Why:** an immutable criterion can encode a false premise. It still has to be
tested as written, but the brief must disclose that the leg is not the real guard.
**How to apply:** when a criterion names a magic string/prefix/format owned by a
third party, verify it against the VENDOR's docs before designing the test. Related:
[[feedback_measure_dont_assert_claims]].

**2. The real paper-only guard IS offline-assertable — don't call it untestable.**
`alpaca-py 0.43.2`: `TradingClient("dummy","dummy",paper=True)` constructs with
**no network**, and `client._base_url` is `BaseURL.TRADING_PAPER` ==
`"https://paper-api.alpaca.markets"`. `TradingClient.__init__` maps
`url_override if url_override else BaseURL.TRADING_PAPER if paper else
BaseURL.TRADING_LIVE` — so `url_override` is a real escape hatch worth asserting is
never passed.

**3. Backend restart cost, MEASURED (not estimated).** From `backend.log`
2026-08-06: `Shutting down` 13:46:15,324 -> `Application startup complete`
13:46:17,779 = **2.455 s**. `com.pyfinagent.backend-watchdog` needs **3 consecutive
60s-spaced /api/health failures (~3 min)** before acting and **does not page** (it
SIGUSR1-dumps then kickstarts), so a normal restart cannot trip it.
**The binding constraint is the Slack evening digest**: `scheduler.py:239-248`
schedules it `hour=settings.evening_digest_hour (default 17), minute=0,
timezone=America/New_York` = **17:00 ET = 23:00 CEST**. A non-200 gives an empty
digest silently, but **connection-refused raises -> `_route_exception_to_p1`
(`:636`) PAGES P1**. Restart outside ~22:58-23:02 CEST, or after "Evening digest
sent" appears.

**4. `launchctl kickstart -k` does NOT pick up plist edits.** It restarts the
service from the already-loaded job definition; an edited `EnvironmentVariables`
block needs `launchctl unload/load` (or `bootout`/`bootstrap`). Getting this wrong
makes a new provenance banner print `source=default` when the code is correct, and
the criterion looks failed for the wrong reason.

**5. Grep trap in settings.py.** `# --- Execution Mode ---` at
`backend/config/settings.py:222` introduces **Celery** (`use_celery`), not the
execution backend. Anyone grepping "execution" in settings.py gets 3 hits and none
is a field — there is no `execution_backend` field at all. Combined with
`extra:"ignore"` (`:632`), an operator typing `EXECUTION_BACKEND=` into
`backend/.env` gets **no error, no warning, no effect**. Related:
[[project_funnel_zero_trade_66_2]], [[project_real_fill_runway_68_0]].

**Sandbox note:** `backend/.env` is DENIED to the researcher sandbox (a Bash call
merely touching it was blocked). Delegate any `.env` grep to Main — see
[[project_backend_restart_safety]].
