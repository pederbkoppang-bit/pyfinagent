---
step: phase-23.5.7
title: Cron job verification — daily_price_refresh — experiment results
date: 2026-05-09
verdict_class: PASS_PENDING_QA (clean — with adjacent finding to surface)
verification_command: 'python3 tests/verify_phase_23_5_7.py'
---

# Experiment Results — phase-23.5.7

## What was done

Verification-only step. **No code changes.** One artifact:

1. `tests/verify_phase_23_5_7.py` — replayable verifier.

## Verification command — verbatim from `.claude/masterplan.json::23.5.7`

```
python3 -c 'import json,sys,urllib.request as u; r=json.load(u.urlopen("http://localhost:8000/api/jobs/all")); j=next((x for x in r["jobs"] if x["id"]=="daily_price_refresh"), None); assert j is not None, "job missing"; assert j.get("status") != "manifest", f"status still manifest: {j}"; assert j.get("next_run") is not None, f"next_run null: {j}"; print("OK", j["id"], j["status"], j["next_run"])'
```

## Verbatim result

```
$ <verbatim immutable command>
OK daily_price_refresh scheduled 2026-05-10T01:00:00+02:00
EXIT=0

$ python tests/verify_phase_23_5_7.py
OK daily_price_refresh status=scheduled next_run=2026-05-10T01:00:00+02:00
EXIT=0
```

## Live `/api/jobs/all` entry

```json
{
  "id": "daily_price_refresh",
  "source": "slack_bot",
  "schedule": "phase-9.2 cron",
  "next_run": "2026-05-10T01:00:00+02:00",
  "last_run": null,
  "status": "scheduled",
  "description": "Daily refresh of universe price snapshots"
}
```

`last_run: null` because the job hasn't fired since the 10:20 CEST
daemon restart; next fire is 01:00 CEST tomorrow (= 23:00 UTC =
19:00 ET, market-close-aligned).

## Why the criterion is satisfied

- `daily_price_refresh.py` makes ZERO HTTP calls (pure Python
  function); `heartbeat()` has no URL; cross-process push goes via
  `_aps_to_heartbeat()` using `_HEARTBEAT_URL = "http://127.0.0.1:8000/api/jobs/heartbeat"`.
- Bridge surfaces `status="scheduled"` from the registry's startup
  seed.
- CronTrigger computed `next_run` correctly.
- No Docker-alias bug applies — researcher confirmed by reading
  the function body verbatim.

## CRITICAL adjacent finding (DO NOT MISS — surfaced for next step)

Researcher's tail of `handoff/logs/slack_bot.log` revealed:

> `format_evening_digest` raised
> **`KeyError: slice(None, 10, None)`** at `formatters.py:376`
> at **23:00:01 CEST today** (= 17:00:01 ET, evening digest fire
> time post-23.5.3.1).

This is a **runtime bug** that 23.5.4's PASS could not catch
because the criterion only tests scheduling, not actual delivery.
Phase-23.5.3.1 fixed the Docker-alias hostname so the httpx call
finally lands → response is parsed → `format_evening_digest` is
invoked → it crashes on a slice access (likely `trades_data[:10]`
where `trades_data` is a dict instead of a list, or an empty
container that doesn't accept slicing).

**This is an HONEST limitation of the immutable criterion**
(scheduling vs end-to-end delivery), NOT a defect of 23.5.4's
verdict. Recommend opening `phase-23.5.7.1` (or sibling) immediately
after this step closes, BEFORE running 23.5.8, to fix the slice
bug. The next evening digest fire is tomorrow 23:00 CEST — there's
~24h to ship the fix.

## Production-stub limitation (researcher noted, NOT in scope)

`daily_price_refresh.py:44-50` uses `_default_fetch` and
`_default_write` STUBS that don't call yfinance or BQ. A successful
fire writes no real data. Coverage gap; deferred to a hardening
task. Not a verification defect — the criterion as written tests
scheduling, which IS satisfied.

## Sibling verifiers — no regressions

| Verifier | Result |
|----------|--------|
| 23.5.1, 23.5.2, 23.5.2.5, 23.5.2.6, 23.5.3, 23.5.3.1, 23.5.4, 23.5.5, 23.5.6 | PASS |
| 23.5.7 (this step) | PASS, EXIT=0 |

## What this step does NOT do

- Fix the `format_evening_digest` KeyError (deferred to next step).
- Wire real yfinance / BQ in `daily_price_refresh` production path.
- The 6 sibling phase-9 jobs.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.5.7-research-brief.md`
- `tests/verify_phase_23_5_7.py`

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_5_7.py
```
