# ~~Pending~~ COMPLETED restart -- 2026-08-18

> **DONE 2026-08-18 13:54:49 CEST**, operator-requested, ~6h before the
> 14:00 ET / 20:00 CEST book cycle (safe margin).
>
> ```
> launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
> listener pid 89340 -> 25117      (PID CHANGED, so the restart took)
> ps lstart: tir. 18 aug. 13.54.49 2026
> launchctl list: 25117  com.pyfinagent.backend
> health: {"status":"ok", version 6.93.237, mcp data/backtest/signals all ok}
> ```
>
> **BOTH fixes VERIFIED IN FORCE, measured on the running process:**
> - `GET /api/charts/DELL?period=1y` -> HTTP 200 with real OHLCV data (was
>   500 before the restart) -- the NaN-safe response class is live.
> - `/api/settings/flags` -> `claude_rail_cooldown_default_hours: 6.0,
>   in_force: true` -- the 86.120 cooldown module loaded successfully and
>   its new Settings field is live.
>
> **Note on 86.120 specifically**: that masterplan step is PARKED (3rd-
> consecutive-CONDITIONAL rule, `escalation_86.120_third_conditional.md`),
> not flipped to `done`. Its code is now live in the running process
> regardless -- restarting picks up everything on disk, not just
> masterplan-`done` work. This is a purely additive, defensive addition
> (a new cooldown that only engages on a specific classified CLI failure
> message; the existing phase-66.1 generic breaker is unchanged for every
> other failure class) that three independent Q/A mutation passes found
> zero product defects in across three cycles -- flagged here for the
> record, not as a concern.

**Original pending-state record below, for history.**

**~~NOT YET IN FORCE~~.** Recorded per the standing rule that backend restarts
batch to session end.

## Running process

```
$ ps -o pid,lstart -p 89340
89340   tir. 18 aug. 08.26.53 2026
```

**pid 89340 started 2026-08-18 08:26:53** and has not re-read `backend/api/charts.py`
since -- the fix below is on disk, not in the running process.

## What is committed-to-disk but not active

| file | change | effect once restarted |
|---|---|---|
| `backend/api/charts.py` | Added `default_response_class=NaNSafeJSONResponse` on the `charts` router (same phase-80.1 pattern already applied to `backend/api/signals.py`) | `GET /api/charts/{ticker}?period=...` stops 500ing when a response contains a non-finite float (nulls it instead of raising during ASGI JSON rendering) |

## Why this bug appeared ("newly introduced")

Reproduced live: `GET /api/charts/DELL?period=1y` -> HTTP 500. `backend.log`
(2026-08-18 09:25:02) shows the exact traceback:

```
ValueError('Out of range float values are not JSON compliant: nan')
when serializing dict item 'Open'
when serializing list item 250
```

Root cause chain:
- `backend/tools/yfinance_tool.py::get_price_history` returns
  `df.reset_index().to_dict(orient="records")` with **zero NaN sanitisation** --
  whatever yfinance/pandas hands back goes straight into the response dict.
- `backend/api/charts.py`'s router had no `default_response_class`, so Starlette's
  default `JSONResponse.render` runs, which hardcodes `json.dumps(..., allow_nan=False)`
  -- any non-finite float in the payload is a guaranteed 500 that the route's own
  `try/except` cannot catch (the failure happens during ASGI rendering, AFTER the
  route already returned its dict).
- This is the **identical** defect class fixed for `GET /api/signals/{ticker}` in
  phase-80.1 (`backend/api/_json_safe.py`, `NaNSafeJSONResponse`) -- that fix was
  applied to the `signals` router only ("deliberately NOT applied app-wide"), and
  `charts.py` was simply never given the same treatment.
- It "newly" surfaced now because yfinance returned a NaN `Open` for DELL near the
  end of the series (list item 250 of ~252 for period=1y, and item 62 of ~63 for
  period=3mo -- i.e. the most recent bar(s) in both periods), most likely a
  partial/incomplete intraday bar. Confirmed at 22:00-22:01 on 2026-08-17 the same
  endpoint returned HTTP 200 for the same ticker -- so the trigger is fresh upstream
  data landing with a gap, not a code change to charts.py itself (git log on that
  file shows nothing since squashed early history). The latent bug (no sanitiser on
  this router) has been there all along; today's NaN in fresh DELL data is what
  exposed it.

Scope note: this is the same **display-layer-only** caveat `_json_safe.py`'s own
docstring states for signals.py -- `yfinance_tool.get_price_history` is NOT shared
with the Layer-1 trading orchestrator (charts are a UI-only read path), so this fix
does not touch any trading input. No further "step 80.27"-style pipeline follow-up
is implied by this fix.

## Verification once restarted

```
curl -s http://127.0.0.1:8000/api/charts/DELL?period=1y -w '\nHTTP_STATUS:%{http_code}\n'
# expect 200, and the 'Open' field for the NaN row to read null instead of 500ing
curl -s http://127.0.0.1:8000/api/charts/DELL?period=3mo -w '\nHTTP_STATUS:%{http_code}\n'
# expect 200 (same NaN-bearing tail row)
ps -o pid,lstart -p <new pid>   # expect a pid AFTER this commit, started after 08:26:53
```

A pid that has not changed means the restart did not take.
