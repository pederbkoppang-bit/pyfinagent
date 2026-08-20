---
name: quota-exhaustion-86-120
description: Claude Code CLI quota-exhaustion detection + rail breaker cooldown -- subtype is useless (85/85 "success"), api_error_status is the real signal but undocumented and unread by the backend, and the breaker has ZERO cooldown
metadata:
  type: project
---

Step 86.120 research. Measured against the 85-file `handoff/away_ops/session_*.json` corpus
and the four in-scope source files.

**`subtype` cannot detect ANY failure on this rail.** Across all 85 away sessions the only
subtype value observed is `"success"` -- including all **36** with `api_error_status 401` AND
the single one with `429`. `run_away_session.sh:207` already said this for 401; the 429 case
generalises it. Anything keyed on subtype is dead code for failure detection.

**`api_error_status` is the machine-readable signal and it is UNDOCUMENTED** -- absent from
both `code.claude.com/docs/en/errors` and `/headless`. Live-observed, and already load-bearing
for the 401 latch (`run_away_session.sh:150,211`; `away_ops/auth_state.py:75`;
`healthcheck.sh:88`; `redact_secrets.py` explicitly preserves it). **ZERO readers in
`backend/agents/` or `backend/services/`** -- the trading rail throws it away, because
`claude_code_invoke` raises at the returncode gate (`claude_code_client.py:439`) *before*
`json.loads` at `:465`, so on rc!=0 the envelope is never parsed at all.

**The ONE real limit envelope** (`session_pm_20260707T200007Z.json`, rc=1, 768 bytes):
`subtype:"success"`, `is_error:true`, `api_error_status:429`, `stop_reason:"stop_sequence"`,
`num_turns:1`, `total_cost_usd:0`, `duration_ms:735`,
`result:"You've hit your session limit · resets 1am (Europe/Oslo)"`.
Note the live string carries an **IANA timezone** the official doc's examples omit, and has
**no date** -- reconstructing an absolute reset needs next-occurrence arithmetic.

**The breaker has NO cooldown -- it has per-cycle amnesia.** `_RAIL_GUARD` is in-process module
state (`claude_code_client.py:102`); `rail_guard_reset()` (`:115-122`) builds a **fresh**
`_RailGuardState()` and is called unconditionally at `autonomous_loop.py:476`. Repo-wide grep
for `rail_cooldown|rail_backoff|breaker_cooldown` = **zero hits**. So a 7-day weekly exhaustion
costs 20 doomed `subprocess.run` spawns *per cycle, forever* (threshold default 20,
`settings.py:180`). No Open->Half-Open timer exists; the reset is by fiat, not by evidence.

**The health probe is structurally blind to quota.** `claude_code_health_probe` (`:494-539`)
runs `claude auth status` -- "free, token-less". A quota-exhausted Max account is still
**authenticated**, so the probe passes and the probe gate never engages for quota.

**`run_away_session.sh:242`'s LIMIT_HIT regex has no `weekly` and no `Opus` alternative**
(`usage limit|session limit|credit.*(exhaust|limit)|out of credit`). Weekly caps shipped
2025-08-28, *after* that detector was written -- a dating artefact, not an oversight. A weekly
hit logs as generic `"claude exited rc=$rc (crash or limit)"` (`:203`). Independently
falsifiable defect.

**Two hypotheses of mine were REFUTED by measurement, both recorded in the brief:** (a) I
expected the limit sentence to pass the subtype gate and be returned as a fabricated analysis
-- it does not, rc=1 makes `:439` fire first; (b) I expected the 150-char truncation at `:456`
to sever `api_error_status` -- it does not, the key sits at byte 77. Measure before claiming.

**Why:** the step needs a cooldown/backoff design that gates retries after a *known-exhausted*
limit; the governing external pattern is Azure's **"accelerated circuit breaking"** (a failure
response carrying enough info to trip immediately and stay tripped), not AWS jitter -- backoff
cannot manufacture quota.

**How to apply:** detect on `api_error_status` + corroborating text (quota vs capacity: the
docs separate "hit your ... limit" from "Server is temporarily limiting requests (not your
usage limit)"); persist the latch cross-process using the existing `auth_page_state.json`
idiom, since anything in `_RAIL_GUARD` dies at the next cycle; reuse the fail-forward path
(`autonomous_loop.py:2607-2637`, `:2720`) which already reads rail state via `_rail_dead_reason()`.
See [[research-gate-discipline]].

Brief: `handoff/current/research_brief_86.120.md` (36.7KB, 6 sources read in full).
