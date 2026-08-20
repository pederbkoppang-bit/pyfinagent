# Research Brief -- step 86.120

**Topic:** Claude Code CLI / Claude Max weekly usage-limit exhaustion detection, and
circuit-breaker cooldown/backoff design for gating retries after a *known-exhausted*
rate limit.
**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-18. **Researcher:** Layer-3 researcher (Workflow rail).

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 31,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.120.md",
  "gate_passed": true
}
```

Flipped to `COMPLETE` as the final act of the session. `coverage` is informational:
the caller declared this step NOT audit-class, so `coverage.dry` does not gate.
Round 2 (the recency pass) still yielded 4 new findings, so `dry` is honestly `false`.

---

## Search-query composition (three-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | current-year frontier | (pending) |
| 2 | last-2-year window | (pending) |
| 3 | year-less canonical | (pending) |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| (pending) | | | | | |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| (pending) | | |

---

## Recency scan (2024-2026)

(pending)

---

## Key findings

(pending)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| (pending) | | | |

---

## Application to pyfinagent

(pending)

---

## Research Gate Checklist

- [ ] >=5 authoritative external sources READ IN FULL via WebFetch
- [ ] 10+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Full pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim

---

# ROUND 1 -- external sources read in full (1-4)

## S1. Claude Code error reference (OFFICIAL DOC, tier 2)
`https://code.claude.com/docs/en/errors` -- accessed 2026-08-18, WebFetch full page.

Verbatim limit strings the CLI emits:

```
You've hit your session limit - resets 3:45pm
You've hit your weekly limit - resets Mon 12:00am
You've hit your Opus limit - resets 3:45pm
API Error: Server is temporarily limiting requests (not your usage limit)
API Error: Request rejected (429) - this may be a temporary capacity issue. If it persists, check https://status.claude.com.
Credit balance is too low
API Error: Usage credits required for 1M context - run /usage-credits to turn them on, or /model to switch to standard context
```

Load-bearing findings:
- **Three DISTINCT limit scopes**, and they are not interchangeable: *session* (5-hour),
  *weekly*, and *Opus-only*. "Session and weekly limits are shared across all models;
  the Opus limit applies only to Opus requests."
- **The reset time is embedded in the human-readable message string, not a field.**
  Session -> `resets HH:MMam/pm`; weekly -> `resets Day HH:MMam/pm`; spend -> full
  `YYYY-MM-DD HH:MM UTC`. A parser gets a *local wall-clock* string with no date for
  the session case and no year for the weekly case.
- **No documented programmatic detection in headless mode.** The doc "does not describe
  JSON output shapes for usage limit errors in `--output-format json` mode", no
  `stop_reason` value, no `subtype` code, no stderr prefix. **This is the single most
  important negative finding for 86.120: text-pattern matching is not a shortcut, it is
  the ONLY documented surface.**
- **Quota exhaustion is NOT retried by default**, and the doc explicitly separates a
  throttle from a quota: "Temporary 429 throttles, but not a gateway's spend-limit
  `429`, which isn't a throttle."
- `CLAUDE_CODE_RETRY_WATCHDOG=1` "enables indefinite retry for `429` and `529` capacity
  errors in unattended sessions" -- i.e. an env var that would make an away-ops run retry
  a *capacity* error forever. Relevant as a thing to NOT set for a quota error.
- `/usage` is the only documented query surface for remaining allowance; it is a slash
  command (interactive), not a `--print` flag.

## S2. Anthropic API rate limits (OFFICIAL DOC, tier 2)
`https://platform.claude.com/docs/en/api/rate-limits` -- accessed 2026-08-18, WebFetch full page.
(Reached via a 301 from `docs.claude.com/en/api/rate-limits`; the redirect host is recorded
because the old URL is what most secondary sources still cite.)

- On the DIRECT API rail, exhaustion is fully machine-readable: "If you exceed any of the
  rate limits you will get a **429 error** describing which rate limit was exceeded, along
  with a **`retry-after`** header indicating how long to wait."
- Header set (verbatim): `retry-after` = "The number of seconds to wait until you can retry
  the request. **Earlier retries will fail.**"; `anthropic-ratelimit-requests-remaining`;
  `anthropic-ratelimit-requests-reset` ("provided in **RFC 3339** format");
  `anthropic-ratelimit-tokens-remaining` / `-reset`; plus input/output token variants and
  `anthropic-priority-*`.
- "The API uses the **token bucket algorithm** ... capacity is **continuously replenished**
  up to your maximum limit, rather than being reset at fixed intervals."
- **The two rails are asymmetric and this is the crux of the step.** The direct API gives a
  `retry-after` in seconds and an RFC-3339 reset. The Claude Code CLI subscription rail
  (S1) gives a **prose sentence with a wall-clock time**. Any design that assumes the CLI
  rail exposes API-style headers is wrong; the CLI is a subprocess and the headers never
  reach the caller.
- Spend limits are a separate axis from rate limits ("API usage pauses until the next
  month") -- three orthogonal exhaustion classes total (rate / spend / subscription quota).

## S3. AWS -- Exponential Backoff And Jitter (OFFICIAL ENG BLOG, tier 2/3)
`https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/` -- accessed
2026-08-18, WebFetch full page.

Formulas verbatim:

```
No jitter:          sleep = min(cap, base * 2^attempt)
Full jitter:        sleep = random(0, min(cap, base * 2^attempt))
Equal jitter:       sleep = min(cap, base*2^attempt)/2 + random(0, min(cap, base*2^attempt)/2)
Decorrelated jitter: sleep = min(cap, random(base, sleep * 3))
```

- Plain capped exponential backoff still clusters: "there are still clusters of calls.
  Instead of reducing the number of clients competing in every round, we've just introduced
  times when no client is competing." Total work grows as **N^2** with N contending clients.
- Full Jitter "reduced our call count by more than half" vs no jitter and completes faster
  than Equal Jitter. Recommendation: **Full Jitter**.
- **Scope caveat for 86.120:** this paper is about *contention* among many clients. A
  single-host, single-cycle rail exhausting a weekly subscription quota is NOT a contention
  problem -- backoff cannot manufacture quota. Jitter matters here only for the *probe*
  cadence (avoiding a synchronized probe storm across cycles/processes), not for the
  decision to stop.

## S4. Azure Architecture Center -- Circuit Breaker pattern (OFFICIAL DOC, tier 2)
`https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker` -- accessed
2026-08-18, WebFetch full page. (This is the doc `claude_code_client.py:116` already cites
as "Azure circuit-breaker per-window semantics".)

- Three states verbatim: **Closed** (count failures; over threshold within a period -> Open,
  start time-out timer), **Open** ("The request from the application fails immediately and an
  exception is returned"), **Half-Open** ("A limited number of requests ... are allowed to
  pass through ... If any request fails, the circuit breaker ... reverts to the **Open**
  state. It restarts the time-out timer").
- **"Accelerated circuit breaking"** -- the exact named pattern this step needs, verbatim:
  "Sometimes a failure response can contain **enough information for the circuit breaker to
  trip immediately and stay tripped for a minimum amount of time**. For example, the error
  response from a shared resource that's overloaded can indicate that the application should
  instead try again in a few minutes, instead of immediately retrying."
- **"Types of exceptions"** -- verbatim: "A circuit breaker might be able to examine the types
  of exceptions that occur and **adjust its strategy based on the nature of these exceptions**."
- **"Recoverability"** -- "configure the circuit breaker to match the likely recovery pattern
  of the operation that it protects ... a circuit breaker can fluctuate and reduce the response
  times of applications if it switches from the **Open** state to the **Half-Open** state too
  quickly."
- Retry-vs-breaker separation, verbatim: "the retry logic should be **sensitive to any
  exceptions that the circuit breaker returns and stop retry attempts if the circuit breaker
  indicates that a fault isn't transient**."
- **"Resource differentiation"** -- one breaker per resource type is wrong when multiple
  independent providers sit behind it (already cited at `autonomous_loop.py:2645`).
- **Manual override** is called out as a requirement when "the recovery time for a failing
  operation is extremely variable".
- **Increasing time-out** is explicitly sanctioned: "you can apply an increasing time-out
  timer to a circuit breaker ... a few seconds initially. If the failure isn't resolved,
  increase the time-out to a few minutes."
- Open state "can return a default value that's meaningful to the application" -- the
  documented basis for pyfinagent's fail-forward/degraded path.

## S5. Resilient Microservices: A Systematic Review of Recovery Patterns (PREPRINT, tier 1)
`https://arxiv.org/html/2512.16959` -- accessed 2026-08-18, WebFetch of the arXiv NATIVE HTML
(per research-gate.md the `/pdf/` URL was deliberately not fetched).

- Head-to-head simulation (Section VI-H), verbatim: "Exponential backoff *without jitter*
  exhibited P99 = 2600 ms and a 17% error rate due to retry amplification. Backoff *with
  jitter* reduced P99 to 1400 ms and errors to 6%. **Combining bounded retries with a circuit
  breaker yielded the best results: P99 = 1100 ms and a 3% error rate.**"
- Theme T1 "Failure-Mode-Pattern Fit": "Pattern effectiveness depends on failure semantics;
  **over-tight circuit-breaker thresholds reduce throughput**."
- Theme T3: "Naive backoff without jitter causes retry storms; adding jitter and budgets
  smooths recovery."
- Timeout guidance: "Tune slightly above p95 and adjust from real metrics."
- **Honest gap (recorded, not papered over):** this review "does not explicitly discuss
  rate-limit-aware circuit breaking" and gives "no taxonomy of error types (transient vs.
  permanent) for differential breaking behavior", and provides **no algorithm for cooldown
  duration**. So the peer-reviewed layer supports *breaker + bounded retry + jitter* but does
  NOT supply the quota-specific rule; that has to come from Azure's "accelerated circuit
  breaking" (S4) plus the CLI's own signal (S1/S6).

## S6. Run Claude Code programmatically / headless (OFFICIAL DOC, tier 2)
`https://code.claude.com/docs/en/headless` -- accessed 2026-08-18, WebFetch full page.

- Exit-code contract, verbatim: "Claude Code exits with code 0 on success and a **non-zero
  code when the run fails**, so your scripts can branch on the exit status." And critically:
  "**When a failure happens inside the run, such as missing authentication, Claude Code prints
  the failure as the result on stdout.**" -- i.e. the failure text lands in `result`, the same
  field a successful answer uses.
- SIGTERM exits with code 143. Piped stdin capped at 10MB.
- **The only documented machine-readable rate-limit classifier is in `stream-json`, not
  `json`.** The `system/api_retry` event carries: `attempt`, `max_retries`,
  `retry_delay_ms` (integer, "milliseconds until the next attempt"), `error_status`
  ("HTTP status code, or `null`"), and `error` -- an enumerated **error category**:
  `authentication_failed`, `oauth_org_not_allowed`, `billing_error`, **`rate_limit`**,
  `overloaded`, `invalid_request`, `model_not_found`, `server_error`, `max_output_tokens`,
  `unknown`.
- **Design consequence:** `--output-format json` (what `claude_code_invoke` uses,
  `claude_code_client.py:373-374`) does NOT expose `error`/`retry_delay_ms`. Getting the
  enumerated classifier would require switching the rail to `stream-json` and reassembling
  the final `result` message -- a real option, and a real cost, that the contract should
  weigh explicitly rather than assume.
- Note `--bare` is NOT usable here: "In bare mode, Claude Code never reads OAuth credentials
  or the system keychain" -- it would break the Max-subscription rail. `claude_code_client.py:367-369`
  already records this.

---

# MEASURED: the one real limit-exhaustion envelope in this repo

Corpus: `handoff/away_ops/session_*.json`, **85 files**. Exactly **one** carries a rate-limit
status. Parsed in full (`handoff/away_ops/session_pm_20260707T200007Z.json`):

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": true,
  "api_error_status": 429,
  "stop_reason": "stop_sequence",
  "num_turns": 1,
  "total_cost_usd": 0,
  "duration_ms": 735,
  "result": "You've hit your session limit · resets 1am (Europe/Oslo)"
}
```

Corresponding away-ops log lines (`handoff/away_ops/*.log`, 2026-07-07):

```
[2026-07-07T20:00:10Z] [pm] claude exited rc=1 (crash or limit)
[2026-07-07T20:00:10Z] [pm] COST total_cost_usd=0 out=session_pm_20260707T200007Z.json
[2026-07-07T20:00:10Z] [pm] LIMIT_HIT -- Agent SDK credit or session limit reached; session output truncated
```

**M1. `subtype` is worthless as a failure signal on this rail.** Across all 85 sessions the
ONLY subtype value observed is `"success"` (67 `"subtype":"success"` + 13 `"subtype": "success"`
= 80 spaced/tight variants; the remainder carry no subtype key). That includes all **36**
sessions with `api_error_status 401` AND the one with `429`. `run_away_session.sh:207-208`
already encodes this ("never key on subtype -- 401 sessions carry subtype `success`"); this
measurement generalises it from the 401 case to the 429 case with an 85-file denominator.

**M2. `api_error_status` is the machine-readable signal, and it is UNDOCUMENTED.** It appears
in neither S1 nor S6. It is live-observed here and already load-bearing for the 401 latch.
Readers today (`grep -rn api_error_status`): `run_away_session.sh:150,211`,
`away_ops/auth_state.py:75`, `away_ops/healthcheck.sh:88`, `away_ops/redact_secrets.py:24,94,130-140`,
`backend/tests/test_phase_85_3_auth_latch_freshness.py:26-27`.
**ZERO readers in `backend/agents/` or `backend/services/`** -- the trading rail does not look
at it at all.

**M3. `is_error: true` IS correct here.** The `claude_code_client.py:358` docstring warns
`is_error` "has known mis-flag history"; on this envelope it is right. It is still weaker than
`api_error_status` because it does not distinguish 401 from 429 from a tool error.

**M4. The limit case exits rc=1, so `claude_code_invoke` raises at the RETURNCODE gate, not the
subtype gate.** `claude_code_client.py:439` (`if completed.returncode != 0`) fires first; the
`subtype != "success"` check at `:476` is never reached. **Good news: no fabrication risk** --
the limit sentence cannot be returned as an analysis. (I hypothesised the opposite and the
measurement refuted it.)

**M5. The limit signal DOES survive into `_RAIL_GUARD.last_error` -- barely, and by accident.**
`:447` builds `_out_snip = stdout.strip()[:300]`; `:456` embeds `_out_snip[:150]` in the
`ClaudeCodeError`; `:771` passes `str(exc)` to `_rail_guard_record_failure`. Measured on the
real 768-byte envelope, `stdout[:150]` is:

```
{"type":"result","subtype":"success","is_error":true,"api_error_status":429,"duration_ms":735,"duration_api_ms":0,"num_turns":1,"result":"You've hit y
```

So `api_error_status":429` survives the 150-char cut, but the human sentence is severed at
`"You've hit y`. **This is positional luck, not a contract** -- JSON key order is not
guaranteed, and one extra early key pushes the status out of the window. Any detector built on
`str(exc)` inherits that fragility. (Caveat disclosed: this file predates the phase-86.79
redactor, so it is raw CLI stdout, not a Python re-serialisation.)

**M6. There is ZERO local evidence of a WEEKLY limit.** 1 of 85 sessions hit any limit, and it
was the 5-hour *session* limit. The weekly string shape is known only from S1. A design must
not assume the weekly message is byte-identical in structure to the session one beyond what S1
states, and must not claim the weekly case is reproduced locally.

**M7. The live message carries an IANA timezone the official doc's examples omit.** Doc (S1):
`resets 3:45pm`. Live: `resets 1am (Europe/Oslo)`. Parseable -- but there is **no DATE** on the
session form and **no year** on the weekly form, so reconstructing an absolute reset instant
requires host-local "next occurrence of" arithmetic. Per auto-memory
`reference_stat_SB_prints_local_time` and `reference_fixed_offset_tz_fixture_is_hour_dependent`,
this is exactly the class of parse that has produced wrong answers in this repo before.

**M8. Secondary discriminators are strong and cheap.** `total_cost_usd: 0`, `num_turns: 1`,
`duration_ms: 735`. A real rail answer costs > 0 and takes 60-90s (`claude_code_client.py:584-591`
records 88.9s live). A near-instant, zero-cost, one-turn failure is a distinctive fingerprint --
but it is shared with the 401 case, so it classifies "cheap structural failure", not "quota".

---

# Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/claude_code_client.py` | 821 (read in full) | Rail guard + breaker + `claude_code_invoke` | LIVE; no quota classification, no cooldown |
| `backend/services/autonomous_loop.py` | 3981 (regions :455-514, :1947-1990, :2600-2760 read) | Cycle start, probe, `_rail_dead_reason`, `_select_lite_analyzer` | LIVE; resets breaker every cycle |
| `scripts/away_ops/run_away_session.sh` | 248 (read :125-248 + full grep) | Away-session wrapper; 401 latch; `LIMIT_HIT` text scan | LIVE; latch exists for 401 ONLY |
| `backend/agents/llm_client.py` | 2456 (grep) | `make_client` routing + routing-breach guards | LIVE; two guards, `:2211-2218` and `:2327-2335` |
| `backend/config/settings.py` | (grep) | `claude_rail_breaker_threshold` `:180`; fail-forward flag `:200` | LIVE; threshold default 20, no cooldown knob exists |
| `handoff/away_ops/session_pm_20260707T200007Z.json` | 768 bytes (parsed) | The ONE real 429 envelope | EVIDENCE |
| `scripts/away_ops/auth_state.py` | (grep `:75`) | 401 latch state reader | LIVE; the reusable latch idiom |
| `scripts/away_ops/healthcheck.sh` | (grep `:88`) | Scans newest session JSON for 401 | LIVE |
| `scripts/away_ops/redact_secrets.py` | (grep `:24,:94,:130-140`) | Redactor; explicitly preserves `api_error_status` | LIVE; its self-test asserts the numeric status is untouched |

## The five structural gaps (each anchored)

**G1 -- The breaker has NO cooldown; it has a per-cycle amnesia instead.**
`_RAIL_GUARD` is module-global in-process state (`claude_code_client.py:102`), and
`rail_guard_reset()` (`:115-122`) replaces it with a **fresh `_RailGuardState()`** -- called
unconditionally at cycle start from `autonomous_loop.py:476`, BEFORE the probe. There is no
Open->Half-Open timer, no `time.monotonic()` deadline, no persistence to disk or BQ. Repo-wide
grep for `rail_cooldown|rail_backoff|breaker_cooldown` returns **zero hits**. Against Azure
(S4) this is the "switches from the Open state to the Half-Open state too quickly" antipattern
in its limiting form: the cooldown is effectively **zero**, and the reset is by fiat rather
than by evidence of recovery.

**G2 -- Consequence of G1: a weekly exhaustion costs 20 doomed subprocess calls per cycle,
forever.** `_rail_breaker_threshold()` returns `claude_rail_breaker_threshold` default **20**
(`claude_code_client.py:110`, `settings.py:180`). The breaker trips at 20 consecutive failures
(`:182`) and skips the rest of *that cycle* (`:150-162`). Next cycle: counter back to 0, 20
more real `subprocess.run` spawns. This is a smaller version of exactly the incident the guard
was built for -- `claude_code_client.py:84-86` records "~162 doomed 5s subprocess calls per
cycle for three weeks with zero pages". The guard bounded the *within-cycle* blast radius and
left the *across-cycle* one untouched.

**G3 -- The health probe cannot see a quota exhaustion.** `claude_code_health_probe`
(`:494-539`) runs `claude auth status`, which is explicitly "free, token-less" (`:495-496`) and
returns ok on exit-0 + `loggedIn`. A quota-exhausted Max account is still **authenticated** --
auth is healthy, quota is gone. So the probe gate (`autonomous_loop.py:477-486`) engages for
credential death and **never** for quota death. The one gate that runs before any tokens are
spent is blind to the exact failure this step targets.

**G4 -- The rail discards the machine-readable signal it already receives.** On rc!=0
(`claude_code_client.py:439`), the envelope is **never `json.loads`-ed** -- parsing happens at
`:465`, after the returncode gate. `api_error_status` therefore survives only as an accidental
substring of a 150-char truncation (M5). The 401 latch in `run_away_session.sh:211` proves the
idiom works; the backend rail simply does not use it.

**G5 -- Two different detectors, two different vocabularies, neither covering weekly.**
`run_away_session.sh:242` scans for `usage limit|session limit|credit.*(exhaust|limit)|out of
credit`. Measured against the corpus this **does** match the real 429 file. But note what it
does NOT contain: **`weekly`**. Against S1's documented weekly string `"You've hit your weekly
limit - resets Mon 12:00am"`, the regex matches only via the bare word `limit` inside
`credit.*(exhaust|limit)` -- which requires the literal `credit` first, and there is none. So
**a WEEKLY limit hit would log as generic `rc=$rc (crash or limit)` and never emit `LIMIT_HIT`**
(`:203`). Meanwhile the backend rail has no text detector at all. The `Opus limit` string is
likewise unmatched. This is a concrete, falsifiable defect independent of any new design.

---

# Recency scan (2024-2026) -- MANDATORY SECTION

Searched explicitly for last-2-year material (queries listed below). **Result: 4 findings that
materially change or constrain the design; none supersede the canonical Azure/AWS patterns,
which remain correct but insufficient on their own.**

1. **Weekly limits are a 2025-08-28 addition** stacked on top of the pre-existing rolling
   5-hour window (snippet: apidog, explainx, Slashdot 2025-07-29). This is why the codebase has
   *session*-limit handling and no *weekly* handling: `run_away_session.sh`'s detector predates
   the weekly cap's existence. G5 is a dating artefact, not an oversight.
2. **Anthropic explicitly targeted unattended 24/7 automation** with the weekly cap (snippet:
   apidog / Slashdot). pyfinagent's away-ops + autonomous loop is precisely that shape, so the
   weekly ceiling is a *design constraint on this project*, not a remote edge case.
3. **`CLAUDE_CODE_RETRY_WATCHDOG=1`** (S1, current docs) enables *indefinite* retry for 429/529
   in unattended sessions. It is a 2026-era knob that would make G2 dramatically worse if set;
   worth an explicit "do not set" in the contract.
4. **The `system/api_retry` stream event with its enumerated `error` categories** (S6) is the
   current-docs classifier, including a literal `rate_limit` value and `retry_delay_ms`.
   This did not exist in earlier CLI docs and is the strongest *documented* alternative to text
   matching -- at the cost of moving the rail to `--output-format stream-json`.

No 2024-2026 source found that contradicts Azure "accelerated circuit breaking" or AWS Full
Jitter. The 2025-12 systematic review (S5) reinforces breaker+bounded-retry+jitter empirically
while explicitly leaving the quota-specific case unaddressed.

## Queries run (three-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | `Claude Code weekly usage limit reached detection CLI error message` |
| 2 | year-less canonical | `exponential backoff jitter circuit breaker half-open state design` |
| 3 | current-year frontier | `circuit breaker pattern cooldown backoff rate limit 429 retry-after 2026` |
| 4 | last-2-year window | `Claude Max subscription weekly rate limit exhaustion 2025 unattended agent detect reset time` |

---

# Source tables

## Read in full (6; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://code.claude.com/docs/en/errors | 2026-08-18 | official doc | WebFetch, full page | Verbatim session/weekly/Opus limit strings; NO documented JSON detection in headless mode; quota != throttle |
| 2 | https://platform.claude.com/docs/en/api/rate-limits | 2026-08-18 | official doc | WebFetch, full page (via 301) | `retry-after` + `anthropic-ratelimit-*-reset` RFC 3339 exist on the API rail ONLY; token-bucket, continuous replenishment |
| 3 | https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ | 2026-08-18 | official eng blog | WebFetch, full page | Four jitter formulas verbatim; Full Jitter recommended; N^2 work under contention |
| 4 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-08-18 | official doc | WebFetch, full page | "Accelerated circuit breaking"; "types of exceptions"; Open->Half-Open too quickly = antipattern; manual override |
| 5 | https://arxiv.org/html/2512.16959 | 2026-08-18 | preprint (arXiv native HTML) | WebFetch, full page | Breaker + bounded retry + jitter = P99 1100ms / 3% err, beating jitter-only (1400ms/6%) and no-jitter (2600ms/17%); NO quota taxonomy |
| 6 | https://code.claude.com/docs/en/headless | 2026-08-18 | official doc | WebFetch, full page | `system/api_retry` carries enumerated `error` incl. `rate_limit` + `retry_delay_ms` -- but only in `stream-json`; failures print as `result` on stdout |

## Identified but snippet-only (31; does NOT count toward the gate)

| # | URL | Kind | Why not fetched in full |
|---|-----|------|-------------------------|
| 1 | https://www.truefoundry.com/blog/claude-code-limits-explained | vendor blog | Secondary restatement of S1 |
| 2 | https://arte.itlibra.com/en/articles/claude-code-usage-limit-reached | blog | Community-tier, superseded by S1 |
| 3 | https://github.com/anthropics/claude-code/issues/41084 | issue tracker | Phantom-usage bug report, not a design source |
| 4 | https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages | official support | Overlaps S1; consumer-facing framing |
| 5 | https://www.enqurious.com/blog/why-claude-says-usage-limit-reached-and-how-to-fix-it | blog | Community-tier |
| 6 | https://medium.com/@dev_tips/claude-code-keeps-hitting-its-limit-heres-how-to-fix-it-ad625bbaf5b1 | blog | Community-tier |
| 7 | https://developer.apple.com/forums/thread/809155 | forum | Xcode-specific, out of scope |
| 8 | https://truto.one/blog/best-practices-for-handling-api-rate-limits-and-retries-across-multiple-third-party-apis/ | industry blog | Header-normalisation angle; covered by S2 |
| 9 | https://www.sitepoint.com/claude-api-429-error-handling-python/ | blog | Direct-API path, not the CLI rail |
| 10 | https://zylos.ai/research/2026-02-20-graceful-degradation-ai-agent-systems/ | industry research | Degradation framing; covered by S4 Open-state default value |
| 11 | https://www.buildmvpfast.com/blog/agent-timeout-circuit-breaker-patterns-runaway-ai-workflows-2026 | blog | Low authority |
| 12 | https://apiscout.dev/blog/api-resilience-circuit-breakers-retries-bulkheads-2026 | blog | Low authority |
| 13 | https://ofox.ai/blog/ai-api-error-handling-troubleshooting-guide-2026/ | blog | Low authority |
| 14 | https://apistatuscheck.com/blog/api-rate-limiting-429-errors-production | blog | Low authority |
| 15 | https://1xapi.com/blog/resilient-api-circuit-breaker-bulkhead-retry-nodejs-2026 | blog | Node-specific |
| 16 | https://github.com/App-vNext/Polly/issues/1895 | issue tracker | Polly `BreakDurationGenerator` = prior art for exponential open-duration; noted, not load-bearing |
| 17 | https://medium.com/@rafaeljcamara/downstream-resiliency-the-timeout-retry-and-circuit-breaker-patterns-d8c02dc72c40 | blog | Restates S4 |
| 18 | https://dev.to/rafaeljcamara/downstream-resiliency-the-timeout-retry-and-circuit-breaker-patterns-2bej | blog | Duplicate of #17 |
| 19 | https://zuplo.com/learning-center/api-gateway-resilience-fault-tolerance | vendor doc | Gateway layer, not applicable to a subprocess rail |
| 20 | https://repost.aws/knowledge-center/bedrock-retry-exponential-backoff-api | official KB | Bedrock-specific |
| 21 | https://arxiv.org/pdf/2602.00887 | preprint | EffGen -- small-LM agents, off-topic |
| 22 | https://aiopsschool.com/blog/exponential-backoff/ | blog | Restates S3 |
| 23 | https://github.com/IBM/mcp-context-forge/issues/258 | issue tracker | Feature request, no findings |
| 24 | https://apidog.com/blog/weekly-rate-limits-claude-pro-max-guide/ | blog | Recency-scan evidence for the 2025-08-28 weekly-cap date |
| 25 | https://www.explainx.ai/blog/claude-usage-limits-2026-timeline-explained | blog | Recency-scan timeline corroboration |
| 26 | https://github.com/anthropics/claude-code/issues/9424 | issue tracker | User impact of weekly caps |
| 27 | https://portkey.ai/blog/claude-code-limits/ | vendor blog | Secondary |
| 28 | https://developers.slashdot.org/story/25/07/29/0156200/claude-code-users-hit-with-weekly-rate-limits | news | Dates the weekly-cap announcement |
| 29 | https://github.com/anthropics/claude-code/issues/9094 | issue tracker | 2025-09-29 limit-change meta-report |
| 30 | https://tech.yahoo.com/ai/articles/anthropic-were-glad-claude-code-164658123.html | news | Low weight |
| 31 | https://tech.yahoo.com/ai/articles/anthropic-putting-limit-claude-ai-092932036.html | news | Low weight |

**URL total: 6 read in full + 31 snippet-only = 37 unique URLs.**

---

# Application to pyfinagent (external findings -> file:line anchors)

**A1. Detect on `api_error_status`, corroborate with text -- not the reverse.** Parse the
envelope on the rc!=0 branch too (`claude_code_client.py:439-457` currently raises without ever
calling `json.loads`, which happens later at `:465`). `api_error_status == 429` is the only
numeric signal (M2) and the 401 latch (`run_away_session.sh:211`) is the proven in-repo idiom.
Text matching stays as a SECOND signal to separate *quota* (`hit your ... limit`) from
*capacity* (S1: "Server is temporarily limiting requests (not your usage limit)") -- because
both are 429 and S1/S6 say they must be treated differently. Do not rely on `str(exc)`
truncation (M5); parse the envelope.

**A2. This is Azure's "accelerated circuit breaking" almost verbatim (S4).** A quota message
carries "enough information for the circuit breaker to trip immediately and stay tripped for a
minimum amount of time". So: trip at **N=1** on a confirmed quota signal, not at the 20-failure
threshold (`claude_code_client.py:110`, `settings.py:180`) -- 19 of those 20 calls are known-
doomed. Keep 20 for *unclassified* failures; S4 "types of exceptions" explicitly sanctions a
per-exception-class strategy, and S5 warns "over-tight circuit-breaker thresholds reduce
throughput", so the tight threshold must apply ONLY to the classified quota case.

**A3. The cooldown should be derived from the reset time, with backoff as the FALLBACK.** S2
establishes the principle ("Earlier retries will fail"). The CLI gives a wall-clock string with
no date (M7), so: parse -> next-occurrence-in-named-TZ -> use as the Open-state deadline. When
parsing fails (and it must be assumed to fail sometimes), fall back to capped exponential
backoff with **Full Jitter** (S3), which is also the right shape for the *probe* cadence.
Explicitly NOT justified by the literature: retrying faster in the hope of quota returning --
S3's contention model does not apply (backoff cannot manufacture quota), and S5 confirms the
win comes from breaker+bounded retry, not from retry alone.

**A4. Cooldown state MUST outlive the cycle, or it does nothing.** G1/G2: `rail_guard_reset()`
(`claude_code_client.py:115-122`) is called unconditionally at `autonomous_loop.py:476` and
wipes everything. A weekly reset is ~7 days; the cycle is intra-day. Any quota latch therefore
needs cross-cycle (and ideally cross-process, since away-ops is a separate process) persistence
-- the `auth_page_state.json` latch (`run_away_session.sh:139-158`, `away_ops/auth_state.py:75`)
is the existing, working, in-repo precedent for exactly this, including automatic recovery on a
successful probe. Reusing that shape is cheaper and better-evidenced than inventing one.

**A5. Do not route the quota case through the existing probe.** G3: `claude auth status`
(`:494-539`) is token-free and passes on a quota-exhausted account. If a half-open probe is
wanted it must be a real (cheap) inference call, which costs quota -- so it should be rate-
limited by the cooldown itself, per S4 "Failed operations testing".

**A6. Fail-forward already exists and is the correct Open-state behaviour.** S4: the Open state
"can return a default value that's meaningful to the application".
`_select_lite_analyzer` (`autonomous_loop.py:2607-2637`) + `_run_failforward_analysis` (`:2720`)
already do this behind `paper_rail_failforward_enabled`, reading rail state through the strict
reader `_rail_dead_reason()` (`:2640-2658`). A quota latch that sets the same
`rail_skipped`/`breaker_tripped` state gets fail-forward **for free** with no new wiring. Note
`settings.py:200` records the flag is DARK and Vertex calls are METERED -- promotion is an
operator decision, and per the standing away-ops `$0 metered` constraint the contract must not
assume it is on.

**A7. Fix G5 regardless.** `run_away_session.sh:242`'s regex has no `weekly` and no `Opus`
alternative, so the two limit classes introduced in 2025-08 log as generic
`"claude exited rc=$rc (crash or limit)"` (`:203`). This is a one-line, independently
verifiable defect with a documented target string (S1) and should be queued as its own step if
out of scope here, per the standing "queue discovered defects in the masterplan" rule.

**A8. Alerting is already correct in shape -- do not duplicate it.** The page fires on the
closed->open TRANSITION only (`claude_code_client.py:174-190`, Fowler/PagerDuty
alert-on-transition) with a one-shot latch, and the probe-failure path deliberately consumes
that latch (`:131-135`) to avoid double-paging. A quota trip should reuse the same latch, with a
distinct `error_type`, so a quota incident pages once per incident and not once per cycle.

---

# Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6** (4 official docs, 1 official eng blog, 1 arXiv preprint; zero community-tier in the read-in-full set)
- [x] 10+ unique URLs total -- **37** (6 full + 31 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- 4 findings, section present above
- [x] Full pages read (not abstracts); arXiv fetched via native `/html/`, never `/pdf/`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (all 4 named files, plus 5 more reached from them)
- [x] Contradictions noted -- S5 explicitly does NOT cover the quota case (recorded as a gap, not smoothed over); my own M4/M5 hypotheses were refuted by measurement and the refutation is recorded
- [x] Claims cited per-claim with URL or file:line
- [ ] GAP: the WEEKLY limit envelope is UNOBSERVED locally (M6). Only the 5-hour session form
      is measured. Any weekly-specific field claim would be inference, not measurement.
- [ ] GAP: `api_error_status` is undocumented (M2). It is live-observed and already trusted by
      the 401 latch, but it carries no compatibility guarantee.
