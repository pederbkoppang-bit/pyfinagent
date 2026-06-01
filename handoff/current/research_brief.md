# Research Brief — phase-54.1: Cross-layer cron-health audit for the unattended away-week

**Tier:** moderate
**Step:** phase-54.1 (audit ALL cron jobs end-to-end; operator REMOTE 2026-06-01 → 2026-06-08, Slack-only)
**Author:** researcher subagent
**Date:** 2026-06-01

---

## 0. Bottom line

The two failed launchd crons (`autoresearch` exit 1, `ablation` exit 1) have a
**single shared root cause**: the launchd wrappers `set -a; . backend/.env` —
**shell-sourcing** a `.env` whose `PAPER_MARKETS` value was changed to the
JSON form `["US","EU","KR"]` on the 2026-06-01 multi-market go-live. Bash mangles
the unquoted bracket/quote value on `source`, so the env var that reaches the
process is non-JSON; pydantic-settings' `EnvSettingsSource` then `json.loads()`-es
it and raises `SettingsError: error parsing value for field "paper_markets"`
(`Expecting value: line 1 column 2 (char 1)`). The live backend is unaffected
because uvicorn lets pydantic read `.env` *natively* (the dotenv source, which
accepts comma form) — only **shell-sourced** crons break. **Fix is non-LLM,
non-destructive** (escape/quote the value in the cron path) but touches a
shared secrets file, so it is ESCALATED with a precise, low-risk patch, not forced.

`mas-harness` is **NOT a failure** — it is a `StartInterval 1800` job showing PID
`-` (idle between fires) with last-exit 0. Per launchd semantics a `-` PID + exit
0 means "loaded, finished cleanly, waiting for the next 30-min tick." Healthy.

The **APScheduler layer is fully healthy** — `/api/jobs/all` shows all 4 core +
7 phase-9 jobs ran today with `status=ok` and sane next-fires; `morning_digest`
delivered to Slack today 12:00 UTC. The digests are **template/data-only (no LLM,
$0)** — decisive for 54.2: the daily operator digest is NOT operator-gated and can
run freely all week.

The **single biggest away-week risk is the slack_bot process itself**: it runs
`python -m backend.slack_bot.app` with **no launchd plist** (PPID 1, orphaned),
so unlike backend/frontend/proxy it has **no auto-restart**. If it dies or the Mac
sleeps and the bot doesn't survive, every digest, the watchdog, and all 11
APScheduler jobs silently stop, and the operator goes blind. This is the
load-bearing finding for 54.2 (give the slack_bot a `KeepAlive` launchd plist +
an external dead-man's-switch heartbeat). `gate_passed: true`.

---

## 1. External sources — READ IN FULL (research-gate floor ≥5; 9 read)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://healthchecks.io/docs/monitoring_cron_jobs/ | 2026-06-01 | official doc (vendor) | WebFetch (full) | Canonical dead-man's-switch: job pings on completion; "when Healthchecks.io does not receive the HTTP request at the expected time, it notifies you." Catches machine-down, **daemon not running / invalid config**, non-zero exit, ran-too-long. Grace time = expected duration + buffer. |
| 2 | https://www.launchd.info/ | 2026-06-01 | authoritative ref (de-facto launchd manual) | WebFetch (full) | `launchctl list`: col1 PID — "`-` … means that while the job is loaded it is currently not running"; col2 — "0 … finished successfully, a positive number … reported an error, a negative number … terminated … received a signal." "Only when RunAtLoad or KeepAlive have been specified … launchd will start the job unconditionally." StartInterval/StartCalendarInterval coalesce on wake from sleep. |
| 3 | https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-06-01 | official doc | WebFetch (full) | `misfire_grace_time` = seconds-late a job may still fire; `coalesce` default **False** (collapse missed runs to one, no misfire events); `max_instances` default **1** (next run is a misfire if prior still running). **MemoryJobStore: "Jobs exist only during the application's runtime. They are lost upon process restart."** `get_jobs()` returns Job instances. |
| 4 | https://docs.slack.dev/apis/web-api/rate-limits/ | 2026-06-01 | official doc | WebFetch (full) | "apps may post **no more than one message per second per channel**"; short bursts allowed. HTTP 429 returns `Retry-After` (seconds) — wait then retry. `chat.postMessage` is "Special Tier". No 2026 change affects `chat.postMessage` (the May-2025 / Mar-2026 tightening targets `conversations.history`). |
| 5 | https://docs.slack.dev/reference/methods/chat.postMessage/ | 2026-06-01 | official doc | WebFetch (full) | Needs `chat:write`. Success → `"ok": true` + `ts`. Errors: `channel_not_found`, `not_in_channel`, `missing_scope`, `token_revoked`. **When using blocks, include the top-level `text` fallback** — "used as a fallback string to display in notifications" + read by screen readers. Implement exponential backoff on `rate_limited`. |
| 6 | https://medium.com/@kinjaldand/your-cron-job-didnt-crash-it-vanished-...-08b4d46d912c | 2026-06-01 | practitioner blog | WebFetch (full) | "The absence of an error is not the same as the presence of success." A vanished job logs nothing, dashboard stays green. Invert monitoring: push heartbeat **only on success** (`work && curl ping`), external watcher alerts on absence. GitLab backup incident: 4/5 silent failures for months. |
| 7 | https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html | 2026-06-01 | official doc (Apple) | WebFetch (full) | **The 10-second throttle rule (verbatim):** "If your daemon shuts down too quickly after being launched, launchd may think it has crashed. Daemons that continue this behavior may be **suspended and not launched again** … do not shut down for at least 10 seconds after launch." StartCalendarInterval missing keys = wildcard (cron-like). KeepAlive subkeys point to launchd.plist(5). |
| 8 | https://incident.io/blog/on-call-best-practices-guide-2026 | 2026-06-01 | authoritative blog (vendor SRE) | WebFetch (full) | A handoff/status must carry **active incidents (status+next steps+severity), silenced alerts + upcoming risky changes, specific runbook/dashboard URLs** — "Handoffs fail when they rely on memory." Classify every alert **actionable / informational / noise**; Google SRE benchmark **2-3 actionable/shift**; group cascading alerts into ONE thread. |
| 9 | https://www.watchflow.io/blog/why-cron-jobs-fail-silently/ | 2026-06-01 | practitioner blog | WebFetch (full) | "Silent Failures: nothing crashes, nobody gets paged, and you only notice when data is missing." **"Logs are an internal signal. They often fail together with the system that's supposed to produce them"** — so logs ≠ monitoring. Daily job: interval 24h + **grace 30-60 min**. Send start/success/fail pings; payload catches "ran, but wrong." |

## 2. External sources — snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://healthchecks.io/docs/ | doc | Index; specifics covered by source 1 |
| https://github.com/Kriss-V/deadmancheck | repo | "alerts when jobs run but do nothing" — corroborates source 6/9 "ran but wrong"; snippet sufficient |
| https://oneuptime.com/blog/post/2026-03-02-how-to-monitor-cron-job-execution-and-alerting-on-ubuntu/view | blog (2026) | Ubuntu-cron specific; pattern identical to source 1 |
| https://www.manpagez.com/man/5/launchd.plist/ | man page | launchd.plist(5); KeepAlive subkeys; corroborates sources 2/7 |
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | man page | launchd.plist(5) mirror; snippet |
| https://github.com/tjluoma/launchd-keepalive | repo | KeepAlive plist examples incl SuccessfulExit; corroborates source 7 |
| https://developer.apple.com/.../CreatingLaunchdJobs (KeepAlive subkeys) | doc | Subkeys explicitly deferred to launchd.plist(5) by source 7 |
| https://apscheduler.readthedocs.io/en/3.x/modules/job.html | doc | `Job.next_run_time` attribute confirmation; snippet |
| https://github.com/agronholm/apscheduler/issues/296 | issue | max_instances → EVENT_JOB_MISSED interaction; corroborates source 3 |
| https://docs.slack.dev/apis/web-api/rate-limits (Special Tier detail) | doc | Already read in full (source 4) |
| https://medium.com/slack-developer-blog/handling-rate-limits-with-slacks-apis-f6f8a63bdbdc | blog | Retry-After handling worked example; corroborates source 4 |
| https://code.dblock.org/2026/03/12/ai-slop-a-slack-api-rate-limiting-disaster.html | blog (2026) | Cautionary: naive retry storms; reinforces backoff in source 5 |
| https://sre.google/workbook/on-call/ | book (Google) | The 2-3 actionable/shift benchmark origin cited by source 8 |
| https://rootly.com/sre/devops-on-call-tools-that-cut-alert-fatigue-in-2025 | blog (2025) | Alert-fatigue tooling; corroborates source 8 |
| https://devops.com/the-end-of-alert-fatigue-...-2026/ | article (2026) | 500-1200 alerts/day stat; reinforces signal-vs-noise |
| https://uptimelabs.io/learn/reduce-on-call-burnout/ | blog | Structural noise reduction; snippet |
| https://www.watchflow.io/blog/why-cron-jobs-fail-silently/ (already full) | — | (listed in read-in-full) |

**Unique URLs collected: 25** (9 read-in-full + 16 snippet-only). Floor is 10.

## 3. Search-query variants run (3 per topic)

- **Topic 1 (heartbeat/dead-man's-switch):** current-year → "...detect cron stopped firing Healthchecks.io" (2026 OneUptime hit); last-2-year → covered by watchflow 2026 + dblock 2026; **year-less canonical** → "cron job monitoring heartbeat dead man switch" (surfaced Healthchecks.io docs, Dead Man's Snitch, the canonical Medium piece).
- **Topic 2 (launchd):** current-year → "launchctl print ... 2026"; last-2-year → "macOS launchd agent crashed not restarted KeepAlive 2025/unattended" (Apple dev doc + tjluoma KeepAlive repo); **year-less canonical** → launchd.info + launchd.plist(5) man pages.
- **Topic 3 (APScheduler):** **year-less canonical** → the official 3.x user guide (version-pinned, not year-locked) + job module doc + issue #296. (APScheduler is a stable lib; a year suffix adds noise — flagged per research-gate rule.)
- **Topic 4 (Slack + SRE handoff):** current-year → "Slack chat.postMessage rate limits ... 2026" (official rate-limits + the 2026 disaster post-mortem); last-2-year → "SRE on-call handoff daily status alert fatigue 2025" (incident.io 2026 guide, Rootly 2025, Catchpoint 2025 stat); **year-less canonical** → Slack official method/rate-limit docs + Google SRE Workbook on-call chapter.

## 4. Recency scan (last 2 years, 2024–2026) — PERFORMED

Findings in the 2024-2026 window:
1. **Slack rate-limit tightening (eff. 2025-05-29 / 2026-03-03)** — applies to
   `conversations.history` / `conversations.replies` for **newly-created
   non-Marketplace apps**, NOT `chat.postMessage`. So pyfinagent's digest/alert
   posting is **unaffected**; the 1-msg/sec/channel limit is unchanged. (source 4)
2. **incident.io On-Call Best Practices 2026** (source 8) and **Catchpoint SRE
   Report 2025** (snippet): ~70% of SREs report on-call burnout; 500-1200
   alerts/day typical. Reinforces the away-week design imperative: the digest
   must be **signal, not noise** — the watchdog already does state-transition
   gating (alert only on change), which matches the 2026 guidance exactly.
3. **watchflow "Why Cron Jobs Fail Silently" (2026)** + **OneUptime (Mar 2026)**
   (sources 9 / snippet): both re-confirm the dead-man's-switch is still the
   2026 state of the art for catching vanished schedulers, and add the
   "logs fail with the system" insight — directly relevant because pyfinagent's
   only away-week visibility today is Slack + on-disk logs.
4. **Apple "Creating Launch Daemons and Agents"** (source 7): still the official
   reference; the 10-second-throttle caveat is unchanged and is a real risk for
   any KeepAlive plist added in 54.2 (a fast-crashing slack_bot would get
   suspended by launchd).

No source CONTRADICTS the recommended design. The dead-man's-switch /
state-transition-alert / KeepAlive-with-throttle-guard patterns are all current.

## 5. Key external findings

1. **A "vanished" job is the hardest failure and needs an EXTERNAL, push-based
   dead-man's-switch — internal logs/dashboards cannot catch it** (sources 1, 6,
   9). "The absence of an error is not the same as the presence of success"
   (source 6); "Logs … often fail together with the system that's supposed to
   produce them" (source 9). The job pings on success (`work && curl …`); an
   independent watcher alerts when the ping is absent within `interval + grace`.
   This catches the exact failure pyfinagent fears: the slack_bot dying and
   taking the whole APScheduler layer (incl. the operator's only window) with it.

2. **launchd status semantics let you distinguish all three pyfinagent states
   deterministically** (source 2): `launchctl list` col1 `-` = loaded-but-not-
   running (idle interval job, e.g. mas-harness), a number = running; col2 `0` =
   clean, **positive = job reported an error (ablation/autoresearch=1)**, negative
   = killed by signal (backend=-15 = a prior SIGTERM, now re-running under
   KeepAlive). So "ran-and-failed" vs "idle" vs "killed" are all readable from
   two columns — the basis for the repeatable audit method (§9).

3. **macOS launchd will COALESCE missed calendar/interval fires on wake, and a
   too-fast-crashing KeepAlive job gets SUSPENDED** (sources 2, 7). For the
   away week on a Mac that may sleep: StartCalendarInterval jobs (autoresearch
   02:00, ablation 03:00) fire once on wake rather than skipping — good. But any
   KeepAlive plist added for the slack_bot in 54.2 MUST honor Apple's rule: "do
   not shut down for at least 10 seconds after launch," else launchd throttles
   it. Mitigation: `ThrottleInterval` (backend uses 5) + ensure the bot doesn't
   exit-on-import-error within 10s.

4. **APScheduler's MemoryJobStore loses ALL jobs on process restart** (source 3).
   pyfinagent's scheduler uses the default in-memory store (confirmed in code),
   so a slack_bot restart re-seeds from `start_scheduler()` — fine for the cron
   schedule, BUT any one-shot/catch-up state is lost. The code already mitigates
   the one job where this matters (daily_price_refresh catch-up-on-start,
   scheduler.py:297-314). `coalesce=True` + `misfire_grace_time` (set on the
   phase-9 jobs) prevent a restart from stale-firing a missed tick.

5. **Slack `chat.postMessage` is rate-safe for a daily digest and a state-gated
   watchdog, but blocks-messages MUST set the `text` fallback** (sources 4, 5).
   1 msg/sec/channel is far above pyfinagent's volume (2 digests + transition-only
   alerts). On 429, honor `Retry-After` with backoff. The digests already pass a
   `text=` fallback (scheduler.py:354/399) — compliant. Risk is only a retry
   storm if an alert loop misfires; the AlertDeduper + state-transition gating
   already prevent that.

6. **An away-week status digest should be signal-dense and actionable** (sources
   8): active state (NAV/P&L, open positions), anything silenced/abnormal, and
   concrete links — and should suppress steady-state noise (Google: 2-3
   actionable/shift). pyfinagent's watchdog already posts only on transitions;
   the daily digest is the "all-clear heartbeat." The away-week design should add
   a **cron-health line to the digest** (or a once-daily cron-health ping) so
   "all jobs green" is itself the dead-man's-switch the operator sees in Slack.

---

## 6. Internal cron inventory (launchd + APScheduler) — file:line anchored

### 6a. launchd jobs (`~/Library/LaunchAgents/com.pyfinagent.*`, non-.bak)

| Plist | Program | Schedule keys | live `launchctl list` | StdErr log | Status |
|-------|---------|---------------|------------------------|------------|--------|
| `com.pyfinagent.mas-harness` | `scripts/mas_harness/run_cycle.sh` | StartInterval 1800; RunAtLoad false | PID `-` / exit 0 | `handoff/mas-harness.launchd.log` (empty) | **HEALTHY (idle between 30-min fires)** |
| `com.pyfinagent.autoresearch` | `scripts/autoresearch/run_nightly.sh` → `run_memo.py` | StartCalendarInterval 02:00; RunAtLoad false | not running / **exit 1** | `handoff/autoresearch.log` | **FAILED (paper_markets parse)** |
| `com.pyfinagent.ablation` | inline bash → `scripts/ablation/run_ablation.py --next-untested` | StartCalendarInterval 03:00; RunAtLoad false | not running / **exit 1** | `handoff/ablation.log` | **FAILED (paper_markets parse)** |
| `com.pyfinagent.backend-watchdog` | `scripts/launchd/backend_watchdog.sh` | StartInterval 60; RunAtLoad true; ProcessType Background | PID `-` / exit 0 | `handoff/logs/backend-watchdog.log` | HEALTHY (idle between 60s fires) |
| `com.pyfinagent.backend` | `caffeinate -i -s … uvicorn backend.main:app :8000` | KeepAlive true; RunAtLoad true; ThrottleInterval 5 | PID 36338 / exit **-15** | `backend.log` | RUNNING (was SIGTERM'd earlier; KeepAlive restarted it) |
| `com.pyfinagent.frontend` | `next dev --port 3000` | KeepAlive true; RunAtLoad true; ThrottleInterval 5 | PID 11636 / exit 0 | `frontend.log` | RUNNING |
| `com.pyfinagent.claude-code-proxy` | `node ~/.openclaw/claude-code-proxy.js` | KeepAlive{SuccessfulExit false}; RunAtLoad true | PID 1269 / exit 0 | `~/.openclaw/logs/claude-code-proxy.{log,err}` | RUNNING |

Notes:
- `.bak-harness-ABCD` / `.bak` copies exist for mas-harness/autoresearch/ablation/backend — **not loaded**, ignore (per task scope).
- launchd col2 semantics per source 2: `1` = job reported an error; `-15` = killed by SIGTERM (signal 15); `-` PID = loaded-not-running.

### 6b. APScheduler in-process jobs

**Process A — "main" scheduler** (in the backend uvicorn process, PID 36338),
registered at `backend/main.py:262` via `init_scheduler(scheduler)` +
`_register_cron_scheduler("main", scheduler)` (`main.py:263`):

| Job id | Trigger | next-fire (live) | Status (live) |
|--------|---------|------------------|---------------|
| `paper_trading_daily` | cron (daily trade cycle; id at `paper_trading.py:38`) | 2026-06-01T14:00-04:00 | scheduled (HEALTHY) |
| `ticket_queue_process_batch` | interval (`main.py:309`) | 2026-06-01T15:58+02:00 | scheduled (HEALTHY) |

**Process B — slack_bot scheduler** (in `python -m backend.slack_bot.app`, PID
42151), `start_scheduler()` at `scheduler.py:187`; 4 core jobs + 7 phase-9 jobs
(`register_phase9_jobs` at `scheduler.py:784`, mapping at `:856-871`):

| Job id | Trigger (scheduler.py) | grace/coalesce | last_run (live) | next_run (live) | Status |
|--------|------------------------|----------------|-----------------|-----------------|--------|
| `morning_digest` | cron 08:00 ET (`:199`) | — | 2026-06-01T12:00 UTC | 2026-06-02T08:00 ET | **ok (delivered)** |
| `evening_digest` | cron 17:00 ET (`:211`) | — | None (not fired today yet) | None* | registered; fires 17:00 ET |
| `watchdog_health_check` | interval 15 min (`:223`) | — | 2026-06-01T13:50 UTC | +15 min | ok |
| `prompt_leak_redteam` | cron 03:15 ET (`:235`) | — | 2026-06-01T07:15 UTC | 2026-06-02T03:15 ET | ok |
| `daily_price_refresh` | cron 01:00 UTC (`:858`) | 21600s / True | 2026-06-01T01:00 UTC | 2026-06-02T01:00 UTC | ok |
| `weekly_fred_refresh` | cron Sun 02:00 UTC (`:860`) | 7200s / True | None | None* | registered; fires Sunday |
| `nightly_mda_retrain` | cron 03:00 UTC (`:862`) | 3600s / True | 2026-06-01T03:00 UTC | 2026-06-02T03:00 UTC | ok |
| `hourly_signal_warmup` | cron :05 UTC (`:864`) | 600s / True | 2026-06-01T13:05 UTC | 2026-06-01T14:05 UTC | ok |
| `nightly_outcome_rebuild` | cron 04:00 UTC (`:866`) | 3600s / True | 2026-06-01T04:00 UTC | 2026-06-02T04:00 UTC | ok |
| `weekly_data_integrity` | cron Mon 05:00 UTC (`:868`) | 7200s / True | 2026-06-01T05:00 UTC | 2026-06-08T05:00 UTC | ok |
| `cost_budget_watcher` | cron 06:00 UTC (`:870`) | 3600s / True | 2026-06-01T06:00 UTC | 2026-06-02T06:00 UTC | ok |
| `daily_price_refresh_catchup` | one-shot +20s on start (`:302`) | 3600s | (per-restart) | — | catch-up only |

\* `next_run=None` for evening_digest/weekly_fred is a **registry artifact**, not a
fault: the heartbeat registry only stores `next_run` after a job's first fire or
the on-start seed (`_seed_next_run_registry`, `:157`), and it resets when the
slack_bot restarts. Both jobs ARE registered in `start_scheduler`/`register_phase9_jobs`
and will fire at their cron time. Delivery history in `handoff/logs/slack_bot.log`
shows evening_digest firing daily through 2026-05-27.

### 6c. Process / liveness snapshot (live, read-only)

- **Slack bot:** RUNNING — PID 42151, `python -m backend.slack_bot.app`, **PPID 1
  (orphaned to launchd), NO launchd plist** (`launchctl list | grep slack` → none).
  Started manually ~Thu; stdout/stderr → `backend_slack.log` (live), NOT the stale
  `handoff/logs/slack_bot.log` (last write 2026-05-27 20:44 — log rotated/changed on
  the restart). `start_scheduler(app)` at `app.py:56`; `asyncio.run(main())` at `:77`.
- **Backend :8000:** RUNNING — PID 36338 under `caffeinate`, KeepAlive (auto-restart).
- **Frontend :3000:** RUNNING — PID 11636, KeepAlive.
- **What drives the digests if there's no slack launchd job?** The digests +
  watchdog + all 11 phase-9/core jobs run **inside the orphaned slack_bot process
  (PID 42151)**. There is **no supervisor** for it — the away-week single point of
  failure.

## 7. Root-cause: unhealthy jobs

### autoresearch (exit 1) + ablation (exit 1) — SAME root cause
Both launchd wrappers do `set -a; . backend/.env; set +a` (run_nightly.sh body;
ablation inline `&& set -a && . backend/.env && set +a`). The 2026-06-01
multi-market go-live wrote `PAPER_MARKETS` to `.env` in JSON form. `paper_markets`
is declared `list[str] = Field(default_factory=lambda: ["US"])` at
`backend/config/settings.py:55`. pydantic-settings treats `list[str]` as a
**complex field** and `json.loads()`-es it from the OS env. When bash *sources* a
`.env` line like `PAPER_MARKETS=["US","EU","KR"]`, the unquoted brackets/quotes are
mangled by shell word-splitting/globbing, so the env var that reaches Python is
non-JSON → `json.decoder.JSONDecodeError: Expecting value: line 1 column 2
(char 1)` → `pydantic_settings.exceptions.SettingsError: error parsing value for
field "paper_markets"`. Verbatim from `handoff/autoresearch.log` (2026-06-01
02:00, rc=1) and `handoff/ablation.log` (same trace, `run_ablation.py:305
get_settings()`).

**Proof it is the shell-source path, not the value itself:** loading `Settings()`
directly (no shell source — the uvicorn path) succeeds RIGHT NOW:
`paper_markets = ['US', 'EU', 'KR']`. So the `.env` value is JSON-valid; only the
`set -a; . backend/.env` shell re-export corrupts it. Reproduced:
`json.loads("['US','EU','KR']")` → fails at char 1 (matches the live error);
`json.loads('["US","EU","KR"]')` → OK. The mangling turns the JSON into the
single-quote/bare form that fails.

**Fix (NON-LLM, non-destructive, ESCALATED — touches the shared .env path):**
Do NOT re-source `.env` blindly. Lowest-risk options (operator picks one):
- (a) In both cron wrappers, **unset PAPER_MARKETS after sourcing** (`. backend/.env;
  unset PAPER_MARKETS`) so pydantic falls back to its native `.env` read /
  `default_factory=["US"]`. Ablation/autoresearch don't need multi-market.
- (b) Add a `field_validator(mode="before")` on `paper_markets` in settings.py that
  accepts the comma form (`"US,EU,KR".split(",")`) so BOTH the dotenv and env
  sources parse — the robust, permanent fix (also future-proofs any other
  shell-sourcing caller). ~8 lines, no new dep, default-safe.
- (c) Quote the value in `.env` as a single JSON token AND switch wrappers to
  `export PAPER_MARKETS='["US","EU","KR"]'` — brittle (still shell-quoting-fragile).
Recommend **(b)** (permanent, covers all callers) or **(a)** (zero-settings-risk).
Escalated because it edits a shared secrets-adjacent file / shared settings model.

NOTE (secondary, latent): `run_memo.py` (autoresearch) has a documented
huggingface-import dependency issue (auto-memory `project_cron_maintenance_jobs`,
phase-51.4). It is NOT the active blocker — the crash happens earlier, at
`get_settings()` (`run_memo.py:152` → `model_tiers.py:131` → `settings.py:469`),
before any HF import. After the paper_markets fix, re-verify autoresearch for the
HF issue separately.

### mas-harness (PID `-`) — NOT a failure
`StartInterval 1800` + `RunAtLoad false`, last-exit 0. Per source 2, a `-` PID with
exit 0 = "loaded, not currently running" — i.e. cleanly idle between 30-minute
fires. `/api/jobs/all` reports it `ok`. No action. (If the operator wants it to run
DURING the away week for autonomous cycles, that is a separate go/no-go, not a
health defect.)

## 8. Digest: LLM-backed or template/data-only? — DECISIVE: TEMPLATE/DATA-ONLY ($0 LLM)

`backend/slack_bot/formatters.py` imports only `math` + `datetime` (`:6-7`) — no
`llm_client`, no anthropic/openai/gemini, no `generate`/`complete`.
`format_morning_digest(portfolio_data, recent_reports)` (`:323`) and
`format_evening_digest(portfolio_data, trades_today)` (`:391`) are pure Block Kit
builders that read dict fields from two backend HTTP GETs
(`/api/paper-trading/portfolio`, `/api/reports`, `/api/paper-trading/trades`) and
format numbers/strings. **No token spend.**

**Implication for 54.2:** the daily operator digest is **NOT operator-gated** (LLM
spend is the only gated axis per active_goal.md). It can run freely for the entire
away week. The only spend-gated cron is the autonomous trade cycle / any
LLM-routine — separate from the digest. The digest is the safe, free, always-on
"all-clear heartbeat" the operator should rely on.

## 9. Recommended audit method (repeatable cross-layer check)

A repeatable "are all my jobs healthy?" check has three legs (all $0, read-only):

1. **launchd leg** — `launchctl list | grep com.pyfinagent` → parse col1 (PID:
   number=running, `-`=loaded/idle) + col2 (0=clean, >0=errored, <0=signal-killed)
   per source 2. Flag any `>0` exit, and any job that SHOULD be a long-running
   daemon (KeepAlive/RunAtLoad) showing PID `-`. For interval jobs, PID `-` is
   normal. Optionally `launchctl print gui/$(id -u)/com.pyfinagent.<job>` for the
   detailed JSON-like view incl. `last exit code` + `runs`.
2. **APScheduler leg** — `GET /api/jobs/all` (already built, `cron_dashboard_api.py:410`)
   merges the "main" + slack_bot schedulers' `get_jobs()` + the heartbeat registry.
   For each job assert: `status != failed`, `next_run` is non-null and in the future
   (or a known cron-day), and `last_run` is within `interval + grace` of now (the
   dead-man's-switch test). The `never_run`/`next_run=None` rows need the
   firing-history cross-check (slack_bot.log) to avoid false alarms (see §6b note).
3. **Liveness leg** — confirm the three host processes are up (backend :8000 health
   200, frontend :3000, **slack_bot PID present**) AND that the slack_bot's heartbeats
   in `/api/jobs/all` are fresh (its jobs' `last_run` advancing) — the only way to
   detect the orphaned-bot-died failure from inside the system.

The artifact `handoff/current/live_check_54.1.md` is exactly this three-leg table
(see that file). This method is the basis for an automated daily cron-health check
in 54.2.

## 10. Recommended away-week monitoring design (heartbeat / dead-man's-switch fit)

Ranked by leverage for the 2026-06-01 → 06-08 window:

1. **Supervise the slack_bot with launchd (highest priority).** Add
   `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` with `RunAtLoad true` +
   `KeepAlive true` (mirroring the backend plist) so the bot auto-restarts on
   crash/sleep-wake. HONOR Apple's 10-second rule (source 7): set `ThrottleInterval`
   (backend uses 5) and ensure the bot doesn't exit-on-bad-config within 10s, or
   launchd will suspend it. This closes the single point of failure that would
   blind the operator. (Adding a launchd plist = `launchctl load` = operator-gated
   per the masterplan; ESCALATE the plist + load command, don't force.)
2. **External dead-man's-switch on the digest (the operator's eyes).** Per sources
   1/6/9, an internal check can't catch the host going fully down. Have the
   morning/evening digest (or a tiny daily cron) `curl` a free Healthchecks.io /
   Cronitor / Dead Man's Snitch ping **on success**; configure interval 24h +
   30-60 min grace; route the absence-alert to email/SMS (a channel independent of
   Slack + the Mac). Then even "Mac asleep / slack_bot dead / no Slack" still
   reaches the operator. (Pip/account = operator-gated; the curl-ping itself is
   trivial and free — ESCALATE the choice of service.)
3. **Fold a cron-health line into the daily digest.** Per source 8, the digest is
   the away-week handoff: add one line summarizing `/api/jobs/all` (e.g. "Crons:
   17/19 green; FAILED: autoresearch, ablation"). That makes "all green" the visible
   all-clear, and surfaces any newly-failed job in the channel the operator watches —
   without paging noise (state-gated, like the existing watchdog).
4. **Keep the existing state-transition gating** (watchdog posts only on
   None→False/True→False/False→True, `scheduler.py:437-462`; AlertDeduper
   fingerprint `type:endpoint`, `:49`). This already matches the 2026 anti-fatigue
   guidance (source 8: actionable-only, group cascades). Do not add steady-state
   spam.
5. **Fix the two failed crons before the operator leaves** (§7) so the away-week
   baseline is all-green; otherwise the daily digest's cron-health line will cry
   wolf at 02:00/03:00 every night.

## 11. Application to pyfinagent (external findings → file:line anchors)

| External finding | pyfinagent anchor / action |
|---|---|
| Dead-man's-switch catches vanished schedulers (src 1,6,9) | slack_bot PID 42151 is orphaned (no plist) → add KeepAlive plist + external heartbeat on digest (`scheduler.py:351`/`:396` post sites) |
| launchctl col1/col2 semantics (src 2) | audit method §9 leg 1; explains ablation/autoresearch exit 1, backend exit -15, mas-harness `-` |
| Apple 10-sec throttle rule (src 7) | new slack-bot plist must set ThrottleInterval + not fast-crash on bad config (settings.py:55 parse must not abort <10s) |
| MemoryJobStore loses jobs on restart (src 3) | slack_bot uses default in-memory store; `daily_price_refresh_catchup` (`scheduler.py:302`) is the only restart-survival path; coalesce+grace on phase-9 jobs (`:856-871`) prevent stale fires |
| chat.postMessage 1/sec + text fallback (src 4,5) | digests already pass `text=` (`scheduler.py:354`,`:399`); volume is safe; keep AlertDeduper to avoid 429 storms |
| Away-status = signal not noise, 2-3 actionable/shift (src 8) | watchdog state-gating (`:437-462`) already compliant; add cron-health line to digest (`format_*_digest`, `formatters.py:323`/`:391`); digest is $0 (template-only) |

---

## 12. Research Gate Checklist

Hard blockers — `gate_passed` false if any unchecked:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (**9** read in full; official Apple/Slack/APScheduler/Healthchecks docs lead the set)
- [x] 10+ unique URLs total incl. snippet-only (**25** collected)
- [x] Recency scan (last 2 years) performed + reported (§4)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (settings.py:55; scheduler.py:56/187/199/211/223/235/297-314/351/396/437-462/784/856-871; main.py:262-263/309; paper_trading.py:38; cron_dashboard_api.py:410/85; formatters.py:6-7/323/391; app.py:56/77; plist paths + launchctl/ps/lsof/`/api/jobs/all` live output)

Soft checks:
- [x] Internal exploration covered every relevant module (7 launchd plists, scheduler.py full, jobs/* inventory, formatters digest fns, cron_dashboard_api, settings fields, main.py lifespan, live launchctl/ps/lsof/`/api/jobs/all`)
- [x] Contradictions / consensus noted (no external contradiction; the "mas-harness not running" alarm in the prompt is REFUTED as a false positive via launchd semantics)
- [x] All claims cited per-claim (URL for external, file:line / command output for internal)

---

```json
{"tier":"moderate","external_sources_read_in_full":9,"snippet_only_sources":16,"urls_collected":25,"recency_scan_performed":true,"internal_files_inspected":12,"report_md":"handoff/current/research_brief.md","gate_passed":true}
```
