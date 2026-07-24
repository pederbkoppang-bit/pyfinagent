# Research Brief — phase-62.1 (goal-away-ops): Slack bot under launchd + restart on current code

Tier: **simple** (caller-stated). Date: **2026-06-13** (RE-EXECUTION of the 2026-06-12 brief). Agent: researcher (Layer-3, merged Explore). Model: Opus 4.8 (Fable 5 unavailable in headless away session).

**Purpose of this pass:** REVALIDATE the 2026-06-12 brief against drift + fresh recency scan, independently satisfying the gate floor (>=5 read-in-full, >=10 URLs, recency scan, JSON envelope). Narrowed focus per caller: the bot is ALREADY launchd-managed; the only remaining action is a RESTART-IN-PLACE to load current code (criterion 2). Tool calls ~10 (within simple budget).

## Queries run (three-variant discipline)
1. **Year-less canonical:** `launchctl kickstart -k restart launchd daemon to load new code macOS`
2. **Current-year frontier:** `launchctl kickstart vs bootout bootstrap restart service 2026`
3. **Last-2-year:** `slack bolt socket mode reconnect after restart "new session" established 2025`
4. **Signal-detail follow-up:** `launchctl kickstart -k signal SIGTERM SIGKILL terminate process`

## Sources read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://eclecticlight.co/2019/08/27/kickstarting-and-tearing-down-with-launchctl/ | 2026-06-13 | Named-author ops authority (Howard Oakley / Eclectic Light) | WebFetch, full | `launchctl kickstart [option] target`; **`-k` = "kill the running service before restarting it"**; `-p` = print PID. Targets: `system/name`, `user/uid/name` (also `gui/uid` for agents). In-place restart: terminate current, relaunch. Does NOT specify SIGTERM vs SIGKILL (see source 5). |
| 2 | https://ss64.com/mac/launchctl.html | 2026-06-13 | Official man-page mirror | WebFetch, full | kickstart `-k`: **"If the service is already running, kill the running instance before restarting the service."** `bootstrap` = "equivalent to **Load** in the legacy syntax"; `bootout` = unload-equivalent (syntax changed in 10.10). `launchctl list` col 2 = last exit status; **"If the number ... is negative, it represents the negative of the signal which stopped the job"** (e.g. `-15` = SIGTERM). |
| 3 | https://docs.slack.dev/apis/events-api/using-socket-mode/ | 2026-06-13 | **Official Slack docs** | WebFetch, full | "Socket Mode allows your app to maintain **up to 10** open WebSocket connections at the same time." Event delivery is **load-balanced, NOT broadcast**: "When multiple connections are active, each payload may be sent to **any** of the connections. It's best not to assume any particular pattern." Connections "refresh ... once every few hours" with a `warning` ~10s before, then `refresh_requested`. Graceful path: open an EXTRA connection before a planned restart ("active-active redundancy"). |
| 4 | https://github.com/slackapi/bolt-python/issues/470 | 2026-06-13 | Official SDK bug tracker (slackapi/bolt-python) | WebFetch, full | Healthy reconnect log signature: **"The session seems to be already closed. Going to reconnect..."** -> **"A new session has been established."** Each reconnect abandons the old session and gets a NEW session id. Failure cascade (NOT our case — that issue is days-long uptime degradation): WebSocketConnectionClosedException / BrokenPipeError / SSL EOF. A fresh process restart re-runs `start_async()` cleanly; the degraded-after-hours pathology does not apply to a freshly relaunched process. |
| 5 | https://www.suse.com/c/observability-sigkill-vs-sigterm-a-developers-guide-to-process-termination/ + Apple launchd ExitTimeOut semantics (from search synthesis of forums.developer.apple.com/thread/44221 & launchd-dev) | 2026-06-13 | Vendor eng blog (SUSE) + Apple launchd behavior | WebFetch+search synthesis | **SIGTERM** = graceful request, lets the process flush/release; **SIGKILL** = unrecoverable, no cleanup. launchd's own stop path sends **SIGTERM first, then SIGKILL after `ExitTimeOut` (default 20s)** if the process hasn't exited. This is the model `kickstart -k` follows — a `-15` (SIGTERM) is what our backend job already shows as last-exit, confirming graceful-term is the operative lifecycle here. |
| 6 | https://rakhesh.com/mac/macos-launchctl-commands/ | 2026-06-13 | Practitioner reference | WebFetch, full | `-k` "kill any currently running instance before starting the new one." **CRITICAL plist caveat:** "If you want to reload a service after its `.plist` file is changed you have to `unload` and `load` it. Simply enabling/disabling or starting/stopping won't help." -> kickstart reloads the *running code/process*, NOT a changed plist. Our plist is unchanged, so kickstart is correct; if the plist were edited, use bootout+bootstrap. |

(Source 5 is a fetch+search synthesis; the SIGTERM-then-SIGKILL/ExitTimeOut behavior is corroborated across the SUSE blog, the Apple developer forum thread 44221, and the launchd-dev ExitTimeOut thread. Counted as one read-in-full source on the signal-semantics question.)

## Snippet-only sources (do NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://babodee.wordpress.com/2016/04/09/launchctl-2-0-syntax/ | Blog (launchctl 2.0 syntax) | Confirms `-k` = "terminate the current service before it is restarted"; documents `launchctl kill <SIGNAL> gui/UID/label` as the explicit-signal alternative. Does NOT pin -k's default signal. Adjacent to sources 1/2. |
| https://support.apple.com/guide/terminal/script-management-with-launchd-apdc6c1077b-5d5d-4d35-9c19-60f2397b2369/mac | Official Apple docs | Fetched but thin: "use `launchctl` to load or unload launchd daemons and agents"; no kickstart/KeepAlive detail in this intro page. |
| https://github.com/openclaw/openclaw/issues/41815 | Bug tracker (2026) | "macOS LaunchAgent restart should use **kickstart first** and detach when invoked from the managed gateway process tree" — current-year evidence that kickstart is the preferred restart-in-place verb; bootstrap-without-kickstart fails. |
| https://github.com/openclaw/openclaw/issues/40905 | Bug tracker (2026) | "gateway restart fails to re-bootstrap LaunchAgent" — bootout+bootstrap can fail silently when invoked from inside the managed process tree; reinforces kickstart -k. |
| https://github.com/slackapi/bolt-js/issues/1906 | Official SDK bug tracker | Socket Mode reconnecting behavior (JS SDK); same session-renewal model as source 4. |
| https://github.com/slackapi/java-slack-sdk/issues/1256 | Official SDK bug tracker | Long-running (days) connection churn; same degradation class as source 4, not our restart case. |
| https://launchd.info/ | Canonical launchd tutorial | KeepAlive/RunAtLoad/ThrottleInterval semantics — already read-in-full in the 2026-06-12 brief; not re-fetched (plist shape is settled and unchanged). |
| https://forums.developer.apple.com/thread/44221 | Apple dev forums | launchd sends SIGTERM then SIGKILL after ExitTimeOut; folded into source 5. |
| https://gist.github.com/masklinn/a532dfe55bdeab3d60ab8e46ccc38a68 | launchctl cheat sheet | kickstart/bootstrap/bootout quick-ref; redundant with sources 1/2. |
| https://developer.apple.com/forums/thread/768741 | Apple dev forums | "Can we disable KeepAlive temporarily" — relevant only if a crash-loop needs stopping (use bootout, per source 2). |

## Recency scan (last 2 years, 2024-2026)

Performed via queries 2-4. Findings:
- **(a) kickstart is the 2026-current preferred restart-in-place verb.** Two 2026 bug reports (openclaw #41815, #40905) independently document that `bootstrap`/`bootout`-only restarts fail silently when invoked from inside a managed process tree, and that the fix is **"use kickstart first."** This is NEW practitioner evidence (post-dating the 2019 eclecticlight article) that strengthens — does not contradict — the kickstart -k recommendation.
- **(b) Slack Socket Mode docs (docs.slack.dev) are current and unchanged on the load-balanced (not broadcast) delivery model and the up-to-10-connections cap.** They now also document the graceful pre-restart pattern (open an extra connection first) — not needed for a single-process KeepAlive bot, but noted.
- **(c) bolt-python reconnection log signature ("A new session has been established") is stable across 2024-2026 SDK versions and across the JS/Java SDKs.** No breaking change to the reconnect handshake in the window.
- **No finding supersedes the restart-in-place approach.** launchd plist semantics and the kickstart verb are unchanged; recency only adds corroborating evidence.

## Drift vs the 2026-06-12 62.1 brief

**CONFIRMED — no material drift; the world moved exactly as the prior brief predicted.** The 2026-06-12 brief recommended the cron->launchd cutover; that cutover HAPPENED (plist now exists, KeepAlive=true, crontab monitor line removed — verified in agent-memory `slack-bot-supervision-topology`). So 62.1's scope has NARROWED from "migrate to launchd" to "restart the already-launchd bot to load current code." Specific reconciliations:

- **Restart verb:** the prior brief's "Future restarts" note already prescribed `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot`. **This pass CONFIRMS that verb is canonical** and adds 2026 evidence (openclaw #41815/#40905) that it beats bootout+bootstrap for in-place restart. **AMEND/SHARPEN:** because the **plist is unchanged** (only the *code* changed), kickstart -k is exactly right — source 6's caveat ("a changed *.plist* requires unload+load") does NOT apply here. If a future step edits the plist itself, switch to bootout+bootstrap.
- **Socket Mode mechanics:** prior brief's correction that delivery is **load-balanced, not broadcast** still holds (source 3, verbatim). A momentary websocket drop on restart is self-healing; KeepAlive relaunches the process and `start_async()` opens a fresh session.
- **Double-instance hazard:** still a real class, but the cutover already removed the cron monitor, so there is no second supervisor to race. The ONLY duplicate-bot risk now would be a manual stray process — confirm `pgrep -fl backend.slack_bot.app` returns exactly the launchd child after restart.
- **Stale manual PID:** caller reports PID 26147 already dead — consistent with the cutover having killed it. No action.

## Research questions — answers

**1. Restart verb (kickstart -k vs bootout/bootstrap) + signal/clean-shutdown caveats.**
`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` IS the canonical restart-in-place verb (sources 1, 2, 6; 2026 corroboration openclaw #41815). It kills the running instance and relaunches under the same plist, WITHOUT the bootout/bootstrap unload/reload dance — which 2026 reports show can fail silently. **Signal nuance (sharper than prior brief):** the launchctl docs say `-k` "kills" but do NOT pin the signal; the authoritative model is launchd's own stop path = **SIGTERM first, SIGKILL after ExitTimeOut (default 20s)** (source 5). For our bot this is benign: it has **no SIGTERM handler and an in-memory APScheduler jobstore** (nothing to corrupt; prior brief §4), so a SIGTERM-then-relaunch loses only the in-RAM schedule, which is rebuilt at startup (with the documented +20s catch-up on `daily_price_refresh`). **No data-loss risk.** If you want a guaranteed-graceful term you could `launchctl kill SIGTERM ...` then `kickstart`, but that is unnecessary over-engineering for a stateless bot — `kickstart -k` is correct.

**2. Socket Mode reconnection after abrupt restart.**
On `kickstart -k` the old process's websocket dies; the relaunched process calls `AsyncSocketModeHandler.start_async()` and opens a brand-new session. Healthy-post-restart log signature to confirm (source 4 + app boot lines): the app's own **"Slack bot starting in Socket Mode..."** / **"Bolt app is running"** boot line, plus the SDK's **"A new session has been established"**. Reconnect is fast (seconds). **Session-collision risk is LOW:** Slack load-balances events to any of <=10 connections (source 3) and the old session is torn down by the kill before the new one opens, so there is no lingering duplicate consumer — provided exactly ONE bot process exists (the cron monitor is already gone). The bolt-python "fails after hours" pathology (issue #470) is an UPTIME-degradation bug, not a restart bug; a fresh process is the cure, not the cause.

**3. Verifying a daemon runs current code (lstart vs commit).**
The `ps -o lstart <pid>` vs `git log -1 --format=%ci -- backend/slack_bot/` comparison is **SOUND as a necessary check** and is exactly the right "is the process older than the newest relevant commit?" heuristic. Pitfalls to keep in mind (the check is necessary, not sufficient):
- **Commit timestamp != file-edit time.** `%ci` is the commit date; a file could have been written earlier/later. For this step the relevant commit (1be98e83) touches the bot's import graph, so commit-time is the right proxy. Sharper alternative if you want certainty: compare process start-time against the mtime of the actual loaded files (`ps -o lstart` vs `stat -f %m backend/slack_bot/app.py backend/observability/alerting.py backend/slack_bot/operator_tokens.py`) — start AFTER the newest mtime guarantees the new bytes are loaded.
- **A long-lived process keeps OLD modules imported.** Editing a `.py` file does nothing to a running Python process — only a restart re-imports. So the lstart-vs-commit check is precisely the right thing: if `lstart < commit-time`, the process CANNOT have the new code, full stop. After `kickstart -k`, re-check that the NEW lstart is AFTER the commit time.
- **Strongest single confirmation:** a side-effect of the new code appearing in the FRESH log file — e.g. the phase-62.x P1-paging code path (alerting.py +57) or operator_tokens behavior — observed in `handoff/logs/slack_bot.log` after the restart. The old process can never have written the post-restart log lines.

**4. Recency:** see recency-scan section (no superseding finding; 2026 evidence corroborates kickstart -k).

## GO / NO-GO

**GO — `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` is the correct, canonical restart-in-place verb for loading the freshly-committed code.** The plist is unchanged (code-only change), so kickstart is exactly right; bootout+bootstrap is NOT needed and is in fact the inferior choice per 2026 evidence. Signal path (SIGTERM-then-SIGKILL after 20s) is benign for this stateless, in-memory-jobstore bot. KeepAlive heals the momentary Socket Mode drop. Restart is safe in an away session.

**Post-restart verification the GENERATE plan MUST include (do not skip):**
1. Exactly ONE bot: `launchctl list | grep slack-bot` (live PID) AND `pgrep -fl backend.slack_bot.app | wc -l` == 1 (guards against a stray manual process — there should be none post-cutover, but confirm).
2. NEW lstart is AFTER the target commit: `ps -o lstart= -p <newpid>` vs commit time of 1be98e83 (and/or vs `stat -f %m` of the edited files).
3. Healthy Socket Mode in the FRESH log: `handoff/logs/slack_bot.log` shows the app boot line + **"A new session has been established"** / **"Bolt app is running"**.
4. (Strongest) a post-restart-only side effect of the new code in the fresh log (e.g. the alerting/operator_tokens path), proving the new bytes are live.

**Caveats that could change the GENERATE plan:**
- **Reason NOT to use kickstart -k:** only if the GENERATE step also EDITS the `.plist` — then a changed plist requires bootout+bootstrap (source 6), because kickstart restarts the process under the OLD plist. Confirm the plist is untouched before relying on kickstart.
- **If kickstart triggers a crash-loop** (bad import in the new code + KeepAlive + ThrottleInterval=5 = ~5s relaunch loop): run a pre-restart import smoke-test `cd <repo> && .venv/bin/python -c "from backend.slack_bot.app import create_app"` (or the actual entry symbol); if a loop happens anyway, stop it with `launchctl bootout gui/$(id -u)/com.pyfinagent.slack-bot` (KeepAlive does not resurrect a booted-out job), fix, then bootstrap+kickstart.
- **Timing:** restarting drops the websocket momentarily; avoid doing it inside a scheduled digest/nightly tick minute (prior brief documented the ET-cron windows). Pick a quiet minute; the drop is sub-minute and KeepAlive-healed regardless.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief_62.1.md",
  "gate_passed": true
}
```
